import hashlib
import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from functools import lru_cache

from ..models import (
    ActivityResult,
    ChainStructureSummary,
    CoordinateBounds,
    LowConfidenceRegion,
    ModelEvidence,
    PdbAtomRecord,
    PhysChemResult,
    ResidueConfidence,
    StructureResult,
    ToxicityResult,
)
from ..state import PeptideState

ESM_ATLAS_FOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
LOCAL_ESMFOLD_MODEL = os.getenv("PEPTIDE_HELPER_ESMFOLD_MODEL", "facebook/esmfold_v1")
ESMFOLD_BACKEND = os.getenv("PEPTIDE_HELPER_ESMFOLD_BACKEND", "auto").lower()
ESMFOLD_LOCAL_FILES_ONLY = os.getenv("PEPTIDE_HELPER_ESMFOLD_LOCAL_FILES_ONLY", "1") != "0"
MODEL_TIMEOUT_SECONDS = int(os.getenv("PEPTIDE_HELPER_MODEL_TIMEOUT", "120"))

TOXICITY_MODEL_SPECS = (
    {
        "name": "ToxinPred3",
        "task": "toxicity",
        "weight": 1.4,
        "source": "Computers in Biology and Medicine 2024; GitHub raghavagps/toxinpred3",
        "command_env": "PEPTIDE_HELPER_TOXINPRED3_CMD",
    },
    {
        "name": "ToxinPred2",
        "task": "toxicity",
        "weight": 1.3,
        "source": "Briefings in Bioinformatics 2022; GitHub raghavagps/toxinpred2",
        "command_env": "PEPTIDE_HELPER_TOXINPRED2_CMD",
    },
    {
        "name": "ToxIBTL",
        "task": "toxicity",
        "weight": 1.15,
        "source": "Bioinformatics 2022; GitHub WLYLab/ToxIBTL",
        "command_env": "PEPTIDE_HELPER_TOXIBTL_CMD",
    },
    {
        "name": "ToxTeller",
        "task": "toxicity",
        "weight": 1.0,
        "source": "ACS Omega 2024; GitHub comics-asiis/ToxicPeptidePrediction",
        "command_env": "PEPTIDE_HELPER_TOXTELLER_CMD",
    },
    {
        "name": "SafetyExtension",
        "task": "safety",
        "weight": 0.7,
        "source": "Optional allergenicity, immunogenicity, or hemolysis predictor",
        "command_env": "PEPTIDE_HELPER_SAFETY_EXTENSION_CMD",
    },
)

ACTIVITY_MODEL_SPECS = (
    {
        "name": "AMPScannerV2",
        "task": "antimicrobial_activity",
        "weight": 1.3,
        "source": "Bioinformatics 2018; GitHub dan-veltri/amp-scanner-v2",
        "command_env": "PEPTIDE_HELPER_AMPSCANNER_CMD",
    },
    {
        "name": "AMPlify",
        "task": "antimicrobial_activity",
        "weight": 1.3,
        "source": "BMC Genomics 2022; GitHub BirolLab/AMPlify",
        "command_env": "PEPTIDE_HELPER_AMPLIFY_CMD",
    },
    {
        "name": "amPEPpy",
        "task": "antimicrobial_activity",
        "weight": 1.0,
        "source": "Bioinformatics 2020; GitHub tlawrence3/amPEPpy",
        "command_env": "PEPTIDE_HELPER_AMPEPPY_CMD",
    },
    {
        "name": "iAMPCN",
        "task": "multi_activity",
        "weight": 1.1,
        "source": "Briefings in Bioinformatics 2023; GitHub joy50706/iAMPCN",
        "command_env": "PEPTIDE_HELPER_IAMPCN_CMD",
    },
    {
        "name": "Deep-AmPEP30",
        "task": "short_peptide_antimicrobial_activity",
        "weight": 0.9,
        "source": "Molecular Therapy - Nucleic Acids 2020",
        "command_env": "PEPTIDE_HELPER_DEEP_AMPEP30_CMD",
    },
)


def _log_agent(agent_name: str) -> None:
    """统一节点日志，便于后续接入结构化日志。"""

    print(f"[Agent] {agent_name} 执行中...")


def _unavailable_evidence(spec: dict, reason: str) -> ModelEvidence:
    return ModelEvidence(
        name=spec["name"],
        task=spec["task"],
        weight=spec["weight"],
        status="unavailable",
        source=spec["source"],
        error=reason,
    )


def _parse_model_payload(spec: dict, output: str) -> ModelEvidence:
    payload = json.loads(output)
    if not isinstance(payload, dict):
        raise ValueError("prediction payload must be an object")
    if "label" not in payload and "prediction" not in payload and "score" not in payload and "probability" not in payload:
        raise ValueError("missing prediction payload")
    label = str(payload.get("label", payload.get("prediction", "unknown"))).lower()
    score = payload.get("score", payload.get("probability"))
    return ModelEvidence(
        name=spec["name"],
        task=spec["task"],
        label=label,
        score=float(score) if score is not None else None,
        weight=spec["weight"],
        status="available",
        source=payload.get("source") or spec["source"],
    )


def _run_configured_model(spec: dict, sequence: str) -> ModelEvidence:
    command_template = os.getenv(spec["command_env"], "").strip()
    if not command_template:
        return _unavailable_evidence(spec, f"未配置 {spec['command_env']}")

    command = shlex.split(command_template.replace("{sequence}", sequence))
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        return _parse_model_payload(spec, completed.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return _unavailable_evidence(spec, "模型命令未输出有效 JSON")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return _unavailable_evidence(spec, exc.__class__.__name__)


def _toxicity_proxy(sequence: str) -> ModelEvidence:
    residues = Counter(sequence)
    hydrophobic = sum(residues[aa] for aa in "AILMFWVY")
    cationic = sum(residues[aa] for aa in "KRH")
    cysteine = residues["C"]
    length = max(len(sequence), 1)
    score = (
        0.25
        + 0.35 * hydrophobic / length
        + 0.2 * cationic / length
        + 0.2 * min(cysteine / 2, 1)
    )
    score = round(min(max(score, 0.0), 1.0), 3)
    return ModelEvidence(
        name="BuiltinToxicityProxy",
        task="toxicity",
        label="toxic" if score >= 0.55 else "non_toxic",
        score=score,
        weight=0.35,
        status="proxy",
        source="内置低权重启发式，只用于外部模型缺失时保持流程可运行",
    )


def _activity_proxy(sequence: str) -> ModelEvidence:
    residues = Counter(sequence)
    hydrophobic = sum(residues[aa] for aa in "AILMFWVY")
    cationic = sum(residues[aa] for aa in "KRH")
    acidic = sum(residues[aa] for aa in "DE")
    length = max(len(sequence), 1)
    charge_bias = max(cationic - acidic, 0) / length
    amphipathic_proxy = min((hydrophobic / length) * 1.7, 1.0)
    score = round(min(0.15 + 0.45 * charge_bias + 0.4 * amphipathic_proxy, 1.0), 3)
    return ModelEvidence(
        name="BuiltinActivityProxy",
        task="antimicrobial_activity",
        label="active" if score >= 0.55 else "inactive",
        score=score,
        weight=0.35,
        status="proxy",
        source="内置低权重启发式，只用于外部模型缺失时保持流程可运行",
    )


def _risk_score(evidence: ModelEvidence) -> float | None:
    if evidence.score is None:
        if evidence.label in {"toxic", "active", "amp", "positive", "yes", "1"}:
            return 1.0
        if evidence.label in {"non_toxic", "nontoxic", "inactive", "non_amp", "negative", "no", "0"}:
            return 0.0
        return None

    score = evidence.score
    if evidence.label in {"non_toxic", "nontoxic", "inactive", "non_amp", "negative", "no", "0"}:
        return 1.0 - score if score > 0.5 else score
    return score


def _weighted_consensus(model_results: list[ModelEvidence]) -> tuple[float, str, str]:
    usable = [item for item in model_results if item.status in {"available", "proxy"}]
    scored = [(item, _risk_score(item)) for item in usable]
    scored = [(item, score) for item, score in scored if score is not None]
    if not scored:
        return 0.0, "insufficient_models", "没有可用模型证据。"

    weight_sum = sum(item.weight for item, _score in scored)
    score = sum(item.weight * score for item, score in scored) / weight_sum
    high = sum(1 for _item, item_score in scored if item_score >= 0.55)
    low = sum(1 for _item, item_score in scored if item_score < 0.45)
    available_count = sum(1 for item in usable if item.status == "available")

    if len(scored) < 3 or available_count == 0:
        consensus = "insufficient_models"
    elif high == len(scored) or low == len(scored):
        consensus = "strong_agreement"
    elif high and low:
        consensus = "conflict"
    else:
        consensus = "moderate_agreement"

    summary = (
        f"可用模型 {available_count} 个，参与综合证据 {len(scored)} 个；"
        f"高风险/高活性证据 {high} 个，低风险/低活性证据 {low} 个。"
    )
    return round(score, 3), consensus, summary


def _collect_model_evidence(specs: tuple[dict, ...], sequence: str) -> list[ModelEvidence]:
    return [_run_configured_model(spec, sequence) for spec in specs]


def _fold_sequence_with_esm_atlas(sequence: str) -> tuple[str, str]:
    req = urllib.request.Request(
        ESM_ATLAS_FOLD_URL,
        data=sequence.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        content_type = resp.headers.get("content-type", "")
        return resp.read().decode("utf-8"), content_type


def _select_esmfold_device():
    import torch

    requested_device = os.getenv("PEPTIDE_HELPER_ESMFOLD_DEVICE")
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_local_esmfold_model():
    import torch
    from transformers import EsmForProteinFolding

    device = _select_esmfold_device()
    if ESMFOLD_LOCAL_FILES_ONLY:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    model = EsmForProteinFolding.from_pretrained(
        LOCAL_ESMFOLD_MODEL,
        local_files_only=ESMFOLD_LOCAL_FILES_ONLY,
        use_safetensors=False,
    )
    model = model.eval().to(device)
    return model, device


def _fold_sequence_with_local_esmfold(sequence: str) -> tuple[str, str]:
    import torch

    model, _device = _load_local_esmfold_model()
    with torch.no_grad():
        pdb_text = model.infer_pdb(sequence)
    return pdb_text, f"local/esmfold; model={LOCAL_ESMFOLD_MODEL}"


def _fold_sequence(sequence: str) -> tuple[str, str]:
    if ESMFOLD_BACKEND == "http":
        return _fold_sequence_with_esm_atlas(sequence)
    if ESMFOLD_BACKEND == "auto":
        try:
            return _fold_sequence_with_local_esmfold(sequence)
        except Exception as exc:
            print(f"[ESMFold] 本地模型调用失败，回退到 ESM Atlas HTTP: {exc.__class__.__name__}")
            return _fold_sequence_with_esm_atlas(sequence)
    return _fold_sequence_with_local_esmfold(sequence)


def _normalize_plddt(mean_score: float) -> float:
    if mean_score <= 1:
        mean_score *= 100
    return round(mean_score, 2)


def _quality_label(mean_plddt: float) -> str:
    if mean_plddt >= 90:
        return "整体结构置信度很高，可用于较强结构解释。"
    if mean_plddt >= 70:
        return "整体结构较可信，适合初步结构分析；低置信局部需谨慎。"
    if mean_plddt >= 50:
        return "整体结构置信度中等，只适合粗略参考。"
    return "整体结构置信度较低，不建议做强结构结论。"


def _parse_atom_record(line: str) -> PdbAtomRecord | None:
    try:
        return PdbAtomRecord(
            serial=int(line[6:11]),
            atom_name=line[12:16].strip(),
            residue_name=line[17:20].strip(),
            chain_id=line[21].strip() or "_",
            residue_number=int(line[22:26]),
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
            occupancy=float(line[54:60]),
            b_factor=_normalize_plddt(float(line[60:66])),
            element=line[76:78].strip() if len(line) >= 78 else "",
        )
    except ValueError:
        return None


def _confidence_counts(residues: list[ResidueConfidence]) -> dict[str, int]:
    return {
        "high": sum(1 for residue in residues if residue.plddt >= 90),
        "medium": sum(1 for residue in residues if 70 <= residue.plddt < 90),
        "low": sum(1 for residue in residues if 50 <= residue.plddt < 70),
        "very_low": sum(1 for residue in residues if residue.plddt < 50),
    }


def _low_confidence_regions(residues: list[ResidueConfidence]) -> list[LowConfidenceRegion]:
    regions = []
    current = []

    for residue in residues:
        if residue.plddt < 70:
            if current and (
                current[-1].chain_id != residue.chain_id
                or current[-1].position + 1 != residue.position
            ):
                regions.append(_build_low_confidence_region(current))
                current = []
            current.append(residue)
            continue

        if current:
            regions.append(_build_low_confidence_region(current))
            current = []

    if current:
        regions.append(_build_low_confidence_region(current))

    return regions


def _build_low_confidence_region(residues: list[ResidueConfidence]) -> LowConfidenceRegion:
    mean_plddt = sum(residue.plddt for residue in residues) / len(residues)
    return LowConfidenceRegion(
        chain_id=residues[0].chain_id,
        start=residues[0].position,
        end=residues[-1].position,
        residue_count=len(residues),
        mean_plddt=round(mean_plddt, 2),
    )


def _chain_summaries(
    atom_records: list[PdbAtomRecord],
    residues: list[ResidueConfidence],
) -> list[ChainStructureSummary]:
    atoms_by_chain = Counter(atom.chain_id for atom in atom_records)
    residues_by_chain = defaultdict(list)
    for residue in residues:
        residues_by_chain[residue.chain_id].append(residue.plddt)

    summaries = []
    for chain_id in sorted(residues_by_chain):
        scores = residues_by_chain[chain_id]
        summaries.append(
            ChainStructureSummary(
                chain_id=chain_id,
                residue_count=len(scores),
                atom_count=atoms_by_chain[chain_id],
                mean_plddt=round(sum(scores) / len(scores), 2),
                min_plddt=round(min(scores), 2),
                max_plddt=round(max(scores), 2),
            )
        )
    return summaries


def _coordinate_bounds(atom_records: list[PdbAtomRecord]) -> CoordinateBounds | None:
    if not atom_records:
        return None

    return CoordinateBounds(
        min_x=min(atom.x for atom in atom_records),
        max_x=max(atom.x for atom in atom_records),
        min_y=min(atom.y for atom in atom_records),
        max_y=max(atom.y for atom in atom_records),
        min_z=min(atom.z for atom in atom_records),
        max_z=max(atom.z for atom in atom_records),
    )


def _parse_esmfold_pdb(
    pdb_text: str,
    sequence: str,
    content_type: str = "",
    source: str = "ESM Atlas",
) -> StructureResult:
    lines = pdb_text.splitlines()
    record_counts = Counter((line[:6].strip() or "BLANK") for line in lines)
    header = " ".join(line[10:].strip() for line in lines if line.startswith("HEADER"))
    title = " ".join(line[10:].strip() for line in lines if line.startswith("TITLE"))
    remarks = [line[10:].strip() for line in lines if line.startswith("REMARK") and line[10:].strip()]

    atom_records = []
    ca_by_residue = {}
    for line in lines:
        if not line.startswith("ATOM"):
            continue

        atom = _parse_atom_record(line)
        if atom is None:
            continue

        atom_records.append(atom)
        if atom.atom_name == "CA":
            ca_by_residue[(atom.chain_id, atom.residue_number)] = atom

    per_residue_plddt = [
        ResidueConfidence(
            position=atom.residue_number,
            residue=atom.residue_name,
            chain_id=atom.chain_id,
            plddt=atom.b_factor,
        )
        for atom in sorted(ca_by_residue.values(), key=lambda item: (item.chain_id, item.residue_number))
    ]
    plddt_scores = [residue.plddt for residue in per_residue_plddt]
    confidence = round(sum(plddt_scores) / len(plddt_scores), 2) if plddt_scores else 0.0
    confidence_counts = _confidence_counts(per_residue_plddt)

    return StructureResult(
        pdb_id=_pdb_result_id(sequence),
        confidence=confidence,
        source=source,
        content_type=content_type,
        sequence_length=len(sequence),
        pdb_line_count=len(lines),
        pdb_char_count=len(pdb_text),
        record_counts=dict(record_counts),
        header=header,
        title=title,
        remarks=remarks[:20],
        atom_count=len(atom_records),
        residue_count=len(per_residue_plddt),
        chain_ids=sorted({atom.chain_id for atom in atom_records}),
        chain_summaries=_chain_summaries(atom_records, per_residue_plddt),
        coordinate_bounds=_coordinate_bounds(atom_records),
        min_plddt=round(min(plddt_scores), 2) if plddt_scores else 0.0,
        max_plddt=round(max(plddt_scores), 2) if plddt_scores else 0.0,
        high_confidence_residue_count=confidence_counts["high"],
        medium_confidence_residue_count=confidence_counts["medium"],
        low_confidence_residue_count=confidence_counts["low"],
        very_low_confidence_residue_count=confidence_counts["very_low"],
        per_residue_plddt=per_residue_plddt,
        low_confidence_residues=[
            residue for residue in per_residue_plddt if residue.plddt < 70
        ],
        low_confidence_regions=_low_confidence_regions(per_residue_plddt),
        atom_records=atom_records,
        structure_quality=_quality_label(confidence),
        pdb_preview="\n".join(lines[:20]),
    )


def _pdb_result_id(sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
    return f"esmfold_{digest}"


def phys_chem_node(_state: PeptideState) -> dict:
    _log_agent("🔬 理化专家")
    return {"phys_chem_res": PhysChemResult(mw=1205.4, pi=6.8)}


def toxicity_node(_state: PeptideState) -> dict:
    _log_agent("☠️ 毒性专家")
    sequence = _state.get("sequence", "").strip().upper()
    model_results = _collect_model_evidence(TOXICITY_MODEL_SPECS, sequence)
    model_results.append(_toxicity_proxy(sequence))
    score, consensus_level, evidence_summary = _weighted_consensus(model_results)
    return {
        "toxicity_res": ToxicityResult(
            is_toxic=score >= 0.55,
            score=score,
            consensus_level=consensus_level,
            evidence_summary=evidence_summary,
            model_results=model_results,
        )
    }


def activity_node(_state: PeptideState) -> dict:
    _log_agent("⚔️ 活性专家")
    sequence = _state.get("sequence", "").strip().upper()
    model_results = _collect_model_evidence(ACTIVITY_MODEL_SPECS, sequence)

    if len(sequence) > 30:
        model_results = [
            item.model_copy(update={"status": "skipped", "error": "序列长度超过 30 aa"})
            if item.name == "Deep-AmPEP30"
            else item
            for item in model_results
        ]

    model_results.append(_activity_proxy(sequence))
    score, consensus_level, evidence_summary = _weighted_consensus(model_results)
    return {
        "activity_res": ActivityResult(
            antimicrobial_score=score,
            consensus_level=consensus_level,
            evidence_summary=evidence_summary,
            model_results=model_results,
        )
    }


def esmfold_node(state: PeptideState) -> dict:
    _log_agent("🧬 结构专家")
    sequence = state.get("sequence", "").strip().upper()

    try:
        pdb_text, content_type = _fold_sequence(sequence)
        source = "Local ESMFold" if content_type.startswith("local/") else "ESM Atlas"
        result = _parse_esmfold_pdb(pdb_text, sequence, content_type, source)
    except Exception as exc:
        result = StructureResult(
            pdb_id=_pdb_result_id(sequence or "empty"),
            confidence=0.0,
            error=f"ESMFold 调用失败: {exc.__class__.__name__}",
        )

    return {"structure_res": result}
