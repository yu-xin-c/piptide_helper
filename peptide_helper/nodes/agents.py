import hashlib
import urllib.error
import urllib.request
from collections import Counter, defaultdict

from ..models import (
    ActivityResult,
    ChainStructureSummary,
    CoordinateBounds,
    LowConfidenceRegion,
    PdbAtomRecord,
    PhysChemResult,
    ResidueConfidence,
    StructureResult,
    ToxicityResult,
)
from ..state import PeptideState

ESM_ATLAS_FOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def _log_agent(agent_name: str) -> None:
    """统一节点日志，便于后续接入结构化日志。"""

    print(f"[Agent] {agent_name} 执行中...")


def _fold_sequence_with_esm_atlas(sequence: str) -> tuple[str, str]:
    req = urllib.request.Request(
        ESM_ATLAS_FOLD_URL,
        data=sequence.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        content_type = resp.headers.get("content-type", "")
        return resp.read().decode("utf-8"), content_type


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


def _parse_esmfold_pdb(pdb_text: str, sequence: str, content_type: str = "") -> StructureResult:
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
    return {"toxicity_res": ToxicityResult(is_toxic=False, score=0.15)}


def activity_node(_state: PeptideState) -> dict:
    _log_agent("⚔️ 活性专家")
    return {"activity_res": ActivityResult(antimicrobial_score=0.88)}


def esmfold_node(state: PeptideState) -> dict:
    _log_agent("🧬 结构专家")
    sequence = state.get("sequence", "").strip().upper()

    try:
        pdb_text, content_type = _fold_sequence_with_esm_atlas(sequence)
        result = _parse_esmfold_pdb(pdb_text, sequence, content_type)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        result = StructureResult(
            pdb_id=_pdb_result_id(sequence or "empty"),
            confidence=0.0,
            error=f"ESM Atlas 调用失败: {exc.__class__.__name__}",
        )

    return {"structure_res": result}
