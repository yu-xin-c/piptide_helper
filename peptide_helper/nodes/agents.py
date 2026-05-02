from ..models import (
    ActivityResult,
    PhysChemResult,
    StructureResult,
    ToxicityResult,
)
from ..state import PeptideState


def _log_agent(agent_name: str) -> None:
    """统一节点日志，便于后续接入结构化日志。"""

    print(f"[Agent] {agent_name} 执行中...")


def phys_chem_node(_state: PeptideState) -> dict:
    _log_agent("🔬 理化专家")
    return {"phys_chem_res": PhysChemResult(mw=1205.4, pi=6.8)}


def toxicity_node(_state: PeptideState) -> dict:
    _log_agent("☠️ 毒性专家")
    return {"toxicity_res": ToxicityResult(is_toxic=False, score=0.15)}


def activity_node(_state: PeptideState) -> dict:
    _log_agent("⚔️ 活性专家")
    return {"activity_res": ActivityResult(antimicrobial_score=0.88)}


def esmfold_node(_state: PeptideState) -> dict:
    _log_agent("🧬 结构专家")
    return {"structure_res": StructureResult(pdb_id="mock_pdb_001", confidence=92.3)}
