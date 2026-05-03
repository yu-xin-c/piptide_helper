from typing import Annotated, List, Optional, TypedDict

from .models import (
    ActivityResult,
    PhysChemResult,
    SequenceAnalysis,
    StructureResult,
    ToxicityResult,
)

DEFAULT_REQUIRED_TASKS = ["phys_chem_node", "toxicity_node"]


def merge_multi_results(
    current: List[SequenceAnalysis],
    update: List[SequenceAnalysis],
) -> List[SequenceAnalysis]:
    """LangGraph reducer：按 sequence 合并并行节点的分析结果。

    当多个 expert node 并行返回 multi_results 时，
    同一序列的结果会被合并到同一个 SequenceAnalysis 对象中。
    """
    result = {sa.sequence: sa.model_copy() for sa in current}
    for sa in update:
        if sa.sequence in result:
            existing = result[sa.sequence]
            merged = existing.model_dump()
            for field in ("phys_chem_res", "toxicity_res", "activity_res", "structure_res"):
                new_val = getattr(sa, field)
                if new_val is not None:
                    merged[field] = new_val
            result[sa.sequence] = SequenceAnalysis(**merged)
        else:
            result[sa.sequence] = sa
    return list(result.values())


class PeptideState(TypedDict, total=False):
    """LangGraph 全局状态，使用 total=False 兼容渐进式填充。"""

    # --- 输入 ---
    sequence: str                    # 单条序列（向后兼容）
    sequences: List[str]             # 多条序列列表（Planner 提取）
    user_request: str                # 用户自然语言需求
    required_tasks: List[str]        # Planner 分配的节点名列表

    # --- 单序列结果（向后兼容） ---
    phys_chem_res: Optional[PhysChemResult]
    toxicity_res: Optional[ToxicityResult]
    activity_res: Optional[ActivityResult]
    structure_res: Optional[StructureResult]

    # --- 多序列结果（带 reducer，支持并行节点合并） ---
    multi_results: Annotated[List[SequenceAnalysis], merge_multi_results]

    # --- 核心处理与报告流转结果 ---
    pepcore_context: str
    text1_analysis: str
    final_report: str


def create_initial_state(
    sequence: str = "",
    user_request: str = "",
    sequences: Optional[List[str]] = None,
) -> PeptideState:
    """统一初始化入口，避免主程序依赖隐式默认值。

    支持两种调用方式：
    1. 单序列：create_initial_state(sequence="ACDEF", user_request="...")
    2. 多序列：create_initial_state(user_request="...", sequences=["ACDEF", "GHIKL"])
    """

    seqs = sequences or []
    if sequence and sequence not in seqs:
        seqs = [sequence] + seqs

    return {
        "sequence": sequence,
        "sequences": seqs,
        "user_request": user_request,
        "required_tasks": [],
        "phys_chem_res": None,
        "toxicity_res": None,
        "activity_res": None,
        "structure_res": None,
        "multi_results": [],
        "pepcore_context": "",
        "text1_analysis": "",
        "final_report": "",
    }
