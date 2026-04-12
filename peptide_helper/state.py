from typing import List, Optional, TypedDict

from .models import (
    ActivityResult,
    PhysChemResult,
    StabilityResult,
    StructureResult,
    ToxicityResult,
)

DEFAULT_REQUIRED_TASKS = ["phys_chem_node", "toxicity_node"]


class PeptideState(TypedDict, total=False):
    """LangGraph 全局状态，使用 total=False 兼容渐进式填充。"""

    sequence: str
    user_request: str
    required_tasks: List[str]

    # 子 Agent 的返回结果
    phys_chem_res: Optional[PhysChemResult]
    toxicity_res: Optional[ToxicityResult]
    activity_res: Optional[ActivityResult]
    stability_res: Optional[StabilityResult]
    structure_res: Optional[StructureResult]

    # 核心处理与报告流转结果
    pepcore_context: str
    text1_analysis: str
    final_report: str


def create_initial_state(sequence: str, user_request: str) -> PeptideState:
    """统一初始化入口，避免主程序依赖隐式默认值。"""

    return {
        "sequence": sequence,
        "user_request": user_request,
        "required_tasks": [],
        "phys_chem_res": None,
        "toxicity_res": None,
        "activity_res": None,
        "stability_res": None,
        "structure_res": None,
        "pepcore_context": "",
        "text1_analysis": "",
        "final_report": "",
    }
