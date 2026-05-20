import json
from typing import Iterable, List, Tuple

from ..models import SequenceAnalysis
from ..state import PeptideState

RESULT_SECTION_MAPPING: Tuple[Tuple[str, str], ...] = (
    ("phys_chem_res", "理化性质"),
    ("toxicity_res", "毒性预测"),
    ("activity_res", "活性评估"),
    ("neuropeptide_res", "神经肽预测"),
    ("structure_res", "3D结构"),
)


def _serialize_result(result) -> str:
    """将单个结果序列化为 JSON 字符串。"""
    if hasattr(result, "analysis_summary"):
        return json.dumps(result.analysis_summary(), ensure_ascii=False, indent=2)
    return result.model_dump_json(indent=2)


def _build_context_sections(state: PeptideState) -> Iterable[str]:
    """单序列模式：按任务类型聚合（向后兼容）。"""
    for state_key, title in RESULT_SECTION_MAPPING:
        result = state.get(state_key)
        if result is None:
            continue
        yield f"### {title}\n{_serialize_result(result)}"


def _build_multi_sequence_context(multi_results: List[SequenceAnalysis]) -> Iterable[str]:
    """多序列模式：按序列分组聚合。"""
    for sa in multi_results:
        parts = [f"## 序列: {sa.sequence}"]
        for field_name, title in RESULT_SECTION_MAPPING:
            result = getattr(sa, field_name, None)
            if result is None:
                continue
            parts.append(f"### {title}\n{_serialize_result(result)}")
        if len(parts) > 1:  # 至少有一项结果
            yield "\n\n".join(parts)


def pepcore_node(state: PeptideState) -> dict:
    """
    数据上下文聚合节点
    - 多序列模式：按序列分组输出
    - 单序列模式：按任务类型输出（向后兼容）
    """
    print("[PepCore] 🧠 正在聚合上下文数据 (Fan-in)...")

    multi_results = state.get("multi_results", [])

    if multi_results:
        # 多序列模式
        context_parts = list(_build_multi_sequence_context(multi_results))
        seq_count = len(multi_results)
        if not context_parts:
            context_str = "未检测到任何专家的分析数据。"
        else:
            context_str = "\n\n---\n\n".join(context_parts)
        print(f"[PepCore] ✅ 聚合完成，共 {seq_count} 条序列，{len(context_parts)} 项专业数据。")
    else:
        # 单序列模式（向后兼容）
        context_parts = list(_build_context_sections(state))
        if not context_parts:
            context_str = "未检测到任何专家的分析数据。"
        else:
            context_str = "\n\n".join(context_parts)
        print(f"[PepCore] ✅ 聚合完成，共包含 {len(context_parts)} 项专业数据。")

    return {"pepcore_context": context_str}
