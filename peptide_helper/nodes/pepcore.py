from typing import Iterable, Tuple

from ..state import PeptideState

RESULT_SECTION_MAPPING: Tuple[Tuple[str, str], ...] = (
    ("phys_chem_res", "理化性质"),
    ("toxicity_res", "毒性预测"),
    ("activity_res", "活性评估"),
    ("stability_res", "稳定性"),
    ("structure_res", "3D结构"),
)


def _build_context_sections(state: PeptideState) -> Iterable[str]:
    for state_key, title in RESULT_SECTION_MAPPING:
        result = state.get(state_key)
        if result is None:
            continue
        yield f"### {title}\n{result.model_dump_json(indent=2)}"


def pepcore_node(state: PeptideState) -> dict:
    """
    数据上下文聚合节点
    判断 state 中哪些 *_res 字段有值，将有值的部分拼接成一段结构清晰的 Markdown 字符串。
    """
    print("[PepCore] 🧠 正在聚合上下文数据 (Fan-in)...")
    context_parts = list(_build_context_sections(state))

    if not context_parts:
        context_str = "未检测到任何专家的分析数据。"
    else:
        context_str = "\n\n".join(context_parts)

    print(f"[PepCore] ✅ 聚合完成，共包含 {len(context_parts)} 项专业数据。")
    return {"pepcore_context": context_str}
