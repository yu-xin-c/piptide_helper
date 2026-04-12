import os

from ..state import PeptideState

DEFAULT_LLM_MODEL = os.getenv("PEPTIDE_HELPER_MODEL", "gpt-4o-mini")


def _get_llm():
    """延迟创建 LLM 客户端，避免没有依赖时影响纯 Mock 运行。"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=DEFAULT_LLM_MODEL, timeout=30, max_retries=1)


def _safe_invoke(prompt: str) -> str:
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("未配置 OPENAI_API_KEY，自动降级到 Mock 流程。")
    return llm.invoke(prompt).content


def _format_error(exc: Exception) -> str:
    """避免把底层异常原文直接暴露到最终报告。"""

    message = str(exc)
    if "Incorrect API key provided" in message:
        return "OPENAI_API_KEY 无效"
    return exc.__class__.__name__


def _mock_text1(context: str, error: str = "") -> str:
    prefix = "【模拟推演过程】"
    if error:
        prefix = f"【LLM调用失败，启用Mock推演】\n原因：{error}"
    return (
        f"{prefix}\n"
        f"基于收集到的多肽专业数据：\n{context}\n\n"
        "该分子在已触发的专项指标中表现出较稳定的一致性，"
        "建议结合湿实验继续验证活性与安全性的平衡关系。"
    )


def _mock_text2(analysis: str, error: str = "") -> str:
    title = "# Peptide Helper 综合评估报告"
    if error:
        title = f"【LLM调用失败，启用Mock报告】\n{title}"
    return (
        f"{title}\n\n"
        "## 1. 核心推演分析\n"
        f"{analysis}\n\n"
        "## 2. 最终结论\n"
        "经过规划、并行分析与核心聚合，本多肽序列已形成可交付的初步评估结果。"
        "建议将当前结论作为下一步实验设计和专家复核的输入。"
    )


def text1_node(state: PeptideState) -> dict:
    """
    初步推演思考节点 (Text1)
    分析 pepcore_context 数据间的生物学关联。
    """
    print("[Reporter] 📝 Text1: 正在进行初步推演思考...")
    context = state.get("pepcore_context", "")

    prompt = (
        "你是一个顶级结构生物学家，请分析以下多肽特征数据之间的生物学关联，"
        f"进行内部推演：\n\n{context}"
    )
    try:
        analysis = _safe_invoke(prompt)
    except Exception as exc:
        analysis = _mock_text1(context, _format_error(exc))

    return {"text1_analysis": analysis}


def text2_node(state: PeptideState) -> dict:
    """
    最终报告生成节点 (Text2)
    基于前置推演思考步骤，产出最终排版好的报告。
    """
    print("[Reporter] 📄 Text2: 正在生成最终交付报告...")
    analysis = state.get("text1_analysis", "")

    prompt = (
        "你是一个资深医药咨询顾问。请将以下科学推演内容翻译成一篇逻辑严密、"
        f"排版清晰的 Markdown 评估报告：\n\n{analysis}"
    )
    try:
        report = _safe_invoke(prompt)
    except Exception as exc:
        report = _mock_text2(analysis, _format_error(exc))

    return {"final_report": report}
