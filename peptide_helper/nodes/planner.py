import json
import os
import re
from typing import Dict, List

from ..prompts import PLANNER_INTENT_PROMPT
from ..state import DEFAULT_REQUIRED_TASKS, PeptideState

DEFAULT_LLM_MODEL = os.getenv("PEPTIDE_HELPER_MODEL", "gpt-4o-mini")

TASK_KEYWORDS: Dict[str, List[str]] = {
    "phys_chem_node": ["理化"],
    "toxicity_node": ["毒性"],
    "activity_node": ["活性", "抗菌"],
    "stability_node": ["稳定"],
    "esmfold_node": ["结构", "3D"],
}

VALID_NODES = set(TASK_KEYWORDS.keys())


def _get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=30,
        max_retries=1,
    )


def _parse_json_response(text: str) -> List[str]:
    # 尝试从 markdown 代码块中提取 JSON
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = code_block.group(1) if code_block else text.strip()

    # 尝试直接匹配 JSON 对象
    if not json_str.startswith("{"):
        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            json_str = brace_match.group()

    data = json.loads(json_str)
    tasks = data.get("required_tasks", [])

    # 只保留合法节点名
    return [t for t in tasks if t in VALID_NODES]


def _keyword_fallback(request: str) -> List[str]:
    # 原有关键词规则，作为 LLM 不可用时的降级方案
    tasks: List[str] = []
    for task_name, keywords in TASK_KEYWORDS.items():
        if any(keyword in request for keyword in keywords):
            tasks.append(task_name)
    return tasks or list(DEFAULT_REQUIRED_TASKS)


def planner_node(state: PeptideState) -> dict:
    """
    意图识别与任务分发节点（LLM 路由 + 关键词兜底）
    优先通过 LLM 理解用户意图并路由到正确节点，
    LLM 不可用或解析失败时降级到关键词规则。
    """
    request = state.get("user_request", "")
    tasks: List[str] = []
    used_llm = False

    llm = _get_llm()
    if llm is not None:
        try:
            prompt = PLANNER_INTENT_PROMPT.format(user_request=request)
            response = llm.invoke(prompt).content
            tasks = _parse_json_response(response)
            used_llm = True
        except Exception as exc:
            print(f"[Planner] ⚠️ LLM 路由失败，降级到关键词规则: {exc}")

    if not tasks:
        tasks = _keyword_fallback(request)

    source = "LLM" if used_llm else "关键词兜底"
    print(f"[Planner] 🧠 分析用户需求: '{request}'")
    print(f"[Planner] 🎯 分配专家任务: {tasks} (来源: {source})")

    return {"required_tasks": tasks}
