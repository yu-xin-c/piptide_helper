import json
import os
import re
from typing import Dict, List

from ..prompts import PLANNER_INTENT_PROMPT
from ..state import DEFAULT_REQUIRED_TASKS, PeptideState

DEFAULT_LLM_MODEL = os.getenv("PEPTIDE_HELPER_MODEL", "gpt-4o-mini")

TASK_KEYWORDS: Dict[str, List[str]] = {
    "phys_chem_node": [
        "理化",
        "分子量",
        "等电点",
        "pi",
        "pI",
        "溶解度",
        "疏水",
        "亲水",
        "电荷",
        "净电荷",
        "氨基酸组成",
        "组成",
        "脂溶",
        "性质",
        "参数",
    ],
    "toxicity_node": [
        "毒性",
        "毒理",
        "有毒",
        "无毒",
        "毒副",
        "毒害",
        "安全",
        "风险",
        "溶血",
        "细胞毒",
        "免疫原",
        "致敏",
        "副作用",
        "不良反应",
        "耐受",
        "小鼠",
        "动物实验",
        "湿实验",
        "临床前",
        "把关",
        "筛查",
    ],
    "activity_node": [
        "活性",
        "抗菌",
        "抗肿瘤",
        "抗癌",
        "抗病毒",
        "杀菌",
        "抑菌",
        "免疫调节",
        "靶向",
        "药效",
        "MIC",
        "病原",
        "金黄色葡萄球菌",
        "杀伤",
        "抑制",
    ],
    "esmfold_node": [
        "结构",
        "3D",
        "三维",
        "构象",
        "折叠",
        "二级结构",
        "三级结构",
        "空间",
        "PDB",
        "ESMFold",
        "螺旋",
        "beta",
        "β",
    ],
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
    normalized_request = request.lower()
    tasks: List[str] = []
    for task_name, keywords in TASK_KEYWORDS.items():
        if any(keyword.lower() in normalized_request for keyword in keywords):
            tasks.append(task_name)

    drug_development_terms = ("成药", "候选药", "候选分子", "开发价值", "药物潜力", "先导", "可行性")
    if any(term in request for term in drug_development_terms):
        for task_name in ("activity_node", "toxicity_node"):
            if task_name not in tasks:
                tasks.append(task_name)

    early_screen_terms = ("初筛", "预筛", "快速筛", "第一轮筛选")
    if any(term in request for term in early_screen_terms):
        for task_name in DEFAULT_REQUIRED_TASKS:
            if task_name not in tasks:
                tasks.append(task_name)

    design_terms = ("优化", "改造", "设计建议", "突变建议")
    if any(term in request for term in design_terms) and "esmfold_node" not in tasks:
        tasks.append("esmfold_node")

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
