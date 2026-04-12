from typing import Dict, List

from ..state import DEFAULT_REQUIRED_TASKS, PeptideState

TASK_KEYWORDS: Dict[str, List[str]] = {
    "phys_chem_node": ["理化"],
    "toxicity_node": ["毒性"],
    "activity_node": ["活性", "抗菌"],
    "stability_node": ["稳定"],
    "esmfold_node": ["结构", "3D"],
}


def planner_node(state: PeptideState) -> dict:
    """
    意图识别与任务分发节点
    如果 state["user_request"] 中包含"毒性"字眼，则将 "toxicity_node" 加入 required_tasks，以此类推。
    如果用户没有明确需求，默认执行理化和毒性。
    """
    request = state.get("user_request", "")
    tasks: List[str] = []

    for task_name, keywords in TASK_KEYWORDS.items():
        if any(keyword in request for keyword in keywords):
            tasks.append(task_name)

    # 如果没有识别到明确的专业需求，兜底使用默认任务。
    if not tasks:
        tasks = list(DEFAULT_REQUIRED_TASKS)

    print(f"[Planner] 🧠 分析用户需求: '{request}'")
    print(f"[Planner] 🎯 分配专家任务: {tasks}")

    return {"required_tasks": tasks}
