from typing import Dict, List

from langgraph.graph import END, StateGraph

from .nodes import (
    activity_node,
    esmfold_node,
    pepcore_node,
    phys_chem_node,
    planner_node,
    text1_node,
    text2_node,
    toxicity_node,
)
from .state import DEFAULT_REQUIRED_TASKS, PeptideState

EXPERT_NODES: Dict[str, object] = {
    "phys_chem_node": phys_chem_node,
    "toxicity_node": toxicity_node,
    "activity_node": activity_node,
    "esmfold_node": esmfold_node,
}


def route_tasks(state: PeptideState) -> List[str]:
    """根据 Planner 结果进行动态扇出，未命中时回退到默认任务。"""

    requested_tasks = state.get("required_tasks") or list(DEFAULT_REQUIRED_TASKS)
    tasks = [task for task in requested_tasks if task in EXPERT_NODES]
    return tasks or ["pepcore_node"]


def build_app():
    """集中构建图谱，避免模块导入阶段散落编排细节。"""

    workflow = StateGraph(PeptideState)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("pepcore_node", pepcore_node)
    workflow.add_node("text1_node", text1_node)
    workflow.add_node("text2_node", text2_node)

    for node_name, node_handler in EXPERT_NODES.items():
        workflow.add_node(node_name, node_handler)

    workflow.set_entry_point("planner_node")
    workflow.add_conditional_edges(
        "planner_node",
        route_tasks,
        {**{node_name: node_name for node_name in EXPERT_NODES}, "pepcore_node": "pepcore_node"},
    )

    for node_name in EXPERT_NODES:
        workflow.add_edge(node_name, "pepcore_node")

    workflow.add_edge("pepcore_node", "text1_node")
    workflow.add_edge("text1_node", "text2_node")
    workflow.add_edge("text2_node", END)
    return workflow.compile()


app = build_app()
