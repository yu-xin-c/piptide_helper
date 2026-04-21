from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import DefaultDict, FrozenSet, Iterable, List, Sequence

from .nodes.planner import planner_node
from .prompts import BENCHMARK_TEST_PROMPT
from .state import DEFAULT_REQUIRED_TASKS, create_initial_state

DEFAULT_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class BenchmarkCase:
    """单条意图识别评估样本。"""

    name: str
    category: str
    user_request: str
    expected_tasks: FrozenSet[str]


def _task_set(*tasks: str) -> FrozenSet[str]:
    return frozenset(tasks)


# 这组样本专门评测 planner 的“路由正确率”，而不是下游 mock 结果。
# 1. direct_property: 用户直接点名某个性质，考察显式路由。
# 2. business_goal: 用户给业务目标，planner 需要倒推出应跑哪些子节点。
# 3. fallback: 用户没有明确目标时，是否走默认路由。
BENCHMARK_CASES: Sequence[BenchmarkCase] = (
    BenchmarkCase(
        name="等电点查询",
        category="direct_property",
        user_request="帮我看看这条序列的等电点是多少。",
        expected_tasks=_task_set("phys_chem_node"),
    ),
    BenchmarkCase(
        name="分子量查询",
        category="direct_property",
        user_request="请给我算一下这条多肽的分子量。",
        expected_tasks=_task_set("phys_chem_node"),
    ),
    BenchmarkCase(
        name="毒性风险评估",
        category="direct_property",
        user_request="帮我评估一下这条序列的毒性风险。",
        expected_tasks=_task_set("toxicity_node"),
    ),
    BenchmarkCase(
        name="抗菌活性评估",
        category="direct_property",
        user_request="请判断这条序列有没有抗菌活性。",
        expected_tasks=_task_set("activity_node"),
    ),
    BenchmarkCase(
        name="稳定性评估",
        category="direct_property",
        user_request="我想看一下这条多肽的稳定性表现。",
        expected_tasks=_task_set("stability_node"),
    ),
    BenchmarkCase(
        name="三维结构预测",
        category="direct_property",
        user_request="请给我做一下这条序列的3D结构预测。",
        expected_tasks=_task_set("esmfold_node"),
    ),
    BenchmarkCase(
        name="显式双任务",
        category="direct_property",
        user_request="帮我一起评估抗菌活性和毒性风险。",
        expected_tasks=_task_set("activity_node", "toxicity_node"),
    ),
    BenchmarkCase(
        name="显式三任务",
        category="direct_property",
        user_request="请同时分析理化性质、稳定性和结构。",
        expected_tasks=_task_set("phys_chem_node", "stability_node", "esmfold_node"),
    ),
    BenchmarkCase(
        name="口服抗癌药评估",
        category="business_goal",
        user_request="评估这条序列是否能作为口服的抗癌药。",
        expected_tasks=_task_set("activity_node", "toxicity_node", "stability_node"),
    ),
    BenchmarkCase(
        name="口服抗菌候选",
        category="business_goal",
        user_request="判断这条多肽能不能作为口服抗菌候选分子。",
        expected_tasks=_task_set("activity_node", "toxicity_node", "stability_node"),
    ),
    BenchmarkCase(
        name="体内给药安全性",
        category="business_goal",
        user_request="如果要做体内给药，先帮我看安全性和稳定性。",
        expected_tasks=_task_set("toxicity_node", "stability_node"),
    ),
    BenchmarkCase(
        name="成药性初判",
        category="business_goal",
        user_request="这条序列值不值得往成药方向推进？",
        expected_tasks=_task_set("activity_node", "toxicity_node", "stability_node"),
    ),
    BenchmarkCase(
        name="构效关系起点",
        category="business_goal",
        user_request="我想先看三维构象，再结合活性做构效关系分析。",
        expected_tasks=_task_set("esmfold_node", "activity_node"),
    ),
    BenchmarkCase(
        name="湿实验前风险筛查",
        category="business_goal",
        user_request="上湿实验前，先评估一下毒性和结构风险。",
        expected_tasks=_task_set("toxicity_node", "esmfold_node"),
    ),
    BenchmarkCase(
        name="没有明确目标的初筛",
        category="fallback",
        user_request="先帮我做个初步筛查。",
        expected_tasks=_task_set(*DEFAULT_REQUIRED_TASKS),
    ),
    BenchmarkCase(
        name="泛化整体判断",
        category="fallback",
        user_request="先给我一个整体判断。",
        expected_tasks=_task_set(*DEFAULT_REQUIRED_TASKS),
    ),
    BenchmarkCase(
        name="信息不足时兜底",
        category="fallback",
        user_request="帮我看看这条序列怎么样。",
        expected_tasks=_task_set(*DEFAULT_REQUIRED_TASKS),
    ),
)


PROMPT_DESIGN_GUIDE = BENCHMARK_TEST_PROMPT


def _predict_tasks(user_request: str) -> FrozenSet[str]:
    # 直接复用项目真实入口，避免 benchmark 和线上逻辑不一致。
    state = create_initial_state(sequence=DEFAULT_SEQUENCE, user_request=user_request)
    # 屏蔽节点内部日志，避免 benchmark 输出被过程日志淹没。
    with StringIO() as buffer, redirect_stdout(buffer):
        result = planner_node(state)
    return frozenset(result.get("required_tasks", []))


def _format_tasks(tasks: Iterable[str]) -> str:
    ordered_tasks = sorted(tasks)
    return ", ".join(ordered_tasks) if ordered_tasks else "(空)"


def run_intent_benchmark(cases: Sequence[BenchmarkCase] = BENCHMARK_CASES) -> None:
    total_cases = len(cases)
    exact_hits = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    failures: List[str] = []
    category_stats: DefaultDict[str, dict] = defaultdict(
        lambda: {"total": 0, "exact_hits": 0}
    )

    for index, case in enumerate(cases, start=1):
        predicted_tasks = _predict_tasks(case.user_request)
        expected_tasks = case.expected_tasks
        is_exact_hit = predicted_tasks == expected_tasks

        if is_exact_hit:
            exact_hits += 1
            category_stats[case.category]["exact_hits"] += 1
        else:
            failures.append(
                f"{index:02d}. {case.name} | category={case.category}\n"
                f"    request: {case.user_request}\n"
                f"    expected: {_format_tasks(expected_tasks)}\n"
                f"    predicted: {_format_tasks(predicted_tasks)}"
            )

        category_stats[case.category]["total"] += 1
        true_positive += len(predicted_tasks & expected_tasks)
        false_positive += len(predicted_tasks - expected_tasks)
        false_negative += len(expected_tasks - predicted_tasks)

    exact_match_rate = exact_hits / total_cases if total_cases else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print("=" * 72)
    print("Intent Benchmark Result")
    print("=" * 72)
    print(f"总样本数: {total_cases}")
    print(f"严格命中率: {exact_match_rate:.2%} ({exact_hits}/{total_cases})")
    print(f"任务级 Precision: {precision:.2%}")
    print(f"任务级 Recall: {recall:.2%}")
    print(f"任务级 F1: {f1:.2%}")
    print("-" * 72)
    print("分类别结果:")

    for category in sorted(category_stats):
        stats = category_stats[category]
        category_rate = (
            stats["exact_hits"] / stats["total"] if stats["total"] else 0.0
        )
        print(
            f"  - {category}: {category_rate:.2%} "
            f"({stats['exact_hits']}/{stats['total']})"
        )

    print("-" * 72)
    print("失败样例:")

    if not failures:
        print("  - 无")
        return

    for failure in failures:
        print(failure)
        print("-" * 72)


if __name__ == "__main__":
    run_intent_benchmark()
