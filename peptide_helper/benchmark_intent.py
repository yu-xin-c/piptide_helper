import json
import os
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from typing import DefaultDict, Dict, FrozenSet, Iterable, List

from .nodes.planner import planner_node
from .state import create_initial_state

DEFAULT_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


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


def run_scientific_benchmark(dataset_path: str = "peptide_helper/eval_dataset.json") -> None:
    if not os.path.exists(dataset_path):
        print(f"找不到数据集文件：{dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_cases = len(dataset)
    exact_matches = 0

    # 宏平均 (Macro) 统计：对每个类别计算指标再平均
    # 微平均 (Micro) 统计：汇总全局的 TP, FP, FN 再计算指标
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    
    # 汉明损失 (Hamming Loss) 统计
    total_labels = 5  # 当前共有 5 个可选专家节点
    total_incorrect_labels = 0

    failures: List[str] = []
    category_stats: DefaultDict[str, dict] = defaultdict(
        lambda: {"total": 0, "exact_matches": 0, "tp": 0, "fp": 0, "fn": 0, "incorrect_labels": 0}
    )

    for item in dataset:
        case_id = item["id"]
        category = item["type"]
        request = item["user_request"]
        expected = frozenset(item["ground_truth"])

        predicted = _predict_tasks(request)

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        
        incorrect_labels_in_case = fp + fn

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        total_incorrect_labels += incorrect_labels_in_case

        category_stats[category]["total"] += 1
        category_stats[category]["tp"] += tp
        category_stats[category]["fp"] += fp
        category_stats[category]["fn"] += fn
        category_stats[category]["incorrect_labels"] += incorrect_labels_in_case

        is_exact_match = (predicted == expected)
        if is_exact_match:
            exact_matches += 1
            category_stats[category]["exact_matches"] += 1
        else:
            failures.append(
                f"[{case_id}] 类型={category}\n"
                f"    用户指令: {request}\n"
                f"    Ground Truth: {_format_tasks(expected)}\n"
                f"    Predicted   : {_format_tasks(predicted)}\n"
                f"    [TP={tp}, FP={fp}, FN={fn}]"
            )

    # 计算全局指标
    emr = exact_matches / total_cases if total_cases else 0.0

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    
    hamming_loss = total_incorrect_labels / (total_cases * total_labels) if total_cases else 0.0

    print("=" * 80)
    print("🔬 Planner 多标签分类科研级评测报告 (Multi-label Classification)")
    print("=" * 80)
    print(f"数据集样本总数: {total_cases}")
    print(f"Exact Match Ratio (EMR, 严格命中率): {emr:.2%} ({exact_matches}/{total_cases})")
    print(f"Hamming Loss (HL, 汉明损失)    : {hamming_loss:.4f} (越接近0越好)")
    print(f"Micro-Precision: {micro_precision:.2%}")
    print(f"Micro-Recall: {micro_recall:.2%}")
    print(f"Micro-F1 Score: {micro_f1:.2%}")
    print("-" * 80)

    print("📊 细分类别表现 (按 Intent Type):")
    for category, stats in sorted(category_stats.items()):
        total = stats["total"]
        cat_emr = stats["exact_matches"] / total if total else 0.0
        c_tp, c_fp, c_fn = stats["tp"], stats["fp"], stats["fn"]

        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) else 0.0
        c_hl = stats["incorrect_labels"] / (total * total_labels) if total else 0.0

        print(f"  [{category}] 样本数: {total}")
        print(f"    - EMR: {cat_emr:.2%} ({stats['exact_matches']}/{total})")
        print(f"    - HL : {c_hl:.4f}")
        print(f"    - F1 : {c_f1:.2%} (P={c_p:.2%}, R={c_r:.2%})")

    print("-" * 80)
    print("❌ 错误路由样例分析 (Error Analysis):")
    if not failures:
        print("  - 无路由错误，全部精准命中！")
    else:
        for failure in failures:
            print(failure)
            print("-" * 80)


if __name__ == "__main__":
    run_scientific_benchmark()
