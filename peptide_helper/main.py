from peptide_helper.graph import app
from peptide_helper.state import create_initial_state


def run_demo(sequence: str, user_request: str) -> None:
    state = create_initial_state(sequence=sequence, user_request=user_request)

    print("=" * 60)
    print("🧬 欢迎使用 Peptide Helper 柔性多智能体生产线沙盘")
    print("=" * 60)
    print(f"\n[Input] 多肽序列: {state['sequence']}")
    print(f"[Input] 需求指令: {state['user_request']}\n")
    print("-" * 60)
    print("🚀 开始动态流转图谱执行：\n")

    result = app.invoke(state)

    print("\n" + "=" * 60)
    print("🎉 流转完成！最终输出报告如下：")
    print("=" * 60)
    print(result.get("final_report", "未生成报告。"))
    print("=" * 60)


def main():
    run_demo(
        sequence="ACDEFGHIKLMNPQRSTVWY",
        user_request="帮我评估这条序列的抗菌活性和毒性风险",
    )

if __name__ == "__main__":
    main()
