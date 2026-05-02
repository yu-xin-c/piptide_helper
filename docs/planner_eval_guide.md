# Planner 多标签分类评测全流程指南

在多智能体系统中，Planner（规划器）的意图识别和任务路由是整个流水线的第一环。如果路由错误，后续 Agent 模型再强大也无法产生有效价值，还会浪费大量算力。

对于 Planner 输出的“子任务列表（List of Tasks）”，在机器学习领域属于经典的**多标签分类（Multi-label Classification）**任务。为了严谨地评估 Planner 的路由能力，本项目设计了一套科研级的自动化评测框架。

---

## 1. 评测集构建 (Golden Dataset)

评测的第一步是构建“黄金评测集”。数据集采用 JSON 格式，存放在 `peptide_helper/eval_dataset.json`。

### 1.1 数据集结构

每条测试样本包含以下字段：
- `id`: 样本唯一标识符。
- `type`: 样本类型（显式、隐式、模糊/兜底）。
- `user_request`: 用户输入的自然语言指令。
- `ground_truth`: 人工标注的真实期望路由节点列表。

### 1.2 样本设计策略

为了保证评测的全面性，必须故意设计以下三种 case：

1. **显式指令**（好猜）：用户明确点名要分析哪些性质（如“测一下等电点”、“看看毒性”）。考察基础的语义匹配能力。
2. **隐式指令**（难猜）：用户使用业务黑话或目标导向的描述（如“口服药潜力”、“实验前筛查”）。考察 Planner 的知识注入和业务逻辑倒推能力。
3. **模糊/兜底指令**（防过度预测）：用户什么都没具体说（如“随便跑一下”）。考察系统是否会克制，不乱调节点，而是走默认兜底逻辑。

### 1.3 数据集示例

```json
[
  {
    "id": "T01",
    "type": "显式",
    "user_request": "测一下等电点，然后看看有没有毒。",
    "ground_truth": ["phys_chem_node", "toxicity_node"]
  },
  {
    "id": "T02",
    "type": "隐式-成药性",
    "user_request": "老板让我评估这条多肽作为候选药的开发价值。",
    "ground_truth": ["activity_node", "toxicity_node"]
  },
  {
    "id": "T04",
    "type": "模糊/兜底",
    "user_request": "跑一下这条序列：ACDEFGH",
    "ground_truth": ["phys_chem_node", "toxicity_node"]
  }
]
```

---

## 2. 评测指标计算逻辑

由于这是一个多标签分类任务，不能简单地用“对/错”来衡量。对于单条数据的评测，我们将真实期望（`ground_truth`）和模型预测（`predicted`）这两个集合进行运算：

### 2.1 基础集合运算

- **True Positives (TP, 命中)**：既在期望里，又被预测出来的节点。`len(predicted & ground_truth)`
- **False Positives (FP, 多跑/幻觉)**：不在期望里，但 Planner 瞎召唤出来的节点。`len(predicted - ground_truth)`
- **False Negatives (FN, 漏跑/遗漏)**：在期望里，但 Planner 忘了召唤的节点。`len(ground_truth - predicted)`

### 2.2 全局评价指标 (Micro-Averaging)

本评测框架采用微平均（Micro-Averaging）方法，先汇总所有样本的 TP、FP、FN，然后再计算整体指标。这在类别分布不均衡的多标签任务中更为严谨。

1. **Micro-Precision (微平均精确率)**
   - **公式**: `Sum(TP) / (Sum(TP) + Sum(FP))`
   - **含义**: Planner “猜对的里面有多少是真需要的”。高 Precision 意味着系统**不乱调**，不浪费算力。

2. **Micro-Recall (微平均召回率)**
   - **公式**: `Sum(TP) / (Sum(TP) + Sum(FN))`
   - **含义**: Planner “真需要的里面有多少被猜出来了”。高 Recall 意味着系统**不漏调**，能满足业务需求。

3. **Micro-F1 Score**
   - **公式**: `2 * (Precision * Recall) / (Precision + Recall)`
   - **含义**: Precision 和 Recall 的调和平均数，反映整体综合性能。

4. **Exact Match Ratio (EMR, 严格命中率)**
   - **公式**: 预测完全等于期望的样本数 / 总样本数
   - **含义**: 最严苛的指标，要求必须“一条不多、一条不少”才算对。

5. **Hamming Loss (HL, 汉明损失)**
   - **公式**: `(Sum(FP) + Sum(FN)) / (总样本数 * 总标签数)`
   - **含义**: 衡量“被错误预测的标签比例”。在有 4 个可用 Agent 的情况下，如果错了 1 个，对了 3 个，汉明损失就是 1/4 = 0.25。
   - **科研意义**: 越接近 0 越好。它能反映 Planner 犯错的“粒度”。如果 EMR 很低（比如经常多预测一个节点导致严格命中失败），但 Hamming Loss 也很低，说明 Planner 虽然常常达不到完美，但每次也只是“差之毫厘”，而不是全盘崩溃。

---

## 3. 自动化评测执行

评测逻辑实现在 `peptide_helper/benchmark_intent.py` 中。

### 3.1 运行评测

在项目根目录下执行以下命令：

```bash
./.venv/bin/python -m peptide_helper.benchmark_intent
```

### 3.2 评测报告解读

脚本运行后，会输出一份详细的评测报告：

1. **全局指标概览**：展示总样本数、EMR 以及 Micro P/R/F1。
2. **细分类别表现**：按 Intent Type（显式、隐式等）分别统计 EMR 和 F1，帮助你发现系统在处理哪类指令时存在短板。
3. **错误分析 (Error Analysis)**：列出所有未能严格命中（EMR=0）的失败样例，并详细打印出 TP、FP、FN 的数值，方便进行针对性的 Prompt 调优。

**示例输出：**
```text
================================================================================
🔬 Planner 多标签分类科研级评测报告 (Multi-label Classification)
================================================================================
数据集样本总数: 130
Exact Match Ratio (EMR, 严格命中率): 81.54% (106/130)
Micro-Precision: 87.07%
...
--------------------------------------------------------------------------------
❌ 错误路由样例分析 (Error Analysis):
[T101] 类型=困难-排除意图
    用户指令: 只看毒性，不要做结构预测。
    Ground Truth: toxicity_node
    Predicted   : esmfold_node, toxicity_node
    [TP=1, FP=1, FN=0]
...
```

---

## 4. 持续优化循环

这套评测框架的设计初衷是为了支持 **“Prompt Engineering -> Benchmark -> Error Analysis -> Prompt Engineering”** 的快速迭代闭环。

1. 扩充 `eval_dataset.json`，积累更多的真实业务场景语料。
2. 运行 benchmark，查看错误样例。
3. 如果发现特定的 FP 或 FN：
   - 调整 `peptide_helper/prompts.py` 中的 `PLANNER_INTENT_PROMPT`。
   - 增加业务规则解释（如“口服药不需要看活性”）。
   - 提供 few-shot 示例（In-Context Learning）。
## 5. 最新基准评测结果 (Latest Baseline)

以下是基于当前 130 条黄金数据集（包含显式、组合、隐式、兜底、困难压力样本）使用关键词兜底路由的最新基准结果：

```text
================================================================================
🔬 Planner 多标签分类科研级评测报告 (Multi-label Classification)
================================================================================
数据集样本总数: 130
Exact Match Ratio (EMR, 严格命中率): 81.54% (106/130)
Hamming Loss (HL, 汉明损失)    : 0.0615 (越接近0越好)
Micro-Precision: 87.07%
Micro-Recall: 99.02%
Micro-F1 Score: 92.66%
--------------------------------------------------------------------------------
📊 细分类别表现 (按 Intent Type):
  原有显式、组合、隐式、兜底样本保持 100.00% EMR。
  新增困难压力样本暴露出否定/排除、英文缩写、能力边界等短板。
--------------------------------------------------------------------------------
❌ 错误路由样例分析 (Error Analysis):
[T101] 类型=困难-排除意图
    用户指令: 只看毒性，不要做结构预测。
    Ground Truth: toxicity_node
    Predicted   : esmfold_node, toxicity_node
    [TP=1, FP=1, FN=0]
```

### 结果解读
- **基础路由仍然稳定**：原有 100 条样本继续全部命中，说明常规显式、组合、隐式、兜底场景未回退。
- **压力样本拉低了完美分数**：新增 30 条困难样本后，EMR 降至 `81.54%`，Micro-F1 为 `92.66%`，更接近真实系统评估。
- **主要短板清晰**：错误集中在否定/排除意图、英文缩写、语义歧义，以及稳定性专家移除后的能力边界处理。
