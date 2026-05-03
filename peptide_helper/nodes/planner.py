import json
import os
import re
from typing import Dict, List

from ..prompts import PLANNER_INTENT_PROMPT
from ..state import DEFAULT_REQUIRED_TASKS, PeptideState

# 标准氨基酸单字母代码集合
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

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


def _extract_sequences(text: str) -> List[str]:
    """从用户输入中提取多肽序列。

    支持的格式：
    - 纯大写氨基酸字符串（长度 >= 2，>= 80% 为标准氨基酸）
    - FASTA 格式（>header 行 + 序列行）
    - 逗号/空格/换行分隔的多条序列
    """
    candidates: List[str] = []

    # 1. 处理 FASTA 格式：按 > 分割
    if ">" in text:
        fasta_blocks = re.split(r"^>", text, flags=re.MULTILINE)
        for block in fasta_blocks:
            block = block.strip()
            if not block:
                continue
            # 去掉 header 行
            lines = block.splitlines()
            seq_lines = [line.strip() for line in lines[1:] if line.strip() and not line.startswith("#")]
            seq = "".join(seq_lines).upper()
            seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)
            if len(seq) >= 2:
                candidates.append(seq)
        return candidates

    # 2. 非 FASTA：尝试从文本中提取连续氨基酸字符串
    # 先按常见分隔符拆分
    tokens = re.split(r"[,;，；\s]+", text)

    for token in tokens:
        token = token.strip().upper()
        if not token:
            continue
        # 去除明显的非序列词（含数字、太多非标准字符等）
        if re.search(r"\d", token):
            continue
        # 计算标准氨基酸比例
        aa_count = sum(1 for ch in token if ch in _AMINO_ACIDS)
        if len(token) >= 2 and aa_count / len(token) >= 0.8:
            # 清洗为纯氨基酸序列
            clean = "".join(ch for ch in token if ch in _AMINO_ACIDS)
            if len(clean) >= 2:
                candidates.append(clean)

    # 去重保序
    seen = set()
    unique = []
    for seq in candidates:
        if seq not in seen:
            seen.add(seq)
            unique.append(seq)

    return unique


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
    1. 从用户输入中提取多肽序列列表
    2. 通过 LLM / 关键词规则确定需要调用的专家节点
    """
    request = state.get("user_request", "")
    existing_sequences = state.get("sequences", [])
    existing_single = state.get("sequence", "")

    # --- 提取序列 ---
    extracted = _extract_sequences(request)

    # 合并：已有的 sequences + 提取的 + 单条 sequence
    merged = list(existing_sequences)
    if existing_single and existing_single not in merged:
        merged.insert(0, existing_single)
    for seq in extracted:
        if seq not in merged:
            merged.append(seq)

    # 如果仍然没有序列，尝试把整个 user_request 当作序列清洗
    if not merged:
        from .agents import _clean_sequence
        cleaned = _clean_sequence(request)
        if len(cleaned) >= 2:
            merged.append(cleaned)

    # --- 任务路由 ---
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
    print(f"[Planner] 🧬 识别到 {len(merged)} 条多肽序列: {merged}")
    print(f"[Planner] 🎯 分配专家任务: {tasks} (来源: {source})")

    return {
        "sequences": merged,
        "sequence": merged[0] if merged else existing_single,  # 向后兼容
        "required_tasks": tasks,
    }
