"""Peptide Helper 包入口。"""

import os

# 1. 加载 .env 文件（可选）
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ModuleNotFoundError:
    pass

# 2. 加载 config.toml 配置（注入到环境变量，不覆盖已有环境变量）
_CONFIG_LOADED = False


def _load_config_toml():
    """从 config.toml 同步配置到 os.environ（环境变量优先）。"""
    global _CONFIG_LOADED
    if _CONFIG_LOADED:
        return
    _CONFIG_LOADED = True

    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return

    # 搜索 config.toml 路径
    config_paths = [
        os.path.join(os.path.dirname(__file__), "config.toml"),  # 包内同级
        os.path.join(os.getcwd(), "peptide_helper", "config.toml"),
        os.path.join(os.getcwd(), "config.toml"),
    ]
    config_path = None
    for p in config_paths:
        if os.path.isfile(p):
            config_path = p
            break

    if not config_path:
        return

    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        return

    # ── LLM 配置 ──
    llm = cfg.get("llm", {})
    _maybe_setenv("OPENAI_API_KEY", llm.get("openai_api_key"))
    _maybe_setenv("OPENAI_BASE_URL", llm.get("openai_base_url"))
    _maybe_setenv("PEPTIDE_HELPER_MODEL", llm.get("model"))

    # ── ESMFold ──
    esm = cfg.get("esmfold", {})
    _maybe_setenv("PEPTIDE_HELPER_ESMFOLD_BACKEND", esm.get("backend"))
    _maybe_setenv("PEPTIDE_HELPER_ESMFOLD_MODEL", esm.get("model"))
    _maybe_setenv("PEPTIDE_HELPER_ESMFOLD_DEVICE", esm.get("device"))
    if esm.get("local_files_only"):
        _maybe_setenv("PEPTIDE_HELPER_ESMFOLD_LOCAL_ONLY", str(esm.get("local_files_only")))

    # ── 模型超时 ──
    model_cfg = cfg.get("model", {})
    if model_cfg.get("timeout_seconds"):
        _maybe_setenv("PEPTIDE_HELPER_MODEL_TIMEOUT", str(model_cfg.get("timeout_seconds")))

    # ── 毒性模型命令 ──
    tox = cfg.get("toxicity_models", {})
    _maybe_setenv("PEPTIDE_HELPER_TOXINPRED3_CMD", tox.get("toxinpred3_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_TOXINPRED2_CMD", tox.get("toxinpred2_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_TOXIBTL_CMD", tox.get("toxibtl_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_TOXTELLER_CMD", tox.get("toxteller_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_SAFETY_EXTENSION_CMD", tox.get("safety_extension_cmd"))

    # ── 活性模型命令 ──
    act = cfg.get("activity_models", {})
    _maybe_setenv("PEPTIDE_HELPER_AMPSCANNER_CMD", act.get("ampscanner_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_AMPLIFY_CMD", act.get("amplify_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_AMPEPPY_CMD", act.get("ampeppy_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_IAMPCN_CMD", act.get("iampcn_cmd"))
    _maybe_setenv("PEPTIDE_HELPER_DEEP_AMPEP30_CMD", act.get("deep_ampep30_cmd"))


def _maybe_setenv(key: str, value):
    """从 config.toml 注入配置（config.toml 优先于环境变量）。"""
    if value and str(value).strip():
        os.environ[key] = str(value).strip()


# 模块导入时自动执行
_load_config_toml()
