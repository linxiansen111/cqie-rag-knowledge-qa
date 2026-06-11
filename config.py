"""
配置管理模块
管理LLM提供商、API Key、Chroma路径等全局配置
"""

from pathlib import Path
import os
from typing import Optional

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 知识文档目录
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"

# ChromaDB 持久化目录
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# 静态文件目录
STATIC_DIR = BASE_DIR / "static"

# 嵌入模型配置
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================
# LLM 提供商配置
# ============================================================
# 可选值: "ollama" | "openai" | "mock"
#   ollama - 本地 Ollama 模型，完全免费离线运行（默认）
#   openai - 任何兼容 OpenAI 格式的 API 服务
#            （DeepSeek / 硅基流动 / Groq 等）
#   mock   - 模板回答，无需任何安装和API Key（调试用）
# 可通过环境变量设置：set LLM_PROVIDER=openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ----------------------------------------------------------
# OpenAI 兼容 API 配置（LLM_PROVIDER=openai 时生效）
# ----------------------------------------------------------
# API Key（DeepSeek / 硅基流动 / Groq 等平台的 Key）
API_KEY = os.getenv("API_KEY", "")

# API 地址（换成对应平台的地址）
# DeepSeek:   https://api.deepseek.com/v1/chat/completions
# 硅基流动:   https://api.siliconflow.cn/v1/chat/completions
# Groq:       https://api.groq.com/openai/v1/chat/completions
# GitHub:     https://models.inference.ai.azure.com/chat/completions
API_URL = os.getenv("API_URL", "https://api.deepseek.com/v1/chat/completions")

# 模型名称（根据平台和需求选择）
# DeepSeek:   deepseek-chat
# 硅基流动:   Qwen/Qwen2.5-7B-Instruct / deepseek-ai/DeepSeek-V3
# Groq:       qwen-2.5-32b / llama-3.3-70b-versatile
# GitHub:     gpt-4o-mini / DeepSeek-R1 (免费有限额)
API_MODEL = os.getenv("API_MODEL", "deepseek-chat")

# ----------------------------------------------------------
# Ollama 本地模型配置（LLM_PROVIDER=ollama 时生效）
# ----------------------------------------------------------
# Ollama 默认地址（安装后运行即可，无需修改）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# 推荐的中文模型（选一个安装即可）:
#   qwen2.5:7b      - 通义千问7B，中文效果好，推荐 (约4GB)
#   qwen2.5:3b      - 轻量版，速度快 (约2GB)
#   deepseek-r1:7b  - DeepSeek R1 蒸馏版
# 安装方法: 打开终端运行 ollama pull qwen2.5:7b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# 检索配置
TOP_K_RETRIEVAL = 3  # 默认检索Top-K数量
CHUNK_SIZE = 300     # 文档分块大小（字符数）
CHUNK_OVERLAP = 50   # 文档分块重叠大小

# 对话配置
MAX_HISTORY_ROUNDS = 10  # 最多保留最近10轮对话

# 服务配置
HOST = "0.0.0.0"
PORT = 8081

# ============================================================
# 用户持久化设置（在 UI 中切换后保存到此文件）
# ============================================================
USER_SETTINGS_PATH = BASE_DIR / "user_settings.json"


def load_user_settings() -> dict:
    """从 user_settings.json 加载用户持久化设置"""
    if USER_SETTINGS_PATH.exists():
        try:
            import json
            return json.loads(USER_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_settings(provider: str = "", api_key: Optional[str] = None) -> None:
    """保存用户设置到 user_settings.json

    Args:
        provider: LLM 提供商（ollama/openai），为空则不更新
        api_key: DeepSeek API Key
            - 传入字符串：更新 Key
            - 传入 None：保留现有的 Key（不修改）
            - 显式传入 ""：清空 Key
    """
    import json
    settings = {}
    if USER_SETTINGS_PATH.exists():
        try:
            settings = json.loads(USER_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            settings = {}
    if provider:
        settings["provider"] = provider
    if api_key is not None:
        settings["api_key"] = api_key
    USER_SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# === 加载用户持久化设置（覆盖环境变量默认值） ===
_user_settings = load_user_settings()
if _user_settings.get("provider"):
    LLM_PROVIDER = _user_settings["provider"]
if _user_settings.get("api_key"):
    API_KEY = _user_settings["api_key"]
