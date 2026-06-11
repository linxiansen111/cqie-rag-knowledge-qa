"""
企业私有知识问答系统 - FastAPI 应用入口
以重庆工程学院为例的RAG知识库智能问答系统
"""

import logging
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    STATIC_DIR,
    HOST,
    PORT,
    API_MODEL as DEEPSEEK_MODEL,
    OLLAMA_MODEL,
    save_user_settings,
)
from knowledge_base import KnowledgeBase
from chat_engine import ChatEngine
from rag_engine import RAGEngine, OllamaLLM, RealLLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 全局组件实例
# ============================================================
knowledge_base = KnowledgeBase()
chat_engine = ChatEngine()
rag_engine = RAGEngine(knowledge_base, chat_engine)


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    logger.info("正在启动企业私有知识问答系统...")
    try:
        # 启动时自动加载知识库
        result = knowledge_base.load()
        logger.info(f"知识库加载完成: 共 {result['doc_count']} 个文档, {result['chunk_count']} 个片段")
    except Exception as e:
        logger.warning(f"知识库自动加载失败: {e}，服务已启动但知识库为空")
    yield
    logger.info("服务正在关闭...")


# 创建FastAPI应用
app = FastAPI(
    title="企业私有知识问答系统",
    description="基于RAG的企业知识库智能问答系统 - 重庆工程学院示例",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 请求/响应模型
# ============================================================
class SendMessageRequest(BaseModel):
    """发送消息请求"""
    session_id: str = Field(default="", description="会话ID，为空则创建新会话")
    message: str = Field(..., description="用户消息内容")


class SendMessageResponse(BaseModel):
    """发送消息响应"""
    session_id: str
    reply: str
    intent: str
    intent_label: str
    sources: list
    tools_used: list
    history: list


class CreateSessionResponse(BaseModel):
    """创建会话响应"""
    session_id: str
    created_at: str


class HistoryResponse(BaseModel):
    """会话历史响应"""
    session_id: str
    messages: list


class KnowledgeReloadResponse(BaseModel):
    """知识库重载响应"""
    status: str
    doc_count: int
    chunk_count: int


class SessionListItem(BaseModel):
    """会话列表项"""
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str


class LLMConfigRequest(BaseModel):
    """LLM 配置请求"""
    provider: str = Field(..., description="LLM提供商: ollama | openai")
    api_key: str = Field(default="", description="DeepSeek API Key（openai模式需要）")


class LLMConfigResponse(BaseModel):
    """LLM 配置响应"""
    provider: str
    model: str
    has_api_key: bool
    status: str


class LLMConfigInfo(BaseModel):
    """LLM 配置信息"""
    provider: str
    model: str
    has_api_key: bool
    available_providers: list


# ============================================================
# API 路由
# ============================================================
@app.get("/")
async def index():
    """返回前端页面"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在")
    return FileResponse(str(index_path))


@app.post("/api/chat/send", response_model=SendMessageResponse)
async def chat_send(request: SendMessageRequest):
    """
    发送消息并获取智能回答

    处理流程：
    1. 意图识别
    2. 工具调用（检索、摘要等）
    3. RAG生成回答
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    try:
        result = rag_engine.process_question(
            session_id=request.session_id,
            question=request.message,
        )
        return SendMessageResponse(
            session_id=result["session_id"],
            reply=result["reply"],
            intent=result["intent"],
            intent_label=result["intent_label"],
            sources=result["sources"],
            tools_used=result["tools_used"],
            history=result["history"],
        )
    except Exception as e:
        logger.error(f"处理消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理消息失败: {str(e)}")


@app.post("/api/session/create", response_model=CreateSessionResponse)
async def create_session():
    """创建新的对话会话"""
    try:
        session = chat_engine.create_session()
        return CreateSessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
        )
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@app.get("/api/session/list", response_model=List[SessionListItem])
async def list_sessions():
    """获取所有会话列表"""
    try:
        return chat_engine.get_all_sessions()
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.get("/api/session/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: str):
    """获取指定会话的历史记录"""
    session = chat_engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    try:
        return HistoryResponse(
            session_id=session_id,
            messages=session.get_history(),
        )
    except Exception as e:
        logger.error(f"获取会话历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话历史失败: {str(e)}")


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str):
    """重置指定会话（清空历史记录）"""
    success = chat_engine.reset_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "ok", "session_id": session_id}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    success = chat_engine.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "ok", "session_id": session_id}


@app.post("/api/knowledge/reload", response_model=KnowledgeReloadResponse)
async def reload_knowledge():
    """重新加载知识库（重新读取文档并向量化）"""
    try:
        result = knowledge_base.reload()
        return KnowledgeReloadResponse(
            status="ok",
            doc_count=result["doc_count"],
            chunk_count=result["chunk_count"],
        )
    except Exception as e:
        logger.error(f"重新加载知识库失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重新加载知识库失败: {str(e)}")


@app.get("/api/config", response_model=LLMConfigInfo)
async def get_llm_config():
    """获取当前 LLM 配置信息"""
    llm = rag_engine._llm
    if isinstance(llm, OllamaLLM):
        provider = "ollama"
        model = llm.model
    elif isinstance(llm, RealLLM):
        provider = "openai"
        model = DEEPSEEK_MODEL
    else:
        provider = "mock"
        model = ""

    from config import load_user_settings
    user_cfg = load_user_settings()
    # 优先使用保存的 Key，其次运行时内存中的 Key
    has_key = bool(user_cfg.get("api_key")) or bool(getattr(rag_engine, "_deepseek_api_key", ""))

    return LLMConfigInfo(
        provider=provider,
        model=model,
        has_api_key=has_key,
        available_providers=["ollama", "openai", "mock"],
    )


@app.post("/api/config/llm", response_model=LLMConfigResponse)
async def set_llm_config(request: LLMConfigRequest):
    """切换 LLM 提供商（Ollama <-> DeepSeek）"""
    if request.provider not in ("ollama", "openai", "mock"):
        raise HTTPException(status_code=400, detail=f"不支持的 LLM 提供商: {request.provider}")

    try:
        rag_engine.set_llm_provider(request.provider, request.api_key)

        # 持久化保存用户设置（重启后自动恢复）
        # 切换到 Ollama 时不清空 Key，方便用户随时切回 DeepSeek
        save_key = request.api_key if request.provider == "openai" else None
        save_user_settings(provider=request.provider, api_key=save_key)

        llm = rag_engine._llm
        model = getattr(llm, "model", "")
        return LLMConfigResponse(
            provider=request.provider,
            model=model,
            has_api_key=bool(getattr(rag_engine, "_deepseek_api_key", "")),
            status="ok",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"切换 LLM 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"切换失败: {str(e)}")


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    llm = rag_engine._llm
    if isinstance(llm, OllamaLLM):
        llm_mode = "ollama"
        llm_label = f"本地 Ollama ({llm.model})"
    elif isinstance(llm, RealLLM):
        llm_mode = "real"
        llm_label = f"DeepSeek API ({DEEPSEEK_MODEL})"
    else:
        llm_mode = "mock"
        llm_label = "模拟模式(模板回答)"

    return {
        "status": "ok",
        "knowledge_loaded": knowledge_base.is_loaded,
        "doc_count": knowledge_base.document_count,
        "llm_mode": llm_mode,
        "llm_label": llm_label,
    }


# ============================================================
# 挂载静态文件（用于前端资源）
# ============================================================
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info(f"服务启动于 http://{HOST}:{PORT}")
    logger.info(f"知识库目录: 待加载")
    llm = rag_engine._llm
    if isinstance(llm, OllamaLLM):
        llm_label = f"本地 Ollama ({llm.model})"
    elif isinstance(llm, RealLLM):
        llm_label = f"DeepSeek API"
    else:
        llm_label = "模拟模式(模板回答)"
    logger.info(f"LLM模式: {llm_label}")

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
