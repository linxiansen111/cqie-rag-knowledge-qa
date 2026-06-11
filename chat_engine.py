"""
对话引擎模块
管理多轮对话上下文、会话创建和历史记录
"""

from typing import List, Dict, Optional
from datetime import datetime
import uuid
import logging

from config import MAX_HISTORY_ROUNDS

logger = logging.getLogger(__name__)


class ChatSession:
    """单个对话会话"""

    def __init__(self, session_id: Optional[str] = None):
        """
        初始化会话

        Args:
            session_id: 会话ID，不传则自动生成
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.messages: List[Dict] = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        添加消息到对话历史

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 附加元数据（意图、来源、工具等）
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message.update(metadata)

        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

        # 限制历史记录长度（只保留最近N轮）
        self._trim_history()

    def _trim_history(self):
        """裁剪历史记录，只保留最近 MAX_HISTORY_ROUNDS 轮对话"""
        # 统计轮数（每轮包含一条user和一条assistant消息）
        user_count = sum(1 for m in self.messages if m["role"] == "user")
        if user_count > MAX_HISTORY_ROUNDS:
            # 需要删除的最早用户消息数量
            excess = user_count - MAX_HISTORY_ROUNDS
            removed = 0
            while removed < excess and self.messages:
                msg = self.messages[0]
                # 每移除一条user消息，也移除它后面的assistant消息
                if msg["role"] == "user":
                    removed += 1
                    # 移除这条user消息和它后面的assistant消息
                    self.messages.pop(0)
                    if self.messages and self.messages[0]["role"] == "assistant":
                        self.messages.pop(0)
                else:
                    self.messages.pop(0)

    def get_history(self, max_turns: Optional[int] = None) -> List[Dict]:
        """获取对话历史"""
        if max_turns is None:
            return list(self.messages)

        # 获取最近max_turns轮对话
        recent = []
        turn_count = 0
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                turn_count += 1
                if turn_count > max_turns:
                    break
            recent.insert(0, msg)
        return recent

    def get_context_for_llm(self) -> List[Dict]:
        """获取供LLM使用的上下文格式"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages
            if msg["role"] in ("user", "assistant")
        ]

    def to_dict(self) -> Dict:
        """将会话转换为字典"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "preview": self.messages[-1]["content"][:100] if self.messages else "",
        }


class ChatEngine:
    """对话引擎：管理所有会话"""

    def __init__(self):
        """初始化对话引擎"""
        self._sessions: Dict[str, ChatSession] = {}

    def create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """
        创建新会话

        Args:
            session_id: 可选，指定会话ID

        Returns:
            创建的会话对象
        """
        session = ChatSession(session_id)
        self._sessions[session.session_id] = session
        logger.info(f"创建新会话: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        获取指定会话

        Args:
            session_id: 会话ID

        Returns:
            会话对象，不存在返回None
        """
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """
        获取已有会话或创建新会话

        Args:
            session_id: 会话ID

        Returns:
            会话对象
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(session_id)

    def get_all_sessions(self) -> List[Dict]:
        """获取所有会话列表（按更新时间降序排列）"""
        sessions = [
            session.to_dict()
            for session in self._sessions.values()
        ]
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False

    def reset_session(self, session_id: str) -> bool:
        """重置会话（清空历史记录）"""
        session = self._sessions.get(session_id)
        if session:
            session.messages = []
            session.updated_at = datetime.now().isoformat()
            logger.info(f"重置会话: {session_id}")
            return True
        return False

    def add_user_message(self, session_id: str, content: str) -> bool:
        """添加用户消息到会话"""
        session = self._sessions.get(session_id)
        if session:
            session.add_message("user", content)
            return True
        return False

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        intent: str = "general",
        sources: Optional[List[Dict]] = None,
        tools_used: Optional[List[str]] = None,
    ):
        """添加助手回复到会话"""
        session = self._sessions.get(session_id)
        if session:
            metadata = {
                "intent": intent,
                "sources": sources or [],
                "tools_used": tools_used or [],
            }
            session.add_message("assistant", content, metadata)
            return True
        return False

    def get_history_for_intent(self, session_id: str) -> List[Dict]:
        """获取用于意图识别的上下文历史"""
        session = self._sessions.get(session_id)
        if not session:
            return []

        context = []
        for msg in reversed(session.messages):
            if msg["role"] == "assistant" and "intent" in msg:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "intent": msg["intent"],
                })
            elif msg["role"] == "user":
                context.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            if len(context) >= 4:  # 最多取最近2轮
                break

        return list(reversed(context))
