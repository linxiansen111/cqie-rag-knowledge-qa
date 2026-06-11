"""
工具调用管理模块
实现文档检索、摘要生成、按意图查询等工具
"""

from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ToolManager:
    """工具管理器：管理和执行RAG相关的工具调用"""

    def __init__(self, retrieve_fn: Callable, summarize_fn: Callable, query_by_intent_fn: Callable):
        """
        初始化工具管理器

        Args:
            retrieve_fn: 文档检索函数
            summarize_fn: 文档摘要函数
            query_by_intent_fn: 按意图查询函数
        """
        self._tools: Dict[str, Dict[str, Any]] = {
            "retrieve_docs": {
                "name": "retrieve_docs",
                "description": "从知识库中检索与查询相关的文档片段",
                "parameters": {
                    "query": {"type": "string", "description": "检索查询语句"},
                    "top_k": {"type": "integer", "description": "返回的文档数量", "default": 3},
                },
                "fn": retrieve_fn,
            },
            "summarize_docs": {
                "name": "summarize_docs",
                "description": "对检索到的文档片段进行智能摘要和整合",
                "parameters": {
                    "docs": {"type": "array", "description": "需要摘要的文档片段列表"},
                },
                "fn": summarize_fn,
            },
            "query_by_intent": {
                "name": "query_by_intent",
                "description": "根据意图类别路由到相应的知识库区域进行查询",
                "parameters": {
                    "intent": {"type": "string", "description": "意图类别"},
                    "params": {"type": "object", "description": "查询参数"},
                },
                "fn": query_by_intent_fn,
            },
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取所有可用工具列表"""
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for info in self._tools.values()
        ]

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """获取工具的用途描述"""
        tool = self._tools.get(tool_name)
        return tool["description"] if tool else None

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 工具不存在时抛出
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"未知工具: {tool_name}，可用工具: {list(self._tools.keys())}")

        logger.info(f"调用工具: {tool_name}, 参数: {kwargs}")
        try:
            result = tool["fn"](**kwargs)
            logger.info(f"工具 {tool_name} 执行成功")
            return result
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {e}")
            raise

    def auto_select_tools(self, intent: str, question: str) -> List[str]:
        """
        根据意图和问题自动选择需要调用的工具组合

        Args:
            intent: 意图类别
            question: 用户问题

        Returns:
            工具名称列表（按调用顺序）
        """
        # 基础工具链：先检索，再摘要
        base_tools = ["retrieve_docs", "summarize_docs"]

        # 根据意图添加特定工具
        intent_tool_map = {
            "school_info": ["query_by_intent"],
            "major_query": ["query_by_intent"],
            "admission": ["query_by_intent"],
            "campus_life": ["query_by_intent"],
            "contact": ["query_by_intent"],
        }

        tools = base_tools.copy()
        if intent in intent_tool_map:
            # 将按意图查询工具插到检索之前，用于优化检索范围
            tools = intent_tool_map[intent] + tools

        return tools


def create_default_retrieve_fn(knowledge_base) -> Callable:
    """创建默认的文档检索工具函数"""
    import logging
    logger = logging.getLogger(__name__)

    def retrieve_docs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """从知识库中检索文档"""
        logger.info(f"执行文档检索: query='{query}', top_k={top_k}")
        try:
            results = knowledge_base.search(query, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return []

    return retrieve_docs


def create_default_summarize_fn() -> Callable:
    """创建默认的文档摘要工具函数"""
    import logging
    logger = logging.getLogger(__name__)

    def summarize_docs(docs: List[Dict[str, Any]]) -> str:
        """对文档片段进行摘要整合"""
        if not docs:
            return "未检索到相关文档。"

        logger.info(f"执行文档摘要: 共{len(docs)}个文档片段")

        combined = []
        seen = set()
        for doc in docs:
            # 跳过综合摘要自身（防止循环引用）
            if doc.get("title") == "综合摘要":
                continue
            content = doc.get("content", "").strip()
            if content and content not in seen:
                combined.append(content)
                seen.add(content)

        # 清理格式：去掉 markdown 标题标记，保持可读性
        def clean_text(text: str) -> str:
            lines = text.split("\n")
            cleaned = []
            for line in lines:
                line = line.strip().lstrip("#").strip()
                if line:
                    cleaned.append(line)
            return "\n".join(cleaned)

        summary_parts = []
        for content in combined:
            cleaned = clean_text(content[:500])
            if cleaned:
                summary_parts.append(cleaned)

        return "\n\n".join(summary_parts) if summary_parts else "未检索到相关文档。"

    return summarize_docs


def create_default_query_by_intent_fn(knowledge_base) -> Callable:
    """创建默认的按意图查询工具函数"""
    import logging
    logger = logging.getLogger(__name__)

    # 意图到检索关键词的映射
    INTENT_QUERY_MAP = {
        "school_info": ["重庆工程学院", "学校概况", "简介", "校区"],
        "major_query": ["专业", "学院", "院系", "学科"],
        "admission": ["招生", "录取", "学费", "报考"],
        "campus_life": ["宿舍", "食堂", "校园", "生活", "设施"],
        "contact": ["电话", "地址", "联系方式", "邮箱"],
    }

    def query_by_intent(intent: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """根据意图从知识库检索"""
        logger.info(f"执行按意图查询: intent='{intent}', params={params}")

        keywords = INTENT_QUERY_MAP.get(intent, ["重庆工程学院"])
        # 使用意图关键词联合检索
        all_results = []
        seen_content = set()

        for keyword in keywords:
            results = knowledge_base.search(keyword, top_k=3)
            for r in results:
                content = r.get("content", "").strip()
                if content and content not in seen_content:
                    all_results.append(r)
                    seen_content.add(content)

        logger.info(f"按意图查询到 {len(all_results)} 个结果")
        return all_results[:5]  # 最多返回5个

    return query_by_intent
