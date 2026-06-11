"""
RAG引擎模块
整合检索、意图识别、工具调用和回答生成
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import re

from config import (
    LLM_PROVIDER as LLM_MODE,
    API_KEY as DEEPSEEK_API_KEY,
    API_URL as DEEPSEEK_API_URL,
    API_MODEL as DEEPSEEK_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from intent_recognizer import IntentRecognizer
from tool_manager import (
    ToolManager,
    create_default_retrieve_fn,
    create_default_summarize_fn,
    create_default_query_by_intent_fn,
)

logger = logging.getLogger(__name__)


class MockLLM:
    """模拟LLM：基于模板和检索结果生成回答，无需API Key"""

    # 意图回答模板
    INTENT_TEMPLATES = {
        "school_info": (
            "根据重庆工程学院的资料，以下是为您整理的相关信息：\n\n{content}\n\n"
            "如您需要了解更多关于学校的信息，请继续提问。"
        ),
        "major_query": (
            "关于重庆工程学院的院系专业信息如下：\n\n{content}\n\n"
            "如果您想了解某个专业的详细信息，请进一步询问。"
        ),
        "admission": (
            "关于重庆工程学院的招生信息，为您整理如下：\n\n{content}\n\n"
            "如需了解更多招生详情，建议访问学校招生官网 http://zs.cqie.edu.cn 或拨打招生咨询电话 023-62846626。"
        ),
        "campus_life": (
            "关于重庆工程学院的校园生活，为您介绍如下：\n\n{content}\n\n"
            "学校致力于为学生提供良好的学习和生活环境。"
        ),
        "contact": (
            "重庆工程学院的联系方式如下：\n\n{content}\n\n"
            "如有更多问题，欢迎随时联系学校。"
        ),
        "general": (
            "以下是根据相关文档找到的信息：\n\n{content}\n\n"
            "如果您的问题未得到准确解答，请尝试换个角度提问。"
        ),
    }

    def generate(
        self,
        question: str,
        intent: str,
        retrieved_docs: List[Dict[str, Any]],
        context: Optional[List[Dict]] = None,
    ) -> str:
        """
        根据模板和检索结果生成回答

        Args:
            question: 用户问题
            intent: 意图类别
            retrieved_docs: 检索到的文档片段
            context: 对话上下文

        Returns:
            生成的回答文本
        """
        # ---- 1. 提取对话上下文，识别追问关系 ----
        prev_question = ""
        prev_answer_snippet = ""
        last_intent = ""
        follow_up = False

        if context:
            # 注意：当前用户的问题已被 add_user_message 提前写入上下文
            # 所以 context 的最后一个 user 消息就是当前问题，需要跳过
            recent = context[-5:] if len(context) >= 5 else context
            user_count = 0
            for msg in reversed(recent):
                if msg.get("role") == "user":
                    user_count += 1
                    if user_count == 1:
                        # 跳过最近一条用户消息（就是当前问题）
                        continue
                    if not prev_question:
                        prev_question = msg["content"]
                elif msg.get("role") == "assistant":
                    if not prev_answer_snippet:
                        prev_answer_snippet = msg.get("content", "")[:150]

            # 判断当前是否为追问：
            # a) 问题长度<25字符 且 有上文
            # b) 以"那/这/它/哪个/还有/然后"等连词代词开头
            follow_up_re = re.compile(
                r"^(那|这|它|他|她|该|其|哪个|哪些|什么|多少|"
                r"还有|然后|另外|其他|别的|最大|最小|最好|"
                r"怎么|如何|能不能|可不可以|有(没有|几个|哪些))"
            )
            follow_up = (
                (len(question) < 25 and bool(prev_question))
                or bool(follow_up_re.match(question))
            )

        # ---- 2. 整理检索内容 ----
        content_parts = []
        seen_titles = set()
        for doc in retrieved_docs:
            title = doc.get("title", "未知来源")
            content = doc.get("content", "").strip()
            if content and title not in seen_titles:
                seen_titles.add(title)
                content_parts.append(f"【{title}】\n{content}")

        if not content_parts:
            return (
                f"抱歉，我没有在知识库中找到与「{question}」相关的信息。\n\n"
                f"您可以尝试以下操作：\n"
                f"1. 换个角度重新描述您的问题\n"
                f"2. 询问关于重庆工程学院的其他方面\n"
                f"3. 联系学校获取更准确的信息"
            )

        # ---- 3. 智能整合 ----
        formatted_content = "\n\n".join(content_parts)

        if intent == "school_info" and "校区" in question:
            formatted_content = self._extract_campus_info(retrieved_docs, question)
        elif intent == "major_query":
            formatted_content = self._extract_major_info(retrieved_docs)

        # ---- 4. 构建回答（融入上下文） ----
        template = self.INTENT_TEMPLATES.get(
            intent, self.INTENT_TEMPLATES["general"]
        )

        answer = template.format(content=formatted_content)

        # 如果是追问且有上文信息，在回答前加自然衔接
        if follow_up and prev_question:
            answer = (
                f"您之前问的是「{prev_question}」，"
                f"现在接着回答您的问题：\n\n{answer}"
            )

        return answer

    def _extract_campus_info(
        self, docs: List[Dict[str, Any]], question: str
    ) -> str:
        """提取校区相关信息"""
        campus_parts = []
        for doc in docs:
            content = doc.get("content", "")
            # 提取包含"校区"的行
            lines = content.split("\n")
            relevant = [
                line.strip()
                for line in lines
                if "校区" in line or "面积" in line or "位于" in line
            ]
            if relevant:
                title = doc.get("title", "")
                campus_parts.append(f"【{title}】\n" + "\n".join(relevant))
            else:
                campus_parts.append(f"【{doc.get('title', '')}】\n{content[:200]}")

        return "\n\n".join(campus_parts) if campus_parts else "\n\n".join(
            [d.get("content", "")[:200] for d in docs]
        )

    def _extract_major_info(self, docs: List[Dict[str, Any]]) -> str:
        """提取专业信息"""
        for doc in docs:
            content = doc.get("content", "")
            title = doc.get("title", "")
            if "院系" in title or "专业" in content[:100]:
                return content
        return "\n\n".join(
            [f"【{d.get('title', '')}】\n{d.get('content', '')[:300]}" for d in docs]
        )


class RealLLM:
    """真实LLM：通过API调用DeepSeek生成回答

    支持重试、超时控制、详细的错误诊断。
    使用 requests 库，需确保已安装: pip install requests
    DeepSeek API 兼容 OpenAI 格式，可无缝切换其他兼容服务。
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未设置")
        import requests as req_lib
        self._requests = req_lib
        # 创建带重试的 Session
        self._session = self._build_session()

    def _build_session(self):
        """构建带重试机制的 requests Session"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = self._requests.Session()
        retry_strategy = Retry(
            total=3,                      # 最多重试3次
            backoff_factor=1,              # 退避因子: 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def generate(
        self,
        question: str,
        intent: str,
        retrieved_docs: List[Dict[str, Any]],
        context: Optional[List[Dict]] = None,
    ) -> str:
        """通过 DeepSeek API 生成回答

        流程：构建消息 → 调用 API → 解析响应 → 返回文本
        """
        # ---- 1. 构建系统提示 ----
        system_prompt = (
            "你是一个企业知识库智能问答助手，负责回答关于重庆工程学院的问题。\n"
            f"用户的问题意图是: {intent}\n\n"
            "请严格遵循以下规则：\n"
            "1. 优先基于知识库文档回答，不编造文档中不存在的信息\n"
            "2. 回答要简洁准确，使用中文，分点列出\n"
            "3. 如果参考文档与用户问题无关（如用户问助手自身的问题：你是谁、你是什么模型等），\n"
            "   请忽略参考文档直接回答，告知用户你是基于 DeepSeek 大模型的企业知识库智能问答助手\n"
            "4. 如果参考文档不足以回答问题，如实告知用户不清楚，不要强行套用无关内容\n"
            "5. 结合对话上下文理解追问含义（如\"那\"\"它\"\"哪个最大\"等指代）\n"
            "6. 在回答末尾标注引用的知识来源文档名称"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # ---- 2. 注入对话历史（最近 3 轮） ----
        if context:
            for msg in context[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # ---- 3. 注入检索结果 ----
        doc_sections = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.get("title", "未知")
            content = doc.get("content", "").strip()
            if content:
                doc_sections.append(f"[{i}] 来源:《{title}》\n{content}")

        doc_context = "参考文档：\n\n" + "\n\n".join(doc_sections)

        # 将检索结果和用户问题一起发给 LLM
        messages.append({
            "role": "user",
            "content": f"{doc_context}\n\n=====\n用户问题：{question}",
        })

        # ---- 4. 估算 token 用量并截断 ----
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars > 30000:  # 估算约7500+ tokens，留安全余量
            logger.warning(f"上下文过长({total_chars}字符)，截断至30000字符")
            # 截断最长的消息
            for m in messages:
                if len(str(m.get("content", ""))) > 8000:
                    m["content"] = m["content"][:8000] + "\n...(已截断)"

        # ---- 5. 调用 DeepSeek API ----
        data = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.3,          # 知识问答用较低温度，减少幻觉
            "max_tokens": 2048,
            "top_p": 0.9,
        }

        logger.info(f"调用 DeepSeek API: model={DEEPSEEK_MODEL}, messages={len(messages)}, "
                     f"retrieved={len(retrieved_docs)}, chars={total_chars}")

        try:
            resp = self._session.post(
                DEEPSEEK_API_URL,
                json=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=60,
            )

            # ---- 6. 处理响应 ----
            if resp.status_code == 401:
                logger.error("DeepSeek API Key 无效或已过期")
                return "抱歉，API认证失败，请检查 DeepSeek API Key 是否正确配置。"
            elif resp.status_code == 429:
                logger.error("DeepSeek API 请求频率超限")
                return "抱歉，当前请求过于频繁，请稍后再试。"
            elif resp.status_code >= 500:
                logger.error(f"DeepSeek 服务端错误: {resp.status_code}")
                return "抱歉，AI服务暂时不可用（服务端错误），请稍后再试。"

            resp.raise_for_status()
            result = resp.json()
            reply = result["choices"][0]["message"]["content"]
            logger.info(f"DeepSeek 回答完成: tokens={result.get('usage', {})}")
            return reply

        except self._requests.exceptions.Timeout:
            logger.error("DeepSeek API 请求超时")
            return "抱歉，AI服务响应超时，请稍后重试或简化您的问题。"
        except self._requests.exceptions.ConnectionError as e:
            logger.error(f"DeepSeek API 连接失败: {e}")
            return (f"抱歉，无法连接到 DeepSeek API。请检查网络连接，"
                    f"或在 config.py 中配置 HTTPS_PROXY 环境变量。")
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {e}", exc_info=True)
            return f"抱歉，AI服务暂时不可用，请稍后再试。错误信息: {str(e)}"


class OllamaLLM:
    """本地 Ollama 模型，完全离线运行

    通过 Ollama Python 客户端调用本地的 Ollama 服务，
    支持所有 Ollama 兼容模型（qwen2.5, deepseek-r1, llama 等）。
    需先安装: pip install ollama
    """

    def __init__(self, base_url: str, model: str):
        """
        Args:
            base_url: Ollama 服务地址（默认 http://localhost:11434）
            model: 模型名称（如 qwen2.5:7b）
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = None
        logger.info(f"OllamaLLM 初始化: url={base_url}, model={model}")

    def _get_client(self):
        """延迟初始化 Ollama 客户端"""
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def generate(
        self,
        question: str,
        intent: str,
        retrieved_docs: List[Dict[str, Any]],
        context: Optional[List[Dict]] = None,
    ) -> str:
        """通过本地 Ollama 模型生成回答

        流程：构建消息 → 调用 Ollama → 返回回答文本
        """
        # ---- 1. 构建系统提示 ----
        system_prompt = (
            "你是一个企业知识库智能问答助手，负责回答关于重庆工程学院的问题。\n"
            f"用户的问题意图是: {intent}\n\n"
            "请严格遵循以下规则：\n"
            "1. 优先基于知识库文档回答，不编造文档中不存在的信息\n"
            "2. 回答要简洁准确，使用中文，分点列出\n"
            "3. 如果参考文档不足以回答问题，如实告知用户不清楚\n"
            "4. 结合对话上下文理解追问含义（如「那」「它」「哪个最大」等指代）\n"
            "5. 在回答末尾标注引用的知识来源文档名称"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # ---- 2. 注入对话历史（最近 3 轮） ----
        if context:
            for msg in context[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # ---- 3. 注入检索结果 ----
        doc_sections = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.get("title", "未知")
            content = doc.get("content", "").strip()
            if content:
                doc_sections.append(f"[{i}] 来源:《{title}》\n{content}")

        doc_context = "参考文档：\n\n" + "\n\n".join(doc_sections)

        messages.append({
            "role": "user",
            "content": f"{doc_context}\n\n=====\n用户问题：{question}",
        })

        # ---- 4. 调用 Ollama ----
        logger.info(
            f"调用 Ollama: model={self.model}, "
            f"messages={len(messages)}, retrieved={len(retrieved_docs)}"
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.3,   # 知识问答用较低温度，减少幻觉
                    "top_p": 0.9,
                },
            )
            reply = response["message"]["content"]
            logger.info(f"Ollama 回答完成: {len(reply)} 字符")
            return reply

        except Exception as e:
            logger.error(f"Ollama 调用失败: {e}", exc_info=True)
            return (
                f"抱歉，本地 AI 模型暂时不可用。\n\n"
                f"错误信息: {str(e)}\n\n"
                f"请确认：\n"
                f"1. Ollama 服务已启动（ollama serve）\n"
                f"2. 模型已下载（ollama pull {self.model}）\n"
                f"3. Ollama 地址配置正确（当前: {self.base_url}）"
            )


class RAGEngine:
    """RAG引擎：整合检索、工具调用和回答生成的主引擎"""

    def __init__(self, knowledge_base, chat_engine):
        """
        初始化RAG引擎

        Args:
            knowledge_base: KnowledgeBase实例
            chat_engine: ChatEngine实例
        """
        self.knowledge_base = knowledge_base
        self.chat_engine = chat_engine
        self.intent_recognizer = IntentRecognizer()

        # 创建工具
        retrieve_fn = create_default_retrieve_fn(knowledge_base)
        summarize_fn = create_default_summarize_fn()
        query_by_intent_fn = create_default_query_by_intent_fn(knowledge_base)
        self.tool_manager = ToolManager(retrieve_fn, summarize_fn, query_by_intent_fn)

        # 初始化LLM
        self._llm = self._create_llm()

    def _create_llm(self):
        """根据配置创建LLM实例"""
        if LLM_MODE == "ollama":
            try:
                logger.info(f"使用本地 Ollama 模式: {OLLAMA_MODEL}")
                return OllamaLLM(OLLAMA_BASE_URL, OLLAMA_MODEL)
            except Exception as e:
                logger.warning(f"Ollama 初始化失败，回退到模拟模式: {e}")

        elif (LLM_MODE == "openai" or LLM_MODE == "real") and DEEPSEEK_API_KEY:
            try:
                logger.info("使用真实LLM模式（DeepSeek）")
                return RealLLM(DEEPSEEK_API_KEY)
            except Exception as e:
                logger.warning(f"真实LLM初始化失败，回退到模拟模式: {e}")

        logger.info("使用模拟LLM模式")
        return MockLLM()

    def set_llm_provider(self, provider: str, api_key: str = ""):
        """运行时动态切换 LLM 提供商

        Args:
            provider: "ollama" | "openai" | "mock"
            api_key: DeepSeek API Key（provider="openai" 时需要）
        """
        if provider == "ollama":
            self._llm = OllamaLLM(OLLAMA_BASE_URL, OLLAMA_MODEL)
            logger.info(f"已切换到本地 Ollama 模式: {OLLAMA_MODEL}")

        elif provider == "openai":
            if not api_key:
                # 尝试使用之前保存的 Key
                api_key = getattr(self, "_deepseek_api_key", "")
            if not api_key:
                raise ValueError("使用 DeepSeek API 需要提供 API Key")
            self._deepseek_api_key = api_key
            self._llm = RealLLM(api_key)
            logger.info("已切换到 DeepSeek API 模式")

        elif provider == "mock":
            self._llm = MockLLM()
            logger.info("已切换到模拟模式")

        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

    def process_question(
        self, session_id: str, question: str
    ) -> Dict[str, Any]:
        """
        处理用户问题：意图识别 -> 工具调用 -> 检索 -> 生成回答

        Args:
            session_id: 会话ID
            question: 用户问题

        Returns:
            包含回答、意图、来源、工具使用等信息的字典
        """
        if not question or not question.strip():
            return {
                "reply": "请输入您的问题。",
                "intent": "general",
                "sources": [],
                "tools_used": [],
            }

        question = question.strip()
        logger.info(f"处理问题: session={session_id}, question='{question}'")

        # 获取会话和上下文
        session = self.chat_engine.get_or_create_session(session_id)
        intent_context = self.chat_engine.get_history_for_intent(session_id)

        # 1. 意图识别
        intent = self.intent_recognizer.recognize(question, intent_context)
        intent_label = self.intent_recognizer.get_intent_label(intent)
        logger.info(f"识别意图: {intent} ({intent_label})")

        # 2. 保存用户消息
        self.chat_engine.add_user_message(session_id, question)

        # 3. 自动选择工具
        tools_to_use = self.tool_manager.auto_select_tools(intent, question)
        logger.info(f"选择工具: {tools_to_use}")

        # 4. 执行工具调用
        retrieved_docs = []
        tools_executed = ["intent_recognition"]

        for tool_name in tools_to_use:
            try:
                if tool_name == "query_by_intent":
                    result = self.tool_manager.call_tool(
                        "query_by_intent",
                        intent=intent,
                        params={"question": question},
                    )
                    if result:
                        retrieved_docs.extend(result)
                    tools_executed.append(tool_name)

                elif tool_name == "retrieve_docs":
                    result = self.tool_manager.call_tool(
                        "retrieve_docs",
                        query=question,
                        top_k=3,
                    )
                    if result:
                        # 合并结果，去重
                        existing_titles = {d.get("title") for d in retrieved_docs}
                        for r in result:
                            if r.get("title") not in existing_titles:
                                retrieved_docs.append(r)
                                existing_titles.add(r.get("title"))
                    tools_executed.append(tool_name)

                elif tool_name == "summarize_docs":
                    if retrieved_docs:
                        summary_text = self.tool_manager.call_tool(
                            "summarize_docs",
                            docs=retrieved_docs,
                        )
                        # 将摘要作为综合整合结果注入到检索结果中
                        # 放在首位，让 LLM 优先看到全局摘要
                        if summary_text and summary_text != "未检索到相关文档。":
                            retrieved_docs.insert(0, {
                                "title": "综合摘要",
                                "content": summary_text[:500],
                                "score": 1.0,
                                "source": "",
                            })
                        tools_executed.append(tool_name)
                    else:
                        summary_text = ""

            except Exception as e:
                logger.error(f"工具 {tool_name} 调用失败: {e}")

        # 5. 获取对话上下文
        context = session.get_context_for_llm()

        # 6. 生成回答
        try:
            reply = self._llm.generate(
                question=question,
                intent=intent,
                retrieved_docs=retrieved_docs,
                context=context,
            )
        except Exception as e:
            logger.error(f"回答生成失败: {e}")
            reply = f"抱歉，生成回答时遇到问题: {str(e)}"

        # 7. 去重并限制来源数量
        seen = set()
        unique_sources = []
        for doc in retrieved_docs:
            title = doc.get("title", "未知来源")
            if title not in seen:
                seen.add(title)
                unique_sources.append({
                    "title": title,
                    "content": doc.get("content", "")[:300],
                    "score": doc.get("score", 0),
                })
                if len(unique_sources) >= 3:
                    break

        # 8. 保存助手回复
        self.chat_engine.add_assistant_message(
            session_id,
            reply,
            intent=intent,
            sources=unique_sources,
            tools_used=tools_executed,
        )

        # 9. 构建响应
        response = {
            "session_id": session_id,
            "reply": reply,
            "intent": intent,
            "intent_label": intent_label,
            "sources": unique_sources,
            "tools_used": tools_executed,
            "history": session.get_history(max_turns=5),
        }

        logger.info(f"回答完成: intent={intent}, tools={tools_executed}, sources={len(unique_sources)}")
        return response
