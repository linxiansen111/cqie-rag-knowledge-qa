"""
知识库管理模块
负责文档加载、拆分、向量化存储和检索
优先使用 ChromaDB ONNX 嵌入模型，自动降级到关键词检索（无需任何模型下载）
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re
import math
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from config import (
    KNOWLEDGE_DIR,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
)

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """
    基于关键词匹配的轻量级检索器（BM25风格）
    无需任何外部模型或依赖，纯Python实现
    """

    def __init__(self):
        self._chunks: List[Dict[str, Any]] = []
        self._vocab: set = set()
        self._doc_freq: Dict[str, int] = {}
        self._avg_len: float = 0.0
        self._k1: float = 1.5
        self._b: float = 0.75

    def _tokenize(self, text: str) -> List[str]:
        """中文+英文分词"""
        # 中文单字+英文单词
        tokens = []
        # 匹配中文单字和英文单词
        for part in re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9_]+', text.lower()):
            tokens.append(part)
        return tokens

    def index(self, chunks: List[Dict[str, Any]]):
        """建立倒排索引"""
        self._chunks = chunks
        doc_count = len(chunks)

        # 统计文档频率
        doc_freq = Counter()
        doc_lengths = []
        for chunk in chunks:
            tokens = self._tokenize(chunk["content"])
            doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freq[t] += 1

        self._doc_freq = dict(doc_freq)
        self._vocab = set(doc_freq.keys())
        self._avg_len = sum(doc_lengths) / max(doc_count, 1)
        logger.info(f"关键词索引完成: {doc_count} 个片段, {len(self._vocab)} 个词汇")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """BM25检索"""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._chunks:
            return []

        n = len(self._chunks)
        scores = []

        for i, chunk in enumerate(self._chunks):
            doc_tokens = self._tokenize(chunk["content"])
            doc_len = len(doc_tokens)
            term_freq = Counter(doc_tokens)

            score = 0.0
            for qt in set(query_tokens):
                if qt not in self._vocab:
                    continue
                df = self._doc_freq.get(qt, 1)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = term_freq.get(qt, 0)
                if tf > 0:
                    score += idf * (tf * (self._k1 + 1)) / (
                        tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_len)
                    )

            scores.append((i, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results = []
        for idx, score in top_scores:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "score": round(min(score / 5.0, 1.0), 4),  # 归一化
                "source": chunk["source"],
            })

        return results


class KnowledgeBase:
    """知识库管理器：文档加载、拆分、向量化和检索"""

    def __init__(self):
        """初始化知识库管理器"""
        self._embedding_function = None
        self._vector_store = None
        self._chroma_collection = None
        self._keyword_retriever = KeywordRetriever()
        self._chunks_meta: List[Dict[str, Any]] = []
        self._documents: List[Dict[str, Any]] = []
        self._is_loaded: bool = False
        self._use_vector: bool = False  # 是否使用向量检索

    def _try_init_vector_store(self, chunks: List[LangChainDocument]):
        """
        尝试初始化向量存储（ChromaDB + 嵌入模型）。

        如果网络可达，会自动下载 sentence-transformers/all-MiniLM-L6-v2 模型。
        如果网络不可达或下载失败，自动降级为关键词检索（BM25），不影响服务正常运行。

        如需启用向量检索，建议：
        1. 配置 HTTP 代理: set HTTPS_PROXY=http://your-proxy:port
        2. 或手动预先下载模型到本地 cache 目录
        """
        try:
            logger.info("正在加载嵌入模型: all-MiniLM-L6-v2（首次约需几分钟下载）")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=str(CHROMA_DB_DIR.parent / "onnx_models"),
            )
            logger.info("嵌入模型加载成功")

            # 将 LangChain Document 转为 Chroma 格式
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [
                {"title": chunk.metadata.get("title", "未知来源"),
                 "source": chunk.metadata.get("source", "")}
                for chunk in chunks
            ]
            ids = [f"chunk_{i}" for i in range(len(chunks))]

            import chromadb
            from chromadb.config import Settings

            chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DB_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = chroma_client.get_or_create_collection(
                name="enterprise_qa",
                metadata={"hnsw:space": "cosine"},
            )

            # 检查集合是否已有数据
            existing_count = collection.count()
            if existing_count == 0:
                # 嵌入
                embeddings = model.encode(texts, show_progress_bar=True).tolist()
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(f"向量库写入完成: {len(chunks)} 个片段")
            else:
                logger.info(f"向量库已有数据，跳过写入（{existing_count} 条）")

            self._chroma_collection = collection
            self._use_vector = True
            return True

        except ImportError as e:
            logger.warning(f"嵌入模型依赖缺失: {e}，自动降级为关键词检索")
            return False
        except Exception as e:
            logger.warning(f"向量存储初始化失败: {e}，自动降级为关键词检索（BM25）")
            return False

    def _load_markdown_files(self) -> List[LangChainDocument]:
        """
        加载 KNOWLEDGE_DIR 目录下的所有 markdown 文件

        Returns:
            LangChain Document 列表
        """
        if not KNOWLEDGE_DIR.exists():
            logger.warning(f"知识库目录不存在: {KNOWLEDGE_DIR}")
            return []

        documents = []
        md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))

        if not md_files:
            logger.warning(f"知识库目录中没有markdown文件: {KNOWLEDGE_DIR}")
            return []

        for file_path in md_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                doc = LangChainDocument(
                    page_content=content,
                    metadata={
                        "title": file_path.name,
                        "source": str(file_path),
                    },
                )
                documents.append(doc)
                logger.info(f"加载文档: {file_path.name} ({len(content)} 字符)")
            except Exception as e:
                logger.error(f"加载文档失败 {file_path}: {e}")

        return documents

    def _split_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """将文档拆分成小块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n---\n", "\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"文档拆分完成: {len(documents)} 个文档 -> {len(chunks)} 个片段")
        return chunks

    def load(self) -> Dict[str, int]:
        """
        加载知识库：读取文档、拆分、向量化并存入ChromaDB
        如果向量化失败，自动降级到关键词检索

        Returns:
            加载结果统计: {"doc_count": 文档数, "chunk_count": 片段数}
        """
        logger.info("开始加载知识库...")

        # 加载原始文档
        raw_docs = self._load_markdown_files()
        if not raw_docs:
            logger.warning("未加载到任何文档")
            return {"doc_count": 0, "chunk_count": 0}

        # 拆分文档
        chunks = self._split_documents(raw_docs)

        # 保存文档元数据
        self._documents = [
            {
                "title": doc.metadata.get("title", "unknown"),
                "content": doc.page_content[:200] + "...",
            }
            for doc in raw_docs
        ]

        # 保存chunk元数据（用于关键词检索降级）
        self._chunks_meta = [
            {
                "title": chunk.metadata.get("title", "未知来源"),
                "content": chunk.page_content.strip(),
                "source": chunk.metadata.get("source", ""),
            }
            for chunk in chunks
        ]

        # 确保ChromaDB目录存在
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        # 尝试向量化，失败则用关键词检索
        vector_ok = self._try_init_vector_store(chunks)
        if not vector_ok:
            logger.info("使用关键词检索降级方案")
            self._keyword_retriever.index(self._chunks_meta)

        self._is_loaded = True

        result = {
            "doc_count": len(raw_docs),
            "chunk_count": len(chunks),
        }
        logger.info(f"知识库加载完成: {result} (检索模式: {'向量检索' if vector_ok else '关键词检索'})")
        return result

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        在知识库中检索与查询最相关的文档片段

        Args:
            query: 检索查询
            top_k: 返回结果数量，默认使用配置值

        Returns:
            检索结果列表，每个结果包含 title、content、score 和 source
        """
        if top_k is None:
            top_k = TOP_K_RETRIEVAL

        if not self._is_loaded:
            logger.warning("知识库尚未加载，尝试自动加载...")
            self.load()

        if not self._is_loaded:
            logger.error("知识库加载失败，无法检索")
            return []

        try:
            if self._use_vector and self._chroma_collection is not None:
                # 向量检索
                results = self._chroma_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )
                items = []
                if results and results.get("documents"):
                    for i in range(len(results["documents"][0])):
                        items.append({
                            "title": results["metadatas"][0][i].get("title", "未知来源") if results.get("metadatas") else "未知来源",
                            "content": results["documents"][0][i].strip(),
                            "score": 1.0 - (results["distances"][0][i] / 2.0) if results.get("distances") else 0.5,
                            "source": results["metadatas"][0][i].get("source", "") if results.get("metadatas") else "",
                        })
                items.sort(key=lambda x: x["score"], reverse=True)
                logger.info(f"向量检索完成: query='{query}', 返回 {len(items)} 个结果")
                return items
            else:
                # 关键词检索降级
                items = self._keyword_retriever.search(query, top_k=top_k)
                logger.info(f"关键词检索完成: query='{query}', 返回 {len(items)} 个结果")
                return items

        except Exception as e:
            logger.error(f"检索失败: {e}")
            # 终极降级：返回空结果
            return []

    def reload(self) -> Dict[str, int]:
        """重新加载知识库（清空后重新加载）"""
        logger.info("重新加载知识库...")

        # 清空ChromaDB目录
        if CHROMA_DB_DIR.exists():
            import shutil
            shutil.rmtree(str(CHROMA_DB_DIR))
            logger.info("已清空旧的ChromaDB数据")

        # 重置状态
        self._vector_store = None
        self._chroma_collection = None
        self._chunks_meta = []
        self._is_loaded = False
        self._use_vector = False

        # 重新加载
        return self.load()

    @property
    def is_loaded(self) -> bool:
        """知识库是否已加载"""
        return self._is_loaded

    @property
    def document_count(self) -> int:
        """知识库中的文档数量"""
        return len(self._documents)

    @property
    def search_mode(self) -> str:
        """当前检索模式"""
        return "vector" if self._use_vector else "keyword"
