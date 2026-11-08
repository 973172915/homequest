"""知识库检索：政策、税费、贷款、过户这类问题走独立索引。

和房源库分开的两个理由：
1. 语料形态不同——房源是一条条结构化记录，知识是长文切块；
2. 问答诉求不同——问税费用不上房源列表，混检索会互相污染。
"""

from __future__ import annotations

from langchain_core.documents import Document

from housing.config import settings
from housing.search.embeddings import get_embeddings, with_query_prefix
from housing.search.faiss_store import load_index

KNOWLEDGE_INDEX = "faiss_knowledge"
_KB_STORE = None


def get_knowledge_store():
    global _KB_STORE
    if _KB_STORE is None:
        _KB_STORE = load_index(settings.index_dir / KNOWLEDGE_INDEX, get_embeddings())
    return _KB_STORE


def search_knowledge(query: str, k: int = 4) -> list[Document]:
    """知识库相似度检索（不做 MMR，长文块本来就不重复）。"""
    store = get_knowledge_store()
    return store.similarity_search(with_query_prefix(query), k=k)
