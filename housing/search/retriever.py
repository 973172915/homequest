"""检索层：混合检索（BM25 + 向量）+ 硬条件过滤 + 业务重排。

为什么不是单一向量检索：
- 向量懂「南北通透、采光好」这类语义，但不懂「80 万以内」这种数值约束；
- BM25 懂「东方天韵」这类精确小区名/学校名，但不懂近义表达；
- 业务重排负责把「价格超预算 30%」这类结果压下去。

三层各司其职，比「上了个 RAG」更接近生产可用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from housing.config import settings
from housing.search.embeddings import get_embeddings, with_query_prefix
from housing.search.tokenizer import tokenize

LISTINGS_INDEX = "faiss_listings"
KNOWLEDGE_INDEX = "faiss_knowledge"


@dataclass
class SearchFilters:
    """从自然语言里解析出的硬条件。全为 None 表示不加过滤。"""

    district: str | None = None
    min_price: float | None = None       # 万元
    max_price: float | None = None
    min_area: float | None = None        # 平方米
    max_area: float | None = None
    rooms: int | None = None
    south_north: bool | None = None
    decoration: str | None = None
    elevator: bool | None = None
    community: str | None = None
    nearby: str | None = None


NUM = r"(\d+(?:\.\d+)?)"

# 用户口语里的简称 → 数据里的正式区县名
DISTRICT_ALIASES: dict[str, str] = {
    "潍城": "潍城区",
    "奎文": "奎文区",
    "坊子": "坊子区",
    "寒亭": "寒亭区",
    "高新": "高新技术产业开发区",
    "经开": "经济技术开发区",
    "经济开发区": "经济技术开发区",
    "寿光": "寿光市",
    "青州": "青州市",
}
DISTRICT_CANONICAL = set(DISTRICT_ALIASES.values())


def _is_district_name(name: str) -> bool:
    """数据里 nearby 存在「寿光市」「青州市」这种和区县同名的脏值，不能当商圈用。"""
    return name in DISTRICT_ALIASES or name in DISTRICT_CANONICAL


_COMMUNITIES: set[str] | None = None


def community_names() -> set[str]:
    """全量小区名，用于从 query 里识别「用户点名了哪个小区」。进程内缓存。"""
    global _COMMUNITIES
    if _COMMUNITIES is None:
        from housing.ingest.clean import load_clean

        df = load_clean()
        _COMMUNITIES = {str(x) for x in df["community"].dropna()}
    return _COMMUNITIES


def detect_community(text: str) -> str | None:
    """找出 query 中出现的小区名，取最长匹配（避免「东方天韵」被「东方」截断）。"""
    best: str | None = None
    for name in community_names():
        if len(name) >= 3 and name in text and (best is None or len(name) > len(best)):
            best = name
    return best


_NEARBY: set[str] | None = None


def nearby_names() -> set[str]:
    """商圈/片区名（如「万达」「新华路北」）。"""
    global _NEARBY
    if _NEARBY is None:
        from housing.ingest.clean import load_clean

        df = load_clean()
        _NEARBY = {str(x) for x in df["nearby"].dropna()}
    return _NEARBY


def detect_nearby(text: str) -> str | None:
    best: str | None = None
    for name in nearby_names():
        if len(name) >= 2 and name in text and (best is None or len(name) > len(best)):
            best = name
    return best


def parse_query(text: str) -> SearchFilters:
    """规则版意图解析，作为 LLM 解析失败时的兜底路径。

    不追求覆盖所有说法，只保证「区县 + 价格 + 面积 + 户型 + 小区」几个高频维度可解析。
    """
    f = SearchFilters()

    # 先识别小区/商圈，再把它们从文本里剔除，
    # 否则「高新实验」（学校名）会被「高新」这个区县简称误吞。
    f.community = detect_community(text)
    masked = text.replace(f.community, " ") if f.community else text
    if f.community is None:
        candidate = detect_nearby(masked)
        # 只有不像区县的才当商圈，否则会把区县名一起抹掉
        if candidate and not _is_district_name(candidate):
            f.nearby = candidate
            masked = masked.replace(candidate, " ")

    # 先匹配正式名，再匹配简称（「高新区」→ 高新技术产业开发区）
    for alias, canonical in sorted(DISTRICT_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in masked:
            f.district = canonical
            break

    m = re.search(rf"预算?\s*{NUM}\s*万|{NUM}\s*万以内|{NUM}\s*万以下|总价\s*{NUM}\s*万", text)
    if m:
        value = next(g for g in m.groups() if g)
        f.max_price = float(value)
    else:
        # 兜底：「经济开发区 60万」这种裸写价格的问法
        m = re.search(rf"(?<![\d.]){NUM}\s*万", text)
        if m:
            f.max_price = float(m.group(1))

    m = re.search(rf"{NUM}\s*(?:平|平方米|㎡)(?:以内|以下|左右)?", text)
    if m:
        f.max_area = float(m.group(1))
        if "左右" in text or "约" in text:
            f.min_area = f.max_area * 0.9  # 「90 平左右」当 81~90 平处理

    m = re.search(r"([一二两三四五12345])\s*(?:居|室)", text)
    if m:
        cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
        f.rooms = cn.get(m.group(1)) or int(m.group(1))

    if "南北通透" in text:
        f.south_north = True
    for dec in ("精装", "简装", "毛坯"):
        if dec in text:
            f.decoration = dec
            break
    if "电梯" in text:
        f.elevator = not any(k in text for k in ("无电梯", "没有电梯", "不带电梯"))

    return f


def matches(doc: Document, filters: SearchFilters | None) -> bool:
    """硬条件判定。数值条件必须精确过滤，交给向量模型是不可靠的。"""
    if filters is None:
        return True
    m = doc.metadata
    if filters.district and m.get("district") != filters.district:
        return False
    if filters.min_price is not None and m.get("price", 0) < filters.min_price:
        return False
    if filters.max_price is not None and m.get("price", 0) > filters.max_price:
        return False
    if filters.min_area is not None and m.get("area", 0) < filters.min_area:
        return False
    if filters.max_area is not None and m.get("area", 0) > filters.max_area:
        return False
    if filters.rooms is not None and m.get("rooms") != filters.rooms:
        return False
    if filters.south_north and not m.get("south_north"):
        return False
    if filters.decoration and m.get("decoration") != filters.decoration:
        return False
    if filters.elevator is not None and bool(m.get("elevator")) != filters.elevator:
        return False
    if filters.community and m.get("community") != filters.community:
        return False
    if filters.nearby and m.get("nearby") != filters.nearby:
        return False
    return True


_STORE_CACHE: dict[str, FAISS] = {}


def get_vectorstore(name: str = LISTINGS_INDEX) -> FAISS:
    """加载已持久化的 FAISS 索引。进程内缓存，避免每次检索都读盘+重建。"""
    from housing.search.faiss_store import load_index

    if name not in _STORE_CACHE:
        _STORE_CACHE[name] = load_index(settings.index_dir / name, get_embeddings())
    return _STORE_CACHE[name]


def get_vector_retriever(k: int) -> BaseRetriever:
    """向量召回：MMR 保证结果多样，避免十套房全来自同一个小区。"""
    store = get_vectorstore()
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 3, 100), "lambda_mult": 0.5},
    )


def get_bm25_retriever(docs: list[Document], k: int) -> BM25Retriever:
    """BM25：中文用字符二元组切分，避免「东方天韵」被拆碎导致漏召回。"""
    retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
    retriever.k = k
    return retriever


_BM25_CACHE: dict[int, BM25Retriever] = {}


def get_cached_bm25(docs: list[Document], k: int) -> BM25Retriever:
    """BM25 建索引要把 1.8 万条文档全切一遍，缓存复用。"""
    if k not in _BM25_CACHE:
        _BM25_CACHE[k] = get_bm25_retriever(docs, k)
    return _BM25_CACHE[k]


def rerank(docs: list[Document], filters: SearchFilters | None = None) -> list[Document]:
    """业务重排：在召回结果上叠加「预算贴合度 / 面积贴合度」微调。

    硬条件已在 matches() 里过滤掉，这里只做排序层面的偏好。
    """

    def score(doc: Document) -> float:
        m = doc.metadata
        s = 0.0
        if filters:
            if filters.max_price:
                # 越接近预算上限但没超，越可能符合「卡预算」的真实需求
                s += 0.1 * (m.get("price", 0) / filters.max_price)
            if filters.max_area and m.get("area"):
                s += 0.1 * (m.get("area", 0) / filters.max_area)
        return s

    return sorted(docs, key=lambda d: -score(d))


def hybrid_search(
    query: str,
    filters: SearchFilters | None = None,
    docs_pool: list[Document] | None = None,
    k: int | None = None,
) -> list[Document]:
    """混合检索主入口（只要结果，不需要知道是否放宽了条件）。"""
    return search_with_flags(query, filters, docs_pool, k)[0]


def search_with_flags(
    query: str,
    filters: SearchFilters | None = None,
    docs_pool: list[Document] | None = None,
    k: int | None = None,
) -> tuple[list[Document], list[str]]:
    """混合检索：BM25 ∪ 向量 → 去重 → 硬过滤 → 重排。

    返回 (结果, 被放宽的条件名)。放宽是因为用户点名的条件在库里没有匹配，
    上层必须把这件事告诉用户，不能静默返回别的小区。
    """
    # LangChain 1.x 把传统 retriever 挪到了 langchain_classic，这里做兼容
    try:
        from langchain.retrievers import EnsembleRetriever  # LangChain < 1.0
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever  # LangChain >= 1.0

    top_k = k or settings.retrieve_top_k
    # 先多召回再过滤：硬条件可能筛掉大部分候选，所以召回量放大 20 倍
    fetch_k = max(top_k * 20, 200)

    if docs_pool is None:
        candidates = get_vector_retriever(fetch_k).invoke(with_query_prefix(query))
    else:
        ensemble = EnsembleRetriever(
            retrievers=[get_cached_bm25(docs_pool, fetch_k), get_vector_retriever(fetch_k)],
            weights=[settings.bm25_weight, settings.vector_weight],
        )
        candidates = ensemble.invoke(with_query_prefix(query))

    seen: set[str] = set()
    unique: list[Document] = []
    for doc in candidates:
        rid = doc.metadata.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(doc)

    filtered = [d for d in unique if matches(d, filters)]
    relaxed: list[str] = []
    if not filtered and filters is not None:
        # 条件太严时逐级放宽，但把放宽了哪些记下来
        for label, candidate_filters in (
            ("community", replace(filters, community=None)),
            ("社区属性", replace(filters, community=None, rooms=None, decoration=None,
                                 elevator=None, south_north=None)),
        ):
            filtered = [d for d in unique if matches(d, candidate_filters)]
            if filtered:
                relaxed.append(label)
                break

    return rerank(filtered, filters)[:top_k], relaxed
