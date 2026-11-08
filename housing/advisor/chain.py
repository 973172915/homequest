"""顾问链：一句人话 → 检索 → 生成 → 带引用的回答。

两条支路：
- 找房类问题 → 房源库（混合检索）
- 政策/税费/贷款类问题 → 知识库（切块长文）
没有配置 DeepSeek Key 时自动降级为「检索结果 + 模板摘要」，
保证面试官 clone 下来不配密钥也能跑通、能截图。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from housing.advisor.llm import get_llm
from housing.advisor.prompts import (
    POLICY_SYSTEM_PROMPT,
    POLICY_USER_TEMPLATE,
    RAG_USER_TEMPLATE,
    SYSTEM_PROMPT,
    format_context,
    format_kb_context,
)
from housing.ingest.build_docs import load_listing_documents
from housing.search.knowledge import search_knowledge
from housing.search.retriever import (
    SearchFilters,
    parse_query,
    search_with_flags,
)

# 政策类问题的关键词。命中且没提具体区域/预算时，走知识库。
POLICY_KEYWORDS = (
    "税", "契税", "增值税", "个税", "满二", "满五", "唯一", "出税",
    "贷款", "首付", "公积金", "利率", "月供", "过户", "限购", "资格",
    "产权", "房改房", "公摊", "得房率", "流程", "材料", "注意",
)


@dataclass
class AdvisorResult:
    answer: str
    sources: list[dict]
    filters: SearchFilters
    mode: str   # "llm" | "fallback"
    kind: str   # "listing" | "policy"
    relaxed: list[str] = field(default_factory=list)


def is_policy_query(query: str, filters: SearchFilters) -> bool:
    """提到税费/贷款，且没说具体区县、小区、商圈、户型、面积、装修 → 当政策问题。

    只出现价格（如「100万的房子契税多少」）不算找房，那是在问税费怎么算。
    """
    if (filters.district or filters.community or filters.nearby
            or filters.rooms or filters.max_area or filters.decoration):
        return False
    return any(kw in query for kw in POLICY_KEYWORDS)


def _fallback_listing(docs, query: str) -> str:
    if not docs:
        return "没有找到符合条件的房源，可以试着放宽预算、面积或换一个区县。"
    lines = [f"根据「{query}」检索到 {len(docs)} 套房源：", "",
             "| 编号 | 小区 | 面积 | 户型 | 总价 | 单价 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for d in docs:
        m = d.metadata
        lines.append(
            f"| [{m['id']}] | {m['community']} | {m['area']:.1f}㎡ | {m['rooms']}室 | "
            f"{m['price']:.1f}万 | {m['unit_price']:.0f}元/㎡ |"
        )
    lines += ["", "> 未配置 DEEPSEEK_API_KEY，当前为检索模式。"]
    return "\n".join(lines)


def _fallback_policy(docs, query: str) -> str:
    if not docs:
        return "知识库里没有找到相关资料。"
    parts = [f"关于「{query}」，资料中相关内容如下：", ""]
    for doc in docs[:3]:
        parts.append(f"**{doc.metadata.get('title', '资料')}**")
        parts.append(doc.page_content.strip())
        parts.append("")
    parts.append("> 未配置 DEEPSEEK_API_KEY，当前直接返回原文片段。")
    return "\n".join(parts)


def build_chain(streaming: bool = False, policy: bool = False):
    """LCEL 链。显式拼装，不用 RetrievalQA 这类一行式封装。"""
    llm = get_llm(streaming=streaming)
    if llm is None:
        return None
    system, user = (
        (POLICY_SYSTEM_PROMPT, POLICY_USER_TEMPLATE) if policy
        else (SYSTEM_PROMPT, RAG_USER_TEMPLATE)
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    return prompt | llm | StrOutputParser()


def _listing_hits(query: str, filters: SearchFilters, k: int | None):
    return search_with_flags(query, filters, docs_pool=load_listing_documents(), k=k)


def _relax_note(relaxed: list[str]) -> str:
    if not relaxed:
        return ""
    return (
        f"【注意】用户点名的条件（{'、'.join(relaxed)}）在库里没有匹配房源，"
        f"以下为放宽后的结果，回答时必须明确说明这一点，不要假装是原条件的结果。\n\n"
    )


def answer(query: str, k: int | None = None) -> AdvisorResult:
    """完整问答：解析条件 → 选择检索源 → 生成。"""
    filters = parse_query(query)

    if is_policy_query(query, filters):
        docs = search_knowledge(query, k=k or 4)
        sources = [dict(d.metadata) for d in docs]
        chain = build_chain(policy=True)
        if chain is None:
            return AdvisorResult(_fallback_policy(docs, query), sources, filters, "fallback", "policy")
        text = chain.invoke({"context": format_kb_context(docs), "question": query})
        return AdvisorResult(text, sources, filters, "llm", "policy")

    docs, relaxed = _listing_hits(query, filters, k)
    sources = [dict(d.metadata) for d in docs]
    chain = build_chain()
    if chain is None:
        text = _relax_note(relaxed) + _fallback_listing(docs, query)
        return AdvisorResult(text, sources, filters, "fallback", "listing", relaxed)
    context = _relax_note(relaxed) + format_context(docs)
    text = chain.invoke({"context": context, "question": query})
    return AdvisorResult(text, sources, filters, "llm", "listing", relaxed)


async def astream_answer(query: str, k: int | None = None):
    """流式版本：先产出 sources，再逐字产出文本。"""
    filters = parse_query(query)
    policy = is_policy_query(query, filters)
    relaxed: list[str] = []

    if policy:
        docs = search_knowledge(query, k=k or 4)
        context = format_kb_context(docs)
    else:
        docs, relaxed = _listing_hits(query, filters, k)
        context = _relax_note(relaxed) + format_context(docs)
    sources = [dict(d.metadata) for d in docs]

    chain = build_chain(streaming=True, policy=policy)
    if chain is None:
        yield {"type": "meta", "sources": sources, "mode": "fallback",
               "kind": "policy" if policy else "listing", "relaxed": relaxed}
        yield {"type": "token",
               "text": _fallback_policy(docs, query) if policy else
               _relax_note(relaxed) + _fallback_listing(docs, query)}
        return

    yield {"type": "meta", "sources": sources, "mode": "llm",
           "kind": "policy" if policy else "listing", "relaxed": relaxed}
    async for chunk in chain.astream({"context": context, "question": query}):
        if chunk:
            yield {"type": "token", "text": chunk}
