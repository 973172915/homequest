"""顾问链：意图识别 → 路由 → 生成。

四条支路：
- find_listing  找房   → 混合检索 + 房源库 RAG
- ask_policy    政策   → 知识库 RAG
- estimate      估价   → LightGBM（缺信息则追问）
- chat          闲聊   → 直接回复 + 引导回房产

有 API Key 时用 LLM 做意图识别；没有则降级为规则版（关键词 + parse_query），
保证 clone 下来不配密钥也能跑通。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from housing.advisor.intent import Intent, clarify_question, classify
from housing.advisor.llm import get_llm, llm_available
from housing.advisor.memory import split_history, summarize
from housing.advisor.prompts import (
    MIXED_SYSTEM_PROMPT,
    MIXED_USER_TEMPLATE,
    POLICY_SYSTEM_PROMPT,
    POLICY_USER_TEMPLATE,
    RAG_USER_TEMPLATE,
    SYSTEM_PROMPT,
    format_context,
    format_kb_context,
)
from housing.ingest.build_docs import load_listing_documents
from housing.search.knowledge import search_knowledge
from housing.search.retriever import SearchFilters, parse_query, search_with_flags

# 政策类问题的关键词（仅用于无 LLM 时的降级路由）
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
    mode: str                      # "llm" | "fallback"
    kind: str                      # listing | policy | estimate | chat | clarify
    relaxed: list[str] = field(default_factory=list)
    intent: str = ""
    missing: list[str] = field(default_factory=list)
    summary: str = ""


def is_policy_query(query: str, filters: SearchFilters) -> bool:
    """无 LLM 时的降级路由：提到税费/贷款，且没说具体房源条件。"""
    if (filters.district or filters.community or filters.nearby
            or filters.rooms or filters.max_area or filters.decoration):
        return False
    return any(kw in query for kw in POLICY_KEYWORDS)


def _condition_count(f: SearchFilters) -> int:
    return sum(1 for v in f.__dict__.values() if v is not None)


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


def _relax_note(relaxed: list[str]) -> str:
    if not relaxed:
        return ""
    return (
        f"【注意】用户点名的条件（{'、'.join(relaxed)}）在库里没有匹配房源，"
        f"以下为放宽后的结果，回答时必须明确说明这一点，不要假装是原条件的结果。\n\n"
    )


def build_chain(streaming: bool = False, policy: bool = False,
                mixed: bool = False, history: list[dict] | None = None):
    """LCEL 链。显式拼装，不用 RetrievalQA 这类一行式封装。"""
    llm = get_llm(streaming=streaming)
    if llm is None:
        return None
    if mixed:
        system, user = MIXED_SYSTEM_PROMPT, MIXED_USER_TEMPLATE
    elif policy:
        system, user = POLICY_SYSTEM_PROMPT, POLICY_USER_TEMPLATE
    else:
        system, user = SYSTEM_PROMPT, RAG_USER_TEMPLATE
    messages: list[tuple[str, str]] = [("system", system)]
    for turn in (history or [])[-4:]:
        role = "human" if turn.get("role") == "user" else "ai"
        content = str(turn.get("content", ""))[:400]
        if content:
            messages.append((role, content))
    messages.append(("human", user))
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt | llm | StrOutputParser()


# ---------------- 四条支路 ----------------

def _listing_answer(query: str, filters: SearchFilters, k: int | None,
                    history: list[dict] | None) -> AdvisorResult:
    docs, relaxed = search_with_flags(
        query, filters, docs_pool=load_listing_documents(), k=k
    )
    sources = [dict(d.metadata) for d in docs]
    context = _relax_note(relaxed) + format_context(docs)

    chain = build_chain(history=history)
    if chain is None:
        text = _relax_note(relaxed) + _fallback_listing(docs, query)
        result = AdvisorResult(text, sources, filters, "fallback", "listing", relaxed)
    else:
        text = chain.invoke({"context": context, "question": query})
        result = AdvisorResult(text, sources, filters, "llm", "listing", relaxed)

    # 条件太少时，答完给一句提示，而不是打断用户去追问
    if _condition_count(filters) == 0:
        result.answer += (
            "\n\n💡 先按全城给你看了几套。想更准的话，"
            "告诉我**想找哪个区**、**预算多少**，我再筛一遍。"
        )
    return result


def _policy_answer(query: str, filters: SearchFilters, k: int | None) -> AdvisorResult:
    docs = search_knowledge(query, k=k or 6)
    sources = [dict(d.metadata) for d in docs]
    chain = build_chain(policy=True)
    if chain is None:
        return AdvisorResult(_fallback_policy(docs, query), sources, filters, "fallback", "policy")
    text = chain.invoke({"context": format_kb_context(docs), "question": query})
    return AdvisorResult(text, sources, filters, "llm", "policy")


def _combined_answer(query: str, filters: SearchFilters, k: int | None,
                     history: list[dict] | None) -> AdvisorResult:
    """复合意图：既要房源，又要把费用算进去。

    两份上下文一起喂给模型——只给其中一份，它就会答「不知道」（这正是最初的 bug）。
    """
    docs, relaxed = search_with_flags(
        query, filters, docs_pool=load_listing_documents(), k=k
    )
    kb_docs = search_knowledge(query, k=6)

    listing_ctx = format_context(docs) or "（无）"
    kb_ctx = format_kb_context(kb_docs) or "（无）"

    sources = [dict(d.metadata) for d in docs]
    sources += [{**dict(d.metadata), "type": "knowledge"} for d in kb_docs]

    chain = build_chain(mixed=True, history=history)
    if chain is None:
        text = (_fallback_listing(docs, query) + "\n\n"
                + _fallback_policy(kb_docs, "费用"))
        return AdvisorResult(text, sources, filters, "fallback", "mixed", relaxed)

    text = chain.invoke({"listings": listing_ctx, "knowledge": kb_ctx, "question": query})
    return AdvisorResult(text, sources, filters, "llm", "mixed", relaxed)


def _estimate_answer(query: str, filters: SearchFilters) -> AdvisorResult:
    from housing.estimate.model import estimate

    area = filters.max_area or filters.min_area or 90.0
    result = estimate(
        community=filters.community or "",
        district=filters.district or "",
        nearby=filters.nearby or "",
        area=area,
        rooms=filters.rooms or 2,
        decoration=filters.decoration or "精装",
        elevator=True if filters.elevator is None else filters.elevator,
        south_north=True if filters.south_north is None else filters.south_north,
    )
    if result is None:
        return AdvisorResult(
            "估价模型还没训练，先运行 `python scripts/train_model.py`。",
            [], filters, "fallback", "estimate",
        )

    total = result.unit_price * area / 10000
    top = "、".join(
        f"{c['feature']} {c['impact_pct']:+.1f}%" for c in result.contributions[:3]
    )
    text = (
        f"按「{filters.community or filters.district or '该区域'} {area:.0f}㎡ "
        f"{filters.rooms or 2}室」估算：\n\n"
        f"- 单价约 **{result.unit_price:.0f} 元/㎡**\n"
        f"- 总价约 **{total:.1f} 万**\n"
        f"- 80% 区间：{result.low:.0f} ~ {result.high:.0f} 元/㎡\n\n"
        f"主要影响因素：{top}。\n\n"
        f"> 基于挂牌价训练，非成交价。"
    )
    return AdvisorResult(text, [], filters, "llm", "estimate")


def _chat_answer(intent: Intent, filters: SearchFilters) -> AdvisorResult:
    text = intent.reply or (
        "我是潍坊二手房助手，可以帮你找房、估房价、回答购房政策。"
        "想看哪个区域的房子？"
    )
    return AdvisorResult(text, [], filters, "fallback" if intent.source == "fallback" else "llm", "chat")


def _clarify_answer(intent: Intent, query: str,
                    history: list[dict] | None) -> AdvisorResult:
    question = clarify_question(intent.missing, intent.filters, history, query)
    return AdvisorResult(question, [], intent.filters,
                         "fallback" if intent.source == "fallback" else "llm",
                         "clarify", missing=intent.missing)


def _route(intent: Intent, query: str, k: int | None,
           history: list[dict] | None) -> AdvisorResult:
    filters = intent.filters
    intents = intent.intents

    if "chat" in intents:
        result = _chat_answer(intent, filters)
    elif "estimate" in intents and intent.missing:
        result = _clarify_answer(intent, query, history)
    elif "estimate" in intents:
        result = _estimate_answer(query, filters)
    elif "find_listing" in intents and "ask_policy" in intents:
        result = _combined_answer(query, filters, k, history)
    elif "find_listing" in intents:
        result = _listing_answer(query, filters, k, history)
    else:
        result = _policy_answer(query, filters, k)

    result.intent = "+".join(intents)
    result.missing = intent.missing
    return result


def _legacy_route(query: str, k: int | None) -> AdvisorResult:
    """无 LLM 时的降级路径：规则解析 + 关键词路由。"""
    filters = parse_query(query)
    if is_policy_query(query, filters):
        return _policy_answer(query, filters, k)
    return _listing_answer(query, filters, k, None)


def _prepare_memory(history: list[dict] | None, summary: str) -> tuple[list[dict], str]:
    """滑动窗口 + 滚动摘要：超出 15 轮的旧内容压进 summary。"""
    overflow, recent = split_history(list(history or []))
    if overflow:
        summary = summarize(summary, overflow)
    return recent, summary


def answer(query: str, k: int | None = None,
           history: list[dict] | None = None,
           last_intents: list[str] | None = None,
           summary: str = "") -> AdvisorResult:
    """完整问答：意图识别 → 路由 → 生成。"""
    if not llm_available():
        return _legacy_route(query, k)

    recent, summary = _prepare_memory(history, summary)
    intent = classify(query, recent, last_intents, summary)
    result = _route(intent, query, k, recent)
    result.summary = summary
    return result


async def astream_answer(query: str, k: int | None = None,
                         history: list[dict] | None = None,
                         last_intents: list[str] | None = None,
                         summary: str = ""):
    """流式版本：先产出 meta，再逐字产出文本。

    只有「找房 / 政策」需要流式（真实生成），闲聊和追问是一次性文本。
    """
    if not llm_available():
        result = _legacy_route(query, k)
        yield {"type": "meta", "sources": result.sources, "mode": result.mode,
               "kind": result.kind, "relaxed": result.relaxed}
        yield {"type": "token", "text": result.answer}
        return

    # 先给一个阶段提示，避免用户干等意图识别 + 检索的几秒钟
    yield {"type": "stage", "text": "正在理解你的问题…"}

    recent, summary = _prepare_memory(history, summary)
    intent = classify(query, recent, last_intents, summary)
    filters = intent.filters
    intents = intent.intents

    # 非生成类：直接一次性返回
    if "chat" in intents or ("estimate" in intents and intent.missing):
        result = _route(intent, query, k, history)
        yield {"type": "meta", "sources": [], "mode": result.mode,
               "kind": result.kind, "relaxed": [], "intent": "+".join(intents), "summary": summary}
        yield {"type": "token", "text": result.answer}
        return

    if "estimate" in intents:
        result = _estimate_answer(query, filters)
        yield {"type": "meta", "sources": [], "mode": "llm", "kind": "estimate", "relaxed": [], "intent": "+".join(intents), "summary": summary}
        yield {"type": "token", "text": result.answer}
        return

    want_listing = "find_listing" in intents
    want_policy = "ask_policy" in intents

    docs, relaxed = [], []
    kb_docs = []
    if want_listing:
        docs, relaxed = search_with_flags(
            query, filters, docs_pool=load_listing_documents(), k=k
        )
    if want_policy:
        kb_docs = search_knowledge(query, k=k or 6)

    sources = [dict(d.metadata) for d in docs]
    sources += [{**dict(d.metadata), "type": "knowledge"} for d in kb_docs]

    kind = "mixed" if (want_listing and want_policy) else ("listing" if want_listing else "policy")

    stage = []
    if want_listing:
        stage.append(f"{len(docs)} 套房源")
    if want_policy:
        stage.append(f"{len(kb_docs)} 条政策资料")
    if stage:
        yield {"type": "stage", "text": "已检索到 " + " + ".join(stage)}

    yield {"type": "meta", "sources": sources, "mode": "llm",
           "kind": kind, "relaxed": relaxed, "intent": "+".join(intents), "summary": summary}

    chain = build_chain(streaming=True, policy=not want_listing, mixed=(kind == "mixed"),
                        history=history)
    if chain is None:
        text = _fallback_listing(docs, query) if want_listing else _fallback_policy(kb_docs, query)
        yield {"type": "token", "text": text}
        return

    if kind == "mixed":
        payload = {
            "listings": format_context(docs) or "（无）",
            "knowledge": format_kb_context(kb_docs) or "（无）",
            "question": query,
        }
    elif want_listing:
        payload = {"context": _relax_note(relaxed) + format_context(docs), "question": query}
    else:
        payload = {"context": format_kb_context(kb_docs), "question": query}

    async for chunk in chain.astream(payload):
        if chunk:
            yield {"type": "token", "text": chunk}

    if want_listing and _condition_count(filters) == 0:
        yield {"type": "token",
               "text": "\n\n💡 想更准的话，告诉我**想找哪个区**、**预算多少**。"}
