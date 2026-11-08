"""对话记忆：滑动窗口 + 滚动摘要。

策略：
- 最近 `max_history_turns` 轮完整保留（不做单条截断，截断正是之前丢信息的根因）
- 超出的旧轮次交给 LLM 压成一段摘要，和下一轮一起带回来
- 摘要只保留「会影响后续检索的事实」：区域、预算、面积、户型、排除项、已确认结论

为什么不用结构化 state：
- history 原文能理解「第二套呢」「这个太贵了」这类模糊指代，state 做不到
- 15 轮 × 2 条 × 700 字 ≈ 7000 token，DeepSeek 1M 上下文完全无压力
"""

from __future__ import annotations

from housing.advisor.llm import get_llm
from housing.config import settings

SUMMARY_PROMPT = """把下面的历史对话压缩成不超过 150 字的摘要。

必须保留（后续检索要用）：
1. 用户的购房条件：区域、预算区间、面积、户型、装修、电梯、朝向
2. 用户明确说不要的东西（如不要顶楼、不要一楼）
3. 已经确认过的结论（如「满二免增值税」「首付最低 15%」）
4. 用户改过条件的轨迹（如「预算从 50 万调整到 100 万」）

不要保留：寒暄、客套话、重复内容、具体房源列表。

只输出摘要正文，不要加「摘要：」这样的前缀。"""


def split_history(history: list[dict], max_turns: int | None = None
                  ) -> tuple[list[dict], list[dict]]:
    """把历史切成 (要压缩的旧轮次, 保留的最近轮次)。

    窗口按「条」算：1 轮 = 用户 + 助手 = 2 条。
    """
    limit = (max_turns or settings.max_history_turns) * 2
    if len(history) <= limit:
        return [], list(history)

    cut = len(history) - limit
    overflow, recent = list(history[:cut]), list(history[cut:])

    # 保证窗口从用户发言开始，避免出现「助手先说话」的错位
    while recent and recent[0].get("role") != "user":
        overflow.append(recent.pop(0))
    return overflow, recent


def _naive_summary(previous: str, turns: list[dict]) -> str:
    """没有 LLM 时的兜底：粗暴拼接用户说过的话。"""
    parts = [previous] if previous else []
    for turn in turns:
        if turn.get("role") == "user":
            text = str(turn.get("content", ""))[:80]
            if text:
                parts.append(text)
    return "；".join(p for p in parts if p)[-600:]


def summarize(previous_summary: str, turns: list[dict]) -> str:
    """把旧轮次压进已有的摘要里。"""
    if not turns:
        return previous_summary

    llm = get_llm()
    if llm is None:
        return _naive_summary(previous_summary, turns)

    lines = []
    if previous_summary:
        lines.append(f"【已有摘要】{previous_summary}")
    lines.append("【新增对话】")
    for turn in turns:
        role = "用户" if turn.get("role") == "user" else "助手"
        content = str(turn.get("content", ""))[:600]
        lines.append(f"{role}：{content}")

    try:
        resp = llm.invoke([("system", SUMMARY_PROMPT), ("human", "\n".join(lines))])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip()
        return text[:600] if text else _naive_summary(previous_summary, turns)
    except Exception:
        return _naive_summary(previous_summary, turns)
