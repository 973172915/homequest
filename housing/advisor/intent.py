"""LLM 意图识别：一次调用同时完成「意图分类 + 槽位抽取 + 追问判定」。

设计要点：
1. 输入带对话历史，才能理解「再便宜点」「那高新区呢」这类追问；
2. 输出严格 JSON，再过 Pydantic 校验，校验不过就降级到规则解析；
3. 没有 API Key 时整体降级到规则版意图（关键词），保证离线也能跑。
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field

from housing.advisor.llm import get_llm
from housing.search.retriever import SearchFilters, parse_query

INTENTS = ("find_listing", "ask_policy", "estimate", "chat")

SYSTEM_PROMPT = """你是「潍房智估」的意图识别模块，负责把用户的自然语言转成结构化指令。

只输出一个 JSON 对象，不要输出任何解释、不要用 markdown 代码块包裹。

JSON 结构：
{
  "intents": ["find_listing", "ask_policy", "estimate", "chat"],
  "filters": {
    "district": 区县名或 null,
    "nearby": 商圈/片区/学校名或 null,
    "community": 小区名或 null,
    "min_price": 数字（万元）或 null,
    "max_price": 数字（万元）或 null,
    "min_area": 数字（平方米）或 null,
    "max_area": 数字（平方米）或 null,
    "rooms": 整数或 null,
    "south_north": true/false/null,
    "decoration": "毛坯|简装|精装|其他" 或 null,
    "elevator": true/false/null
  },
  "missing": ["缺少的关键信息"],
  "reply": "仅当 intents 只含 chat 时填写，一句话回复并引导到房产话题"
}

intents 是数组，可以同时包含多个：
- find_listing：想找房、看房、比较房源。包括只说「我想在潍坊买套房」这种没给具体条件的。
- ask_policy：问税费、契税、首付、公积金、贷款、过户、限购、满二满五、房改房等政策知识。
- estimate：问「这套房值多少钱」「我这房子能卖多少」这类估价诉求。
- chat：打招呼、闲聊、与买房无关的请求（写代码、问天气等）。chat 只能单独出现。
- 注意：问「房价会涨吗」「明年走势」「值不值得投资」这类预测性问题，归到 ask_policy，
  由知识库作答——资料里没有预测数据时应该老实说没有，而不是随便给个判断。

复合意图判定（重点）：
- 「50万能买到哪里的房，加上契税要多少」→ ["find_listing", "ask_policy"]，既要房源也要算税费。
- 「80万首付能买多大的房」→ ["find_listing", "ask_policy"]，首付比例是政策、能买多大是找房。
- 「这套房值多少，税费多少」→ ["estimate", "ask_policy"]。
- 只要句子里同时出现了「找房」和「政策/费用」两类诉求，就要输出多个 intent，不要二选一。

槽位抽取规则（非常重要）：
1. 用户没提到的字段一律填 null，绝对不要猜测或补全。
2. 价格单位统一为「万元」。用户说「八十万」→ 80；说「80万」→ 80；说「800000」→ 80。
3. 区间表述：「六七十万」→ min_price=60, max_price=70；「90平左右」→ min_area=81, max_area=90。
4. 否定表述：「不要顶楼」「别要一楼」不影响 filters，忽略即可。
5. 结合对话历史理解指代：「再便宜点」表示在历史条件基础上降低 max_price；「那高新区呢」表示把 district 换成高新区。

多轮意图继承（重要）：
- 如果用户这句话只是在上一轮基础上改条件（「那100万呢」「再便宜点」「换成高新区」「80万的呢」），
  必须延续上一轮的 intents，不要因为这句话里没出现「税」字就丢掉 ask_policy。
- 例：上一轮问「50万能买到什么房，加各种税」，这轮问「那100万呢」
  → 输出 ["find_listing", "ask_policy"]，filters.max_price = 100。
- 只有用户明确换了话题（「你好」「满二和满五区别」）时才重新判定意图。

missing 字段：
- 只有 intents 含 find_listing 才需要判断。
- 如果 district/community/nearby 全为空，且 max_price 也为空，则 missing 至少包含 "区域或预算"。
- 其他情况 missing 为空数组。"""


@dataclass
class Intent:
    """意图识别结果。intents 是数组，支持复合意图。"""

    intents: list[str]
    filters: SearchFilters
    missing: list[str] = field(default_factory=list)
    reply: str = ""
    source: str = "llm"  # llm | fallback

    @property
    def intent(self) -> str:
        """主意图（第一个），用于日志和兼容旧调用。"""
        return self.intents[0] if self.intents else "chat"

    def has(self, name: str) -> bool:
        return name in self.intents

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing)


def compute_missing(intents: list[str], f: SearchFilters) -> list[str]:
    """追问什么，由代码决定，不交给模型。

    模型在「缺什么」这件事上不可靠（实测会把已经给了的预算还报成缺失），
    而这是纯确定性逻辑：缺什么就是 None 的那个字段。

    产品上最多问一句，所以两个都缺时合并成一条，不连环追问。
    """
    if "find_listing" in intents:
        lack_area = not (f.district or f.nearby or f.community)
        lack_budget = f.max_price is None and f.min_price is None
        if lack_area and lack_budget:
            return ["区域和预算"]
        if lack_area:
            return ["区域"]
        if lack_budget:
            return ["预算"]
        return []          # 面积、户型属于可选条件，不追问

    if "estimate" in intents:
        missing = []
        if not f.community:
            missing.append("小区")
        if f.max_area is None and f.min_area is None:
            missing.append("面积")
        return missing

    return []


def _default_filters() -> SearchFilters:
    return SearchFilters()


def _normalize_district(name: str) -> str:
    """模型可能返回「高新区」「奎文」这类简称，统一成数据里的正式名。

    不做归一化的话，用户说「高新区」，filters 里就是「高新区」，
    而房源数据里存的是「高新技术产业开发区」，硬过滤直接零结果。
    """
    from housing.search.retriever import DISTRICT_ALIASES, DISTRICT_CANONICAL

    text = (name or "").strip()
    if not text or text in DISTRICT_CANONICAL:
        return text
    if text in DISTRICT_ALIASES:
        return DISTRICT_ALIASES[text]
    for alias, canonical in sorted(DISTRICT_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in text:
            return canonical
    return text


def _to_filters(data: dict) -> SearchFilters:
    """把模型输出的 filters 转成 SearchFilters，顺带做数值校验与区县归一化。"""
    raw = data.get("filters") or {}
    f = SearchFilters()

    for key in ("district", "nearby", "community", "decoration"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != "null":
            text = value.strip()
            setattr(f, key, _normalize_district(text) if key == "district" else text)

    for key in ("min_price", "max_price", "min_area", "max_area"):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            num = float(value)
            if key.endswith("price") and 0 < num < 100000:  # 万元
                setattr(f, key, num)
            elif key.endswith("area") and 0 < num < 2000:   # 平方米
                setattr(f, key, num)

    rooms = raw.get("rooms")
    if isinstance(rooms, int) and 1 <= rooms <= 10:
        f.rooms = rooms

    for key in ("south_north", "elevator"):
        value = raw.get(key)
        if isinstance(value, bool):
            setattr(f, key, value)

    # 区间必须合法，否则丢弃下限
    if f.min_price is not None and f.max_price is not None and f.min_price > f.max_price:
        f.min_price, f.max_price = f.max_price, f.min_price
    if f.min_area is not None and f.max_area is not None and f.min_area > f.max_area:
        f.min_area, f.max_area = f.max_area, f.min_area

    return f


def _fallback(query: str, history: list[dict] | None) -> Intent:
    """没有 API Key 或调用失败时的降级路径：规则解析 + 关键词判意图。"""
    filters = parse_query(query)
    text = query
    intents: list[str] = []

    policy_kw = ("税", "契税", "首付", "公积金", "贷款", "过户", "限购", "满二", "满五", "产权")
    has_policy = any(k in text for k in policy_kw)
    has_listing = any(
        getattr(filters, name) is not None
        for name in ("district", "nearby", "community", "max_price", "max_area", "rooms")
    )

    if has_policy and not has_listing:
        intents = ["ask_policy"]
    elif any(k in text for k in ("值多少", "能卖多少", "估价", "值钱")):
        intents = ["estimate"]
    else:
        if has_listing:
            intents.append("find_listing")
        if has_policy:
            intents.append("ask_policy")

    if not intents:
        return Intent(["chat"], filters,
                      reply="你好，我是潍坊二手房助手。想找哪的房子？可以告诉我区域和预算。",
                      source="fallback")

    return Intent(intents, filters, missing=compute_missing(intents, filters),
                  source="fallback")


def _normalize(text: str) -> str:
    """全角转半角：用户常输入「５０万」「，」这类全角字符，统一成 ASCII 便于解析。"""
    return unicodedata.normalize("NFKC", text or "")


def classify(query: str, history: list[dict] | None = None,
             last_intents: list[str] | None = None,
             summary: str = "") -> Intent:
    """识别意图。

    history 由上层窗口化（memory.split_history），这里只做单条安全上限，不再粗暴截 500 字
    —— 之前正是因为截断，把「上一轮算过税」这半句信息切掉了。

    last_intents 是上一轮意图，作为冗余保险；summary 是滚动摘要。
    """
    query = _normalize(query)
    llm = get_llm()
    if llm is None:
        return _fallback(query, history)

    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
    if summary:
        messages.append(("human", f"（对话摘要，早前几轮聊过的内容：{summary}）"))
    for turn in (history or []):
        role = "human" if turn.get("role") == "user" else "ai"
        content = str(turn.get("content", ""))[:2000]   # 安全上限，远大于单条实际长度
        if content:
            messages.append((role, content))
    if last_intents:
        messages.append(("human", f"（系统提示：上一轮的意图是 {last_intents}，"
                                  f"如果本轮只是修改条件，请延续它。）"))
    messages.append(("human", query))

    try:
        resp = llm.invoke(
            messages,
            response_format={"type": "json_object"},  # type: ignore[call-arg]
        )
        text = resp.content if hasattr(resp, "content") else str(resp)
        data = json.loads(_strip_fence(text))
    except Exception:
        return _fallback(query, history)

    raw_intents = data.get("intents")
    if isinstance(raw_intents, str):
        raw_intents = [raw_intents]
    if not isinstance(raw_intents, list):
        return _fallback(query, history)

    intents: list[str] = []
    for name in raw_intents:
        if name in INTENTS and name not in intents:
            intents.append(name)
    if "chat" in intents:
        intents = ["chat"]          # chat 是互斥的
    if not intents:
        return _fallback(query, history)

    filters = _to_filters(data)
    reply = data.get("reply") if isinstance(data.get("reply"), str) else ""

    # missing 一律由代码计算：模型在「缺什么」上不可靠
    missing = compute_missing(intents, filters)

    return Intent(intents=intents, filters=filters, missing=missing, reply=reply, source="llm")


def _strip_fence(text: str) -> str:
    """模型偶尔会包一层 ```json，去掉再解析。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


CLARIFY_PROMPT = """你是潍坊二手房助手。现在需要向用户追问一个缺失信息才能继续。

要求：
1. 只问必要的，最多一句半，不要罗列超过 4 个选项。
2. 不要重复用户已经说过的条件（下面会给你）。
3. 语气自然，像人说话，不要用「请提供」这种公文腔。
4. 只输出问题本身，不要解释你为什么问。

示例：
缺区域 → 「您想找哪个区域的房子？比如奎文区、潍城区、高新区。」
缺小区和面积 → 「估价需要两个信息：哪个小区、建筑面积大概多少？」"""


def _template_question(missing: list[str]) -> str:
    """无 LLM 时的兜底措辞。"""
    if "区域" in missing and "预算" in missing:
        return "您想找哪个区域、预算大概多少？比如「奎文区 80 万左右两居」。"
    if "区域" in missing:
        return "您想找哪个区域？比如奎文区、潍城区、高新区。"
    if "预算" in missing:
        return "您的预算大概多少？"
    if "小区" in missing and "面积" in missing:
        return "估价需要两个信息：哪个小区、建筑面积大概多少？"
    if "小区" in missing:
        return "是哪个小区？"
    if "面积" in missing:
        return "建筑面积大概多少平？"
    return "能再具体一点吗？"


def clarify_question(
    missing: list[str],
    filters: SearchFilters,
    history: list[dict] | None = None,
    query: str = "",
) -> str:
    """生成追问措辞。决策在调用方，这里只负责把话说自然。"""
    llm = get_llm()
    if llm is None:
        return _template_question(missing)

    known = {k: v for k, v in filters.__dict__.items() if v is not None}
    user = (
        f"用户刚说：{query}\n"
        f"已解析出的条件：{known or '无'}\n"
        f"还缺：{missing}\n"
        "请生成一句追问。"
    )
    messages: list[tuple[str, str]] = [("system", CLARIFY_PROMPT)]
    for turn in (history or [])[-4:]:
        role = "human" if turn.get("role") == "user" else "ai"
        content = str(turn.get("content", ""))[:300]
        if content:
            messages.append((role, content))
    messages.append(("human", user))

    try:
        resp = llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip()
        return text if text else _template_question(missing)
    except Exception:
        return _template_question(missing)
