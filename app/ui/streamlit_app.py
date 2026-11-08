"""潍房智估 · 前端（Streamlit）。

启动前先起 API：
    uvicorn app.api.main:app --port 8000
再起界面：
    streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="潍房智估", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    .price-tag { font-size: 1.4rem; font-weight: 700; color: #d9480f; }
    .muted { color: #868e96; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10)
def api_health() -> dict:
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception as exc:
        return {"status": "down", "error": str(exc)}


def post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def stream_events(query: str, history: list[dict], k: int):
    """消费 SSE：先出 meta（房源/模式），再逐字出 token。"""
    payload = {"query": query, "k": k, "history": history}
    with requests.post(
        f"{API_BASE}/advice/stream", json=payload, stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body == "[DONE]":
                break
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                continue


health = api_health()

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.title("🏠 潍房智估")
    st.caption("潍坊二手房智能选房助手")
    if health.get("status") == "ok":
        st.success("API 在线")
        st.caption(f"AI 模型：{'已就绪' if health.get('llm') else '未配置（降级模式）'}")
    else:
        st.error("API 未启动")
        st.code("uvicorn app.api.main:app --port 8000", language="bash")

    st.divider()
    k = st.slider("返回房源数量", 3, 20, 8)

tab_search, tab_estimate, tab_advice = st.tabs(["🔍 找房", "💰 估价", "💬 AI 顾问"])

# ---------------- 找房 ----------------
with tab_search:
    query = st.text_input(
        "用一句话描述你的需求",
        value="奎文区 100平以内 南北通透 精装两居 预算80万",
        placeholder="比如：万达附近 预算60万 两居",
    )

    if st.button("搜索", type="primary", key="btn_search"):
        if health.get("status") != "ok":
            st.error("请先启动 API 服务")
        else:
            with st.spinner("检索中…"):
                try:
                    data = post_json("/search", {"query": query, "k": k})
                except Exception as exc:
                    st.error(f"检索失败：{exc}")
                    data = None

            if data:
                relaxed = data.get("relaxed") or []
                if relaxed:
                    label = {"community": "指定小区", "社区属性": "户型/装修/电梯/朝向"}
                    names = "、".join(label.get(r, r) for r in relaxed)
                    st.warning(
                        f"⚠️ 你指定的条件（**{names}**）在库里没有匹配房源，"
                        f"以下为放宽该条件后的结果，不是精确命中。"
                    )

                filters = data.get("filters") or {}
                if filters:
                    st.markdown(
                        " ".join(f"`{k2}={v}`" for k2, v in filters.items())
                        + f"  ·  {data['count']} 套 · {data['elapsed_ms']}ms"
                    )
                else:
                    st.caption(f"{data['count']} 套 · {data['elapsed_ms']}ms")

                for item in data["listings"]:
                    with st.container(border=True):
                        left, right = st.columns([3, 1])
                        with left:
                            st.markdown(f"**[{item['id']}] {item['community']}**")
                            tags = [
                                item["district"],
                                item["nearby"],
                                f"{item['area']:.1f}㎡",
                                f"{item['rooms']}室",
                                item["decoration"],
                                item["floor_level"] + "楼层",
                                "有电梯" if item["elevator"] else "无电梯",
                                "南北通透" if item["south_north"] else "非南北",
                            ]
                            st.caption(" · ".join(tags))
                        with right:
                            st.markdown(
                                f"<div class='price-tag'>{item['price']:.1f}万</div>"
                                f"<div class='muted'>{item['unit_price']:.0f} 元/㎡</div>",
                                unsafe_allow_html=True,
                            )

# ---------------- 估价 ----------------
with tab_estimate:
    st.caption("基于 1.8 万条挂牌数据训练的 LightGBM 模型：随机切分 MAPE 10.3%，"
               "按小区分组切分 14.7%（新小区场景）。结果为挂牌价估算，非成交价。")

    c1, c2, c3 = st.columns(3)
    with c1:
        est_community = st.text_input("小区", value="东方天韵", key="est_community")
        est_district = st.selectbox(
            "区县",
            ["奎文区", "潍城区", "坊子区", "寒亭区", "高新技术产业开发区",
             "经济技术开发区", "寿光市", "青州市"],
            key="est_district",
        )
    with c2:
        est_area = st.number_input("建筑面积（㎡）", min_value=20.0, max_value=500.0,
                                   value=100.0, step=1.0, key="est_area")
        est_rooms = st.selectbox("户型", [1, 2, 3, 4, 5], index=1, key="est_rooms")
    with c3:
        est_floor = st.radio("楼层", ["低", "中", "高"], index=1, horizontal=True, key="est_floor")
        est_decoration = st.selectbox("装修", ["毛坯", "简装", "精装", "其他"],
                                      index=2, key="est_decoration")

    c4, c5, c6 = st.columns(3)
    with c4:
        est_elevator = st.checkbox("有电梯", value=True, key="est_elevator")
    with c5:
        est_south_north = st.checkbox("南北通透", value=True, key="est_sn")
    with c6:
        est_total_floors = st.number_input("总层数", min_value=1, max_value=60,
                                           value=11, key="est_tf")

    if st.button("估算价格", type="primary", key="btn_estimate"):
        if health.get("status") != "ok":
            st.error("请先启动 API 服务")
        else:
            try:
                data = post_json("/estimate", {
                    "community": est_community,
                    "district": est_district,
                    "nearby": "",
                    "area": est_area,
                    "rooms": est_rooms,
                    "baths": 1,
                    "floor_level": est_floor,
                    "decoration": est_decoration,
                    "elevator": est_elevator,
                    "south_north": est_south_north,
                    "total_floors": int(est_total_floors),
                })
            except Exception as exc:
                st.error(f"估价失败：{exc}")
                data = None

            if data:
                left, right = st.columns([1, 2])
                with left:
                    st.metric("估算单价", f"{data['unit_price']:.0f} 元/㎡")
                    st.metric("估算总价", f"{data['total_price_wan']:.1f} 万")
                    st.caption(f"80% 区间：{data['low']:.0f} ~ {data['high']:.0f} 元/㎡")
                with right:
                    st.markdown("**主要影响因素**")
                    rows = [{"特征": c["feature"], "影响": c["impact_pct"]}
                            for c in data["contributions"]]
                    st.dataframe(rows, hide_index=True, use_container_width=True)

# ---------------- AI 顾问（多轮 + 流式） ----------------
with tab_advice:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    head, tail = st.columns([5, 1])
    with head:
        st.caption("支持多轮对话：可以说「再便宜点」「那高新区呢」；也能一次问找房 + 税费。")
    with tail:
        if st.button("清空", key="btn_clear_chat"):
            st.session_state.chat = []
            st.rerun()

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("问问看，比如：50万能买到什么房，加上税费要多少")
    if prompt:
        if health.get("status") != "ok":
            st.error("请先启动 API 服务")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                meta: dict = {}
                status = st.empty()

                def tokens():
                    for evt in stream_events(prompt, st.session_state.chat[:-1], k):
                        kind = evt.get("type")
                        if kind == "stage":
                            status.caption("⏳ " + evt.get("text", ""))
                        elif kind == "meta":
                            meta.update(evt)
                        elif kind == "token":
                            if status is not None:
                                status.empty()
                            yield evt.get("text", "")

                try:
                    text = st.write_stream(tokens())
                except Exception as exc:
                    status.empty()
                    text = f"生成失败：{exc}"
                    st.error(text)

                if meta.get("relaxed"):
                    label = {"community": "指定小区", "社区属性": "户型/装修/电梯/朝向"}
                    names = "、".join(label.get(r, r) for r in meta["relaxed"])
                    st.warning(f"⚠️ 条件（{names}）没有匹配房源，以上为放宽后的结果。")
                if meta.get("kind") == "mixed":
                    st.caption("✓ 本次回答同时用了房源库和知识库")

            st.session_state.chat.append({"role": "assistant", "content": str(text or "")})
