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


def stream_advice(query: str, k: int):
    """消费 SSE，先返回 meta（房源），再逐字返回文本。"""
    with requests.post(
        f"{API_BASE}/advice/stream", json={"query": query, "k": k}, stream=True, timeout=180
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event["type"] == "token":
                yield event["text"]


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

tab_search, tab_advice = st.tabs(["🔍 找房", "💬 AI 顾问"])

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

# ---------------- AI 顾问 ----------------
with tab_advice:
    question = st.text_area(
        "告诉顾问你的情况",
        value="预算 80 万，在奎文区想买个两居，要求南北通透，有什么推荐？",
        height=100,
    )

    if st.button("咨询顾问", type="primary", key="btn_advice"):
        if health.get("status") != "ok":
            st.error("请先启动 API 服务")
        else:
            try:
                st.write_stream(stream_advice(question, k))
            except Exception as exc:
                st.error(f"生成失败：{exc}")
