"""潍房智估 API。

接口：
    GET  /health          健康检查（模型、索引、LLM 是否就绪）
    POST /search          一句话找房 → 结构化筛选 + 房源列表
    POST /advice          一句话 → AI 顾问完整报告
    POST /advice/stream   同上，SSE 流式返回（打字机效果）

启动：
    uvicorn app.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from housing.advisor.chain import answer, astream_answer
from housing.advisor.llm import llm_available


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热：加载模型和索引，避免第一个请求等 9 秒。"""
    from housing.ingest.build_docs import load_listing_documents
    from housing.search.retriever import get_vectorstore

    t0 = time.perf_counter()
    get_vectorstore()
    load_listing_documents()
    app.state.warmup_seconds = round(time.perf_counter() - t0, 2)
    yield


app = FastAPI(
    title="潍房智估 API",
    description="潍坊二手房智能选房助手：混合检索 + AI 置业顾问",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="自然语言找房需求")
    k: int = Field(10, ge=1, le=50)


class Listing(BaseModel):
    id: str
    district: str
    nearby: str
    community: str
    area: float
    rooms: int
    price: float
    unit_price: float
    decoration: str
    floor_level: str
    elevator: int
    south_north: int


class SearchResponse(BaseModel):
    query: str
    filters: dict[str, Any]
    count: int
    elapsed_ms: int
    relaxed: list[str] = Field(default_factory=list)
    listings: list[Listing]


class AdviceResponse(BaseModel):
    query: str
    mode: str
    kind: str
    relaxed: list[str]
    elapsed_ms: int
    answer: str
    sources: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm": llm_available(),
        "warmup_seconds": getattr(app.state, "warmup_seconds", None),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    from housing.ingest.build_docs import load_listing_documents
    from housing.search.retriever import parse_query, search_with_flags

    t0 = time.perf_counter()
    filters = parse_query(req.query)
    docs, relaxed = search_with_flags(req.query, filters, docs_pool=load_listing_documents(), k=req.k)
    elapsed = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        query=req.query,
        filters={k: v for k, v in filters.__dict__.items() if v is not None},
        count=len(docs),
        elapsed_ms=elapsed,
        relaxed=relaxed,
        listings=[Listing(**{k: d.metadata[k] for k in Listing.model_fields}) for d in docs],
    )


@app.post("/advice", response_model=AdviceResponse)
def advice(req: SearchRequest) -> AdviceResponse:
    t0 = time.perf_counter()
    try:
        result = answer(req.query, k=req.k)
    except Exception as exc:  # 外部 API 失败要让调用方看到明确原因
        raise HTTPException(status_code=502, detail=f"生成失败：{exc}") from exc
    return AdviceResponse(
        query=req.query,
        mode=result.mode,
        kind=result.kind,
        relaxed=result.relaxed,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        answer=result.answer,
        sources=result.sources,
    )


@app.post("/advice/stream")
async def advice_stream(req: SearchRequest) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in astream_answer(req.query, k=req.k):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
