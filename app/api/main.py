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

from fastapi import Depends, FastAPI, Header, HTTPException
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
    # 多轮对话历史：[{"role": "user"/"assistant", "content": "..."}]
    history: list[dict[str, str]] = Field(default_factory=list)
    # 上一轮的意图组合，用于多轮继承（历史文本会被截断）
    last_intents: list[str] = Field(default_factory=list)
    # 滚动摘要（超过 15 轮的旧对话被压缩到这里）
    summary: str = Field(default="", max_length=2000)


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
    intent: str = ""
    relaxed: list[str]
    missing: list[str] = []
    elapsed_ms: int
    answer: str
    sources: list[dict[str, Any]]


class EstimateRequest(BaseModel):
    community: str = Field(..., min_length=1, max_length=50)
    district: str = Field("奎文区", max_length=30)
    nearby: str = Field("", max_length=30)
    area: float = Field(..., gt=10, le=600, description="建筑面积（㎡）")
    rooms: int = Field(2, ge=1, le=8)
    baths: int = Field(1, ge=0, le=5)
    floor_level: str = Field("中", pattern="^(低|中|高)$")
    decoration: str = Field("精装", pattern="^(毛坯|简装|精装|其他)$")
    elevator: bool = True
    south_north: bool = True
    total_floors: int = Field(6, ge=1, le=60)


class EstimateResponse(BaseModel):
    unit_price: float
    total_price_wan: float
    low: float
    high: float
    contributions: list[dict[str, Any]]
    note: str = "挂牌价估算，非成交价"


# ---------------- 认证 ----------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=20)
    password: str = Field(..., min_length=6, max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=20)
    password: str = Field(..., min_length=1, max_length=64)


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str


def current_user(authorization: str | None = Header(default=None)) -> int:
    """从 `Authorization: Bearer <token>` 解析用户；无效返回 401。"""
    from housing.storage.users import resolve_token

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    user_id = resolve_token(authorization.split(" ", 1)[1].strip())
    if user_id is None:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return user_id


@app.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    from housing.storage.users import issue_token
    from housing.storage.users import register as do_register

    user_id, error = do_register(req.username, req.password)
    if user_id is None:
        raise HTTPException(status_code=400, detail=error)
    return AuthResponse(token=issue_token(user_id), user_id=user_id,
                        username=req.username.strip())


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    from housing.storage.users import issue_token, verify

    user_id = verify(req.username, req.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return AuthResponse(token=issue_token(user_id), user_id=user_id,
                        username=req.username.strip())


@app.get("/auth/me")
def me(user_id: int = Depends(current_user)) -> dict[str, Any]:
    from housing.storage.users import get_user

    user = get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    from housing.storage.users import revoke_token

    if authorization and authorization.lower().startswith("bearer "):
        revoke_token(authorization.split(" ", 1)[1].strip())
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm": llm_available(),
        "warmup_seconds": getattr(app.state, "warmup_seconds", None),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, user_id: int = Depends(current_user)) -> SearchResponse:
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
def advice(req: SearchRequest, user_id: int = Depends(current_user)) -> AdviceResponse:
    t0 = time.perf_counter()
    try:
        result = answer(req.query, k=req.k, history=req.history,
                        last_intents=req.last_intents, summary=req.summary)
    except Exception as exc:  # 外部 API 失败要让调用方看到明确原因
        raise HTTPException(status_code=502, detail=f"生成失败：{exc}") from exc
    return AdviceResponse(
        query=req.query,
        mode=result.mode,
        kind=result.kind,
        intent=result.intent,
        relaxed=result.relaxed,
        missing=result.missing,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        answer=result.answer,
        sources=result.sources,
    )


@app.post("/advice/stream")
async def advice_stream(req: SearchRequest, user_id: int = Depends(current_user)) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in astream_answer(req.query, k=req.k, history=req.history,
                                              last_intents=req.last_intents,
                                              summary=req.summary):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/estimate", response_model=EstimateResponse)
def estimate_price(req: EstimateRequest, user_id: int = Depends(current_user)) -> EstimateResponse:
    """单套房估价：返回单价、总价、区间和主要影响因素。"""
    from housing.estimate.model import estimate

    result = estimate(
        community=req.community,
        district=req.district,
        nearby=req.nearby,
        area=req.area,
        rooms=req.rooms,
        floor_level=req.floor_level,
        decoration=req.decoration,
        elevator=req.elevator,
        south_north=req.south_north,
        total_floors=req.total_floors,
        baths=req.baths,
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="估价模型未训练，先运行：python scripts/train_model.py",
        )
    return EstimateResponse(
        unit_price=result.unit_price,
        total_price_wan=round(result.unit_price * req.area / 10000, 1),
        low=result.low,
        high=result.high,
        contributions=result.contributions,
    )


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    """模型评估指标，给界面展示用。"""
    import json as _json

    from housing.config import settings as _settings

    meta_path = _settings.models_dir / "price_model_meta.json"
    if not meta_path.exists():
        return {"trained": False}
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    return {"trained": True, **meta}
