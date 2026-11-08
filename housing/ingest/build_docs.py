"""把房源表序列化成检索文档。

关键设计：**一房一文档，不做文本切块**。
理由：一条房源全部字段拼起来约 100 字，切块会把「小区名」和「价格」拆散，
向量里只剩半套房，召回反而变差；而且 metadata 需要与正文绑定，切块会产生多份副本。
"""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

TEMPLATE = (
    "【房源 {id}】{district} · {nearby} · {community}\n"
    "{size} / {area}㎡ / {floor} / {direction} / {decoration} / {elevator}\n"
    "总价 {total} 万，单价 {unit} 元/㎡\n"
    "卖点：{title}"
)


def _s(value, default: str = "未知") -> str:
    """把 NaN/None/空串统一成可读文本。"""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _direction(row) -> str:
    if bool(row.get("south_north")):
        return "南北通透"
    parts = [d for d, on in (("南", "has_south"), ("北", "has_north"),
                             ("东", "has_east"), ("西", "has_west"))
             if bool(row.get(on))]
    return "".join(parts) + "向" if parts else "朝向未知"


def _elevator(row) -> str:
    if not bool(row.get("has_elevator")):
        return "无电梯"
    elev, units = row.get("elevators"), row.get("units_per_floor")
    if pd.notna(elev) and pd.notna(units):
        return f"有电梯（{int(elev)}梯{int(units)}户）"
    return "有电梯"


def format_listing(row) -> str:
    """单条房源 → 自然语言段落。"""
    return TEMPLATE.format(
        id=_s(row.get("id")),
        district=_s(row.get("district")),
        nearby=_s(row.get("nearby")),
        community=_s(row.get("community")),
        size=f"{int(row['rooms'])}室{int(row['halls'])}厅" if pd.notna(row.get("rooms")) else "户型未知",
        area=f"{row['area']:.1f}",
        floor=f"{_s(row.get('floor_level'))}楼层(共{int(row['total_floors'])}层)"
        if pd.notna(row.get("total_floors")) else _s(row.get("floor")),
        direction=_direction(row),
        decoration=_s(row.get("decoration")),
        elevator=_elevator(row),
        total=f"{row['total_price_wan']:.1f}",
        unit=f"{row['unit_price']:.0f}",
        title=_s(row.get("title"), ""),
    )


def build_metadata(row) -> dict:
    """Chroma 只接受 str/int/float/bool，这里做一次显式转换。"""
    return {
        "id": _s(row.get("id")),
        "district": _s(row.get("district")),
        "nearby": _s(row.get("nearby")),
        "community": _s(row.get("community")),
        "price": float(row["total_price_wan"]),
        "unit_price": float(row["unit_price"]),
        "area": float(row["area"]),
        "rooms": int(row["rooms"]),
        "decoration": _s(row.get("decoration")),
        "elevator": int(bool(row.get("has_elevator"))),
        "floor_level": _s(row.get("floor_level")),
        "total_floors": int(row["total_floors"]),
        "south_north": int(bool(row.get("south_north"))),
    }


def build_listing_documents(df: pd.DataFrame) -> list[Document]:
    """整表 → Document 列表。"""
    return [
        Document(page_content=format_listing(row), metadata=build_metadata(row))
        for _, row in df.iterrows()
    ]


_POOL: list[Document] | None = None


def load_listing_documents() -> list[Document]:
    """加载全量房源文档，进程内缓存（BM25 每次检索都要用）。"""
    global _POOL
    if _POOL is None:
        from housing.ingest.clean import load_clean

        _POOL = build_listing_documents(load_clean())
    return _POOL
