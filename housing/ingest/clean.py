"""清洗原始挂牌数据，产出统一口径的结构化表。

原始数据 18339 条 × 16 列，质量很好（无缺失、无非数值），
主要工作是把 4 个中文串字段解析成结构化特征，并剔除少量异常样本。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from housing.config import settings

CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
          "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

SIZE_PAT = re.compile(r"(\d+)室(\d+)厅(\d+)厨(\d+)卫")
FLOOR_PAT = re.compile(r"(低|中|高)楼层.*?共(\d+)层")
ELEV_PAT = re.compile(r"([一二两三四五六七八九十]+)梯([一二两三四五六七八九十]+)户")

# 装修由差到好，转成有序数值
DECORATION_RANK = {"毛坯": 0, "其他": 1, "简装": 2, "精装": 3}

# 标题里的卖点关键词
KEYWORD_MAP = {
    "has_parking": "车位",
    "has_storage": "储藏",
    "is_urgent": "急",
    "is_tax_free": "出税",
    "is_full_two": "满二",
    "is_ready_to_move": "拎包",
    "is_furnished": "家具",
}

# 单价合理区间，超出视为录入错误（7 条 <2000，0 条 >50000）
UNIT_PRICE_MIN, UNIT_PRICE_MAX = 2000.0, 50000.0


def cn_num(text: str) -> int | None:
    """中文数字转整数：两 -> 2，十四 -> 14，二十八 -> 28。"""
    if not text:
        return None
    if "十" in text:
        tens_part, _, ones_part = text.partition("十")
        tens = CN_NUM.get(tens_part, 1) if tens_part else 1
        ones = CN_NUM.get(ones_part, 0) if ones_part else 0
        return tens * 10 + ones
    return CN_NUM.get(text[0])


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """读取原始 CSV，保持字符串类型，空字符串不转 NaN。"""
    src = path or settings.data_raw
    df = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return df.apply(lambda s: s.str.strip())


def _parse_size(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["size"].str.extract(SIZE_PAT)
    df["rooms"] = pd.to_numeric(parts[0], errors="coerce")
    df["halls"] = pd.to_numeric(parts[1], errors="coerce")
    df["baths"] = pd.to_numeric(parts[3], errors="coerce")
    return df


def _parse_floor(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["floor"].str.extract(FLOOR_PAT)
    df["floor_level"] = parts[0]
    df["total_floors"] = pd.to_numeric(parts[1], errors="coerce")
    # 没有精确楼层号，用档位近似相对位置，供模型和排序使用
    ratio = {"低": 0.15, "中": 0.50, "高": 0.85}
    df["floor_ratio"] = df["floor_level"].map(ratio)
    df["is_top_floor"] = (df["floor_level"] == "高") & (df["total_floors"] >= 7)
    return df


def _parse_elevator(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["elevatorNum"].str.extract(ELEV_PAT)
    df["elevators"] = parts[0].map(lambda x: cn_num(x) if isinstance(x, str) else None)
    df["units_per_floor"] = parts[1].map(lambda x: cn_num(x) if isinstance(x, str) else None)
    df["units_per_elevator"] = df["units_per_floor"] / df["elevators"]
    return df


def _parse_direction(df: pd.DataFrame) -> pd.DataFrame:
    d = df["direction"].fillna("")
    df["has_south"] = d.str.contains("南")
    df["has_north"] = d.str.contains("北")
    df["has_east"] = d.str.contains("东")
    df["has_west"] = d.str.contains("西")
    df["south_north"] = df["has_south"] & df["has_north"]
    df["direction_count"] = d.str.split().map(len)
    return df


def _parse_title(df: pd.DataFrame) -> pd.DataFrame:
    for col, kw in KEYWORD_MAP.items():
        df[col] = df["title"].str.contains(kw)
    return df


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """原始表 → 建模/检索用表。幂等。"""
    df = load_raw() if df is None else df.copy()

    # 数值列
    for src, dst in [("total_price", "total_price_wan"), ("unit_price", "unit_price"),
                     ("square", "area")]:
        df[dst] = pd.to_numeric(df[src], errors="coerce")
    df["total_price"] = df["total_price_wan"] * 10000  # 统一为元

    df = _parse_size(df)
    df = _parse_floor(df)
    df = _parse_elevator(df)
    df = _parse_direction(df)
    df = _parse_title(df)

    df["decoration_rank"] = df["decoration"].map(DECORATION_RANK)
    df["has_elevator"] = df["elevator"].eq("有")
    df["area_segment"] = pd.cut(
        df["area"], bins=[0, 60, 90, 120, 150, 500, 10000],
        labels=["<60", "60-90", "90-120", "120-150", "150-500", ">500"],
    )

    # 异常样本：单价超出合理区间或关键字段缺失
    n0 = len(df)
    df = df.dropna(subset=["area", "unit_price", "rooms", "total_floors"])
    df = df[(df["unit_price"] >= UNIT_PRICE_MIN) & (df["unit_price"] <= UNIT_PRICE_MAX)]
    df = df[df["area"] > 0]
    n_dropped = n0 - len(df)

    df = df.reset_index(drop=True)
    df.attrs["dropped"] = n_dropped
    return df


def save(df: pd.DataFrame, path: Path | None = None) -> Path:
    """保存清洗结果到 parquet。"""
    settings.ensure_dirs()
    target = path or settings.processed / "clean.parquet"
    # parquet 不支持 category，转回字符串
    out = df.copy()
    if "area_segment" in out:
        out["area_segment"] = out["area_segment"].astype(str)
    out.to_parquet(target, index=False)
    return target


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """读取清洗结果；不存在时现场跑一遍。"""
    target = path or settings.processed / "clean.parquet"
    if target.exists():
        return pd.read_parquet(target)
    return clean()
