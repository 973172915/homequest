"""核心逻辑单元测试：分词、意图解析、条件匹配、清洗、路由。

全部在 CPU 上跑，不加载向量模型，几秒钟完成：
    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from housing.advisor.chain import is_policy_query  # noqa: E402
from housing.ingest.build_docs import format_listing  # noqa: E402
from housing.ingest.clean import clean  # noqa: E402
from housing.search.retriever import SearchFilters, matches, parse_query  # noqa: E402
from housing.search.tokenizer import tokenize  # noqa: E402

RAW_COLUMNS = [
    "id", "title", "total_price", "unit_price", "square", "size", "floor",
    "direction", "type", "district", "nearby", "community", "decoration",
    "elevator", "elevatorNum", "ownership",
]


def make_raw(**overrides) -> pd.DataFrame:
    row = {
        "id": "1",
        "title": "东方天韵 精装两居 南北通透",
        "total_price": "75",
        "unit_price": "7503",
        "square": "100",
        "size": "2室2厅1厨1卫",
        "floor": "中楼层 (共11层)",
        "direction": "南 北",
        "type": "板楼",
        "district": "奎文区",
        "nearby": "新华路北",
        "community": "东方天韵",
        "decoration": "精装",
        "elevator": "有",
        "elevatorNum": "一梯两户",
        "ownership": "商品房",
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=RAW_COLUMNS)


# ---------------- 分词 ----------------
def test_tokenize_keeps_bigram():
    tokens = tokenize("东方天韵")
    assert "东" in tokens
    assert "东方" in tokens          # 二元组能命中小区名的局部
    assert "东方天韵" not in tokens  # 不依赖分词词典


def test_tokenize_handles_mixed_text():
    tokens = tokenize("3室2厅 南北通透 90平")
    assert "90" in tokens
    assert "南北" in tokens


# ---------------- 意图解析 ----------------
def test_parse_district_alias():
    f = parse_query("高新区 总价150万 三室 有电梯")
    assert f.district == "高新技术产业开发区"
    assert f.max_price == 150
    assert f.rooms == 3
    assert f.elevator is True


def test_parse_community_not_swallowed_by_district():
    """「东方天韵」不能被当成区县，也不能把「高新实验」识别成高新区。"""
    f = parse_query("东方天韵 90平左右 精装")
    assert f.community == "东方天韵"
    assert f.district is None
    assert f.min_area == pytest.approx(81.0)
    assert f.max_area == 90.0

    f2 = parse_query("高新实验附近 三室 预算140万")
    assert f2.nearby == "高新实验"
    assert f2.district is None


def test_parse_nearby_and_rooms():
    f = parse_query("万达附近 预算60万 两居")
    assert f.nearby == "万达"
    assert f.max_price == 60
    assert f.rooms == 2


def test_parse_bare_price():
    """「经济开发区 60万」这种不带「预算」前缀的也要能解析。"""
    f = parse_query("经济开发区 60万 两室")
    assert f.district == "经济技术开发区"
    assert f.max_price == 60


# ---------------- 条件匹配 ----------------
def _doc(**meta):
    base = {"district": "奎文区", "price": 75.0, "area": 100.0, "rooms": 2,
            "south_north": 1, "decoration": "精装", "elevator": 1,
            "community": "东方天韵", "nearby": "新华路北"}
    base.update(meta)
    return SimpleNamespace(metadata=base)


def test_matches_all_constraints():
    f = SearchFilters(district="奎文区", max_price=80, max_area=100, rooms=2)
    assert matches(_doc(), f)


@pytest.mark.parametrize(
    "meta",
    [
        {"district": "坊子区"},
        {"price": 90.0},
        {"area": 120.0},
        {"rooms": 3},
        {"decoration": "毛坯"},
        {"elevator": 0},
    ],
)
def test_matches_rejects_violation(meta):
    f = SearchFilters(district="奎文区", max_price=80, max_area=100,
                      rooms=2, decoration="精装", elevator=True)
    assert not matches(_doc(**meta), f)


# ---------------- 清洗 ----------------
def test_clean_parses_chinese_fields():
    df = clean(make_raw())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["rooms"] == 2 and row["halls"] == 2 and row["baths"] == 1
    assert row["floor_level"] == "中"
    assert row["total_floors"] == 11
    assert bool(row["south_north"]) is True
    assert row["elevators"] == 1 and row["units_per_floor"] == 2
    assert row["total_price"] == 750000


def test_clean_drops_outlier_unit_price():
    df = clean(make_raw(unit_price="100"))  # 单价 100 元/㎡ 明显是脏数据
    assert len(df) == 0


def test_format_listing_contains_key_fields():
    row = clean(make_raw()).iloc[0]
    text = format_listing(row)
    assert "东方天韵" in text
    assert "75.0" in text


# ---------------- 问题路由 ----------------
def test_policy_question_routes_to_knowledge():
    q = "买一套100万的二手房，满两年，契税要交多少"
    assert is_policy_query(q, parse_query(q)) is True


def test_listing_question_routes_to_search():
    q = "奎文区 100平以内 精装两居 预算80万"
    assert is_policy_query(q, parse_query(q)) is False


# ---------------- 估价 ----------------
def test_estimate_interval_and_contributions():
    """模型存在时应给出合理单价区间和影响因素；不存在则跳过。"""
    from housing.estimate.model import estimate

    result = estimate("东方天韵", "奎文区", "新华路北", 100.0, 2)
    if result is None:
        pytest.skip("估价模型尚未训练")

    assert 1000 < result.unit_price < 60000
    assert result.low < result.unit_price < result.high
    assert len(result.contributions) <= 5
    # 单价 ≈ 该城市二手房的实际量级
    assert 3000 < result.unit_price < 20000


def test_estimate_area_scales_total_price():
    """面积翻倍，总价应接近翻倍（单价模型 + 面积换算）。"""
    from housing.estimate.model import estimate

    small = estimate("恒大名都", "高新技术产业开发区", "市府东南", 100.0, 3)
    big = estimate("恒大名都", "高新技术产业开发区", "市府东南", 200.0, 3)
    if small is None or big is None:
        pytest.skip("估价模型尚未训练")

    ratio = (big.unit_price * 200) / (small.unit_price * 100)
    assert 1.5 < ratio < 2.5
