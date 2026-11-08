"""估价模块：LightGBM 回归 + 可解释性。

设计要点：
1. 目标是 log(单价)，不是总价——总价 = 单价 × 面积，面积做特征就够了；
2. 两种切分都跑：随机切分（同小区房源同时进训练和测试）和按小区分组切分（真实线上场景）；
3. 解释性用 LightGBM 原生的 `pred_contrib`（等价 SHAP），不额外引入 shap 依赖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from housing.config import settings
from housing.ingest.clean import clean

# 类别特征：用整数编码，未知类别统一映射到 0
CAT_FEATURES = ["district", "nearby", "community", "type"]
NUM_FEATURES = [
    "area", "rooms", "baths", "total_floors", "floor_ratio",
    "units_per_elevator", "direction_count", "decoration_rank",
    "has_elevator", "south_north", "has_parking", "has_storage",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

MODEL_DIR = settings.models_dir
MODEL_FILE = "price_model.txt"
ENCODER_FILE = "feature_encoders.json"
META_FILE = "price_model_meta.json"


def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """从清洗后的表里取特征列（未编码）。"""
    out = pd.DataFrame(index=df.index)
    out["area"] = pd.to_numeric(df["area"], errors="coerce")
    out["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    out["baths"] = pd.to_numeric(df["baths"], errors="coerce")
    out["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce")
    out["floor_ratio"] = pd.to_numeric(df["floor_ratio"], errors="coerce")
    out["units_per_elevator"] = pd.to_numeric(df["units_per_elevator"], errors="coerce")
    out["direction_count"] = pd.to_numeric(df["direction_count"], errors="coerce")
    out["decoration_rank"] = pd.to_numeric(df["decoration_rank"], errors="coerce")
    out["has_elevator"] = df["has_elevator"].astype(int)
    out["south_north"] = df["south_north"].astype(int)
    out["has_parking"] = df["has_parking"].astype(int)
    out["has_storage"] = df["has_storage"].astype(int)
    for col in CAT_FEATURES:
        out[col] = df[col].astype(str)
    return out


def fit_encoders(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """每个类别特征建 值→整数 的映射，1 起编，0 留给未知。"""
    encoders: dict[str, dict[str, int]] = {}
    for col in CAT_FEATURES:
        values = frame[col].value_counts().index.tolist()
        encoders[col] = {v: i + 1 for i, v in enumerate(values)}
    return encoders


def encode(frame: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = frame.copy()
    for col in CAT_FEATURES:
        mapping = encoders[col]
        out[col] = out[col].map(mapping).fillna(0).astype(int)
    for col in NUM_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[FEATURES]


def build_dataset(df: pd.DataFrame | None = None):
    """清洗表 → (未编码特征, log 单价)。"""
    df = clean() if df is None else df
    df = df[df["unit_price"] > 0].reset_index(drop=True)
    frame = _base_frame(df)
    y = np.log(df["unit_price"].to_numpy())
    return frame, y, df


def train(
    df: pd.DataFrame | None = None,
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    seed: int = 42,
):
    """训练模型并对两种切分分别评估。"""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    frame, y, meta = build_dataset(df)

    def fit_eval(train_idx, test_idx, tag: str):
        enc = fit_encoders(frame.iloc[train_idx])
        X_tr = encode(frame.iloc[train_idx], enc)
        X_te = encode(frame.iloc[test_idx], enc)

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y[train_idx], categorical_feature=CAT_FEATURES)

        pred = np.exp(model.predict(X_te))
        actual = np.exp(y[test_idx])
        mape = float(np.mean(np.abs(pred - actual) / actual) * 100)
        # log 空间的残差标准差，用来给单套估价造区间
        resid = float(np.std(np.log(pred) - np.log(actual)))
        return {
            "tag": tag,
            "mape": round(mape, 2),
            "mae": round(float(mean_absolute_error(actual, pred)), 0),
            "r2": round(float(r2_score(actual, pred)), 4),
            "residual_std": round(resid, 4),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }, model, enc

    idx = np.arange(len(frame))

    # A. 随机切分（乐观，同小区会同时出现在训练和测试）
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    random_res, _, _ = fit_eval(tr, te, "随机切分")

    # B. 按小区分组切分（真实场景：测试集是没见过的新小区）
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_g, te_g = next(gss.split(idx, groups=frame["community"]))
    group_res, model, enc = fit_eval(tr_g, te_g, "按小区分组切分")

    return model, enc, [random_res, group_res], meta


def save(model, encoders: dict, metrics: list[dict], extra: dict | None = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # LightGBM 的 save_model 走 C++ fopen，中文路径会报 "not available for writes"，
    # 改成取出模型字符串再用 Python 写盘
    (MODEL_DIR / MODEL_FILE).write_text(
        model.booster_.model_to_string(), encoding="utf-8"
    )
    (MODEL_DIR / ENCODER_FILE).write_text(
        json.dumps(encoders, ensure_ascii=False), encoding="utf-8"
    )
    payload = {"metrics": metrics, "features": FEATURES, **(extra or {})}
    (MODEL_DIR / META_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return MODEL_DIR / MODEL_FILE


def load():
    """加载模型和编码器；不存在则返回 None。"""
    import lightgbm as lgb

    model_path = MODEL_DIR / MODEL_FILE
    enc_path = MODEL_DIR / ENCODER_FILE
    if not model_path.exists() or not enc_path.exists():
        return None, None
    booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    encoders = json.loads(enc_path.read_text(encoding="utf-8"))
    return booster, encoders


@dataclass
class Estimate:
    unit_price: float
    low: float
    high: float
    contributions: list[dict] = field(default_factory=list)


def _raw_row(
    community: str, district: str, nearby: str, area: float, rooms: int,
    floor_level: str = "中", decoration: str = "精装", elevator: bool = True,
    south_north: bool = True, total_floors: int = 6, baths: int = 1,
) -> pd.DataFrame:
    """把一条查询拼成原始 CSV 的一行，复用 clean() 的解析逻辑。"""
    row = {
        "id": "-1",
        "title": "",
        "total_price": "100",
        "unit_price": "8000",
        "square": str(area),
        "size": f"{rooms}室2厅1厨{baths}卫",
        "floor": f"{floor_level}楼层 (共{total_floors}层)",
        "direction": "南 北" if south_north else "南",
        "type": "板楼",
        "district": district,
        "nearby": nearby,
        "community": community,
        "decoration": decoration,
        "elevator": "有" if elevator else "无",
        "elevatorNum": "一梯两户",
        "ownership": "商品房",
    }
    return pd.DataFrame([row])


def _infer_location(community: str, district: str, nearby: str) -> tuple[str, str]:
    """只填小区名时，从历史数据里补出区县和商圈（两者都是强特征）。"""
    if district and nearby:
        return district, nearby
    try:
        from housing.ingest.clean import load_clean

        df = load_clean()
        hit = df[df["community"] == community]
        if not hit.empty:
            if not district:
                district = str(hit["district"].mode().iloc[0])
            if not nearby:
                nearby = str(hit["nearby"].mode().iloc[0])
    except Exception:
        pass
    return district, nearby


def estimate(
    community: str, district: str, nearby: str, area: float, rooms: int,
    floor_level: str = "中", decoration: str = "精装", elevator: bool = True,
    south_north: bool = True, total_floors: int = 6, baths: int = 1,
    top_n: int = 5,
) -> Estimate | None:
    """单套房估价：单价 + 区间 + 影响因素。模型不存在时返回 None。"""
    booster, encoders = load()
    if booster is None or encoders is None:
        return None

    district, nearby = _infer_location(community, district, nearby)

    cleaned = clean(_raw_row(community, district, nearby, area, rooms, floor_level,
                             decoration, elevator, south_north, total_floors, baths))
    if cleaned.empty:
        cleaned = clean(_raw_row(community, district, nearby, area, rooms,
                                 "中", "简装", True, True, 6, 1))
    frame = encode(_base_frame(cleaned), encoders)

    # pred_contrib 返回 n_features+1 列，最后一列是基准值（log 空间）
    contrib = booster.predict(frame, pred_contrib=True)[0]
    log_price = float(np.sum(contrib))
    unit_price = float(np.exp(log_price))

    # 用训练残差的标准差造区间（近似 80% 区间）
    meta_path = MODEL_DIR / META_FILE
    resid = 0.12
    if meta_path.exists():
        try:
            resid = json.loads(meta_path.read_text(encoding="utf-8")).get("residual_std", 0.12)
        except Exception:
            pass
    low = float(np.exp(log_price - 1.28 * resid))
    high = float(np.exp(log_price + 1.28 * resid))

    names = FEATURES + ["基准价"]
    pairs = sorted(
        ((names[i], float(contrib[i])) for i in range(len(FEATURES))),
        key=lambda x: -abs(x[1]),
    )
    contributions = [
        {"feature": name, "impact_pct": round((np.exp(v) - 1) * 100, 1)}
        for name, v in pairs[:top_n]
    ]

    return Estimate(unit_price=round(unit_price, 0), low=round(low, 0),
                    high=round(high, 0), contributions=contributions)
