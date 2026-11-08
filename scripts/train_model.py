"""训练估价模型并输出评估报告。

用法：
    python scripts/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from housing.config import settings  # noqa: E402
from housing.estimate.model import build_dataset, save, train  # noqa: E402


def baseline_metrics() -> list[dict]:
    """中位数基线：全区县 / 按商圈 / 按小区，作为模型的下限参照。"""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    frame, y, df = build_dataset()
    price = np.exp(y)
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(idx, groups=frame["community"]))

    out = []
    global_med = float(np.median(price[tr]))
    for label, key in [("全局中位数", None), ("区县中位数", "district"), ("商圈中位数", "nearby")]:
        if key is None:
            pred = np.full(len(te), global_med)
        else:
            med = df.iloc[tr].groupby(key)["unit_price"].median()
            pred = df.iloc[te][key].map(med).fillna(global_med).to_numpy()
        mape = float(np.mean(np.abs(pred - price[te]) / price[te]) * 100)
        out.append({"tag": label, "mape": round(mape, 2)})
    return out


def main() -> None:
    settings.ensure_dirs()
    print("1/3 训练中（约 30~60 秒）…")
    model, encoders, metrics, meta = train()
    resid = metrics[-1]["residual_std"] if metrics else 0.12
    path = save(model, encoders, metrics, extra={"residual_std": resid})
    print(f"   模型已保存: {path}")

    print("2/3 计算中位数基线 …")
    baselines = baseline_metrics()

    print("3/3 写评估报告 …")
    lines = [
        "# 估价模型评估",
        "",
        "目标：预测**挂牌单价**（元/㎡），样本 18332 条。",
        "",
        "## 基线 vs 模型",
        "",
        "| 方法 | 切分方式 | MAPE |",
        "| --- | --- | --- |",
    ]
    for b in baselines:
        lines.append(f"| {b['tag']} | 按小区分组 | {b['mape']}% |")
    for m in metrics:
        lines.append(f"| **LightGBM** | {m['tag']} | **{m['mape']}%** |")

    lines += ["", "## 详细指标", "", "| 切分 | MAPE | MAE(元/㎡) | R² | 训练/测试 |",
              "| --- | --- | --- | --- | --- |"]
    for m in metrics:
        lines.append(
            f"| {m['tag']} | {m['mape']}% | {m['mae']:.0f} | {m['r2']} | "
            f"{m['n_train']}/{m['n_test']} |"
        )

    lines += [
        "",
        "## 怎么读这两个数",
        "",
        "- **随机切分**偏乐观：同一小区的房源会同时出现在训练集和测试集，模型等于「见过这个小区」。",
        "- **按小区分组切分**才是线上真实场景：用户问的房子，往往来自模型没见过的楼盘。",
        "- 两者差距就是「小区记忆」带来的虚高，写进 README 比只报一个好数字更可信。",
        "",
        f"区间估计用 log 残差标准差 {resid:.3f}，约 ±{1.28 * resid * 100:.0f}%（80% 区间）。",
    ]

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"   报告: {out_dir / 'model_eval.md'}")

    print("\n=== 结果 ===")
    for b in baselines:
        print(f"  基线 {b['tag']:<8} MAPE {b['mape']:>6.2f}%")
    for m in metrics:
        print(f"  模型 {m['tag']:<12} MAPE {m['mape']:>6.2f}%  MAE {m['mae']:.0f}  R² {m['r2']}")


if __name__ == "__main__":
    main()
