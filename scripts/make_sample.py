"""从全量数据里抽 2000 条样本，用于提交到仓库。

按区县分层抽样，保证样本的区县分布与全量一致；固定随机种子，结果可复现。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sh_data.csv"
DST = ROOT / "data" / "sample_2000.csv"
N = 2000
SEED = 42

df = pd.read_csv(SRC, dtype=str, keep_default_na=False, encoding="utf-8-sig")
print(f"全量: {len(df)} 条, {df['district'].nunique()} 个区县")

# 按区县分层抽样，各区县按比例取
parts = []
for _district, group in df.groupby("district", sort=False):
    k = max(1, round(len(group) / len(df) * N))
    parts.append(group.sample(n=min(k, len(group)), random_state=SEED))

sample = pd.concat(parts).head(N).sample(frac=1, random_state=SEED).reset_index(drop=True)
sample.to_csv(DST, index=False, encoding="utf-8", quoting=1)
print(f"样本: {len(sample)} 条 -> {DST}")
print("\n区县分布对比（全量 vs 样本）:")
full_pct = df["district"].value_counts(normalize=True) * 100
samp_pct = sample["district"].value_counts(normalize=True) * 100
for d in df["district"].value_counts().index:
    print(f"  {d:<12} {full_pct[d]:5.1f}%  ->  {samp_pct.get(d, 0):5.1f}%")
