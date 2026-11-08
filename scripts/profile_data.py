"""数据体检脚本：跑一遍，就知道 sh_data.csv 能不能直接用来建模。

用法：
    python scripts/profile_data.py

输出：
    控制台摘要 + data/reports/profile_report.txt
"""
import re
import sys
from pathlib import Path

import pandas as pd

try:  # Windows 控制台默认 GBK，中文会乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "sh_data.csv"
OUT_DIR = ROOT / "artifacts" / "reports"
OUT_FILE = OUT_DIR / "profile_report.txt"

CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9}
TEN = "十"

_lines = []


def say(msg=""):
    print(msg)
    _lines.append(str(msg))


def cn_num(text):
    """中文数字转整数：两 -> 2，十四 -> 14，二十八 -> 28，三十七 -> 37。"""
    if not text:
        return None
    if TEN in text:
        tens_part, _, ones_part = text.partition(TEN)
        tens = CN_NUM.get(tens_part, 1) if tens_part else 1
        ones = CN_NUM.get(ones_part, 0) if ones_part else 0
        return tens * 10 + ones
    return CN_NUM.get(text[0])


def parse_elevator(text):
    """「一梯两户」-> (1, 2)；「八梯十四户」-> (8, 14)。"""
    if not isinstance(text, str):
        return (None, None)
    m = re.search(r"([一二两三四五六七八九十]+)梯([一二两三四五六七八九十]+)户", text)
    if not m:
        return (None, None)
    return (cn_num(m.group(1)), cn_num(m.group(2)))


def section(title):
    say()
    say("=" * 70)
    say(title)
    say("=" * 70)


def main():
    if not RAW.exists():
        raise SystemExit(f"找不到原始数据：{RAW}")

    df = pd.read_csv(RAW, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.apply(lambda s: s.str.strip())

    section("1. 基本规模")
    say(f"行数: {len(df)}")
    say(f"列数: {df.shape[1]}")
    say(f"列名: {list(df.columns)}")
    say(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    section("2. 缺失值（空字符串按缺失算）")
    missing = (df == "").sum()
    for col, cnt in missing.items():
        if cnt:
            say(f"  {col:<14} {cnt:>6} 条  ({cnt / len(df) * 100:.2f}%)")
    if missing.sum() == 0:
        say("  没有空字符串字段")

    section("3. 重复与 id 连续性")
    say(f"整行完全重复: {df.duplicated().sum()} 条")
    say(f"同一 title 重复: {df.duplicated(subset=['title']).sum()} 条")
    ids = pd.to_numeric(df["id"], errors="coerce")
    say(f"id 范围: {ids.min()} ~ {ids.max()}，去重后 {ids.nunique()} 个，连续性缺失 {int(ids.max() - ids.min() + 1 - ids.nunique())} 个")

    section("4. 数值字段")
    num = pd.DataFrame({
        "total_price": pd.to_numeric(df["total_price"], errors="coerce"),
        "unit_price": pd.to_numeric(df["unit_price"], errors="coerce"),
        "square": pd.to_numeric(df["square"], errors="coerce"),
    })
    say("非数值无法转换的行数:")
    for col in num.columns:
        bad = num[col].isna().sum()
        say(f"  {col:<12} {bad} 条")
    say()
    say(num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2).to_string())

    section("5. 单价 / 总价 / 面积 一致性校验")
    ok = num.dropna()
    if len(ok):
        implied = ok["total_price"] * 10000 / ok["square"]
        diff = (implied - ok["unit_price"]).abs() / ok["unit_price"]
        for thr in (0.01, 0.05, 0.10):
            say(f"  单价与 (总价/面积) 偏差 > {int(thr * 100):>2}% 的行: {int((diff > thr).sum()):>5} 条")
        say(f"  偏差最大的一行: 单价 {ok.loc[diff.idxmax(), 'unit_price']:.0f}，"
            f"按总价/面积算 {(implied[diff.idxmax()]):.0f}")

    section("6. 异常值筛查（按业务经验阈值）")
    checks = [
        ("单价 < 2000 元/㎡", num["unit_price"] < 2000),
        ("单价 > 50000 元/㎡", num["unit_price"] > 50000),
        ("面积 < 20 ㎡", num["square"] < 20),
        ("面积 > 500 ㎡", num["square"] > 500),
        ("总价 < 5 万", num["total_price"] < 5),
        ("总价 > 1000 万", num["total_price"] > 1000),
    ]
    for name, mask in checks:
        say(f"  {name:<22} {int(mask.sum()):>5} 条")

    section("7. 类别字段分布（Top 15）")
    for col in ["district", "nearby", "type", "decoration", "elevator", "ownership", "direction", "size"]:
        if col not in df.columns:
            continue
        say(f"\n-- {col}  （{df[col].nunique()} 个取值）")
        say(df[col].value_counts().head(15).to_string())

    section("8. 小区（高基数字段）")
    vc = df["community"].value_counts()
    say(f"小区总数: {df['community'].nunique()}")
    say(f"只出现 1 次的房源所在小区: {int((vc == 1).sum())} 个")
    say(f"出现最多的 10 个小区:\n{vc.head(10).to_string()}")

    section("9. 结构化解析可行性")
    size_pat = re.compile(r"(\d+)室(\d+)厅(\d+)厨(\d+)卫")
    parsed = df["size"].str.extract(size_pat)
    say(f"户型可解析: {parsed[0].notna().sum()} / {len(df)} 条")
    say("户型分布（室-厅）:")
    combo = parsed[0].fillna("?") + "室" + parsed[1].fillna("?") + "厅"
    say(combo.value_counts().head(10).to_string())

    floor_pat = re.compile(r"(低|中|高)楼层.*?共(\d+)层")
    fl = df["floor"].str.extract(floor_pat)
    say(f"\n楼层可解析: {fl[0].notna().sum()} / {len(df)} 条")
    say("楼层档位分布:")
    say(fl[0].fillna("(无法解析)").value_counts().to_string())
    floors = pd.to_numeric(fl[1], errors="coerce")
    say(f"总层数: 最少 {floors.min():.0f}，最多 {floors.max():.0f}，中位 {floors.median():.0f}")

    elev = df["elevatorNum"].map(parse_elevator)
    per_elev = pd.DataFrame(elev.tolist(), columns=["elevators", "units"]).dropna()
    say(f"\n梯户比可解析: {len(per_elev)} / {len(df)} 条")
    if len(per_elev):
        ratio = per_elev["units"] / per_elev["elevators"]
        say(f"每梯服务户数: 中位 {ratio.median():.1f}，最大 {ratio.max():.0f}")
        say(f"电梯数: 中位 {per_elev['elevators'].median():.0f}，最大 {per_elev['elevators'].max():.0f}")

    section("10. 文本字段")
    title_len = df["title"].str.len()
    say(f"title 长度: 最短 {title_len.min()}，最长 {title_len.max()}，中位 {title_len.median():.0f}")
    say(f"title 含空格的行数: {int(df['title'].str.contains(' ').sum())}")
    for kw in ["急", "学区", "车位", "储藏", "拎包", "精装", "满二", "出税"]:
        say(f"  title 含「{kw}」: {int(df['title'].str.contains(kw).sum())} 条")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        OUT_FILE.write_text("\n".join(_lines), encoding="utf-8")
        target = OUT_FILE
    except PermissionError:
        target = ROOT / "profile_report.txt"
        target.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n报告已写入: {target}")


if __name__ == "__main__":
    main()
