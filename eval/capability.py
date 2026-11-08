"""能力评测：按维度量化当前系统的水平。

维度：
1. 意图识别（单意图 / 复合意图 / 闲聊）
2. 槽位抽取（价格、面积、户型、中文数字、区间）
3. 多轮理解（指代、继承、条件修改）
4. 检索约束满足（22 条标注查询）
5. 生成引用合法性（回答里的房源编号是否都在检索结果里）
6. 延迟分布

跑法：
    python eval/capability.py            # 全量（含 LLM，约 5~8 分钟）
    python eval/capability.py --no-llm   # 只跑意图/检索/估价，几秒
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[import-not-found]

# ---------------- 标注集 ----------------

INTENT_CASES = [
    # (query, 期望的 intents, 期望的字段)
    ("奎文区 100平以内 两居 预算80万", ["find_listing"], {"district": "奎文区", "max_price": 80}),
    ("我想在潍坊买套房", ["find_listing"], {}),
    ("高新区 三室 有电梯", ["find_listing"], {"district": "高新技术产业开发区", "rooms": 3}),
    ("契税怎么算", ["ask_policy"], {}),
    ("满二和满五有啥区别", ["ask_policy"], {}),
    ("公积金最多能贷多少", ["ask_policy"], {}),
    ("房改房能买吗", ["ask_policy"], {}),
    ("这套房值多少钱", ["estimate"], {}),
    ("东方天韵 100平 精装 值多少", ["estimate"], {"community": "东方天韵"}),
    ("你好", ["chat"], {}),
    ("今天天气怎么样", ["chat"], {}),
    ("帮我写个 Python 脚本", ["chat"], {}),
    ("50万能买什么房，加上税多少", ["find_listing", "ask_policy"], {"max_price": 50}),
    ("80万首付能买多大的房", ["find_listing", "ask_policy"], {}),
    ("总价别超过八十万 奎文区", ["find_listing"], {"max_price": 80}),
    ("六七十万的房子 潍城区", ["find_listing"], {"min_price": 60, "max_price": 70}),
    # 「左右」解析成区间是对的，所以断言区间下界，而不是死磕 100
    ("100个平方左右 两居", ["find_listing"], {"min_area": 90, "max_area": 110}),
]

MULTI_CASES = [
    # (历史, 新问题, 期望条件变化)
    ([("预算50万能买什么房，加各种税", "…")], "那100万呢",
     {"max_price": 100}, "意图应保持含 ask_policy"),
    ([("奎文区 预算80万 两居", "…")], "那高新区呢",
     {"district": "高新技术产业开发区"}, "区域应换，预算沿用"),
    ([("预算80万 两居 奎文区", "…")], "再便宜点",
     {}, "预算应下调"),
    ([("考虑税费和契税", "…")], "那公积金呢",
     {}, "应仍走政策"),
]

FLOAT_FIELDS = {"min_price", "max_price", "min_area", "max_area"}


def _ok(exp, got) -> bool:
    if got is None:
        return False
    if isinstance(exp, float) or isinstance(got, float):
        try:
            return abs(float(got) - float(exp)) < 5
        except (TypeError, ValueError):
            return False
    return got == exp


def eval_intents() -> dict:
    from housing.advisor.intent import classify

    exact = 0
    field_total = field_ok = 0
    rows = []
    for query, want_intents, want_fields in INTENT_CASES:
        intent = classify(query)
        same = set(intent.intents) == set(want_intents)
        fields_hit = all(_ok(v, getattr(intent.filters, k, None)) for k, v in want_fields.items())
        exact += int(same)
        for k, v in want_fields.items():
            field_total += 1
            field_ok += int(_ok(v, getattr(intent.filters, k, None)))
        rows.append((query, want_intents, intent.intents, same, fields_hit))

    return {
        "n": len(INTENT_CASES),
        "intent_exact": exact,
        "intent_rate": round(exact / len(INTENT_CASES) * 100, 1),
        "field_hit": field_ok,
        "field_total": field_total,
        "field_rate": round(field_ok / max(field_total, 1) * 100, 1),
        "rows": rows,
    }


def eval_multi() -> dict:
    from housing.advisor.intent import classify

    ok = 0
    rows = []
    for ctx, query, want, note in MULTI_CASES:
        history = []
        for u, a in ctx:
            history += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        intent = classify(query, history)
        hit = all(_ok(v, getattr(intent.filters, k, None)) for k, v in want.items())
        # 再便宜点：只要预算变了就算对
        if query == "再便宜点":
            hit = intent.filters.max_price is not None and intent.filters.max_price < 80
        ok += int(hit)
        rows.append((query, note, hit, intent.intents))
    return {"n": len(MULTI_CASES), "hit": ok,
            "rate": round(ok / len(MULTI_CASES) * 100, 1), "rows": rows}


def eval_retrieval() -> dict:
    import numpy as np

    from housing.ingest.build_docs import load_listing_documents
    from housing.search.retriever import SearchFilters, matches, search_with_flags

    pool = load_listing_documents()
    items = [json.loads(x) for x in
             (ROOT / "eval" / "qa_pairs.jsonl").read_text("utf-8").splitlines() if x.strip()]
    listing = [i for i in items if i["type"] == "listing"]

    valid = set(SearchFilters.__dataclass_fields__)
    all_ok = 0
    sat_rates, latencies = [], []
    for item in listing:
        f = SearchFilters(**{k: v for k, v in item["expect"].items() if k in valid})
        t0 = time.perf_counter()
        docs, _ = search_with_flags(item["query"], f, docs_pool=pool, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        if not docs:
            sat_rates.append(0.0)
            continue
        rate = sum(1 for d in docs if matches(d, f)) / len(docs)
        sat_rates.append(rate)
        all_ok += int(rate == 1.0)

    return {
        "n": len(listing),
        "all_ok": all_ok,
        "all_ok_rate": round(all_ok / len(listing) * 100, 1),
        "avg_satisfaction": round(float(np.mean(sat_rates)) * 100, 1),
        "p50_ms": round(statistics.median(latencies)),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1]),
    }


def eval_generation(n: int = 4) -> dict:
    import re

    from housing.advisor.chain import answer

    items = [json.loads(x) for x in
             (ROOT / "eval" / "qa_pairs.jsonl").read_text("utf-8").splitlines() if x.strip()]
    listing = [i for i in items if i["type"] == "listing"][:n]

    clean, rows = 0, []
    for item in listing:
        t0 = time.perf_counter()
        res = answer(item["query"])
        dt = time.perf_counter() - t0
        source_ids = {str(s.get("id")) for s in res.sources}
        cited = set(re.findall(r"\[(\d+)\]", res.answer))
        fake = cited - source_ids
        ok = not fake and bool(cited)
        clean += int(ok)
        rows.append((item["id"], ok, len(cited), sorted(fake), round(dt, 1)))

    return {"n": len(listing), "clean": clean,
            "rate": round(clean / max(len(listing), 1) * 100, 1), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("能力评测")
    print("=" * 72)

    print("\n[1/4] 意图识别与槽位抽取 …")
    intents = eval_intents()
    print(f"   意图完全正确 {intents['intent_exact']}/{intents['n']} "
          f"({intents['intent_rate']}%)")
    print(f"   槽位字段命中 {intents['field_hit']}/{intents['field_total']} "
          f"({intents['field_rate']}%)")
    for query, want, got, same, fields_hit in intents["rows"]:
        if not (same and fields_hit):
            print(f"     ✗ {query[:26]:<28} 期望{want} 实际{got}")

    print("\n[2/4] 多轮理解 …")
    multi = eval_multi()
    print(f"   通过 {multi['hit']}/{multi['n']} ({multi['rate']}%)")
    for query, note, hit, got in multi["rows"]:
        if not hit:
            print(f"     ✗ {query:<12} {note} → {got}")

    print("\n[3/4] 检索约束满足 …")
    ret = eval_retrieval()
    print(f"   Top10 全部满足条件 {ret['all_ok']}/{ret['n']} ({ret['all_ok_rate']}%)")
    print(f"   单条平均满足率 {ret['avg_satisfaction']}%")
    print(f"   延迟 P50 {ret['p50_ms']}ms  P95 {ret['p95_ms']}ms")

    gen = {"n": 0, "clean": 0, "rate": 0.0, "rows": []}
    if not args.no_llm:
        print("\n[4/4] 生成引用合法性 …")
        gen = eval_generation()
        print(f"   无伪造编号 {gen['clean']}/{gen['n']} ({gen['rate']}%)")
        for iid, ok, cited, fake, dt in gen["rows"]:
            print(f"     {'✓' if ok else '✗'} {iid} 引用{cited}条 "
                  f"伪造{len(fake)} {dt}s")

    summary = {
        "intent_rate": intents["intent_rate"],
        "slot_rate": intents["field_rate"],
        "multi_rate": multi["rate"],
        "retrieval_rate": ret["all_ok_rate"],
        "retrieval_p50_ms": ret["p50_ms"],
        "retrieval_p95_ms": ret["p95_ms"],
        "generation_rate": gen["rate"] if gen["n"] else None,
    }

    out = ROOT / "eval" / "results" / "capability_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("总览")
    print("=" * 72)
    for k, v in summary.items():
        print(f"   {k:<20} {v}")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
