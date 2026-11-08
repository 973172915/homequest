"""评测脚本：意图解析、检索约束满足率、生成引用合法性、延迟。

用法：
    python eval/run_eval.py              # 全量（含 5 条生成评测）
    python eval/run_eval.py --no-gen     # 只跑检索，不调 LLM（几秒钟）
    python eval/run_eval.py --gen 8      # 生成评测跑 8 条

产出：
    artifacts/reports/eval_report.md / eval_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from housing.config import settings  # noqa: E402
from housing.search.retriever import (  # noqa: E402
    SearchFilters,
    matches,
    parse_query,
    search_with_flags,
)

QA_FILE = ROOT / "eval" / "qa_pairs.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
FLOAT_FIELDS = {"min_price", "max_price", "min_area", "max_area"}


def load_items() -> list[dict]:
    return [json.loads(line) for line in QA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_filters(item: dict) -> SearchFilters:
    """把 expect 里的条件转成 SearchFilters（忽略 kind 这类非条件字段）。"""
    valid = {f.name for f in fields(SearchFilters)}
    return SearchFilters(**{k: v for k, v in item["expect"].items() if k in valid})


def field_equal(name: str, got, want) -> bool:
    if got is None:
        return False
    if name in FLOAT_FIELDS:
        return abs(float(got) - float(want)) < 0.5
    return got == want


def eval_parse(items: list[dict]) -> dict:
    """逐字段对比规则解析器输出和人工标注。"""
    per_field: dict[str, list[bool]] = {}
    exact = 0
    listing_items = [i for i in items if i["type"] == "listing"]

    for item in listing_items:
        parsed = parse_query(item["query"])
        all_ok = True
        for key, want in item["expect"].items():
            if key == "kind":
                continue
            ok = field_equal(key, getattr(parsed, key, None), want)
            per_field.setdefault(key, []).append(ok)
            all_ok = all_ok and ok
        exact += int(all_ok)

    return {
        "total": len(listing_items),
        "exact_match": exact,
        "exact_rate": round(exact / max(len(listing_items), 1) * 100, 1),
        "per_field": {
            k: {"n": len(v), "ok": sum(v), "rate": round(sum(v) / len(v) * 100, 1)}
            for k, v in sorted(per_field.items())
        },
        "failures": [
            {"id": i["id"], "query": i["query"], "parsed": parse_query(i["query"]).__dict__,
             "expect": i["expect"]}
            for i in listing_items
            if not all(field_equal(k, getattr(parse_query(i["query"]), k, None), v)
                       for k, v in i["expect"].items() if k != "kind")
        ],
    }


def eval_retrieval(items: list[dict], pool, k: int = 10) -> dict:
    """用人工标注的条件（而不是解析结果）做检索，单独衡量检索层质量。"""
    rows = []
    latencies = []
    for item in items:
        if item["type"] != "listing":
            continue
        filters = expected_filters(item)
        t0 = time.perf_counter()
        docs, relaxed = search_with_flags(item["query"], filters, docs_pool=pool, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

        if not docs:
            rows.append({"id": item["id"], "query": item["query"], "returned": 0,
                         "satisfied": 0, "rate": 0.0, "all_ok": False, "relaxed": relaxed})
            continue
        ok = sum(1 for d in docs if matches(d, filters))
        rows.append({"id": item["id"], "query": item["query"], "returned": len(docs),
                     "satisfied": ok, "rate": round(ok / len(docs) * 100, 1),
                     "all_ok": ok == len(docs), "relaxed": relaxed})

    total = len(rows)
    all_ok = sum(1 for r in rows if r["all_ok"])
    satisfactions = [r["satisfied"] for r in rows]
    total_returned = sum(r["returned"] for r in rows)
    relaxed_rows = [r for r in rows if r["relaxed"]]
    strict_rows = [r for r in rows if not r["relaxed"]]
    strict_ok = sum(1 for r in strict_rows if r["all_ok"])

    return {
        "total": total,
        "all_satisfied": all_ok,
        "all_satisfied_rate": round(all_ok / max(total, 1) * 100, 1),
        "constraint_satisfaction": round(sum(satisfactions) / max(total_returned, 1) * 100, 1),
        "strict_total": len(strict_rows),
        "strict_ok": strict_ok,
        "strict_rate": round(strict_ok / max(len(strict_rows), 1) * 100, 1),
        "relaxed_cases": [{"id": r["id"], "query": r["query"], "relaxed": r["relaxed"]} for r in relaxed_rows],
        "latency_p50_ms": round(statistics.median(latencies), 0),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 0),
        "rows": rows,
    }


ID_PAT = re.compile(r"\[(\d+)\]")
PRICE_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*万")


def check_generation(item: dict, answer: str, sources: list[dict]) -> dict:
    """幻觉检测：回答里出现的房源编号、价格是否都在检索结果里。"""
    source_ids = {str(s.get("id")) for s in sources}
    cited = set(ID_PAT.findall(answer))
    fake_ids = cited - source_ids

    prices = {round(float(s["price"]), 1) for s in sources if s.get("price") is not None}
    claims = [float(x) for x in PRICE_PAT.findall(answer)]
    # 只校验 20 万以上的价格断言（小额多是税费/中介费，不属于房源价格）
    big_claims = [c for c in claims if c >= 20]
    bad_prices = [
        c for c in big_claims
        if not any(abs(c - p) <= 1.0 for p in prices)
    ]

    return {
        "id": item["id"],
        "query": item["query"],
        "chars": len(answer),
        "cited": len(cited),
        "fake_ids": sorted(fake_ids),
        "price_claims": len(big_claims),
        "bad_prices": bad_prices,
        "ok": not fake_ids and not bad_prices and bool(cited),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=5, help="生成评测条数")
    parser.add_argument("--no-gen", action="store_true")
    args = parser.parse_args()

    settings.ensure_dirs()
    items = load_items()
    print(f"评测集：{len(items)} 条（找房 {sum(1 for i in items if i['type']=='listing')}，"
          f"政策 {sum(1 for i in items if i['type']=='policy')}）\n")

    print("1/3 意图解析 …")
    parse_res = eval_parse(items)
    print(f"   完全匹配 {parse_res['exact_match']}/{parse_res['total']} "
          f"({parse_res['exact_rate']}%)")
    for f, v in parse_res["per_field"].items():
        print(f"     {f:<14} {v['ok']}/{v['n']}  {v['rate']}%")

    print("\n2/3 检索 …")
    from housing.ingest.build_docs import load_listing_documents

    pool = load_listing_documents()
    ret_res = eval_retrieval(items, pool)
    print(f"   全部命中条件 {ret_res['all_satisfied']}/{ret_res['total']} "
          f"({ret_res['all_satisfied_rate']}%)")
    print(f"   严格模式（不含触发放宽） {ret_res['strict_ok']}/{ret_res['strict_total']} "
          f"({ret_res['strict_rate']}%)")
    print(f"   单条约束满足率 {ret_res['constraint_satisfaction']}%")
    print(f"   延迟 P50 {ret_res['latency_p50_ms']:.0f}ms  P95 {ret_res['latency_p95_ms']:.0f}ms")
    for case in ret_res["relaxed_cases"]:
        print(f"     ~ {case['id']} {case['query']}  放宽了「{'、'.join(case['relaxed'])}」")
    for row in ret_res["rows"]:
        if not row["all_ok"] and not row["relaxed"]:
            print(f"     ! {row['id']} {row['query']}  满足 {row['rate']}%")

    gen_res = []
    if not args.no_gen:
        print(f"\n3/3 生成（{args.gen} 条，调一次 LLM 约 5 秒）…")
        from housing.advisor.chain import answer

        listing_items = [i for i in items if i["type"] == "listing"][: args.gen]
        for item in listing_items:
            t0 = time.perf_counter()
            res = answer(item["query"])
            dt = time.perf_counter() - t0
            check = check_generation(item, res.answer, res.sources)
            check["seconds"] = round(dt, 1)
            check["mode"] = res.mode
            gen_res.append(check)
            mark = "✓" if check["ok"] else "✗"
            print(f"   {mark} {item['id']} {dt:4.1f}s 引用{check['cited']}条 "
                  f"伪造编号{len(check['fake_ids'])} 价格异常{len(check['bad_prices'])}")

    ok_gen = sum(1 for g in gen_res if g["ok"])
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"total": len(items), "size": len(pool)},
        "parse": parse_res,
        "retrieval": {k: v for k, v in ret_res.items() if k != "rows"},
        "retrieval_rows": ret_res["rows"],
        "generation": {
            "n": len(gen_res),
            "clean": ok_gen,
            "clean_rate": round(ok_gen / max(len(gen_res), 1) * 100, 1),
            "details": gen_res,
        },
    }

    out_json = RESULTS_DIR / "eval_report.json"
    out_md = RESULTS_DIR / "eval_report.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 评测报告",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 评测集：{len(items)} 条（找房 {len([i for i in items if i['type']=='listing'])} 条，"
        f"政策 {len([i for i in items if i['type']=='policy'])} 条）",
        f"- 房源库：{len(pool)} 条",
        "",
        "## 指标",
        "",
        "| 层 | 指标 | 数值 |",
        "| --- | --- | --- |",
        f"| 意图解析 | 字段完全匹配率 | {parse_res['exact_rate']}% |",
        f"| 检索 | Top10 全部满足条件的比例 | {ret_res['all_satisfied_rate']}% |",
        f"| 检索 | 严格模式（不含放宽） | {ret_res['strict_rate']}% "
        f"({ret_res['strict_ok']}/{ret_res['strict_total']}) |",
        f"| 检索 | 单条结果约束满足率 | {ret_res['constraint_satisfaction']}% |",
        f"| 检索 | 延迟 P50 / P95 | {ret_res['latency_p50_ms']:.0f}ms / {ret_res['latency_p95_ms']:.0f}ms |",
        f"| 生成 | 无伪造编号且价格可溯源 | {summary['generation']['clean_rate']}% "
        f"({ok_gen}/{len(gen_res)}) |",
        "",
    ]
    (out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告：{out_json}\n      {out_md}")


if __name__ == "__main__":
    main()
