"""RAG 评测：检索层 + 生成层。

检索指标（相关性 = 满足全部硬条件）：
    Precision@10  返回的 10 条里有多少条真的合规
    nDCG@10       合规的结果排得够不够靠前
    MRR           第一条合规结果出现在第几位
    无结果率      搜出 0 条的比例

生成指标：
    引用合法率    回答里的 [房源编号] 是否都真实存在于检索结果
    价格一致率    回答里提到的金额是否都能在检索结果里找到
    拒答正确率    资料没有的问题，是否老实说不知道而不是编

跑法：
    python eval/rag_eval.py            # 全量（含 LLM，约 1 分钟）
    python eval/rag_eval.py --no-gen   # 只跑检索
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

# 评测必须可复现：把温度钉死为 0（生产环境仍是 0.3）。
# 之前同一个问题两次跑出不同答案，导致指标在 7/8 和 8/8 之间跳。
os.environ.setdefault("DEEPSEEK_TEMPERATURE", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[import-not-found]

# 知识库里没有答案的问题，用来测「会不会编」
REFUSAL_CASES = [
    "潍坊 2027 年房价会涨吗",
    "东方天韵的物业费一个月多少钱",
    "你能帮我去银行办贷款吗",
]


def _items() -> list[dict]:
    path = ROOT / "eval" / "qa_pairs.jsonl"
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def eval_retrieval() -> dict:
    from housing.ingest.build_docs import load_listing_documents
    from housing.search.retriever import SearchFilters, matches, search_with_flags

    items = [i for i in _items() if i["type"] == "listing"]
    pool = load_listing_documents()
    valid = set(SearchFilters.__dataclass_fields__)

    precisions, ndcgs, mrrs, latencies = [], [], [], []
    empty = 0

    for item in items:
        f = SearchFilters(**{k: v for k, v in item["expect"].items() if k in valid})
        t0 = time.perf_counter()
        docs, _ = search_with_flags(item["query"], f, docs_pool=pool, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)

        if not docs:
            empty += 1
            precisions.append(0.0)
            ndcgs.append(0.0)
            mrrs.append(0.0)
            continue

        rel = [1 if matches(d, f) else 0 for d in docs]
        precisions.append(sum(rel) / len(rel))

        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(rel), len(rel))))
        ndcgs.append(dcg / ideal if ideal else 0.0)

        first = next((i for i, r in enumerate(rel) if r), None)
        mrrs.append(1 / (first + 1) if first is not None else 0.0)

    return {
        "n": len(items),
        "precision": round(float(statistics.mean(precisions)) * 100, 1),
        "ndcg": round(float(statistics.mean(ndcgs)) * 100, 1),
        "mrr": round(float(statistics.mean(mrrs)), 3),
        "empty_rate": round(empty / len(items) * 100, 1),
        "p50_ms": round(statistics.median(latencies)),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1]),
    }


ID_PAT = re.compile(r"\[(\d+)\]")
MONEY_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*万")


def _check_ids(answer: str, sources: list[dict]) -> tuple[int, int]:
    """返回 (引用数量, 伪造数量)。"""
    src = {str(s.get("id")) for s in sources}
    cited = set(ID_PAT.findall(answer))
    return len(cited), len(cited - src)


def _check_prices(answer: str, sources: list[dict]) -> tuple[int, int]:
    """返回 (大额金额数量, 对不上的数量)。只校验 ≥20 万的，小额多是税费。"""
    known = {round(float(s["price"]), 1) for s in sources if s.get("price") is not None}
    claims = [float(x) for x in MONEY_PAT.findall(answer) if float(x) >= 20]
    bad = [c for c in claims if not any(abs(c - k) <= 1.0 for k in known)]
    return len(claims), len(bad)


def eval_generation(n: int = 8) -> dict:
    from housing.advisor.chain import answer

    items = [i for i in _items() if i["type"] == "listing"][:n]
    rows = []
    cited_total = fake_total = claim_total = bad_total = 0
    latencies = []

    for item in items:
        t0 = time.perf_counter()
        res = answer(item["query"])
        latencies.append(time.perf_counter() - t0)

        cited, fake = _check_ids(res.answer, res.sources)
        claims, bad = _check_prices(res.answer, res.sources)
        cited_total += cited
        fake_total += fake
        claim_total += claims
        bad_total += bad
        rows.append({"id": item["id"], "cited": cited, "fake": fake,
                     "claims": claims, "bad": bad,
                     "ok": fake == 0 and bad == 0 and cited > 0,
                     "seconds": round(latencies[-1], 1)})

    # 拒答：只要没给出确定性结论、并说明了能力边界，就算诚实
    refusal_words = ("没有", "无法", "不清楚", "不确定", "不能", "查不到",
                     "帮不上", "做不了", "办不到", "没办法", "不提供", "只能",
                     "得你本人", "建议你向", "建议向")
    refusals = []
    for q in REFUSAL_CASES:
        res = answer(q)
        text = res.answer
        honest = any(w in text for w in refusal_words)
        refusals.append({"query": q, "honest": honest,
                         "kind": res.kind, "head": text[:60].replace("\n", " ")})

    return {
        "n": len(items),
        "cited": cited_total,
        "fake": fake_total,
        "id_precision": round((cited_total - fake_total) / max(cited_total, 1) * 100, 1),
        "claims": claim_total,
        "bad_claims": bad_total,
        "price_precision": round((claim_total - bad_total) / max(claim_total, 1) * 100, 1),
        "refusal_n": len(REFUSAL_CASES),
        "refusal_ok": sum(1 for r in refusals if r["honest"]),
        "p50_s": round(statistics.median(latencies), 1),
        "rows": rows,
        "refusals": refusals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gen", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("检索层")
    print("=" * 70)
    ret = eval_retrieval()
    print(f"   查询数            {ret['n']}")
    print(f"   Precision@10      {ret['precision']}%")
    print(f"   nDCG@10           {ret['ndcg']}%")
    print(f"   MRR               {ret['mrr']}")
    print(f"   无结果率          {ret['empty_rate']}%")
    print(f"   延迟 P50/P95      {ret['p50_ms']}ms / {ret['p95_ms']}ms")

    gen = None
    if not args.no_gen:
        print("\n" + "=" * 70)
        print("生成层")
        print("=" * 70)
        gen = eval_generation()
        print(f"   引用精确率        {gen['id_precision']}%  "
              f"({gen['cited']} 条引用，{gen['fake']} 条伪造)")
        print(f"   金额精确率        {gen['price_precision']}%  "
              f"({gen['claims']} 个金额，{gen['bad_claims']} 个不符)")
        print(f"   拒答正确率        {gen['refusal_ok']}/{gen['refusal_n']}")
        print(f"   端到端 P50        {gen['p50_s']}s")
        print("\n   明细：")
        for r in gen["rows"]:
            print(f"     {'✓' if r['ok'] else '✗'} {r['id']} 引用{r['cited']} 伪造{r['fake']} "
                  f"金额{r['claims']} 不符{r['bad']} {r['seconds']}s")
        print("\n   拒答明细：")
        for r in gen["refusals"]:
            print(f"     {'✓' if r['honest'] else '✗'} [{r['kind']}] {r['query'][:24]} → {r['head']}")

    out = ROOT / "eval" / "results" / "rag_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"retrieval": ret, "generation": gen},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
