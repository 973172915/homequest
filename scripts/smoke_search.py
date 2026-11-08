"""检索冒烟测试：一句话 → Top-K 房源。

用法：
    python scripts/smoke_search.py "奎文区 100平以内 南北通透 精装两居 预算80万"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from housing.ingest.build_docs import load_listing_documents  # noqa: E402
from housing.search.retriever import hybrid_search, parse_query  # noqa: E402

DEFAULT_QUERY = "奎文区 100平以内 南北通透 精装两居 预算80万"


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    filters = parse_query(query)
    print(f"查询：{query}")
    print(f"解析条件：{filters}\n")

    docs = load_listing_documents()
    results = hybrid_search(query, filters, docs_pool=docs, k=10)

    if not results:
        print("没有检索到结果")
        return

    print(f"命中 {len(results)} 条：\n")
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"{i:2d}. [{m['id']}] {m['district']}·{m['nearby']}·{m['community']} "
              f"{m['area']:.1f}㎡ {m['rooms']}室 {m['decoration']} "
              f"{m['price']:.1f}万 单价{m['unit_price']:.0f}")


if __name__ == "__main__":
    main()
