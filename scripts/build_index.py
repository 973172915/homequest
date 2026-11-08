"""构建向量索引：房源库 + 知识库。

用法：
    python scripts/build_index.py            # 已存在则跳过
    python scripts/build_index.py --rebuild  # 全量重建

底层用 FAISS 而不是 Chroma：Chroma 1.5 的 Rust 后端在本机会报
`unable to open database file`（纯 sqlite3 打开同一目录却正常），
FAISS 没有这个问题，且 1.8 万条数据用 IndexFlatIP 暴力检索就足够快。
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_community.vectorstores import FAISS  # noqa: E402

from housing.config import settings  # noqa: E402
from housing.ingest.build_docs import build_listing_documents  # noqa: E402
from housing.ingest.build_kb import split_knowledge_documents  # noqa: E402
from housing.ingest.clean import clean, save  # noqa: E402
from housing.search.embeddings import get_embeddings  # noqa: E402
from housing.search.faiss_store import save_index  # noqa: E402

BATCH = 1024


def build(name: str, docs, embeddings, rebuild: bool) -> int:
    """分批 embedding 后写入 FAISS，边跑边打印进度。"""
    path = settings.index_dir / name
    if path.exists() and not rebuild:
        print(f"  [{name}] 已存在，跳过（加 --rebuild 重建）")
        return 0
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    store: FAISS | None = None
    total = len(docs)
    for start in range(0, total, BATCH):
        batch = docs[start:start + BATCH]
        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)
        print(f"  [{name}] {min(start + BATCH, total)}/{total}", flush=True)

    if store is not None:
        save_index(store, path)
    return total


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    settings.ensure_dirs()

    print("1/4 清洗数据 …")
    df = clean()
    parquet = save(df)
    print(f"   {len(df)} 条 → {parquet}（剔除 {df.attrs.get('dropped', 0)} 条异常）")

    print("2/4 构造检索文档 …")
    listing_docs = build_listing_documents(df)
    kb_docs = split_knowledge_documents()
    print(f"   房源文档 {len(listing_docs)} 条（一房一文档，不切块）")
    print(f"   知识文档 {len(kb_docs)} 块（切块 {settings.kb_chunk_size}/{settings.kb_chunk_overlap}）")

    print("3/4 加载向量模型 …")
    embeddings = get_embeddings()
    print(f"   {settings.embedding_model}")

    print("4/4 建索引 …")
    t0 = time.time()
    build("faiss_listings", listing_docs, embeddings, rebuild)
    if kb_docs:
        build("faiss_knowledge", kb_docs, embeddings, rebuild)
    print(f"完成，用时 {time.time() - t0:.1f}s，索引目录 {settings.index_dir}")


if __name__ == "__main__":
    main()
