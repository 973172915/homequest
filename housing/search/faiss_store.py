"""FAISS 索引的读写封装。

为什么不用 `store.save_local()`：
faiss 的 C++ 层用 `fopen` 打开文件，在 Windows 上按 ANSI 编码解析路径，
项目路径含中文（`F:\\ai应用开发项目`）时会报 "could not open ... No such file or directory"。
这里改用 `faiss.serialize_index` 取字节，再用 Python 的 `open` 写盘（原生支持 Unicode），
读取时反序列化，路径问题彻底消失。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

INDEX_FILE = "index.faiss"
STORE_FILE = "index.pkl"


def save_index(store: FAISS, directory: Path) -> None:
    """把 FAISS 索引和 docstore 写到目录。"""
    directory.mkdir(parents=True, exist_ok=True)
    raw = faiss.serialize_index(store.index)
    (directory / INDEX_FILE).write_bytes(raw.tobytes())

    payload = {
        "docstore": store.docstore,
        "index_to_docstore_id": store.index_to_docstore_id,
    }
    with open(directory / STORE_FILE, "wb") as f:
        pickle.dump(payload, f)


def load_index(directory: Path, embeddings: Embeddings) -> FAISS:
    """从目录恢复 FAISS 检索器。"""
    index_path = directory / INDEX_FILE
    store_path = directory / STORE_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"索引不存在：{index_path}，先运行 python scripts/build_index.py")

    buf = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
    index = faiss.deserialize_index(buf)

    with open(store_path, "rb") as f:
        payload = pickle.load(f)

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=payload["docstore"],
        index_to_docstore_id=payload["index_to_docstore_id"],
    )
