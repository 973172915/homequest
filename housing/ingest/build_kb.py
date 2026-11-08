"""知识库文档切块。

和房源库相反：这里的文档是长文（税费、贷款、过户流程），**必须切块**，
否则一个 chunk 里混三个主题，检索精度会垮。

切块参数：chunk_size=400 / overlap=60，分隔符按中文标点降级。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from housing.config import settings

SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


def split_knowledge_documents(
    directory: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """读取目录下所有 .md，按段落切成检索块。"""
    src = directory or settings.data_kb
    if not src.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.kb_chunk_size,
        chunk_overlap=chunk_overlap or settings.kb_chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )

    docs: list[Document] = []
    for path in sorted(src.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem.split("-", 1)[-1]
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    # 标题写进正文：切块后单看正文容易丢失主题，
                    # 加上「【税费】」这类前缀能让向量更准地理解这一段在讲什么
                    page_content=f"【{title}】{chunk}",
                    metadata={
                        "doc_id": path.stem,
                        "title": title,
                        "chunk_index": i,
                        "source": path.name,
                        "type": "knowledge",
                    },
                )
            )
    return docs
