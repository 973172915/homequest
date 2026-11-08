"""中文分词器：字符二元组 + 英文/数字词。

为什么不用 jieba：
1. jieba 只发布源码包（sdist），安装时要本地构建 wheel，在受限环境下容易失败；
2. 在「找小区名、学校名」这类精确匹配场景里，字符二元组（bi-gram）的召回不输分词，
   而且对未登录词（新小区名）更稳。

代价是索引体积略大，但 1.8 万条房源完全无压力。
"""

from __future__ import annotations

import re

_CJK = r"\u4e00-\u9fff"
_TOKEN_PAT = re.compile(rf"[{_CJK}]+|[A-Za-z]+|\d+(?:\.\d+)?")
_IS_CJK = re.compile(rf"^[{_CJK}]+$")


def tokenize(text: str) -> list[str]:
    """中文按字 + 二元组切分，英文数字按词切分并小写。"""
    tokens: list[str] = []
    for m in _TOKEN_PAT.finditer(text or ""):
        piece = m.group()
        if _IS_CJK.match(piece):
            tokens.extend(piece)
            tokens.extend(piece[i:i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece.lower())
    return tokens
