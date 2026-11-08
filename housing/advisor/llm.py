"""LLM 工厂：没有 Key 时返回 None，由上层降级为模板摘要。

这样面试官 clone 项目后不配密钥也能跑通演示。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from housing.config import settings


def llm_available() -> bool:
    return bool(settings.deepseek_api_key)


def get_llm(streaming: bool = False, effort: str | None = None) -> ChatOpenAI | None:
    """构造 DeepSeek 客户端（OpenAI 兼容协议）。

    reasoning_effort 显式传给 ChatOpenAI（不再塞进 model_kwargs，否则 LangChain 会告警）：
    - "none"：关闭思考模式，RAG 生成场景首选，首字延迟低；
    - "low"/"high"/"max"：留给「对比 3 套房」这类需要推理的复杂任务。
    """
    if not llm_available():
        return None

    return ChatOpenAI(
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        temperature=settings.deepseek_temperature,
        streaming=streaming,
        timeout=60,
        max_retries=2,
        reasoning_effort=effort or settings.deepseek_reasoning_effort,
    )
