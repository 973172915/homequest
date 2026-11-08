FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/artifacts/hf \
    HF_HUB_DISABLE_SYMLINKS=1

# 国内可换成清华源加速构建：docker compose build --build-arg PIP_INDEX_URL=...
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# torch 必须先单独从 CPU 源装：Linux 下 PyPI 的 torch 会带上 2~3 GB 的 CUDA 依赖
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --index-url ${PIP_INDEX_URL} -r requirements.txt

COPY . .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# run.py 会在索引缺失时先建索引，再拉起 API 和界面
CMD ["python", "run.py", "--no-browser"]
