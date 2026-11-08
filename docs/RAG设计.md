# 检索增强（RAG）管线设计

> 技术栈：LangChain（LCEL）+ bge embedding + FAISS + BM25 混合检索 + DeepSeek `deepseek-flash`
> 目标：让「一句人话找房」和「置业顾问」都跑在同一个检索层上
>
> ⚠️ **实现更新（v1.1）**：向量库由 Chroma 改为 **FAISS**——Chroma 1.5 的 Rust 后端在本机对任何路径都报
> `unable to open database file`（同样目录下纯 sqlite3 却能正常打开），属于其自身缺陷；
> 中文分词由 jieba 改为**自研字符二元组**——jieba 只发布源码包，安装需本地编译 wheel。
> 后文若出现 `Chroma`/`jieba` 字样，属于设计推演，以 `housing/search/` 下的实际代码为准。

---

## 一、管线总览

```
                    ┌──────────────────────┐
                    │ data/sh_data.csv     │
                    │ 18339 条房源          │
                    └──────────┬───────────┘
                               │ ① 清洗 + 结构化
                    ┌──────────▼───────────┐
                    │ clean.parquet        │
                    └──────────┬───────────┘
                               │ ② 序列化为自然语言文档
                    ┌──────────▼───────────┐
                    │ Document(page_content,│
                    │  metadata={...})      │  一房一文档，不切
                    └──────────┬───────────┘
                               │ ③ 向量化（bge-small-zh-v1.5）
                    ┌──────────▼───────────┐        ┌──────────────────┐
                    │ Chroma（向量库）      │        │ 知识库：政策/税费 │
                    │ collection=fangyuan   │        │ 切块 400/60       │
                    └──────────┬───────────┘        └────────┬─────────┘
                               │                             │
                    ┌──────────▼─────────────────────────────▼──────┐
                    │ ④ 混合检索 EnsembleRetriever                   │
                    │    BM25(jieba) 0.3  +  向量 0.7  + 元数据过滤   │
                    │    → MMR 去重 → Top-K                          │
                    └──────────────────────┬────────────────────────┘
                                           │
                    ┌──────────────────────▼────────────────────────┐
                    │ ⑤ 生成：ChatPromptTemplate + deepseek-flash    │
                    │    强制引用房源 id，禁止编造                      │
                    └──────────────────────┬────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │ 流式输出 + 房源卡片 + 引用 │
                              └─────────────────────────┘
```

---

## 二、语料设计：**两个库，两套切法**

这是整个管线里最容易做错的地方。很多人上来就 `RecursiveCharacterTextSplitter(chunk_size=1000)`，对表格数据是灾难。

### 2.1 库 A：房源库（18339 条，**不切块**）

每套房序列化成一段约 100 字的自然语言，一条记录 = 一个 Document：

```
【房源 10023】潍坊市奎文区 · 新华路北 · 东方天韵
3室2厅2卫 / 135.5㎡ / 中楼层(共18层) / 南北通透 / 精装 / 有电梯 / 两梯三户
总价 127 万，单价 9374 元/㎡
卖点：人民医院近，楼层好，H 户型，配套齐全
```

```python
Document(
    page_content=text,
    metadata={
        "id": "10023", "district": "奎文区", "nearby": "人民医院",
        "community": "东方天韵", "price": 127.0, "area": 135.49,
        "rooms": 3, "decoration": "精装", "elevator": True, "floor_level": "中",
    },
)
```

**为什么不切**：
- 一条房源总共 100 字，切了反而把「小区名」和「价格」拆散；
- 用户问「东方天韵 130 平的房子」，切开后向量里只剩半套房，召回直接崩；
- 元数据要跟着正文走，切块会让 metadata 复制多份，过滤逻辑变复杂。

**这条决定了整个检索质量**——什么时候不该切块，比会不会调 splitter 更重要。

### 2.2 库 B：知识库（政策/税费/贷款，**必须切块**）

这块才是 `RecursiveCharacterTextSplitter` 的用武之地：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=60,
    separators=["\n\n", "\n", "。", "；", "，", " "],
)
```

建议放 4~6 篇（每篇 800~1500 字），覆盖：
1. 潍坊购房资格与限购政策
2. 二手房交易税费（契税/增值税/个税）怎么算
3. 贷款：商贷 vs 公积金，首付比例
4. 过户流程与时间
5. 「满二 / 满五唯一 / 出税」到底省多少钱
6. 怎么看户型、楼层、朝向

> ⚠️ 这些文档你自己写或让 LLM 起草后人工校对，**README 里注明「示例语料，非官方政策」**，别把幻觉政策当成真事实。

---

## 三、向量化

| 项 | 选择 | 理由 |
|---|---|---|
| 模型 | `BAAI/bge-small-zh-v1.5` | 100MB，CPU 秒级，中文短文本效果好 |
| 升级路径 | `BAAI/bge-m3` 或 `Qwen3-Embedding-0.6B` | 效果更好，但 1~2GB |
| 归一化 | `normalize_embeddings=True` | 用余弦距离，必须归一化 |
| 下载 | `HF_ENDPOINT=https://hf-mirror.com` | 本机 hf.co 超时，镜像可用 |

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

> bge 系列中文检索有个小技巧：**query 前面加指令前缀** `为这个句子生成表示以用于检索相关文章：`，能提 2~3 个点召回。写进代码注释里，是个细节加分项。

---

## 四、检索：混合 + 过滤 + MMR

单一向量检索在「数字条件」上很弱——用户说「80 万以内」，向量相似度根本理解不了价格大小。所以必须混合。

```python
# 1) 向量检索（MMR 保证多样性）
vec_retriever = get_vector_retriever(k=fetch_k)   # FAISS + bge，见 housing/search/retriever.py

# 2) BM25（中文用字符二元组，不依赖 jieba）
from housing.search.tokenizer import tokenize
bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
bm25.k = 10

# 3) 融合
from langchain.retrievers import EnsembleRetriever
retriever = EnsembleRetriever(retrievers=[bm25, vec_retriever], weights=[0.3, 0.7])
```

**再加一层重排**（这个是自己写的，不靠框架）：

```python
def rerank(docs, query_price_range, query_area):
    """在检索结果上叠加业务规则：价格偏离、面积匹配、小区热度。"""
    scored = []
    for d in docs:
        s = 0.0
        if query_price_range and query_price_range[0] <= d.metadata["price"] <= query_price_range[1]:
            s += 0.4
        if query_area and abs(d.metadata["area"] - query_area) <= 10:
            s += 0.3
        scored.append((s, d))
    return [d for _, d in sorted(scored, key=lambda x: -x[0])]
```

> **分工原则**：向量检索解决「语义」，BM25 解决「精确词」（小区名、学校名），业务重排解决「数字条件」。三层各司其职。

---

## 五、生成

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-flash",                       # 官方 id，不是 deepseek-v4.1-flash
    base_url="https://api.deepseek.com",
    api_key=settings.deepseek_api_key,
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是潍坊二手房置业顾问。只能依据【房源】中的信息回答。
规则：
1. 禁止编造房源、价格、面积、小区名；
2. 每推荐一套房，必须标注房源编号，如「[10023]」；
3. 涉及价格结论时，说明是挂牌价；
4. 输出结构：推荐列表 → 对比表 → 每套优缺点 → 看房建议。"""),
    ("human", "【房源】\n{context}\n\n【问题】\n{question}"),
])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**流式**：`async for chunk in chain.astream(q)`，前端做打字机效果。

**引用回传**：不要只返回文本，返回 `{answer, sources: [Document.metadata]}`，前端才能渲染房源卡片——这是「能点开看原房源」的关键，也是防幻觉的兜底。

---

## 六、工具化（为 C 方向的 agent loop 准备）

LangChain 的 `@tool` 装饰器 + `bind_tools`，让模型自己决定调哪个：

```python
from langchain_core.tools import tool

@tool
def search_listings(query: str, max_price: float = None,
                    district: str = None, min_area: float = None) -> list[dict]:
    """按自然语言 + 结构化条件检索房源，返回最多 10 条。"""
    ...  # 内部复用上面的 EnsembleRetriever

@tool
def estimate_price(community: str, area: float, rooms: int,
                   floor_level: str, decoration: str) -> dict:
    """调用估价模型，返回单价、总价区间和主要影响因素。"""
    ...

@tool
def compare_listings(ids: list[str]) -> list[dict]:
    """对比多套房源的面积、单价、楼层、装修、总价。"""
    ...

llm_with_tools = llm.bind_tools([search_listings, estimate_price, compare_listings])
```

用户问「这套 120 平的房子贵不贵，周边有没有更划算的」，模型会自动串起 `estimate_price` → `search_listings` → `compare_listings`。

---

## 七、评测（决定 README 首屏数字）

| 层 | 指标 | 怎么测 |
|---|---|---|
| 检索 | Recall@10、MRR | 30 条人工标注 query，看正确答案是否在 Top10 |
| 生成 | 幻觉率 | 人工检查回答里的房源 id / 价格是否都在上下文中 |
| 生成 | 相关性 / 可读性 | 1~5 分人工打分 |
| 系统 | P95 延迟、单次成本 | 埋点记录 |

目标：`Recall@10 ≥ 85%`、`幻觉率 < 5%`、`P95 < 3s`。

---

## 八、目录与代码骨架

```
src/
├── ingest/
│   ├── build_docs.py        # 房源 → Document（序列化模板）
│   ├── build_kb.py          # 知识库切块
│   └── index.py             # 写 Chroma
├── search/
│   ├── embeddings.py        # bge 封装（含 query 前缀）
│   ├── retriever.py         # 混合检索 + 重排
│   └── rerank.py
├── advisor/
│   ├── prompts.py
│   ├── chain.py             # LCEL 链
│   └── tools.py             # @tool 三个工具
└── eval/
    ├── qa_pairs.jsonl       # 30 条评测问题
    └── run_eval.py
```

依赖（`uv add`）：
```
langchain langchain-core langchain-community faiss-cpu
langchain-huggingface langchain-text-splitters langchain-openai
sentence-transformers pandas pydantic-settings fastapi
```

---

## 九、工期

| 步骤 | 天 |
|---|---|
| 房源序列化 + 建库 + BM25 | 2 |
| 混合检索 + 重排 + 调参 | 2 |
| 生成链 + 引用 + 流式 | 2 |
| 知识库切块 + 语料撰写 | 1 |
| 评测 + 调优 | 1.5 |
| **合计** | **约 8.5 天** |

---

## 十、要注意的三个坑

1. **`deepseek-flash` 默认开思考模式**，做 RAG 生成时会先输出一长段推理，流式体验变怪。简单问答建议 `reasoning_effort="low"` 或关闭思考模式。
2. **LangChain 版本漂移快**，`EnsembleRetriever` 的 import 路径在不同版本有差异，必须用 `uv.lock` 锁版本，否则 clone 下来跑不通。
3. **别把 LangChain 当黑盒**：`VectorstoreIndexCreator` 这类一行式封装会隐藏关键环节。用 LCEL 显式拼装，每个环节都可控、可测。
