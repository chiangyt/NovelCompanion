# NovelCompanion — 项目架构文档

## 项目简介

NovelHelper 是一个 **EPUB 小说阅读 + AI 问答** 应用。用户上传 EPUB 文件后，可以在网页中阅读，并通过 AI 对话向导提问书中内容。系统具备**剧透控制**机制：AI 只会根据用户已读章节作答，不泄露后续情节。

---

## 整体架构

```
用户浏览器
    │  上传 EPUB / 翻页 / 提问
    ▼
frontend/index.html          ← 单页前端（纯 HTML + JS）
    │  HTTP REST API
    ▼
src/api/main.py              ← FastAPI 后端入口
    ├── src/parser/          ← EPUB 解析 & 文本切块
    ├── src/rag/             ← 向量化 & ChromaDB 检索
    └── src/chat/            ← LLM Agent 对话
              │
         data/chromadb/      ← 本地向量数据库（持久化）
```

### 数据流（上传阶段）

```
EPUB 文件
  → parse_epub()        解析成 Chapter 列表（含 HTML + 纯文本）
  → chunk_chapters()    切成 Chunk 列表（600字/块，75字overlap）
  → ingest_chunks()     embed_texts() 向量化 → 写入 ChromaDB
```

### 数据流（对话阶段）

```
用户问题 + current_chapter
  → chat()              LangChain Agent 调度
      ├── search_book()     向量检索 ChromaDB（过滤 chapter <= current）
      └── internet_search() Tavily 搜索互联网
  → 拼装消息历史 → Qwen-plus API → 返回回复文本
```

---

## 目录结构

```
NovelCompanion/
├── run.py                   启动入口
├── api_key.env              API 密钥（不入库）
├── architecture.md          本文件
├── plan.md                  方案设计文档
├── frontend/
│   └── index.html           单页前端
├── src/
│   ├── api/
│   │   └── main.py          FastAPI 路由 & 中间件
│   ├── parser/
│   │   ├── epub_parser.py   EPUB 解析
│   │   └── chunker.py       文本切块
│   ├── rag/
│   │   ├── embedder.py      向量化（bge-m3）
│   │   └── vector_store.py  ChromaDB 读写
│   └── chat/
│       └── assistant.py     LLM Agent 对话
└── data/
    └── chromadb/            向量数据库本地存储
```

---

## 模块详解

### `run.py` — 启动入口

| 作用 | 用 uvicorn 启动 FastAPI，监听 127.0.0.1:8000 |
|------|------|

---

### `frontend/index.html` — 单页前端

纯 HTML + 原生 JS，无框架依赖。

| 区域 | 说明 |
|------|------|
| 顶栏 | 上传 EPUB 按钮、章节下拉跳转 |
| 左侧阅读区 | 用 `<iframe srcdoc>` 渲染章节 HTML，隔离 epub 样式 |
| 进度条 | 控制剧透边界（`currentChapter`），拖动同步到后端 |
| 右侧对话区 | 发送问题、展示 AI 回复（含极简 Markdown 渲染） |

**前端状态对象 `state`：**

| 字段 | 说明 |
|------|------|
| `bookId` | 书籍 ID（文件名去扩展名） |
| `chapters` | 章节列表 `[{index, title}]` |
| `currentChapter` | 剧透控制基准（已读到哪章） |
| `displayChapter` | 当前显示章节（翻页用，可超前浏览） |
| `history` | 对话历史 `[{role, content}]` |

**主要函数：**

| 函数 | 作用 |
|------|------|
| `loadChapters()` | 上传成功后拉取章节列表，初始化进度条 |
| `showChapter(index)` | 加载并显示指定章节 HTML；翻页超过进度时自动推进 `currentChapter` |
| `setCurrentChapter(index)` | 更新剧透基准，同步 POST 到后端 `/progress` |
| `sendMessage()` | 发送问题，收到回复后追加到对话历史 |
| `simpleMarkdown(text)` | 极简 Markdown → HTML 转换（加粗、列表、表格、blockquote） |
| `escapeHtml(t)` | XSS 防护，转义 HTML 特殊字符 |

---

### `src/api/main.py` — FastAPI 后端

**全局状态（内存）：**

| 变量 | 类型 | 说明 |
|------|------|------|
| `_books` | `dict[str, list[Chapter]]` | book_id → 章节列表 |
| `_collection_ids` | `dict[str, str]` | book_id → ChromaDB 安全 collection 名 |
| `_progress` | `dict[str, int]` | book_id → 当前阅读章节编号 |

**辅助函数：**

| 函数 | 作用 |
|------|------|
| `_safe_collection_id(book_id)` | 将任意书名（含中文）转为 ChromaDB 合法名称：`"bk_" + MD5[:16]` |

**API 路由：**

| 方法 | 路径 | 作用 |
|------|------|------|
| `POST` | `/books/{book_id}/upload` | 接收 EPUB，解析→切块→向量化入库，返回章节数/块数 |
| `GET` | `/books/{book_id}/chapters` | 返回章节列表 `[{index, title}]` |
| `GET` | `/books/{book_id}/chapters/{chapter_index}/html` | 返回指定章节原始 HTML |
| `POST` | `/books/{book_id}/chat` | 接收问题，调用 LLM Agent，返回回复 |
| `POST` | `/books/{book_id}/progress` | 保存阅读进度（当前章节编号） |
| `GET` | `/books/{book_id}/progress` | 读取阅读进度 |
| `GET` | `/` | 返回前端 `index.html` |

---

### `src/parser/epub_parser.py` — EPUB 解析

**数据结构：**

```python
@dataclass
class Chapter:
    index: int      # 章节编号（0-based）
    title: str      # 章节标题
    html: str       # 原始 HTML（展示层）
    text: str       # 纯文本（检索层）
```

**函数：**

| 函数 | 作用 |
|------|------|
| `parse_epub(epub_path)` | 读取 EPUB，遍历所有文档，跳过正文 < 50 字的页面（封面/版权页），返回 `Chapter` 列表 |
| `_extract_title(html)` | 从 HTML 中提取标题：优先 h1/h2/h3，fallback 到 `<title>` 标签 |
| `_extract_text(html)` | 去除 script/style，保留段落换行，合并多余空行，返回纯文本 |

---

### `src/parser/chunker.py` — 文本切块

**切块参数：**

| 参数 | 默认值 | 说明 |
|------|------|------|
| `CHUNK_SIZE` | 600 字符 | 目标块大小（≈ 400 token） |
| `OVERLAP` | 75 字符 | 相邻块重叠量（保持上下文连续性） |

**数据结构：**

```python
@dataclass
class Chunk:
    chunk_id: str        # "{chapter_index}-{chunk_index}"
    chapter_index: int
    chapter_title: str
    chunk_index: int     # 在本章内的序号
    text: str
```

**函数：**

| 函数 | 作用 |
|------|------|
| `chunk_chapters(chapters)` | 对所有章节切块，返回完整 Chunk 列表 |
| `_chunk_chapter(chapter, chunk_size, overlap)` | 对单章切块：段落聚合 → 超长段落再切 → 加 overlap |
| `_split_long_paragraph(text, chunk_size)` | 按中文句末标点（。！？…）切分超长段落 |
| `_merge_segments(segments, chunk_size)` | 将短 segment 合并到接近 chunk_size，避免碎片化 |

---

### `src/rag/embedder.py` — 向量化

使用 **`BAAI/bge-m3`** 模型（本地运行，首次自动下载 ~2GB）。

| 函数 | 作用 |
|------|------|
| `get_model()` | 加载 bge-m3（`@lru_cache` 单例，避免重复加载），自动选择 CUDA / CPU |
| `embed_texts(texts, batch_size=64)` | 批量向量化，normalize 后返回 float 列表，>50 条时显示进度条 |
| `embed_query(query)` | 单条查询向量化（normalize） |

---

### `src/rag/vector_store.py` — ChromaDB 向量存储

每本书对应一个 ChromaDB **collection**，使用余弦相似度（`hnsw:space: cosine`）。

数据库持久化路径：`data/chromadb/`

| 函数 | 作用 |
|------|------|
| `_get_client()` | 创建 ChromaDB PersistentClient |
| `get_collection(book_id)` | 获取或创建指定书的 collection |
| `ingest_chunks(book_id, chunks)` | 先删除旧 collection，再批量写入：embed → add（含 metadata） |
| `search(book_id, query, current_chapter, top_k=5)` | 向量检索，**按 `chapter_index <= current_chapter` 过滤**（剧透控制核心），返回含 text/chapter_index/chapter_title/distance 的列表 |

---

### `src/chat/assistant.py` — LLM Agent 对话

使用 **LangChain Agent** + **Qwen-plus**（通义千问，兼容 OpenAI 接口）。

**配置：**

| 参数 | 值 |
|------|------|
| `MODEL` | `qwen-plus` |
| `QWEN_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MAX_HISTORY_TURNS` | 6（防止 context 超长，每轮 = 1 user + 1 assistant） |

**工具（Tools）：**

| 工具 | 触发场景 | 实现 |
|------|------|------|
| `search_book` | 查询书中人物、情节、对话、背景 | 调用 `vector_store.search()`，带 `current_chapter` 过滤 |
| `internet_search` | 查询作者背景、历史背景、现实知识 | 调用 Tavily Search API |

**函数：**

| 函数 | 作用 |
|------|------|
| `make_search_book_tool(book_id, current_chapter)` | 工厂函数，动态生成绑定了 book_id 和 current_chapter 的 `search_book` tool |
| `chat(book_id, query, current_chapter, history)` | 主对话函数：截取最近 N 轮历史 → 构建消息列表 → 调用 Agent → 返回回复文本 |

---

## 关键设计决策

### 剧透控制
`current_chapter` 贯穿整个系统：前端拖进度条 → POST 到后端 → 每次对话请求携带 → `vector_store.search()` 的 `where` 过滤 → `search_book` tool 只返回已读范围内容。

### ChromaDB collection 命名
书名可能含中文、特殊字符，直接用作 collection 名会报错。通过 `_safe_collection_id()` 转为 `bk_<MD5[:16]>` 格式。

### Embedding 单例
`get_model()` 用 `@lru_cache` 保证进程内只加载一次模型，避免每次请求重新加载 2GB 模型。

### 历史截断
对话历史只取最近 `MAX_HISTORY_TURNS * 2 = 12` 条消息传给 API，防止长篇小说讨论中 context 超出模型限制。

### iframe 隔离
章节 HTML 用 `<iframe srcdoc>` 渲染，避免 EPUB 自带 CSS 污染主页面样式。
