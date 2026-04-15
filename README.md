# NovelHelper

中文小说陪伴式阅读工具。上传 EPUB，随时向 AI 提问书中内容，设置阅读进度后自动避免剧透。

## 功能

- 上传 EPUB，自动解析章节、向量化入库
- 左侧渲染原文（保留排版），右侧对话框提问
- 剧透控制：只根据已读章节回答，不透露后续情节
- 基于 bge-m3 向量检索（本地 GPU），Qwen API 生成回答
- Agent 支持网络搜索（Tavily），可回答作者背景、历史背景等书外知识

## 环境要求

- Python 3.10+
- CUDA 环境（用于本地运行 bge-m3，无 GPU 可回退到 CPU 但速度较慢）

## 安装

```bash
pip install -r requirements.txt
```

首次运行会自动下载 bge-m3 模型（约 2GB，存入 HuggingFace 缓存）。

## 配置

在项目根目录创建 `api_key.env`，填入以下 Key：

```env
QWEN_API_KEY=your_qwen_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 启动

```bash
python run.py
```

## 使用

1. 点击顶栏「上传 EPUB」选择文件，等待处理完成（GPU 约 20 秒）
2. 左侧阅读原文，可通过下拉或左右箭头翻章节
3. 底部滑块设置「读到第几章」（翻页时自动跟进）
4. 右侧输入框提问，Enter 发送，Shift+Enter 换行


## 项目结构

```
novelhelper/
├── run.py                  # 启动入口
├── eval.py                 # 评估脚本（实验性，不作为项目功能）
├── requirements.txt
├── frontend/
│   └── index.html          # 前端单页应用
├── src/
│   ├── parser/
│   │   ├── epub_parser.py  # EPUB 解析（双轨：HTML 展示 + 纯文本检索）
│   │   └── chunker.py      # 段落级切块，带章节 metadata
│   ├── rag/
│   │   ├── embedder.py     # bge-m3 向量化（GPU）
│   │   └── vector_store.py # ChromaDB 入库与检索
│   ├── chat/
│   │   └── assistant.py    # LangChain Agent（书内检索 + 网络搜索）
│   └── api/
│       └── main.py         # FastAPI 后端
└── data/
    └── chromadb/           # 向量数据库持久化（自动生成）
```
