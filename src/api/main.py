"""
FastAPI 后端
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 加载 api_key.env
_env_path = Path(__file__).parent.parent.parent / "api_key.env"
load_dotenv(_env_path)

from ..parser.epub_parser import parse_epub
from ..parser.chunker import chunk_chapters
from ..rag.vector_store import ingest_chunks
from ..chat.assistant import chat

BASE_DIR = Path(__file__).parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="NovelHelper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存中维护每本书的章节信息（book_id -> chapters）
_books: dict[str, list] = {}
# book_id -> ChromaDB 安全 collection 名（中文文件名无法直接用）
_collection_ids: dict[str, str] = {}


def _safe_collection_id(book_id: str) -> str:
    """将任意 book_id 转为 ChromaDB 合法的 collection 名。"""
    return "bk_" + hashlib.md5(book_id.encode()).hexdigest()[:16]


# ── 上传 & 索引 ─────────────────────────────────────────────────────────────

@app.post("/books/{book_id}/upload")
async def upload_book(book_id: str, file: UploadFile = File(...)):
    """上传 epub，解析并向量化入库。"""
    if not file.filename.endswith(".epub"):
        raise HTTPException(400, "仅支持 epub 文件")

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        chapters = parse_epub(tmp_path)
        chunks = chunk_chapters(chapters)
        cid = _safe_collection_id(book_id)
        ingest_chunks(cid, chunks)
        _books[book_id] = chapters
        _collection_ids[book_id] = cid
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "book_id": book_id,
        "chapters": len(chapters),
        "chunks": len(chunks),
    }


# ── 章节列表 & 原文 ──────────────────────────────────────────────────────────

@app.get("/books/{book_id}/chapters")
def list_chapters(book_id: str):
    """返回章节列表（index + title）。"""
    chapters = _books.get(book_id)
    if chapters is None:
        raise HTTPException(404, "书籍未找到，请先上传")
    return [{"index": c.index, "title": c.title} for c in chapters]


@app.get("/books/{book_id}/chapters/{chapter_index}/html")
def get_chapter_html(book_id: str, chapter_index: int):
    """返回指定章节的原始 HTML（供前端渲染）。"""
    chapters = _books.get(book_id)
    if chapters is None:
        raise HTTPException(404, "书籍未找到")
    if chapter_index < 0 or chapter_index >= len(chapters):
        raise HTTPException(404, "章节不存在")
    return {"html": chapters[chapter_index].html}


# ── 对话 ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    current_chapter: int
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]


@app.post("/books/{book_id}/chat")
def chat_endpoint(book_id: str, req: ChatRequest):
    """发送问题，返回 assistant 回复。"""
    if book_id not in _books:
        raise HTTPException(404, "书籍未找到，请先上传")

    cid = _collection_ids.get(book_id, _safe_collection_id(book_id))
    reply = chat(
        book_id=cid,
        query=req.query,
        current_chapter=req.current_chapter,
        history=req.history,
    )
    return {"reply": reply}


# ── 阅读进度 ─────────────────────────────────────────────────────────────────

class ProgressRequest(BaseModel):
    current_chapter: int

# 内存中存储阅读进度（book_id -> current_chapter）
_progress: dict[str, int] = {}


@app.post("/books/{book_id}/progress")
def set_progress(book_id: str, req: ProgressRequest):
    _progress[book_id] = req.current_chapter
    return {"book_id": book_id, "current_chapter": req.current_chapter}


@app.get("/books/{book_id}/progress")
def get_progress(book_id: str):
    return {"current_chapter": _progress.get(book_id, 0)}


# ── 前端静态文件 ──────────────────────────────────────────────────────────────

(FRONTEND_DIR / "static").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
