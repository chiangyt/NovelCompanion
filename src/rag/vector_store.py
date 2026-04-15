"""
ChromaDB 向量存储模块
- 每本书对应一个 collection（用书名命名）
- metadata 字段：chapter_index, chapter_title, chunk_index, chunk_id
- 查询时按 chapter_index <= current_chapter 过滤，实现剧透控制
"""

from pathlib import Path

import chromadb

from ..parser.chunker import Chunk
from .embedder import embed_texts, embed_query

# 本地持久化路径
DB_PATH = Path(__file__).parent.parent.parent / "data" / "chromadb"


def _get_client() -> chromadb.PersistentClient:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(book_id: str):
    """获取（或创建）某本书的 collection。"""
    client = _get_client()
    return client.get_or_create_collection(
        name=book_id,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(book_id: str, chunks: list[Chunk], batch_size: int = 64) -> None:
    """
    将 chunks 向量化并写入 ChromaDB。
    已存在的 book_id collection 会先清空再写入（重新索引场景）。
    """
    client = _get_client()

    # 清空旧数据（重新上传同一本书时）
    try:
        client.delete_collection(book_id)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=book_id,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    print(f"开始向量化 {total} 个 chunk...")

    for start in range(0, total, batch_size):
        batch = chunks[start: start + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts)

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "chapter_index": c.chapter_index,
                "chapter_title": c.chapter_title,
                "chunk_index": c.chunk_index,
            } for c in batch],
        )
        print(f"  已写入 {min(start + batch_size, total)}/{total}")

    print(f"入库完成，共 {total} 条。")


def search(
    book_id: str,
    query: str,
    current_chapter: int,
    top_k: int = 5,
) -> list[dict]:
    """
    检索与 query 最相关的 chunk，只返回 chapter_index <= current_chapter 的结果。

    返回列表，每项包含：text, chapter_index, chapter_title, chunk_index, distance
    """
    collection = get_collection(book_id)
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"chapter_index": {"$lte": current_chapter}},
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "chapter_index": meta["chapter_index"],
            "chapter_title": meta["chapter_title"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })

    return hits
