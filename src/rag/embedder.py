"""
Embedding 模块 - 使用 bge-m3
首次运行会自动下载模型（~2GB），之后从本地缓存加载。
"""

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """加载 bge-m3 模型（单例，避免重复加载）。"""
    print(f"加载 embedding 模型: {MODEL_NAME}  device={DEVICE}")
    return SentenceTransformer(MODEL_NAME, device=DEVICE)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    批量生成 embedding。
    返回 list of float vectors，与 texts 顺序一一对应。
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,   # 余弦相似度场景下建议 normalize
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """生成单条查询的 embedding（bge-m3 查询侧无需特殊前缀）。"""
    model = get_model()
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()
