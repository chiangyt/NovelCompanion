"""
段落级切块模块
- 输入：Chapter 列表（来自 epub_parser）
- 输出：Chunk 列表，每个 chunk 带完整 metadata
- 策略：按自然段落切分，超长段落再细分，相邻 chunk 保留 overlap
"""

from dataclasses import dataclass

from .epub_parser import Chapter

# 中文场景下粗略估算：1 token ≈ 1.5 汉字
# 目标 400 token → ~600 字；overlap 50 token → ~75 字
CHUNK_SIZE = 600    # 字符数
OVERLAP = 75        # 字符数


@dataclass
class Chunk:
    chunk_id: str       # 唯一 ID："{chapter_index}-{chunk_index}"
    chapter_index: int
    chapter_title: str
    chunk_index: int    # 在本章内的序号
    text: str           # chunk 纯文本


def chunk_chapters(chapters: list[Chapter],
                   chunk_size: int = CHUNK_SIZE,
                   overlap: int = OVERLAP) -> list[Chunk]:
    """将所有章节切成 Chunk 列表。"""
    all_chunks: list[Chunk] = []
    for chapter in chapters:
        chunks = _chunk_chapter(chapter, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_chapter(chapter: Chapter,
                   chunk_size: int,
                   overlap: int) -> list[Chunk]:
    """对单个章节切块。"""
    # 按自然段落（空行）分割
    paragraphs = [p.strip() for p in chapter.text.split("\n\n") if p.strip()]

    # 将段落聚合成目标大小的 chunk，超长段落强制截断
    raw_segments: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            raw_segments.append(para)
        else:
            # 超长段落按句子切分（。！？为句末）
            raw_segments.extend(_split_long_paragraph(para, chunk_size))

    # 聚合相邻 segment，直到接近 chunk_size
    merged = _merge_segments(raw_segments, chunk_size)

    # 加 overlap：每个 chunk 前面拼上上一个 chunk 的尾部
    chunks: list[Chunk] = []
    for i, text in enumerate(merged):
        if i > 0 and overlap > 0:
            tail = merged[i - 1][-overlap:]
            text = tail + text

        chunks.append(Chunk(
            chunk_id=f"{chapter.index}-{i}",
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            chunk_index=i,
            text=text,
        ))

    return chunks


def _split_long_paragraph(text: str, chunk_size: int) -> list[str]:
    """按句末标点切分超长段落。"""
    sentences = []
    buf = ""
    for char in text:
        buf += char
        if char in ("。", "！", "？", "…", "\u201c", "\u201d") and len(buf) >= chunk_size // 2:
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences or [text]


def _merge_segments(segments: list[str], chunk_size: int) -> list[str]:
    """将短 segment 合并到接近 chunk_size。"""
    merged = []
    buf = ""
    for seg in segments:
        if not buf:
            buf = seg
        elif len(buf) + len(seg) + 1 <= chunk_size:
            buf = buf + "\n" + seg
        else:
            merged.append(buf)
            buf = seg
    if buf:
        merged.append(buf)
    return merged
