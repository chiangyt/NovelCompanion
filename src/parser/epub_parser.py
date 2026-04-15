"""
EPUB 解析模块 - 双轨处理
- 展示层：保留原始 HTML，供前端渲染
- 检索层：提取纯文本，清洗后用于 RAG
"""

import re
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


@dataclass
class Chapter:
    index: int          # 章节编号（0-based）
    title: str          # 章节标题
    html: str           # 原始 HTML（展示层）
    text: str           # 纯文本（检索层）


def parse_epub(epub_path: str | Path) -> list[Chapter]:
    """
    解析 epub 文件，返回章节列表。
    章节顺序按 epub spine 决定，fallback 到文档顺序。
    """
    book = epub.read_epub(str(epub_path))
    chapters = []
    index = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content().decode("utf-8", errors="replace")
        text = _extract_text(html_content)

        # 跳过几乎没有正文的页面（封面、版权页等）
        if len(text.strip()) < 50:
            continue

        title = _extract_title(html_content) or f"第{index + 1}章"
        chapters.append(Chapter(
            index=index,
            title=title,
            html=html_content,
            text=text,
        ))
        index += 1

    return chapters


def _extract_title(html: str) -> str | None:
    """从 HTML 中提取章节标题，优先 h1/h2，fallback 到 title 标签。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in ("h1", "h2", "h3"):
        el = soup.find(tag)
        if el:
            return el.get_text(strip=True)
    title_el = soup.find("title")
    if title_el:
        t = title_el.get_text(strip=True)
        if t:
            return t
    return None


def _extract_text(html: str) -> str:
    """
    从 HTML 提取纯文本（检索层）：
    - 去除脚本、样式
    - 保留段落换行
    - 清理多余空白
    """
    soup = BeautifulSoup(html, "lxml")

    # 移除无用标签
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    # 段落间加换行
    for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4"]):
        tag.append("\n")

    text = soup.get_text(separator="")
    # 合并多余空行，保留段落结构
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
