"""
对话系统 - 阅读向导 assistant
- Agent 按需调用 search_book / internet_search 工具
- 对话历史只保留最近 N 轮传给 API（防止 context 超长）
- UI 层可自行保存完整历史
"""

import os
from typing import Literal

from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from ..rag.vector_store import search

MODEL = "qwen-plus"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_HISTORY_TURNS = 6   # 保留最近 N 轮（1 轮 = 1 user + 1 assistant）
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

SYSTEM_PROMPT = """你是一位专业的小说阅读向导，帮助读者更好地理解正在阅读的小说。

你的职责：
- 基于书中内容回答读者的问题（角色、情节、背景等）
- 只使用读者已读章节的内容，绝对不透露未读部分的剧情
- 回答要简洁、准确，引用原文时注明出自哪一章，不要擅自推测人物性格或故事走向
- 如果已读章节中找不到答案，诚实告知，不要猜测或编造

每次回答前，你会收到从书中检索到的相关段落作为参考。请优先基于这些段落作答。"""


def make_search_book_tool(book_id: str, current_chapter: int):
    @tool
    def search_book(query: str) -> str:
        """在书中检索相关内容，适合查找书中人物、情节、对话、背景设定等"""
        hits = search(book_id, query, current_chapter, top_k=3)
        if not hits:
            return "书中未找到相关内容"
        return "\n\n".join([
            f"[第{h['chapter_index']}章 {h['chapter_title']}]\n{h['text']}"
            for h in hits
        ])
    return search_book


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict:
    """搜索互联网，适合查询作者背景、历史背景、现实世界知识等书中不包含的信息"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def chat(
    book_id: str,
    query: str,
    current_chapter: int,
    history: list[dict],   # [{"role": "user"|"assistant", "content": str}, ...]
) -> str:
    """
    发送一条消息，返回 assistant 回复文本。

    history 是完整对话历史，函数内部截取最近 MAX_HISTORY_TURNS 轮传给 API。
    """
    search_book_tool = make_search_book_tool(book_id, current_chapter)
    tools = [search_book_tool, internet_search]

    model = ChatOpenAI(
        api_key=os.environ.get("QWEN_API_KEY"),
        base_url=QWEN_BASE_URL,
        model=MODEL,
    )

    agent = create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)

    recent_history = history[-(MAX_HISTORY_TURNS * 2):]
    messages = []
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=query))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content
