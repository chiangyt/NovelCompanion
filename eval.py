"""
RAGAS 评估脚本，直接运行：python eval.py

流程：
    1. 解析 epub 并建立向量索引
    2. 对每个问题单独调用 search() 收集 contexts
    3. 调用 chat() 得到 answer
    4. 用 RAGAS 计算 faithfulness / answer_relevancy / context_precision
"""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "api_key.env")

from src.parser.epub_parser import parse_epub
from src.parser.chunker import chunk_chapters
from src.rag.vector_store import ingest_chunks, search
from src.chat.assistant import chat

from openai import OpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL = "qwen-plus"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

EPUB_PATH = Path("玛普尔小姐1　谋杀启事.epub")
CURRENT_CHAPTER = 999      # 限制检索章节范围，999 = 不限制
SKIP_INDEX = True         # 已入库时改为 True 跳过建索引
OUTPUT = "ragas_results.csv"

# ── 测试问题 ──────────────────────────────────────────────────────────────────

QUESTIONS = [
    # 正常检索
    "埃德蒙和斯威腾汉姆太太是什么关系？",
    "被害者是谁？",
    "谋杀案会发生在哪里？",
    "小围场的主人是谁？",
    "这个故事发生在哪个时代？",
    "克拉多克警督为什么出警？",
    "鲁迪·谢尔兹的行动线是？",
    "兰德尔·戈德勒的遗产由谁继承？",
    "马普尔小姐住在谁家？",
    "不完美的完美，冷冰冰的匀称，光辉灿烂的徒劳。这个句子出自哪个作品？",
    "埃德蒙·斯威腾汉姆戴眼镜吗？",
    "米琪做的特色蛋糕具体是什么特色？",
    "谁偷了警督的枪？",
    # 应该找不到答案（书中未提及）
    "马普尔小姐的身高是多少？",

]

def _safe_id(name: str) -> str:
    return "bk_" + hashlib.md5(name.encode()).hexdigest()[:16]


def build_index(epub_path: Path) -> str:
    print(f"解析 {epub_path.name} ...")
    chapters = parse_epub(epub_path)
    chunks = chunk_chapters(chapters)
    cid = _safe_id(epub_path.name)
    ingest_chunks(cid, chunks)
    print(f"入库完成：{len(chapters)} 章，{len(chunks)} 个 chunk")
    return cid


def collect_dataset(book_id: str, current_chapter: int) -> list[dict]:
    dataset = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        hits = search(book_id, q, current_chapter, top_k=3)
        contexts = [h["text"] for h in hits]
        answer = chat(book_id, q, current_chapter, history=[])
        dataset.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
        })
    return dataset


class LocalEmbeddings:
    """用项目已有的 bge-m3 模型包装成 RAGAS 兼容的 embeddings。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from src.rag.embedder import embed_texts
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        from src.rag.embedder import embed_query
        return embed_query(text)


def run_ragas(dataset: list[dict]):

    client = OpenAI(
        api_key=os.environ.get("QWEN_API_KEY"),
        base_url=QWEN_BASE_URL)
    
    llm = llm_factory(MODEL,client=client)
    
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())

    
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
    metrics=[faithfulness, answer_relevancy]
    
    ds = Dataset.from_list(dataset)
    return evaluate(ds, metrics)





def llm_as_judge(dataset: list[dict]) -> list[dict]:
    """用不同于生成答案的 LLM 来做评审，给每个答案打分并说明理由。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    SYSTEM_PROMPT = """你是一个严谨的读者，正在评估一个答案是否符合书中内容。
请根据提供的 question、contexts 和 answer，判断答案是否 faithful（忠实于原文）和 relevant（与问题相关）。
请给出评分（1-5分）和简短的理由。"""

    judge_model = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-5.4-nano",
    )
    evaluation_results = []
    for row in dataset:
        q = row["question"]
        c = "\n\n".join(row["contexts"])
        a = row["answer"]
        query = f"问题：{q}\n\n参考原文：\n{c}\n\n答案：{a}"
        result = judge_model.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
        evaluation_results.append({
            "question": q,
            "answer": a,
            "faithfulness_judge": result.content,
        })
    return evaluation_results
if __name__ == "__main__":
    if not EPUB_PATH.exists():
        raise SystemExit(f"文件不存在：{EPUB_PATH}")

    if SKIP_INDEX:
        book_id = _safe_id(EPUB_PATH.name)
        print(f"跳过建索引，使用已有 collection: {book_id}")
    else:
        book_id = build_index(EPUB_PATH)

    print(f"\n收集评估数据（chapter <= {CURRENT_CHAPTER}）...")
    dataset = collect_dataset(book_id, CURRENT_CHAPTER)

    print("\n运行 RAGAS 评估...")
    result = run_ragas(dataset)

    #result = llm_as_judge(dataset)
    print("\n── 评估结果 ──────────────────────")
    print(result)

    df = result.to_pandas()
    #df = Dataset.from_list(result).to_pandas()
    #df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存到 {OUTPUT}")
