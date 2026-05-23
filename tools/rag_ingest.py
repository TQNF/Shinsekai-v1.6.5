"""
小说原文 RAG 入库脚本 — 将 _temp_split.json 中的 scenario（小说全文）
按段落分段，向量化后存入 Qdrant，供 novel_search 工具检索。

用法:
    python tools/rag_ingest.py [--chunk-size 800] [--overlap 200]

只需运行一次，之后每次对话都能通过 novel_search 检索。
重复运行会先清空旧数据再重新入库。
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEMPLATE_FILE = DATA_DIR / "character_templates" / "_temp_split.json"
QDRANT_PATH = (DATA_DIR / "memory" / "qdrant_rag").as_posix()
COLLECTION_NAME = "novel_rag"
EMBEDDING_DIMS = 384
CHUNK_SIZE = 400
OVERLAP = 100


def _load_novel_text() -> str:
    if not TEMPLATE_FILE.exists():
        print(f"错误: 找不到角色模板文件 {TEMPLATE_FILE}")
        sys.exit(1)
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenario = data.get("scenario", "")
    if not scenario.strip():
        print("错误: scenario 字段为空")
        sys.exit(1)
    return scenario


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    chapter_pattern = re.compile(r"(第[一二三四五六七八九十百千]+章\s+.+)")
    parts = chapter_pattern.split(text)

    chapters: list[tuple[str, str]] = []
    current_title = "前言"
    current_body = ""

    for part in parts:
        if chapter_pattern.match(part):
            if current_body.strip():
                chapters.append((current_title, current_body.strip()))
            current_title = part.strip()
            current_body = ""
        else:
            current_body += part
    if current_body.strip():
        chapters.append((current_title, current_body.strip()))

    chunks = []
    for title, body in chapters:
        paragraphs = re.split(r"\n(?=　　)", body)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        buffer = ""
        for para in paragraphs:
            if buffer:
                buffer += "\n"
            buffer += para
            if len(buffer) >= chunk_size:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": buffer.strip(),
                    "chapter": title,
                    "char_count": len(buffer.strip()),
                })
                buffer = buffer[-overlap:] if overlap > 0 else ""
        if buffer.strip():
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": buffer.strip(),
                "chapter": title,
                "char_count": len(buffer.strip()),
            })

    return chunks


def ingest(chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    print("正在加载小说原文...")
    novel_text = _load_novel_text()
    print(f"小说原文总长度: {len(novel_text)} 字符")

    print(f"正在分段 (chunk_size={chunk_size}, overlap={overlap})...")
    chunks = _split_into_chunks(novel_text, chunk_size, overlap)
    print(f"共分为 {len(chunks)} 个段落")

    print("正在加载 Embedding 模型 (sentence-transformers/all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("正在向量化...")
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)

    import os
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("已清空旧数据")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE),
    )

    print("正在写入 Qdrant...")
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=chunk["id"],
            vector=vector.tolist(),
            payload={
                "text": chunk["text"],
                "chapter": chunk["chapter"],
                "char_count": chunk["char_count"],
                "chunk_index": i,
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\n入库完成！共 {count} 个段落已存入向量数据库")
    print(f"数据库路径: {QDRANT_PATH}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="小说原文 RAG 入库")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每段最大字符数")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help="段间重叠字符数")
    args = parser.parse_args()
    ingest(args.chunk_size, args.overlap)
