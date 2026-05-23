"""
RAG 知识库管理工具 — 查看、搜索、删除向量数据库中的内容

用法:
    python tools/rag_manager.py list              列出所有条目（概览）
    python tools/rag_manager.py list --conv       列出对话历史条目
    python tools/rag_manager.py search <关键词>   搜索小说内容
    python tools/rag_manager.py search <关键词> --conv  搜索对话历史
    python tools/rag_manager.py delete <id>       删除指定ID的条目
    python tools/rag_manager.py clear             清空小说知识库
    python tools/rag_manager.py clear --conv      清空对话历史知识库
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NOVEL_QDRANT_PATH = (DATA_DIR / "memory" / "qdrant_rag").as_posix()
NOVEL_COLLECTION = "novel_rag"
CONV_QDRANT_PATH = (DATA_DIR / "memory" / "qdrant_conv").as_posix()
CONV_COLLECTION = "conversation_history"
EMBEDDING_DIMS = 384


def _get_client(path):
    from qdrant_client import QdrantClient
    return QdrantClient(path=path)


def _get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _is_collection_empty(client, collection):
    try:
        return client.count(collection).count == 0
    except Exception:
        return True


def cmd_list(args):
    use_conv = "--conv" in args
    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "对话历史"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "小说"

    client = _get_client(path)
    if _is_collection_empty(client, collection):
        print(f"{label}知识库为空")
        return

    count = client.count(collection).count
    print(f"=== {label}知识库: {count} 条 ===\n")

    offset = None
    shown = 0
    while True:
        results, offset = client.scroll(collection, limit=20, offset=offset)
        if not results:
            break
        for r in results:
            payload = r.payload or {}
            if use_conv:
                user_text = payload.get("user", "")[:60]
                assistant_text = payload.get("assistant", "")[:60]
                print(f"  ID: {r.id}")
                print(f"    User: {user_text}...")
                print(f"    Assistant: {assistant_text}...")
            else:
                chapter = payload.get("chapter", "?")
                text = payload.get("text", "")[:80]
                print(f"  ID: {r.id}")
                print(f"    Chapter: {chapter}")
                print(f"    Text: {text}...")
            print()
            shown += 1
        if offset is None:
            break
    print(f"共显示 {shown} 条")


def cmd_search(args):
    use_conv = "--conv" in args
    query_words = [a for a in args if not a.startswith("--")]
    if not query_words:
        print("请提供搜索关键词")
        return
    query = " ".join(query_words)

    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "对话历史"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "小说"

    client = _get_client(path)
    if _is_collection_empty(client, collection):
        print(f"{label}知识库为空")
        return

    encoder = _get_encoder()
    vector = encoder.encode(query).tolist()
    results = client.query_points(collection_name=collection, query=vector, limit=5)

    print(f"=== 搜索 '{query}' ({label}) ===\n")
    for r in results.points:
        payload = r.payload or {}
        if use_conv:
            print(f"  Score: {r.score:.4f}  ID: {r.id}")
            print(f"    User: {payload.get('user', '')[:200]}")
            print(f"    Assistant: {payload.get('assistant', '')[:200]}")
        else:
            print(f"  Score: {r.score:.4f}  ID: {r.id}")
            print(f"    Chapter: {payload.get('chapter', '?')}")
            print(f"    Text: {payload.get('text', '')[:200]}")
        print()


def cmd_delete(args):
    target_id = None
    use_conv = "--conv" in args
    id_args = [a for a in args if not a.startswith("--")]
    if id_args:
        target_id = id_args[0]

    if not target_id:
        print("请提供要删除的条目ID")
        return

    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "对话历史"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "小说"

    client = _get_client(path)
    from qdrant_client.models import PointIdsList
    client.delete(collection, points_selector=PointIdsList(points=[target_id]))
    print(f"已从{label}知识库删除: {target_id}")


def cmd_clear(args):
    use_conv = "--conv" in args
    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "对话历史"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "小说"

    confirm = input(f"确认清空{label}知识库？(yes/no): ")
    if confirm.lower() != "yes":
        print("取消")
        return

    client = _get_client(path)
    try:
        client.delete_collection(collection)
        print(f"{label}知识库已清空")
    except Exception as e:
        print(f"清空失败: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "list":
        cmd_list(rest)
    elif cmd == "search":
        cmd_search(rest)
    elif cmd == "delete":
        cmd_delete(rest)
    elif cmd == "clear":
        cmd_clear(rest)
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
