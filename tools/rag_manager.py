"""
RAG + Memory unified manager

Commands:
  list              List novel chunks
  list --conv       List conversation history
  list --mem [NAME] List mem0 memories (NAME=character name, default=all)
  search WORD       Search novel
  search WORD --conv   Search conversations
  search WORD --mem [NAME]  Search mem0 memories
  delete ID         Delete entry by ID from novel/conv RAG
  delete ID --mem   Delete mem0 memory by ID
  clear             Clear novel database
  clear --conv      Clear conversation database
  clear --mem [NAME] Clear mem0 memories (NAME=character name, default=all)
  purge WORD        One-click: search & delete WORD from ALL stores (novel RAG, conv RAG, mem0)
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


def _get_mem0():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm.tools.memory_tools import _get_mem0
    return _get_mem0()


def _resolve_mem_agent(name):
    return name if name and name.strip() else None


# ── list ──────────────────────────────────────────────────────────

def cmd_list(args):
    use_conv = "--conv" in args
    use_mem = "--mem" in args

    if use_mem:
        _list_mem(args)
        return

    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "Conversation History"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "Novel"

    client = _get_client(path)
    if _is_collection_empty(client, collection):
        print(f"{label} database is empty")
        return

    count = client.count(collection).count
    print(f"=== {label}: {count} entries ===\n")

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
    print(f"Total: {shown}")


def _list_mem(args):
    name_args = [a for a in args if not a.startswith("--")]
    char_name = name_args[0] if name_args else None
    mem = _get_mem0()
    agent_id = _resolve_mem_agent(char_name)
    if agent_id:
        raw = mem.get_all(filters={"user_id": agent_id}, limit=200)
    else:
        all_agents = ["auto_extract", "user"]
        try:
            from config.config_manager import ConfigManager
            cfg = ConfigManager()
            for c in cfg.config.characters:
                all_agents.append(c.name)
        except Exception:
            pass
        results_all = []
        for ag in all_agents:
            try:
                r = mem.get_all(filters={"user_id": ag}, limit=200)
                items = r.get("results", []) if isinstance(r, dict) else r
                results_all.extend(items)
            except Exception:
                pass
        raw = {"results": results_all}

    results = raw.get("results", []) if isinstance(raw, dict) else raw
    if not results:
        print("mem0: no memories found")
        return

    print(f"=== mem0 memories: {len(results)} entries ===\n")
    for m in results:
        content = m.get("memory", "")
        mid = m.get("id", "")
        uid = m.get("user_id", "")
        print(f"  ID: {mid}")
        if uid:
            print(f"    User: {uid}")
        print(f"    Content: {content[:100]}...")
        print()
    print(f"Total: {len(results)}")


# ── search ────────────────────────────────────────────────────────

def cmd_search(args):
    use_conv = "--conv" in args
    use_mem = "--mem" in args

    query_words = [a for a in args if not a.startswith("--")]
    if not query_words:
        print("Please provide search keywords")
        return
    query = " ".join(query_words)

    if use_mem:
        _search_mem(query, args)
        return

    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "Conversation History"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "Novel"

    client = _get_client(path)
    if _is_collection_empty(client, collection):
        print(f"{label} database is empty")
        return

    encoder = _get_encoder()
    vector = encoder.encode(query).tolist()
    results = client.query_points(collection_name=collection, query=vector, limit=5)

    print(f"=== Search '{query}' ({label}) ===\n")
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


def _search_mem(query, args):
    name_args = [a for a in args if not a.startswith("--") and a != query]
    char_name = name_args[0] if name_args else None
    mem = _get_mem0()
    agent_id = _resolve_mem_agent(char_name)

    all_items = []
    if agent_id:
        results = mem.search(query, filters={"user_id": agent_id}, limit=10)
        items = results.get("results", []) if isinstance(results, dict) else results
        all_items.extend(items)
    else:
        all_agents = ["auto_extract", "user"]
        try:
            from config.config_manager import ConfigManager
            cfg = ConfigManager()
            for c in cfg.config.characters:
                all_agents.append(c.name)
        except Exception:
            pass
        for ag in all_agents:
            try:
                r = mem.search(query, filters={"user_id": ag}, limit=10)
                items = r.get("results", []) if isinstance(r, dict) else r
                all_items.extend(items)
            except Exception:
                pass
    if not all_items:
        print(f"mem0: no results for '{query}'")
        return

    print(f"=== Search '{query}' (mem0) ===\n")
    for m in all_items:
        content = m.get("memory", "")
        mid = m.get("id", "")
        score = m.get("score", "")
        uid = m.get("user_id", "")
        print(f"  Score: {score}  ID: {mid}")
        if uid:
            print(f"    User: {uid}")
        print(f"    Content: {content[:200]}")
        print()


# ── delete ────────────────────────────────────────────────────────

def cmd_delete(args):
    use_conv = "--conv" in args
    use_mem = "--mem" in args
    id_args = [a for a in args if not a.startswith("--")]
    target_id = id_args[0] if id_args else None

    if not target_id:
        print("Please provide an ID to delete")
        return

    if use_mem:
        mem = _get_mem0()
        try:
            mem.delete(target_id)
            print(f"Deleted from mem0: {target_id}")
        except Exception as e:
            print(f"mem0 delete failed: {e}")
        return

    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "Conversation History"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "Novel"

    client = _get_client(path)
    from qdrant_client.models import PointIdsList
    client.delete(collection, points_selector=PointIdsList(points=[target_id]))
    print(f"Deleted from {label}: {target_id}")


# ── clear ─────────────────────────────────────────────────────────

def cmd_clear(args):
    use_conv = "--conv" in args
    use_mem = "--mem" in args

    if use_mem:
        name_args = [a for a in args if not a.startswith("--")]
        char_name = name_args[0] if name_args else None
        mem = _get_mem0()
        agent_id = _resolve_mem_agent(char_name)

        if agent_id:
            raw = mem.get_all(filters={"user_id": agent_id}, limit=500)
        else:
            raw = mem.get_all(filters={"user_id": "auto_extract"}, limit=500)

        results = raw.get("results", []) if isinstance(raw, dict) else raw
        if not results:
            print("mem0: no memories to clear")
            return

        confirm = input(f"Delete {len(results)} mem0 memories? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled")
            return

        deleted = 0
        for m in results:
            try:
                mem.delete(m.get("id", ""))
                deleted += 1
            except Exception:
                pass
        print(f"Deleted {deleted} mem0 memories")
        return

    if use_conv:
        path = CONV_QDRANT_PATH
        collection = CONV_COLLECTION
        label = "Conversation History"
    else:
        path = NOVEL_QDRANT_PATH
        collection = NOVEL_COLLECTION
        label = "Novel"

    confirm = input(f"Clear {label} database? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled")
        return

    client = _get_client(path)
    try:
        client.delete_collection(collection)
        print(f"{label} database cleared")
    except Exception as e:
        print(f"Clear failed: {e}")


# ── purge (one-click delete from ALL stores) ──────────────────────

def cmd_purge(args):
    query_words = [a for a in args if not a.startswith("--")]
    if not query_words:
        print("Please provide keywords to purge")
        return
    query = " ".join(query_words)

    print(f"=== PURGE '{query}' from ALL stores ===\n")
    total_deleted = 0

    # 1. Novel RAG
    try:
        client = _get_client(NOVEL_QDRANT_PATH)
        if not _is_collection_empty(client, NOVEL_COLLECTION):
            encoder = _get_encoder()
            vector = encoder.encode(query).tolist()
            results = client.query_points(collection_name=NOVEL_COLLECTION, query=vector, limit=20)
            novel_ids = []
            for r in results.points:
                payload = r.payload or {}
                text = payload.get("text", "")
                if query.lower() in text.lower() or r.score > 0.5:
                    novel_ids.append(str(r.id))
                    print(f"  [Novel] Score:{r.score:.4f} Chapter:{payload.get('chapter','?')} {text[:60]}...")
            if novel_ids:
                from qdrant_client.models import PointIdsList
                client.delete(NOVEL_COLLECTION, points_selector=PointIdsList(points=novel_ids))
                print(f"  -> Deleted {len(novel_ids)} from Novel RAG")
                total_deleted += len(novel_ids)
            else:
                print("  [Novel] No matches found")
        else:
            print("  [Novel] Database is empty")
    except Exception as e:
        print(f"  [Novel] Error: {e}")

    print()

    # 2. Conversation RAG
    try:
        client = _get_client(CONV_QDRANT_PATH)
        if not _is_collection_empty(client, CONV_COLLECTION):
            encoder = _get_encoder()
            vector = encoder.encode(query).tolist()
            results = client.query_points(collection_name=CONV_COLLECTION, query=vector, limit=20)
            conv_ids = []
            for r in results.points:
                payload = r.payload or {}
                user_text = payload.get("user", "")
                assistant_text = payload.get("assistant", "")
                if query.lower() in (user_text + assistant_text).lower() or r.score > 0.5:
                    conv_ids.append(str(r.id))
                    print(f"  [Conv] Score:{r.score:.4f} User:{user_text[:40]}... Asst:{assistant_text[:40]}...")
            if conv_ids:
                from qdrant_client.models import PointIdsList
                client.delete(CONV_COLLECTION, points_selector=PointIdsList(points=conv_ids))
                print(f"  -> Deleted {len(conv_ids)} from Conversation RAG")
                total_deleted += len(conv_ids)
            else:
                print("  [Conv] No matches found")
        else:
            print("  [Conv] Database is empty")
    except Exception as e:
        print(f"  [Conv] Error: {e}")

    print()

    # 3. mem0
    try:
        mem = _get_mem0()
        all_agents = ["auto_extract", "user"]
        try:
            from config.config_manager import ConfigManager
            cfg = ConfigManager()
            for c in cfg.config.characters:
                all_agents.append(c.name)
        except Exception:
            pass

        mem_ids = []
        for agent in all_agents:
            try:
                r = mem.search(query, filters={"user_id": agent}, limit=10)
                items = r.get("results", []) if isinstance(r, dict) else r
                for m in items:
                    content = m.get("memory", "")
                    mid = m.get("id", "")
                    if query.lower() in content.lower() or True:
                        mem_ids.append(mid)
                        print(f"  [mem0] Agent:{agent} Content:{content[:60]}...")
            except Exception:
                pass

        if mem_ids:
            for mid in mem_ids:
                try:
                    mem.delete(mid)
                except Exception:
                    pass
            print(f"  -> Deleted {len(mem_ids)} from mem0")
            total_deleted += len(mem_ids)
        else:
            print("  [mem0] No matches found")
    except Exception as e:
        print(f"  [mem0] Error: {e}")

    print(f"\n=== PURGE complete: {total_deleted} total entries deleted ===")


# ── main ──────────────────────────────────────────────────────────

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
    elif cmd == "purge":
        cmd_purge(rest)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
