# compact_manager.py
import json
import tiktoken
import uuid
import threading
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

_CONV_QDRANT_PATH = (Path.cwd() / "data" / "memory" / "qdrant_conv").as_posix()
_CONV_COLLECTION = "conversation_history"
_CONV_DIMS = 384
_conv_client: Any = None
_conv_encoder: Any = None
_conv_lock = threading.Lock()


def _get_conv_client() -> Any:
    global _conv_client
    if _conv_client is not None:
        return _conv_client
    with _conv_lock:
        if _conv_client is not None:
            return _conv_client
        from qdrant_client import QdrantClient
        _conv_client = QdrantClient(path=_CONV_QDRANT_PATH)
        return _conv_client


def _get_conv_encoder() -> Any:
    global _conv_encoder
    if _conv_encoder is not None:
        return _conv_encoder
    with _conv_lock:
        if _conv_encoder is not None:
            return _conv_encoder
        from sentence_transformers import SentenceTransformer
        _conv_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _conv_encoder


def _ensure_conv_collection() -> None:
    from qdrant_client.models import Distance, VectorParams
    client = _get_conv_client()
    try:
        client.get_collection(_CONV_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=_CONV_COLLECTION,
            vectors_config=VectorParams(size=_CONV_DIMS, distance=Distance.COSINE),
        )
        logger.info("对话历史RAG集合已创建: %s", _CONV_COLLECTION)


def _archive_messages_to_rag(messages: List[Dict[str, str]]) -> int:
    if not messages:
        return 0
    try:
        _ensure_conv_collection()
        client = _get_conv_client()
        encoder = _get_conv_encoder()

        from qdrant_client.models import PointStruct

        pairs: List[Dict[str, str]] = []
        i = 0
        while i < len(messages):
            if messages[i].get("role") == "user":
                user_text = messages[i].get("content", "")
                assistant_text = ""
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_text = messages[i + 1].get("content", "")
                    i += 2
                else:
                    i += 1
                if user_text.strip() or assistant_text.strip():
                    pairs.append({"user": user_text, "assistant": assistant_text})
            else:
                i += 1

        if not pairs:
            return 0

        texts_to_encode = []
        for p in pairs:
            snippet_user = p["user"][:200]
            snippet_assistant = p["assistant"][:200]
            texts_to_encode.append(f"User: {snippet_user}\nAssistant: {snippet_assistant}")

        vectors = encoder.encode(texts_to_encode, show_progress_bar=False, batch_size=32)

        points = []
        for idx, (pair, vector) in enumerate(zip(pairs, vectors)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "user": pair["user"],
                    "assistant": pair["assistant"],
                    "type": "conversation",
                },
            ))

        client.upsert(collection_name=_CONV_COLLECTION, points=points)
        logger.info("对话历史归档: %d 轮对话已存入RAG", len(points))
        return len(points)

    except Exception as e:
        logger.error("对话历史归档失败: %s", e)
        return 0


def conversation_search(query: str, limit: int = 5) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"error": "query 不能为空"}
    try:
        _ensure_conv_collection()
        client = _get_conv_client()
        encoder = _get_conv_encoder()
        vector = encoder.encode(q).tolist()
        results = client.query_points(
            collection_name=_CONV_COLLECTION,
            query=vector,
            limit=limit,
        )
        hits = []
        for r in results.points:
            payload = r.payload or {}
            hits.append({
                "score": round(r.score, 4),
                "user": payload.get("user", ""),
                "assistant": payload.get("assistant", ""),
            })
        return {"query": q, "count": len(hits), "results": hits}
    except Exception as e:
        logger.exception("conversation_search 失败")
        return {"error": str(e)}


def _sanitize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """清理消息列表，移除孤立的 tool 消息和不完整的 tool_calls 链"""
    if not messages:
        return messages
    cleaned = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role", "")
        if role == "tool":
            # 检查前面是否有匹配的 assistant(tool_calls)
            tc_id = m.get("tool_call_id", "")
            has_parent = False
            for j in range(len(cleaned) - 1, -1, -1):
                prev = cleaned[j]
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    tc_ids = [tc.get("id", "") for tc in prev["tool_calls"]]
                    if tc_id in tc_ids:
                        has_parent = True
                        break
                elif prev.get("role") in ("user", "system"):
                    break
            if not has_parent:
                logger.warning("移除孤立 tool 消息: tool_call_id=%s", tc_id)
                i += 1
                continue
        elif role == "assistant" and m.get("tool_calls"):
            # 检查后面是否有对应的 tool 消息
            tc_ids = [tc.get("id", "") for tc in m["tool_calls"]]
            has_response = False
            for j in range(i + 1, len(messages)):
                nxt = messages[j]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id", "") in tc_ids:
                    has_response = True
                    break
                elif nxt.get("role") in ("user", "system"):
                    break
            if not has_response:
                logger.warning("移除无响应的 assistant(tool_calls) 消息")
                i += 1
                continue
        cleaned.append(m)
        i += 1
    return cleaned


class CompactManager:
    """管理记忆压缩的类"""

    def __init__(self, llm_adapter, max_tokens: int = 128000, compact_threshold: float = 0.9):
        self.llm_adapter = llm_adapter
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.num_tokens = 0

        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None
            logger.warning("tiktoken not available, using approximate token counting")

    def set_token_count(self, token_count: int):
        self.num_tokens = token_count

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        if self.encoder:
            total_tokens = 0
            for message in messages:
                content = message.get('content', '')
                total_tokens += len(self.encoder.encode(content)) + 4
            return total_tokens
        else:
            total_chars = 0
            for message in messages:
                content = message.get('content', '')
                total_chars += len(content)
            return int(total_chars / 1.5)

    def increase_token_count(self, new_messages: List[Dict[str, str]], token_usage: int = 0):
        if token_usage > 0:
            self.num_tokens = token_usage
        else:
            self.num_tokens += self.count_tokens(new_messages)
        return self.num_tokens

    def needs_compaction(self, messages: List[Dict[str, str]], token_usage: int = 0) -> bool:
        if self.num_tokens == 0:
            self.num_tokens = self.count_tokens(messages)
            logger.info(f"Initial token count: {self.num_tokens} tokens for {len(messages)} messages")
        else:
            self.increase_token_count(messages[-1:], token_usage)
        token_count = self.num_tokens
        threshold_tokens = self.max_tokens * self.compact_threshold
        return token_count > threshold_tokens

    def auto_compact_if_needed(self, messages: List[Dict[str, str]], token_usage: int = 0) -> List[Dict[str, str]]:
        """
        自动检查并归档旧消息到RAG向量数据库。
        当 token 数超过阈值时，把最早的对话存入向量数据库，箱子只留最近几轮。
        归档在后台线程执行，不阻塞主对话流程。
        """
        if not self.needs_compaction(messages, token_usage):
            return messages

        if len(messages) <= 3:
            return messages

        system_message = messages[0] if messages[0].get('role') == 'system' else None
        non_system = messages[1:] if system_message else messages

        keep_recent = 10
        if len(non_system) <= keep_recent:
            return messages

        messages_to_archive = non_system[:-keep_recent]
        messages_to_keep = non_system[-keep_recent:]

        # 向前扫描：如果 messages_to_keep 的开头是 tool 消息，
        # 必须把对应的 assistant(tool_calls) 消息也拉进来，否则 API 报错
        while messages_to_keep:
            first = messages_to_keep[0]
            if first.get("role") == "tool":
                # 需要找到对应的 assistant(tool_calls) 消息
                if messages_to_archive:
                    pulled = messages_to_archive.pop()
                    messages_to_keep.insert(0, pulled)
                else:
                    break
            elif first.get("role") == "assistant" and first.get("tool_calls"):
                # assistant 带 tool_calls，后面必须跟 tool 消息
                # 检查第二条是否是 tool
                if len(messages_to_keep) > 1 and messages_to_keep[1].get("role") == "tool":
                    break  # 正常，不用处理
                else:
                    # tool_calls 后面缺 tool 消息，把这条 assistant 也归档
                    messages_to_archive.append(messages_to_keep.pop(0))
                    continue
            else:
                break

        # 从 messages_to_keep 末尾扫描：确保不以 assistant(tool_calls) 结尾
        # （后面缺 tool 响应的话 API 也会报错）
        while messages_to_keep and messages_to_keep[-1].get("role") == "assistant" and messages_to_keep[-1].get("tool_calls"):
            orphan = messages_to_keep.pop()
            messages_to_archive.append(orphan)

        if not messages_to_keep:
            return messages

        archive_snapshot = [dict(m) for m in messages_to_archive]
        t = threading.Thread(
            target=_archive_messages_to_rag,
            args=(archive_snapshot,),
            daemon=True,
        )
        t.start()
        logger.info(
            "对话归档已启动（后台线程）: %d 条旧对话正在存入RAG, 箱子保留最近 %d 条消息",
            len(messages_to_archive), len(messages_to_keep),
        )

        result = []
        if system_message:
            result.append(system_message)

        result.append({
            'role': 'user',
            'content': (
                "【历史对话已归档】之前的对话已存入对话历史知识库。"
                "当你需要回忆之前的对话内容时，请调用 conversation_search 工具搜索。"
                "例如：conversation_search(\"雾切和我的对话\")"
            ),
        })

        result.extend(messages_to_keep)
        result = _sanitize_messages(result)
        self.num_tokens = self.count_tokens(result)
        return result
