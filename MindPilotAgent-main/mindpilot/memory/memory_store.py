"""
向量记忆存储
============
短期工作记忆 + 长期向量存储，支持跨会话续接。
FAISS 可用时使用向量检索，否则退回关键词匹配。
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class MemoryEntry:
    """记忆条目"""
    entry_id: str
    session_id: str
    agent: str
    content: str              # 文本内容（用于检索）
    payload: Any = None       # 完整数据
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    importance: float = 1.0   # 重要性分数（影响检索权重）

    def to_dict(self) -> dict:
        d = asdict(self)
        if not isinstance(d.get("payload"), (dict, list, str, int, float, bool, type(None))):
            d["payload"] = str(d["payload"])
        return d


class MemoryStore:
    """
    记忆存储系统
    - 短期记忆：当前会话 in-memory list
    - 长期记忆：持久化 JSON + FAISS 向量索引（可选）
    """

    def __init__(self, store_dir: str = "memory/store", session_id: str = "default",
                 embedding_model: str = "all-MiniLM-L6-v2", logger=None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.logger = logger

        self._short_term: list[MemoryEntry] = []   # 当前会话记忆
        self._long_term: list[MemoryEntry] = []    # 持久化记忆
        self._embedder = None
        self._index = None

        # 尝试加载 sentence-transformers
        self._embedding_available = self._try_load_embedder(embedding_model)
        # 加载已有长期记忆
        self._load_long_term()

    def _try_load_embedder(self, model_name: str) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
            if self.logger:
                self.logger.info("Memory", f"向量嵌入模型已加载: {model_name}")
            return True
        except Exception:
            if self.logger:
                self.logger.info("Memory", "sentence-transformers 不可用，使用关键词检索")
            return False

    def add(self, content: str, agent: str = "", payload: Any = None,
            tags: list[str] = None, importance: float = 1.0) -> MemoryEntry:
        """添加记忆条目"""
        import uuid
        entry = MemoryEntry(
            entry_id=str(uuid.uuid4())[:8],
            session_id=self.session_id,
            agent=agent,
            content=content,
            payload=payload,
            tags=tags or [],
            importance=importance,
        )
        self._short_term.append(entry)
        self._long_term.append(entry)
        # 更新 FAISS 索引
        if self._embedding_available:
            self._update_index(entry)
        return entry

    def search(self, query: str, top_k: int = 5,
               agent_filter: Optional[str] = None) -> list[MemoryEntry]:
        """检索相关记忆"""
        pool = self._long_term
        if agent_filter:
            pool = [e for e in pool if e.agent == agent_filter]
        if not pool:
            return []

        if self._embedding_available and self._index is not None:
            return self._vector_search(query, pool, top_k)
        else:
            return self._keyword_search(query, pool, top_k)

    def get_recent(self, n: int = 10, agent_filter: Optional[str] = None) -> list[MemoryEntry]:
        """获取最近 n 条记忆"""
        pool = self._short_term if not agent_filter else [
            e for e in self._short_term if e.agent == agent_filter
        ]
        return sorted(pool, key=lambda e: e.timestamp, reverse=True)[:n]

    def _vector_search(self, query: str, pool: list[MemoryEntry], top_k: int) -> list[MemoryEntry]:
        try:
            import numpy as np
            import faiss
            query_emb = self._embedder.encode([query], normalize_embeddings=True)
            corpus_embs = self._embedder.encode(
                [e.content for e in pool], normalize_embeddings=True
            )
            index = faiss.IndexFlatIP(corpus_embs.shape[1])
            index.add(corpus_embs.astype(np.float32))
            distances, indices = index.search(query_emb.astype(np.float32), min(top_k, len(pool)))
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and dist > 0.1:
                    entry = pool[idx]
                    results.append(entry)
            return results
        except Exception:
            return self._keyword_search(query, pool, top_k)

    def _keyword_search(self, query: str, pool: list[MemoryEntry], top_k: int) -> list[MemoryEntry]:
        """关键词匹配检索（fallback）"""
        q_words = set(query.lower().split())
        scored = []
        for entry in pool:
            words = set(entry.content.lower().split())
            overlap = len(q_words & words) / max(len(q_words), 1)
            score = overlap * entry.importance
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _update_index(self, entry: MemoryEntry):
        """增量更新 FAISS 索引（简化版：直接重建）"""
        pass  # 向量检索在 search 时实时构建

    def _load_long_term(self):
        """从磁盘加载长期记忆"""
        path = self.store_dir / "long_term.jsonl"
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line.strip())
                    self._long_term.append(MemoryEntry(**d))
            if self.logger:
                self.logger.info("Memory", f"加载 {len(self._long_term)} 条长期记忆")
        except Exception as e:
            if self.logger:
                self.logger.warning("Memory", f"记忆加载失败: {e}")

    def save_long_term(self):
        """持久化长期记忆到磁盘"""
        path = self.store_dir / "long_term.jsonl"
        try:
            with open(path, "w", encoding="utf-8") as f:
                for entry in self._long_term:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            if self.logger:
                self.logger.info("Memory", f"已保存 {len(self._long_term)} 条记忆")
        except Exception as e:
            if self.logger:
                self.logger.error("Memory", f"记忆保存失败: {e}")

    def clear_session(self):
        """清空当前会话的短期记忆"""
        self._short_term.clear()

    def stats(self) -> dict:
        return {
            "short_term": len(self._short_term),
            "long_term": len(self._long_term),
            "embedding_available": self._embedding_available,
        }
