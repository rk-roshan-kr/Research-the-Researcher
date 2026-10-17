"""
embedding_cache.py — SQLite-backed global embedding cache using binary BLOBs.
"""

from __future__ import annotations

import os
import sqlite3
import json
import hashlib
import time
import logging
from typing import List, Optional, Union
from threading import Lock
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    SQLite-backed persistent cache for sentence embeddings.
    Saves embeddings as binary BLOBs using numpy float32.tobytes() to minimize storage.
    """

    def __init__(self, cache_dir: str = "outputs", db_name: str = "embedding_store.db"):
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, db_name)
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    text TEXT,
                    embedding BLOB,
                    model_name TEXT,
                    model_revision TEXT,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info(f"Global embedding store cache initialized at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def get_embeddings(
        self, texts: list[str], model_name: str, model_revision: str = "main"
    ) -> list[Optional[np.ndarray]]:
        """
        Check database cache for a batch of texts.
        Returns a list containing the matching float32 numpy arrays, or None if uncached.
        """
        hashes = [
            hashlib.sha256(f"{t}|||{model_name}|||{model_revision}".encode("utf-8")).hexdigest()
            for t in texts
        ]
        results: list[Optional[np.ndarray]] = [None] * len(texts)

        with self._lock, self._connect() as conn:
            chunk_size = 500
            for i in range(0, len(hashes), chunk_size):
                chunk_hashes = hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk_hashes)
                rows = conn.execute(
                    f"SELECT hash, embedding, embedding_dim FROM embeddings WHERE hash IN ({placeholders})",
                    chunk_hashes
                ).fetchall()

                hash_to_emb = {}
                for row in rows:
                    blob = row["embedding"]
                    hash_to_emb[row["hash"]] = np.frombuffer(blob, dtype=np.float32)

                for idx in range(i, min(i + chunk_size, len(texts))):
                    h = hashes[idx]
                    if h in hash_to_emb:
                        results[idx] = hash_to_emb[h]
        return results

    def save_embeddings(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        model_name: str,
        model_revision: str = "main",
    ):
        """
        Store a batch of text embeddings in SQLite.
        """
        with self._lock, self._connect() as conn:
            for t, emb in zip(texts, embeddings):
                h = hashlib.sha256(f"{t}|||{model_name}|||{model_revision}".encode("utf-8")).hexdigest()
                emb_f32 = np.asarray(emb, dtype=np.float32)
                blob = emb_f32.tobytes()
                dim = int(emb_f32.shape[0])
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (hash, text, embedding, model_name, model_revision, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (h, t, blob, model_name, model_revision, dim)
                )
            conn.commit()


_CACHE_INSTANCES: dict[str, EmbeddingCache] = {}
from threading import Lock as ThreadLock
_CACHE_LOCK = ThreadLock()

def get_embedding_cache(cache_dir: str = "outputs") -> EmbeddingCache:
    global _CACHE_INSTANCES
    abs_dir = os.path.abspath(cache_dir)
    with _CACHE_LOCK:
        if abs_dir not in _CACHE_INSTANCES:
            _CACHE_INSTANCES[abs_dir] = EmbeddingCache(cache_dir=abs_dir)
        return _CACHE_INSTANCES[abs_dir]


def cached_encode(
    texts: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    model_revision: str = "main",
    normalize_embeddings: bool = True,
    cache_dir: str = "outputs",
) -> np.ndarray:
    """
    Encode a list of texts using SentenceTransformers with persistent caching in SQLite.
    Optimized to run SentenceTransformers in large batches only for uncached entries.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    texts_cleaned = [str(t) for t in texts]
    cache = get_embedding_cache(cache_dir)
    
    # Retrieve what we can from persistent cache
    cached_embs = cache.get_embeddings(texts_cleaned, model_name, model_revision)
    
    missing_indices = [i for i, emb in enumerate(cached_embs) if emb is None]

    if missing_indices:
        missing_texts = [texts_cleaned[i] for i in missing_indices]
        
        # Resolve SentenceTransformer lazy-loaded via self_model's embedder registry
        from academic_pipeline.self_model import get_embedder_by_name
        embedder = get_embedder_by_name(model_name)
        
        logger.info(f"Embedding Cache Miss: Encoding {len(missing_texts)} texts in batch using {model_name}...")
        new_embs = embedder.encode(
            missing_texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=256,
            show_progress_bar=False,
        )
        
        # Save to database
        cache.save_embeddings(missing_texts, new_embs, model_name, model_revision)
        
        for idx, new_emb in zip(missing_indices, new_embs):
            cached_embs[idx] = new_emb

    # Concatenate list of 1D arrays into a single 2D array
    return np.vstack(cached_embs)
