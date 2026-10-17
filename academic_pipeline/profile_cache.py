"""
profile_cache.py — Cache manager for professor dossiers and clustered research programs.
"""

from __future__ import annotations

import os
import sqlite3
import json
import time
import logging
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

class ProfileCache:
    """
    Persistent SQLite-backed cache for professor dossiers and their processed research programs.
    Avoids expensive OpenAlex fetching, embedding, clustering, and LLM naming on subsequent runs.
    """

    def __init__(self, cache_dir: str = "outputs", db_name: str = "professor_profile_cache.db", ttl_days: float = 14.0):
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, db_name)
        self.ttl = ttl_days * 86400
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS professor_profiles (
                    professor_id TEXT PRIMARY KEY,
                    dossier TEXT NOT NULL,
                    programs TEXT NOT NULL,
                    last_refresh REAL NOT NULL
                )
            """)
            conn.commit()
        logger.info(f"Professor profile and program cache initialized at {self.db_path} (TTL={ttl_days} days)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def get_profile(self, professor_id: str) -> Optional[dict]:
        """
        Retrieve cached profile and programs if present and not expired.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT dossier, programs, last_refresh FROM professor_profiles WHERE professor_id = ?",
                (professor_id,)
            ).fetchone()

        if row is None:
            return None

        if time.time() - row["last_refresh"] > self.ttl:
            logger.info(f"Profile cache expired for professor {professor_id}")
            return None

        try:
            return {
                "dossier": json.loads(row["dossier"]),
                "programs": json.loads(row["programs"])
            }
        except Exception as e:
            logger.warning(f"Error parsing cached profile for {professor_id}: {e}")
            return None

    def save_profile(self, professor_id: str, dossier: dict, programs: list[dict]):
        """
        Save raw dossier and clustered research programs to database.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO professor_profiles (professor_id, dossier, programs, last_refresh)
                VALUES (?, ?, ?, ?)
                """,
                (
                    professor_id,
                    json.dumps(dossier, ensure_ascii=False),
                    json.dumps(programs, ensure_ascii=False),
                    time.time()
                )
            )
            conn.commit()
        logger.info(f"Cached dossier and {len(programs)} programs for professor {professor_id}")
