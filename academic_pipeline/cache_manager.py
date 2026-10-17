"""
cache_manager.py — Two-tier SQLite + in-memory LRU cache.

Tier 1 (hot):  In-memory dict with LRU eviction — nanosecond access.
Tier 2 (cold): SQLite WAL-mode database — persistent, concurrent-read safe.

Tables:
  paper_analyses    (key TEXT PK, data JSON, created_at REAL)
  social_profiles   (key TEXT PK, data JSON, created_at REAL)
  resolved_profiles (key TEXT PK, data JSON, created_at REAL)
  search_results    (key TEXT PK, data JSON, created_at REAL)

A background sweep evicts entries older than `ttl_days` days.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# In-memory LRU
# ──────────────────────────────────────────────────────────────────────────────

class LRUCache:
    """Thread-safe LRU cache backed by an OrderedDict."""

    def __init__(self, maxsize: int = 1000):
        self._store:  OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        self._lock    = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)


# ──────────────────────────────────────────────────────────────────────────────
# SQLite Backend
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS paper_analyses (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_abstracts (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS social_profiles (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS resolved_profiles (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS search_results (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS image_descriptions (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- ── Research Opportunity Discovery System tables ──────────────────────────────

-- User's research identity vector (90-day TTL, rebuilt when papers change)
CREATE TABLE IF NOT EXISTS self_model (
    key        TEXT PRIMARY KEY,     -- always "self_model_v1"
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- InvestigationStream objects — never expire, user-managed lifecycle
CREATE TABLE IF NOT EXISTS investigation_streams (
    id                 TEXT PRIMARY KEY,   -- uuid
    data               TEXT NOT NULL,      -- full InvestigationStream JSON
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL,
    status             TEXT NOT NULL,      -- active/blocked/speculative/queued/resolved
    effective_priority REAL NOT NULL       -- precomputed for fast ordering
);
CREATE INDEX IF NOT EXISTS idx_inv_status   ON investigation_streams(status);
CREATE INDEX IF NOT EXISTS idx_inv_priority ON investigation_streams(effective_priority DESC);

-- Collaboration scores — 14-day TTL (professor output can change)
CREATE TABLE IF NOT EXISTS collaboration_scores (
    key        TEXT PRIMARY KEY,     -- "{model_hash}_{prof_openalex_id}"
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_collab_ts ON collaboration_scores(created_at);

-- Long-horizon professor interaction memory — never expires
CREATE TABLE IF NOT EXISTS professor_memory (
    prof_id    TEXT PRIMARY KEY,     -- openalex_id or sha256(name)
    data       TEXT NOT NULL,        -- full ProfessorMemory JSON
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status     TEXT NOT NULL,        -- discovered/contacted/in_conversation/not_relevant/resolved
    last_score REAL                  -- most recent collaboration score (nullable)
);
CREATE INDEX IF NOT EXISTS idx_mem_status ON professor_memory(status);
CREATE INDEX IF NOT EXISTS idx_mem_score  ON professor_memory(last_score DESC);

-- Research Threads
CREATE TABLE IF NOT EXISTS research_threads (
    id                 TEXT PRIMARY KEY,   -- uuid
    prof_id            TEXT NOT NULL,
    investigation_id   TEXT NOT NULL,
    status             TEXT NOT NULL,      -- active/dormant/ignored/completed
    data               TEXT NOT NULL,      -- full ResearchThread JSON
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_thread_prof ON research_threads(prof_id);
CREATE INDEX IF NOT EXISTS idx_thread_inv  ON research_threads(investigation_id);

-- Research Interactions
CREATE TABLE IF NOT EXISTS research_interactions (
    id                 TEXT PRIMARY KEY,   -- uuid
    thread_id          TEXT NOT NULL,
    prof_id            TEXT NOT NULL,
    investigation_id   TEXT NOT NULL,
    blocker_id         TEXT NOT NULL,
    interaction_number INTEGER NOT NULL,
    status             TEXT NOT NULL,      -- drafted/sent/replied/action_taken/completed/ignored/dormant
    follow_up_date     TEXT,               -- YYYY-MM-DD
    data               TEXT NOT NULL,      -- full ResearchInteraction JSON
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interaction_thread ON research_interactions(thread_id);
CREATE INDEX IF NOT EXISTS idx_interaction_prof ON research_interactions(prof_id);

-- Research Outcomes
CREATE TABLE IF NOT EXISTS research_outcomes (
    id                 TEXT PRIMARY KEY,   -- uuid
    investigation_id   TEXT NOT NULL,
    prof_id            TEXT NOT NULL,
    data               TEXT NOT NULL,      -- full ResearchOutcome JSON
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcome_prof ON research_outcomes(prof_id);

CREATE INDEX IF NOT EXISTS idx_paper_ts    ON paper_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_abstract_ts ON paper_abstracts(created_at);
CREATE INDEX IF NOT EXISTS idx_social_ts   ON social_profiles(created_at);
CREATE INDEX IF NOT EXISTS idx_resolved_ts ON resolved_profiles(created_at);
CREATE INDEX IF NOT EXISTS idx_search_ts   ON search_results(created_at);
CREATE INDEX IF NOT EXISTS idx_images_ts   ON image_descriptions(created_at);

-- Crawled faculty list from OpenAlex for dynamic coverage
CREATE TABLE IF NOT EXISTS crawled_faculty (
    openalex_id      TEXT,
    university       TEXT,
    university_alias TEXT,
    name             TEXT,
    h_index          INTEGER,
    works_count      INTEGER,
    concepts         TEXT,
    created_at       REAL,
    PRIMARY KEY (openalex_id, university)
);
CREATE INDEX IF NOT EXISTS idx_crawled_uni ON crawled_faculty(university);
CREATE INDEX IF NOT EXISTS idx_crawled_alias ON crawled_faculty(university_alias);
"""


class CacheManager:
    """
    Two-tier cache: in-memory LRU (hot) → SQLite (cold).

    All public methods are thread-safe.
    """

    TTL_PAPER   = 30 * 86400   # 30 days
    TTL_SOCIAL  = 7  * 86400   # 7 days (social media changes faster)
    TTL_PROFILE = 7  * 86400
    TTL_SEARCH  = 1  * 86400   # 1 day

    def __init__(self, cache_dir: str = "outputs"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        db_path = os.path.join(cache_dir, "pipeline_cache.db")
        self._db_path = db_path
        self._lock    = Lock()
        self._lru     = LRUCache(maxsize=2000)

        # Initialise SQLite
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Migration to add concepts column to crawled_faculty if not present
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(crawled_faculty)")
                columns = [info[1] for info in cursor.fetchall()]
                if "concepts" not in columns:
                    conn.execute("ALTER TABLE crawled_faculty ADD COLUMN concepts TEXT")
                    conn.commit()
                    logger.info("Migrated crawled_faculty table: added concepts column.")
            except Exception as e:
                logger.error(f"Failed to migrate crawled_faculty table: {e}")

        logger.info(f"Cache initialised at {db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.create_function("lower", 1, lambda s: s.lower() if s is not None else None)
        conn.row_factory = sqlite3.Row
        return conn


    def _get(self, table: str, key: str, ttl: float) -> Optional[dict]:
        lru_key = f"{table}:{key}"

        # Tier 1 — memory
        val = self._lru.get(lru_key)
        if val is not None:
            return val

        # Tier 2 — SQLite
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT data, created_at FROM {table} WHERE key = ?",
                (key,)
            ).fetchone()
        if row is None:
            return None
        if time.time() - row["created_at"] > ttl:
            return None   # stale
        data = json.loads(row["data"])
        self._lru.set(lru_key, data)
        return data

    def _set(self, table: str, key: str, data: dict):
        lru_key = f"{table}:{key}"
        self._lru.set(lru_key, data)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} (key, data, created_at) "
                f"VALUES (?, ?, ?)",
                (key, json.dumps(data, ensure_ascii=False), time.time())
            )
            conn.commit()

    # ── Paper Analyses ────────────────────────────────────────────────────────

    def get_paper_analysis(self, key: str) -> Optional[dict]:
        return self._get("paper_analyses", key, self.TTL_PAPER)

    def save_paper_analysis(self, key: str, data: dict):
        self._set("paper_analyses", key, data)

    # ── Paper Abstracts ───────────────────────────────────────────────────────

    def get_paper_abstract(self, key: str) -> Optional[str]:
        res = self._get("paper_abstracts", key, self.TTL_PAPER)
        return res.get("abstract") if res else None

    def save_paper_abstract(self, key: str, abstract: str):
        self._set("paper_abstracts", key, {"abstract": abstract})

    # ── Social Profiles ───────────────────────────────────────────────────────

    def get_social_analysis(self, key: str) -> Optional[dict]:
        return self._get("social_profiles", key, self.TTL_SOCIAL)

    def save_social_analysis(self, key: str, data: dict):
        self._set("social_profiles", key, data)

    # ── Resolved Profiles ─────────────────────────────────────────────────────

    def get_resolved_profile(self, key: str) -> Optional[dict]:
        return self._get("resolved_profiles", key, self.TTL_PROFILE)

    def save_resolved_profile(self, key: str, data: dict):
        self._set("resolved_profiles", key, data)

    # ── Search Results ────────────────────────────────────────────────────────

    def get_search_results(self, key: str) -> Optional[list]:
        result = self._get("search_results", key, self.TTL_SEARCH)
        return result.get("results") if result else None

    def save_search_results(self, key: str, results: list):
        self._set("search_results", key, {"results": results})

    # ── Image Descriptions ──────────────────────────────────────────────────
    
    def get_image_description(self, key: str) -> Optional[str]:
        res = self._get("image_descriptions", key, self.TTL_PAPER)
        return res.get("description") if res else None

    def save_image_description(self, key: str, description: str):
        self._set("image_descriptions", key, {"description": description})

    # ── Self Model ────────────────────────────────────────────────────────────
    # Uses content-hash invalidation rather than TTL.
    # The self-model is rebuilt whenever paper_count or model_hash changes,
    # not on a fixed schedule. This ensures a newly published paper triggers
    # a rebuild on the next call rather than waiting up to 90 days.

    def get_self_model(self, expected_hash: Optional[str] = None) -> Optional[dict]:
        """
        Returns the stored self-model dict, or None if:
          - no model exists, OR
          - expected_hash is provided and does not match stored model_hash
            (signals the corpus has changed, rebuild required).
        """
        # Use a long TTL purely as a hard expiry backstop (not primary invalidation)
        _BACKSTOP_TTL = 180 * 86400  # 180 days
        data = self._get("self_model", "self_model_v1", _BACKSTOP_TTL)
        if data is None:
            return None
        if expected_hash is not None and data.get("model_hash") != expected_hash:
            logger.info(
                "Self-model hash mismatch (corpus changed). Rebuild required."
            )
            return None
        return data

    def save_self_model(self, data: dict):
        """Persist a self-model. data must contain a 'model_hash' key."""
        if "model_hash" not in data:
            raise ValueError("self_model data must include 'model_hash' for invalidation.")
        self._set("self_model", "self_model_v1", data)

    # ── Collaboration Scores ──────────────────────────────────────────────────

    TTL_COLLAB_SCORE = 14 * 86400  # 14 days

    def get_collaboration_score(self, key: str) -> Optional[dict]:
        return self._get("collaboration_scores", key, self.TTL_COLLAB_SCORE)

    def save_collaboration_score(self, key: str, data: dict):
        self._set("collaboration_scores", key, data)

    # ── Investigation Streams ─────────────────────────────────────────────────
    # Uses raw SQL (not generic _get/_set) because status and effective_priority
    # are separate indexed columns that must be updated independently.

    def get_investigation(self, inv_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM investigation_streams WHERE id = ?", (inv_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_investigations(
        self, status_filter: Optional[str] = None
    ) -> list[dict]:
        """Returns investigations ordered by effective_priority descending."""
        with self._lock, self._connect() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT data FROM investigation_streams "
                    "WHERE status = ? ORDER BY effective_priority DESC",
                    (status_filter,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM investigation_streams "
                    "ORDER BY effective_priority DESC"
                ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def save_investigation(self, inv: dict):
        """Insert or replace a full InvestigationStream dict."""
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO investigation_streams "
                "(id, data, created_at, updated_at, status, effective_priority) "
                "VALUES (?, ?, COALESCE((SELECT created_at FROM investigation_streams WHERE id=?), ?), ?, ?, ?)",
                (
                    inv["id"],
                    json.dumps(inv, ensure_ascii=False),
                    inv["id"], now,   # created_at: keep original if exists
                    now,              # updated_at: always now
                    inv["status"],
                    inv.get("effective_priority", inv.get("base_priority", 0.5)),
                )
            )
            conn.commit()
        lru_key = f"investigation:{inv['id']}"
        self._lru.set(lru_key, inv)

    def delete_investigation(self, inv_id: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM investigation_streams WHERE id = ?", (inv_id,)
            )
            conn.commit()

    def count_active_investigations(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM investigation_streams "
                "WHERE status IN ('active', 'blocked')"
            ).fetchone()
        return row[0] if row else 0

    # ── Professor Memory ──────────────────────────────────────────────────────
    # Never expires — long-horizon memory is permanent until explicitly deleted.

    def get_professor_memory(self, prof_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM professor_memory WHERE prof_id = ?", (prof_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_professor_memories(
        self, status_filter: Optional[str] = None
    ) -> list[dict]:
        """Returns memories ordered by last_score descending.
        NULLs sort last via portable (last_score IS NULL), last_score DESC.
        """
        with self._lock, self._connect() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT data FROM professor_memory "
                    "WHERE status = ? "
                    "ORDER BY (last_score IS NULL), last_score DESC",
                    (status_filter,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM professor_memory "
                    "ORDER BY (last_score IS NULL), last_score DESC"
                ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def save_professor_memory(self, prof_id: str, data: dict):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO professor_memory "
                "(prof_id, data, created_at, updated_at, status, last_score) "
                "VALUES (?, ?, COALESCE((SELECT created_at FROM professor_memory WHERE prof_id=?), ?), ?, ?, ?)",
                (
                    prof_id,
                    json.dumps(data, ensure_ascii=False),
                    prof_id, now,
                    now,
                    data.get("status", "discovered"),
                    data.get("last_score"),
                )
            )
            conn.commit()

    # ── Research Threads ──────────────────────────────────────────────────────

    def get_thread(self, thread_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM research_threads WHERE id = ?", (thread_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_threads(
        self, prof_id: Optional[str] = None, investigation_id: Optional[str] = None
    ) -> list[dict]:
        with self._lock, self._connect() as conn:
            if prof_id and investigation_id:
                rows = conn.execute(
                    "SELECT data FROM research_threads WHERE prof_id = ? AND investigation_id = ? ORDER BY updated_at DESC",
                    (prof_id, investigation_id)
                ).fetchall()
            elif prof_id:
                rows = conn.execute(
                    "SELECT data FROM research_threads WHERE prof_id = ? ORDER BY updated_at DESC",
                    (prof_id,)
                ).fetchall()
            elif investigation_id:
                rows = conn.execute(
                    "SELECT data FROM research_threads WHERE investigation_id = ? ORDER BY updated_at DESC",
                    (investigation_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM research_threads ORDER BY updated_at DESC"
                ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def save_thread(self, thread: dict):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_threads "
                "(id, prof_id, investigation_id, status, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM research_threads WHERE id=?), ?), ?)",
                (
                    thread["id"],
                    thread["professor_id"],
                    thread["investigation_id"],
                    thread["status"],
                    json.dumps(thread, ensure_ascii=False),
                    thread["id"], now,
                    now,
                )
            )
            conn.commit()

    def delete_thread(self, thread_id: str):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM research_threads WHERE id = ?", (thread_id,))
            conn.execute("DELETE FROM research_interactions WHERE thread_id = ?", (thread_id,))
            conn.commit()

    # ── Research Interactions ─────────────────────────────────────────────────

    def get_interaction(self, interaction_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM research_interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_interactions(
        self,
        thread_id: Optional[str] = None,
        prof_id: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> list[dict]:
        query = "SELECT data FROM research_interactions WHERE 1=1"
        params = []
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        if prof_id:
            query += " AND prof_id = ?"
            params.append(prof_id)
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        query += " ORDER BY interaction_number ASC, created_at ASC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def save_interaction(self, interaction: dict):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_interactions "
                "(id, thread_id, prof_id, investigation_id, blocker_id, interaction_number, status, follow_up_date, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM research_interactions WHERE id=?), ?), ?)",
                (
                    interaction["id"],
                    interaction["thread_id"],
                    interaction["prof_id"],
                    interaction["investigation_id"],
                    interaction["blocker_id"],
                    interaction["interaction_number"],
                    interaction["status"],
                    interaction.get("follow_up_date"),
                    json.dumps(interaction, ensure_ascii=False),
                    interaction["id"], now,
                    now,
                )
            )
            conn.commit()

    def delete_interaction(self, interaction_id: str):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM research_interactions WHERE id = ?", (interaction_id,))
            conn.commit()

    # ── Research Outcomes ─────────────────────────────────────────────────────

    def get_outcome(self, outcome_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM research_outcomes WHERE id = ?", (outcome_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_outcomes(
        self, prof_id: Optional[str] = None, investigation_id: Optional[str] = None
    ) -> list[dict]:
        query = "SELECT data FROM research_outcomes WHERE 1=1"
        params = []
        if prof_id:
            query += " AND prof_id = ?"
            params.append(prof_id)
        if investigation_id:
            query += " AND investigation_id = ?"
            params.append(investigation_id)
        query += " ORDER BY created_at DESC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def save_outcome(self, outcome: dict):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_outcomes "
                "(id, investigation_id, prof_id, data, created_at) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM research_outcomes WHERE id=?), ?))",
                (
                    outcome["id"],
                    outcome["investigation_id"],
                    outcome["professor_id"],
                    json.dumps(outcome, ensure_ascii=False),
                    outcome["id"], now,
                )
            )
            conn.commit()

    def delete_outcome(self, outcome_id: str):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM research_outcomes WHERE id = ?", (outcome_id,))
            conn.commit()

    # ── Maintenance ───────────────────────────────────────────────────────────

    def evict_stale(self):
        """Delete expired entries from tables with TTLs."""
        now = time.time()
        with self._lock, self._connect() as conn:
            for table, ttl in [
                ("paper_analyses",     self.TTL_PAPER),
                ("paper_abstracts",    self.TTL_PAPER),
                ("social_profiles",    self.TTL_SOCIAL),
                ("resolved_profiles",  self.TTL_PROFILE),
                ("search_results",     self.TTL_SEARCH),
                ("image_descriptions", self.TTL_PAPER),
                ("self_model",         self.TTL_SELF_MODEL),
                ("collaboration_scores", self.TTL_COLLAB_SCORE),
                # investigation_streams and professor_memory: never evicted
            ]:
                conn.execute(
                    f"DELETE FROM {table} WHERE created_at < ?",
                    (now - ttl,)
                )
            conn.commit()
        logger.info("Stale cache entries evicted.")

    def stats(self) -> dict:
        """Return table row counts for debugging."""
        tables = [
            "paper_analyses", "paper_abstracts", "social_profiles",
            "resolved_profiles", "search_results", "image_descriptions",
            "self_model", "investigation_streams",
            "collaboration_scores", "professor_memory", "crawled_faculty",
        ]
        result = {}
        with self._lock, self._connect() as conn:
            for t in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                result[t] = count
        return result

    # ── Crawled Faculty ───────────────────────────────────────────────────────

    def save_crawled_faculty(self, university: str, university_alias: str, faculty_list: list[dict]):
        """Save complete crawled faculty list from OpenAlex to database."""
        with self._lock, self._connect() as conn:
            for f in faculty_list:
                openalex_id = f.get("id")
                name = f.get("display_name", "Unknown")
                h_index = f.get("summary_stats", {}).get("h_index") or f.get("h_index") or 0
                works_count = f.get("works_count") or 0
                
                # Extract and normalize concepts
                x_concepts = f.get("x_concepts", [])
                concept_ids = []
                for c in x_concepts:
                    cid = c.get("id")
                    if cid:
                        cid_clean = cid.split("/")[-1]
                        if not cid_clean.startswith("C") and cid_clean.isdigit():
                            cid_clean = "C" + cid_clean
                        concept_ids.append(cid_clean)
                concepts_str = ",".join(concept_ids) if concept_ids else None
                
                conn.execute(
                    "INSERT OR REPLACE INTO crawled_faculty (openalex_id, university, university_alias, name, h_index, works_count, concepts, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (openalex_id, university, university_alias, name, h_index, works_count, concepts_str, time.time())
                )
            conn.commit()

    def get_crawled_faculty_count(self, name_or_alias: str) -> int:
        """Return total crawled faculty count for a university name or its alias."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM crawled_faculty WHERE LOWER(university) = ? OR LOWER(university_alias) = ?",
                (name_or_alias.strip().lower(), name_or_alias.strip().lower())
            ).fetchone()
            return row["cnt"] if row else 0

    def list_crawled_faculty(self, name_or_alias: str) -> list[dict]:
        """List crawled faculty profiles for a university name or its alias sorted by h-index."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT openalex_id, university, university_alias, name, h_index, works_count FROM crawled_faculty "
                "WHERE LOWER(university) = ? OR LOWER(university_alias) = ? ORDER BY h_index DESC",
                (name_or_alias.strip().lower(), name_or_alias.strip().lower())
            ).fetchall()
            return [dict(r) for r in rows]

    def get_screened_faculty_count(self, name_or_alias: str, concept_ids: list[str]) -> int:
        """Count how many crawled researchers match the target concepts."""
        if not concept_ids:
            return 0
        clauses = []
        params = []
        for cid in concept_ids:
            clauses.append("concepts LIKE ?")
            params.append(f"%{cid}%")
        
        like_query = " OR ".join(clauses)
        query = (
            "SELECT COUNT(*) as cnt FROM crawled_faculty "
            f"WHERE (LOWER(university) = ? OR LOWER(university_alias) = ?) AND ({like_query})"
        )
        
        full_params = [name_or_alias.strip().lower(), name_or_alias.strip().lower()] + params
        with self._lock, self._connect() as conn:
            row = conn.execute(query, tuple(full_params)).fetchone()
            return row["cnt"] if row else 0

    def get_scored_faculty_count(self, university: str, university_alias: str) -> int:
        """Count how many professors at this university have collaboration scores."""
        count = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT data FROM collaboration_scores").fetchall()
            for r in rows:
                try:
                    data = json.loads(r["data"])
                    inst = data.get("institution", "").strip().lower()
                    if inst and (inst == university.strip().lower() or inst == university_alias.strip().lower()):
                        count += 1
                except Exception:
                    pass
        return count

    def get_resolved_profile_count(self, university: str, university_alias: str) -> int:
        """Count how many profiles are in resolved_profiles for this university."""
        count = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT data FROM resolved_profiles").fetchall()
            for r in rows:
                try:
                    data = json.loads(r["data"])
                    inst = data.get("affiliation", "").strip().lower()
                    if inst and (inst == university.strip().lower() or inst == university_alias.strip().lower()):
                        count += 1
                except Exception:
                    pass
        return count

