"""
research_graph.py — SQLite-backed NetworkX Graph Store for Research Program Graph.
"""

from __future__ import annotations

import os
import sqlite3
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class ClaimNode:
    id: str
    text: str
    type: str  # "ours" | "theirs"
    confidence: float
    paper_doi: str
    method: str
    dataset: str
    metric: str
    program_id: Optional[str] = None

@dataclass
class ContradictionLink:
    id: str
    my_claim_id: str
    their_claim_id: str
    type: str
    resolution_hypothesis: str
    severity: float
    confidence: float
    status: str

@dataclass
class EvidenceNeedNode:
    id: str
    blocker_id: str
    investigation_id: str
    description: str
    method: str
    dataset: str
    metric: str

@dataclass
class ResearchProgramNode:
    id: str
    title: str
    summary: str
    university: str
    professor_name: str
    last_refresh: float = 0.0

class ResearchProgramGraph:
    """
    Manages the Research Program Graph using NetworkX in-memory,
    persisting structure to outputs/research_program_graph.db.
    """

    def __init__(self, cache_dir: str = "outputs", db_name: str = "research_program_graph.db"):
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, db_name)
        self.graph = nx.DiGraph()
        self._init_db()
        self.load_graph_from_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS programs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    university TEXT NOT NULL,
                    professor_name TEXT NOT NULL,
                    last_refresh REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    paper_doi TEXT NOT NULL,
                    method TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    program_id TEXT,
                    FOREIGN KEY(program_id) REFERENCES programs(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contradictions (
                    id TEXT PRIMARY KEY,
                    my_claim_id TEXT NOT NULL,
                    their_claim_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    resolution_hypothesis TEXT NOT NULL,
                    severity REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY(my_claim_id) REFERENCES claims(id) ON DELETE CASCADE,
                    FOREIGN KEY(their_claim_id) REFERENCES claims(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_needs (
                    id TEXT PRIMARY KEY,
                    blocker_id TEXT NOT NULL,
                    investigation_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    method TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    metric TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_probability (
                    need_id TEXT NOT NULL,
                    program_id TEXT NOT NULL,
                    probability REAL NOT NULL,
                    rationale TEXT NOT NULL,
                    PRIMARY KEY(need_id, program_id),
                    FOREIGN KEY(need_id) REFERENCES evidence_needs(id) ON DELETE CASCADE,
                    FOREIGN KEY(program_id) REFERENCES programs(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def load_graph_from_db(self):
        """Sync the in-memory NetworkX DiGraph with SQLite state."""
        self.graph.clear()
        conn = self._connect()
        try:
            # 1. Load Programs
            rows = conn.execute("SELECT * FROM programs").fetchall()
            for r in rows:
                self.graph.add_node(
                    r["id"],
                    type="program",
                    title=r["title"],
                    summary=r["summary"],
                    university=r["university"],
                    professor_name=r["professor_name"],
                    last_refresh=r["last_refresh"]
                )

            # 2. Load Claims & link to programs
            rows = conn.execute("SELECT * FROM claims").fetchall()
            for r in rows:
                self.graph.add_node(
                    r["id"],
                    type="claim",
                    text=r["text"],
                    claim_type=r["type"],
                    confidence=r["confidence"],
                    paper_doi=r["paper_doi"],
                    method=r["method"],
                    dataset=r["dataset"],
                    metric=r["metric"]
                )
                if r["program_id"]:
                    self.graph.add_edge(r["program_id"], r["id"], relation="asserts")

            # 3. Load Contradictions & add edges between claims
            rows = conn.execute("SELECT * FROM contradictions").fetchall()
            for r in rows:
                self.graph.add_edge(
                    r["my_claim_id"],
                    r["their_claim_id"],
                    relation="contradicts",
                    id=r["id"],
                    type=r["type"],
                    resolution_hypothesis=r["resolution_hypothesis"],
                    severity=r["severity"],
                    confidence=r["confidence"],
                    status=r["status"]
                )

            # 4. Load Evidence Needs
            rows = conn.execute("SELECT * FROM evidence_needs").fetchall()
            for r in rows:
                self.graph.add_node(
                    r["id"],
                    type="evidence_need",
                    blocker_id=r["blocker_id"],
                    investigation_id=r["investigation_id"],
                    description=r["description"],
                    method=r["method"],
                    dataset=r["dataset"],
                    metric=r["metric"]
                )

            # 5. Load Evidence Probabilities
            rows = conn.execute("SELECT * FROM evidence_probability").fetchall()
            for r in rows:
                self.graph.add_edge(
                    r["need_id"],
                    r["program_id"],
                    relation="probability_match",
                    probability=r["probability"],
                    rationale=r["rationale"]
                )
        finally:
            conn.close()

    def save_program(self, program: ResearchProgramNode):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO programs (id, title, summary, university, professor_name, last_refresh)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (program.id, program.title, program.summary, program.university, program.professor_name, program.last_refresh)
            )
            conn.commit()
        finally:
            conn.close()
        self.graph.add_node(
            program.id,
            type="program",
            title=program.title,
            summary=program.summary,
            university=program.university,
            professor_name=program.professor_name,
            last_refresh=program.last_refresh
        )

    def save_claim(self, claim: ClaimNode):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO claims (id, text, type, confidence, paper_doi, method, dataset, metric, program_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (claim.id, claim.text, claim.type, claim.confidence, claim.paper_doi, claim.method, claim.dataset, claim.metric, claim.program_id)
            )
            conn.commit()
        finally:
            conn.close()
        self.graph.add_node(
            claim.id,
            type="claim",
            text=claim.text,
            claim_type=claim.type,
            confidence=claim.confidence,
            paper_doi=claim.paper_doi,
            method=claim.method,
            dataset=claim.dataset,
            metric=claim.metric
        )
        if claim.program_id:
            self.graph.add_edge(claim.program_id, claim.id, relation="asserts")

    def save_contradiction(self, link: ContradictionLink):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO contradictions (id, my_claim_id, their_claim_id, type, resolution_hypothesis, severity, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (link.id, link.my_claim_id, link.their_claim_id, link.type, link.resolution_hypothesis, link.severity, link.confidence, link.status)
            )
            conn.commit()
        finally:
            conn.close()
        self.graph.add_edge(
            link.my_claim_id,
            link.their_claim_id,
            relation="contradicts",
            id=link.id,
            type=link.type,
            resolution_hypothesis=link.resolution_hypothesis,
            severity=link.severity,
            confidence=link.confidence,
            status=link.status
        )

    def save_evidence_need(self, need: EvidenceNeedNode):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence_needs (id, blocker_id, investigation_id, description, method, dataset, metric)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (need.id, need.blocker_id, need.investigation_id, need.description, need.method, need.dataset, need.metric)
            )
            conn.commit()
        finally:
            conn.close()
        self.graph.add_node(
            need.id,
            type="evidence_need",
            blocker_id=need.blocker_id,
            investigation_id=need.investigation_id,
            description=need.description,
            method=need.method,
            dataset=need.dataset,
            metric=need.metric
        )

    def save_evidence_probability(self, need_id: str, program_id: str, probability: float, rationale: str):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence_probability (need_id, program_id, probability, rationale)
                VALUES (?, ?, ?, ?)
                """,
                (need_id, program_id, probability, rationale)
            )
            conn.commit()
        finally:
            conn.close()
        self.graph.add_edge(
            need_id,
            program_id,
            relation="probability_match",
            probability=probability,
            rationale=rationale
        )

    def get_contradictions(self) -> List[Dict]:
        """Get all contradictions as dicts."""
        self.load_graph_from_db()
        results = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "contradicts":
                my_claim = self.graph.nodes[u]
                their_claim = self.graph.nodes[v]
                results.append({
                    "id": data["id"],
                    "my_claim": my_claim["text"],
                    "my_claim_doi": my_claim["paper_doi"],
                    "their_claim": their_claim["text"],
                    "their_claim_doi": their_claim["paper_doi"],
                    "type": data["type"],
                    "resolution_hypothesis": data["resolution_hypothesis"],
                    "severity": data["severity"],
                    "confidence": data["confidence"],
                    "status": data["status"]
                })
        return results
