"""
investigation_schemas.py — Core dataclasses for the Research Opportunity Discovery System.

Object hierarchy:
    InvestigationStream
        ├── Contradiction  (severity × confidence → priority escalation)
        └── ResearchBlocker  (specificity_score quality gate → ENRS input)

ProfessorMemory + MemoryEvent live here too (used by research_memory.py).

All objects are JSON-serializable via .to_dict() / cls.from_dict().
Embeddings (list[float]) are stored inline; numpy arrays are never pickled.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Contradiction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Contradiction:
    """
    Two claims that are in tension within an investigation.

    Priority escalation uses severity × confidence, not severity alone.
    A severity=0.95, confidence=0.30 contradiction (i.e. "potentially important
    but weak evidence") should NOT dominate the investigation queue.

    Thresholds:
        signal > 0.81  →  priority += 0.50  (high confidence critical contradiction)
        signal > 0.63  →  priority += 0.30  (moderate confidence major contradiction)
    """
    id:                    str
    claim_yours:           str   # "Our model shows X is learnable online"
    claim_external:        str   # "Chen et al. 2023 show X requires offline curriculum"
    source_yours:          str   # your paper title / "EarthOS design doc" / "our work"
    source_external:       str   # doi or "Author et al. YEAR, paper title"
    resolution_hypothesis: str   # "Difference may be scale-dependent"
    severity:              float  # 0.0–1.0: how important if the contradiction is real
    confidence:            float  # 0.0–1.0: how sure we are the contradiction is genuine
    contradiction_type:    Literal["empirical", "theoretical", "definitional", "scope"]
    status:                Literal["open", "narrowed", "resolved"]
    created_at:            float = field(default_factory=time.time)

    @property
    def effective_signal(self) -> float:
        """severity × confidence — the actual escalation driver."""
        return self.severity * self.confidence

    @classmethod
    def new(
        cls,
        claim_yours: str,
        claim_external: str,
        source_yours: str,
        source_external: str,
        resolution_hypothesis: str,
        severity: float,
        confidence: float,
        contradiction_type: Literal["empirical", "theoretical", "definitional", "scope"] = "empirical",
    ) -> "Contradiction":
        return cls(
            id=str(uuid.uuid4()),
            claim_yours=claim_yours,
            claim_external=claim_external,
            source_yours=source_yours,
            source_external=source_external,
            resolution_hypothesis=resolution_hypothesis,
            severity=max(0.0, min(1.0, severity)),
            confidence=max(0.0, min(1.0, confidence)),
            contradiction_type=contradiction_type,
            status="open",
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim_yours": self.claim_yours,
            "claim_external": self.claim_external,
            "source_yours": self.source_yours,
            "source_external": self.source_external,
            "resolution_hypothesis": self.resolution_hypothesis,
            "severity": self.severity,
            "confidence": self.confidence,
            "contradiction_type": self.contradiction_type,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Contradiction":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────────────
# ResearchBlocker
# ──────────────────────────────────────────────────────────────────────────────

BLOCKER_TYPE = Literal[
    "dataset",          # need data they collected / curated
    "methodology",      # need a technique or protocol they developed
    "domain_knowledge", # need expertise we lack (clinical, biological, etc.)
    "computational",    # need infrastructure or scale we don't have
    "validation",       # need them to run experiments validating our claims
    "theoretical",      # need a formal framework, proof, or theory
]

EVIDENCE_STRENGTH = Literal[
    "paper_mention",    # referenced in passing — weakest
    "pilot_result",     # small-scale preliminary result
    "replicated_result",# independently reproduced
    "published_result", # peer-reviewed, published — strongest
]

# Multipliers applied to ENRS when a professor's paper matches a blocker
BLOCKER_TYPE_MULTIPLIERS: dict[str, float] = {
    "dataset":          1.20,
    "domain_knowledge": 1.15,
    "validation":       1.10,
    "methodology":      1.00,
    "theoretical":      1.00,
    "computational":    0.90,
}

# Minimum specificity_score to enter the ENRS computation
SPECIFICITY_THRESHOLD = 0.40


@dataclass
class ResearchBlocker:
    """
    A specific piece of evidence or capability that is blocking an investigation.

    The whole ENRS architecture depends on blocker quality.
    Blockers with specificity_score < SPECIFICITY_THRESHOLD are rejected at
    input — the user is prompted to refine before they enter the pipeline.

    effective_priority = severity × specificity_score
    This ensures vague blockers (high severity, low specificity) don't
    dominate professor ranking with a weak signal.
    """
    id:               str
    description:      str             # must be specific — validated on creation
    blocker_type:     str             # one of BLOCKER_TYPE
    severity:         float           # 0.0–1.0: how badly this blocks the investigation
    specificity_score: float          # 0.0–1.0: auto-computed by specificity_validator
    resolved:         bool = False
    resolved_by:      Optional[str] = None    # professor name or paper doi
    resolved_at:      Optional[float] = None
    evidence_strength: Optional[str] = None   # set when evidence is recorded
    embedding:        Optional[list[float]] = None  # 384-dim, computed on creation
    required_method:  str = "N/A"
    required_dataset: str = "N/A"
    required_metric:  str = "N/A"
    created_at:       float = field(default_factory=time.time)

    @property
    def effective_priority(self) -> float:
        """Only high-quality blockers get full weight in ENRS."""
        return self.severity * self.specificity_score

    @property
    def is_eligible_for_enrs(self) -> bool:
        """Blocker must meet quality threshold to enter scoring."""
        return self.specificity_score >= SPECIFICITY_THRESHOLD and not self.resolved

    @classmethod
    def new(
        cls,
        description: str,
        blocker_type: str,
        severity: float,
        specificity_score: float,
        embedding: Optional[list[float]] = None,
        required_method: str = "N/A",
        required_dataset: str = "N/A",
        required_metric: str = "N/A",
    ) -> "ResearchBlocker":
        if required_method == "N/A" and required_dataset == "N/A" and required_metric == "N/A":
            try:
                from academic_pipeline.local_llm import LocalLLMClient
                client = LocalLLMClient()
                reqs = client.extract_blocker_requirements_sync(description)
                required_method = reqs.get("required_method", "N/A")
                required_dataset = reqs.get("required_dataset", "N/A")
                required_metric = reqs.get("required_metric", "N/A")
            except Exception:
                pass
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            blocker_type=blocker_type,
            severity=max(0.0, min(1.0, severity)),
            specificity_score=max(0.0, min(1.0, specificity_score)),
            embedding=embedding,
            required_method=required_method,
            required_dataset=required_dataset,
            required_metric=required_metric,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "blocker_type": self.blocker_type,
            "severity": self.severity,
            "specificity_score": self.specificity_score,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "evidence_strength": self.evidence_strength,
            "embedding": self.embedding,
            "required_method": self.required_method,
            "required_dataset": self.required_dataset,
            "required_metric": self.required_metric,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchBlocker":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────────────
# InvestigationStream
# ──────────────────────────────────────────────────────────────────────────────

INVESTIGATION_STATUS = Literal["active", "blocked", "speculative", "queued", "resolved"]

# Hard limit on simultaneous active/blocked investigations.
# Not configurable — can be relaxed after the system is proven in practice.
MAX_ACTIVE_INVESTIGATIONS = 5


@dataclass
class InvestigationStream:
    """
    An open research investigation the user is actively pursuing.

    This is the primary unit of collaboration matching.
    A professor is scored against each active investigation's blockers,
    not against the researcher's identity as a whole.

    Status transitions:
        speculative → queued → active ↔ blocked → resolved
    Only "active" and "blocked" investigations enter ENRS scoring.
    MAX_ACTIVE_INVESTIGATIONS = 5 is enforced by InvestigationEngine.
    """
    id:                str
    title:             str
    description:       str
    status:            str   # INVESTIGATION_STATUS
    base_priority:     float                     # user-set, 0.0–1.0
    effective_priority: float                    # computed by _compute_effective_priority
    contradictions:    list[Contradiction] = field(default_factory=list)
    blockers:          list[ResearchBlocker] = field(default_factory=list)
    source_papers:     list[str] = field(default_factory=list)
    related_topics:    list[str] = field(default_factory=list)
    created_at:        float = field(default_factory=time.time)
    updated_at:        float = field(default_factory=time.time)
    resolved_at:       Optional[float] = None
    resolution_notes:  Optional[str] = None

    @classmethod
    def new(
        cls,
        title: str,
        description: str,
        base_priority: float,
        source_papers: Optional[list[str]] = None,
        related_topics: Optional[list[str]] = None,
        status: str = "queued",
    ) -> "InvestigationStream":
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            status=status,
            base_priority=max(0.0, min(1.0, base_priority)),
            effective_priority=max(0.0, min(1.0, base_priority)),
            source_papers=source_papers or [],
            related_topics=related_topics or [],
        )

    def compute_effective_priority(self) -> float:
        """
        Recompute effective_priority from base + contradiction escalation.
        Uses severity × confidence (not severity alone) to avoid false escalation.

        Thresholds:
            signal > 0.81  →  +0.50
            signal > 0.63  →  +0.30
        """
        base = self.base_priority
        max_signal = max(
            (c.effective_signal for c in self.contradictions if c.status == "open"),
            default=0.0,
        )
        if max_signal > 0.81:
            return min(1.0, base + 0.50)
        elif max_signal > 0.63:
            return min(1.0, base + 0.30)
        return base

    def refresh_priority(self):
        """Update effective_priority in-place. Call after any contradiction change."""
        self.effective_priority = self.compute_effective_priority()
        self.updated_at = time.time()

    @property
    def open_blockers(self) -> list[ResearchBlocker]:
        """Blockers that are unresolved and eligible for ENRS."""
        return [b for b in self.blockers if b.is_eligible_for_enrs]

    @property
    def is_scoreable(self) -> bool:
        """Only active and blocked investigations with eligible blockers enter ENRS."""
        return (
            self.status in ("active", "blocked")
            and len(self.open_blockers) > 0
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "base_priority": self.base_priority,
            "effective_priority": self.effective_priority,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "blockers": [b.to_dict() for b in self.blockers],
            "source_papers": self.source_papers,
            "related_topics": self.related_topics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "resolution_notes": self.resolution_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InvestigationStream":
        contradictions = [
            Contradiction.from_dict(c) for c in d.get("contradictions", [])
        ]
        blockers = [
            ResearchBlocker.from_dict(b) for b in d.get("blockers", [])
        ]
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            status=d["status"],
            base_priority=d["base_priority"],
            effective_priority=d.get("effective_priority", d["base_priority"]),
            contradictions=contradictions,
            blockers=blockers,
            source_papers=d.get("source_papers", []),
            related_topics=d.get("related_topics", []),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            resolved_at=d.get("resolved_at"),
            resolution_notes=d.get("resolution_notes"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Research Memory objects
# ──────────────────────────────────────────────────────────────────────────────

MEMORY_EVENT_TYPE = Literal[
    "dossier_viewed",
    "score_computed",
    "email_drafted",
    "email_sent",
    "new_paper_found",
    "blocker_resolution_suggested",  # system suggests, user must confirm
    "blocker_resolved",              # user confirmed resolution
    "contradiction_flagged",
    "evidence_recorded",
]

PROFESSOR_STATUS = Literal[
    "discovered",
    "contacted",
    "in_conversation",
    "not_relevant",
    "resolved",
]


@dataclass
class MemoryEvent:
    timestamp:           float = field(default_factory=time.time)
    event_type:          str = "dossier_viewed"   # MEMORY_EVENT_TYPE
    notes:               str = ""
    investigation_id:    Optional[str] = None
    blocker_id:          Optional[str] = None
    related_paper_doi:   Optional[str] = None
    evidence_strength:   Optional[str] = None     # EVIDENCE_STRENGTH if applicable
    impact_score:        Optional[float] = None
    reply_quality:       Optional[str] = None     # no_reply | short_reply | technical_reply | advice_given | meeting_offered | collaboration_offer

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "notes": self.notes,
            "investigation_id": self.investigation_id,
            "blocker_id": self.blocker_id,
            "related_paper_doi": self.related_paper_doi,
            "evidence_strength": self.evidence_strength,
            "impact_score": self.impact_score,
            "reply_quality": self.reply_quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProfessorMemory:
    """
    Long-horizon persistent record of all interactions with one professor.
    Never expires — the value compounds over years of use.
    """
    professor_name:             str
    openalex_id:                Optional[str]
    institution:                str
    first_seen_at:              float = field(default_factory=time.time)
    last_updated_at:            float = field(default_factory=time.time)
    last_score:                 Optional[float] = None
    status:                     str = "discovered"   # PROFESSOR_STATUS
    events:                     list[MemoryEvent] = field(default_factory=list)
    resolved_blocker_ids:       list[str] = field(default_factory=list)
    contradictions_introduced:  list[str] = field(default_factory=list)
    papers_at_last_check:       list[str] = field(default_factory=list)  # doi list
    questions_worth_asking:     list[str] = field(default_factory=list)
    profile_version:            int = 1
    last_refresh_date:          Optional[str] = None
    reply_count:                int = 0
    outcome_count:              int = 0
    interaction_count:          int = 0
    last_response_date:         Optional[str] = None
    h_index:                    int = 0
    career_stage:               str = "unknown"
    last_publication_year:      Optional[int] = None
    student_supervision_signal: float = 0.0

    @classmethod
    def new(
        cls,
        professor_name: str,
        institution: str,
        openalex_id: Optional[str] = None,
    ) -> "ProfessorMemory":
        return cls(
            professor_name=professor_name,
            openalex_id=openalex_id,
            institution=institution,
        )

    def add_event(self, event: MemoryEvent):
        self.events.append(event)
        self.last_updated_at = time.time()

    @property
    def prof_id(self) -> str:
        """Stable key for SQLite storage."""
        if self.openalex_id:
            return self.openalex_id
        import hashlib
        return "name_" + hashlib.sha256(
            self.professor_name.lower().encode()
        ).hexdigest()[:16]

    def get_relationship_strength(self) -> float:
        """
        Computes relationship strength dynamically based on reply quality and response metrics.

        If any MemoryEvent has a reply_quality field, uses the new classification model:
          no_reply          → +0
          short_reply       → +1
          technical_reply   → +4
          advice_given      → +6
          meeting_offered   → +10
          collaboration_offer → +15
        Total reply score is scaled: reply_contrib = total * 0.04 (capped at 1.0).

        Falls back to legacy reply_count * 0.4 model if no events have reply_quality
        (backward-compatible with existing databases).
        """
        if self.status == "ignored":
            return 0.0

        QUALITY_SCORES = {
            "no_reply": 0,
            "short_reply": 1,
            "technical_reply": 4,
            "advice_given": 6,
            "meeting_offered": 10,
            "collaboration_offer": 15,
        }

        quality_events = [e for e in self.events if e.reply_quality]
        if quality_events:
            # New reply-quality model
            total_reply_score = sum(
                QUALITY_SCORES.get(e.reply_quality, 0) for e in quality_events
            )
            reply_contrib = min(1.0, total_reply_score * 0.04)
            base = min(1.0, reply_contrib + (self.outcome_count * 0.3) + (self.interaction_count * 0.1))
        else:
            # Legacy fallback
            base = min(1.0, (self.reply_count * 0.4) + (self.outcome_count * 0.3) + (self.interaction_count * 0.1))

        if self.status == "dormant":
            base *= 0.5

        return round(base, 2)

    def get_research_influence_score(self) -> float:
        """
        Computes research influence dynamically based on tested outcomes.
        Sum of impact scores of all tested outcomes (evidence_recorded events).
        """
        score = 0.0
        for e in self.events:
            if e.event_type == "evidence_recorded" and e.impact_score is not None:
                score += e.impact_score
        return round(score, 2)

    def get_collaboration_readiness_score(self, lab_accessibility: float = 0.5) -> float:
        """
        Computes the collaboration readiness score dynamically.
        Factors:
          - CRM status: in_conversation (+0.35), contacted (+0.10), resolved (+0.50)
          - reply_count: +0.10 per reply (max 0.20)
          - outcome_count: +0.15 per outcome (max 0.30)
          - lab_accessibility: scale dynamic score by 0.5 + (lab_accessibility * 0.5)
          - dormant status: score is halved
        """
        if self.status in ("ignored", "not_relevant"):
            return 0.0
            
        score = 0.05  # base
        if self.status == "contacted":
            score += 0.10
        elif self.status == "in_conversation":
            score += 0.35
        elif self.status == "resolved":
            score += 0.50
            
        score += min(0.20, self.reply_count * 0.10)
        score += min(0.30, self.outcome_count * 0.15)
        
        # Scale based on lab accessibility
        accessibility_factor = 0.5 + (lab_accessibility * 0.5)
        score *= accessibility_factor
        
        if self.status == "dormant":
            score *= 0.5
            
        return round(min(1.0, score), 2)

    def to_dict(self) -> dict:
        return {
            "professor_name": self.professor_name,
            "openalex_id": self.openalex_id,
            "institution": self.institution,
            "first_seen_at": self.first_seen_at,
            "last_updated_at": self.last_updated_at,
            "last_score": self.last_score,
            "status": self.status,
            "events": [e.to_dict() for e in self.events],
            "resolved_blocker_ids": self.resolved_blocker_ids,
            "contradictions_introduced": self.contradictions_introduced,
            "papers_at_last_check": self.papers_at_last_check,
            "questions_worth_asking": self.questions_worth_asking,
            "profile_version": self.profile_version,
            "last_refresh_date": self.last_refresh_date,
            "reply_count": self.reply_count,
            "outcome_count": self.outcome_count,
            "interaction_count": self.interaction_count,
            "last_response_date": self.last_response_date,
            "h_index": self.h_index,
            "career_stage": self.career_stage,
            "last_publication_year": self.last_publication_year,
            "student_supervision_signal": self.student_supervision_signal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProfessorMemory":
        events = [MemoryEvent.from_dict(e) for e in d.get("events", [])]
        return cls(
            professor_name=d["professor_name"],
            openalex_id=d.get("openalex_id"),
            institution=d.get("institution", ""),
            first_seen_at=d.get("first_seen_at", time.time()),
            last_updated_at=d.get("last_updated_at", time.time()),
            last_score=d.get("last_score"),
            status=d.get("status", "discovered"),
            events=events,
            resolved_blocker_ids=d.get("resolved_blocker_ids", []),
            contradictions_introduced=d.get("contradictions_introduced", []),
            papers_at_last_check=d.get("papers_at_last_check", []),
            questions_worth_asking=d.get("questions_worth_asking", []),
            profile_version=d.get("profile_version", 1),
            last_refresh_date=d.get("last_refresh_date"),
            reply_count=d.get("reply_count", 0),
            outcome_count=d.get("outcome_count", 0),
            interaction_count=d.get("interaction_count", 0),
            last_response_date=d.get("last_response_date"),
            h_index=d.get("h_index", 0),
            career_stage=d.get("career_stage", "unknown"),
            last_publication_year=d.get("last_publication_year"),
            student_supervision_signal=d.get("student_supervision_signal", 0.0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Research CRM & Grouping Layer Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchThread:
    id: str                           # UUID
    professor_id: str                 # Link to professor
    investigation_id: str             # Link to investigation
    subject: str                      # Topic of discussion
    status: str = "active"            # active | dormant | ignored | completed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "professor_id": self.professor_id,
            "investigation_id": self.investigation_id,
            "subject": self.subject,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchThread":
        return cls(
            id=d["id"],
            professor_id=d["professor_id"],
            investigation_id=d["investigation_id"],
            subject=d["subject"],
            status=d.get("status", "active"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


@dataclass
class ResearchInteraction:
    id: str                           # UUID
    thread_id: str                    # Link to ResearchThread
    professor_name: str
    prof_id: str                      # OpenAlex ID or name hash
    investigation_id: str             # Link to investigation
    blocker_id: str                   # Link to ResearchBlocker
    interaction_number: int           # e.g., 1 for cold contact, 2 for reply
    question: str                     # The intellectual tension question
    response: Optional[str] = None    # The email response body
    evidence_gained: Optional[str] = None
    next_action: str = ""
    follow_up_date: Optional[str] = None # YYYY-MM-DD
    status: str = "drafted"           # drafted | sent | replied | action_taken | completed | ignored | dormant
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "professor_name": self.professor_name,
            "prof_id": self.prof_id,
            "investigation_id": self.investigation_id,
            "blocker_id": self.blocker_id,
            "interaction_number": self.interaction_number,
            "question": self.question,
            "response": self.response,
            "evidence_gained": self.evidence_gained,
            "next_action": self.next_action,
            "follow_up_date": self.follow_up_date,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchInteraction":
        return cls(
            id=d["id"],
            thread_id=d["thread_id"],
            professor_name=d["professor_name"],
            prof_id=d["prof_id"],
            investigation_id=d["investigation_id"],
            blocker_id=d["blocker_id"],
            interaction_number=d["interaction_number"],
            question=d["question"],
            response=d.get("response"),
            evidence_gained=d.get("evidence_gained"),
            next_action=d.get("next_action", ""),
            follow_up_date=d.get("follow_up_date"),
            status=d.get("status", "drafted"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


@dataclass
class ResearchOutcome:
    id: str                           # UUID
    investigation_id: str             # UUID
    professor_id: str
    suggestion: str                   # What they suggested
    contribution_type: str = "other"  # paper_recommendation | method_suggestion | etc.
    implementation_notes: str = ""
    impact_score: float = 0.0
    resulting_artifact: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "investigation_id": self.investigation_id,
            "professor_id": self.professor_id,
            "suggestion": self.suggestion,
            "contribution_type": self.contribution_type,
            "implementation_notes": self.implementation_notes,
            "impact_score": self.impact_score,
            "resulting_artifact": self.resulting_artifact,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchOutcome":
        return cls(
            id=d["id"],
            investigation_id=d["investigation_id"],
            professor_id=d["professor_id"],
            suggestion=d["suggestion"],
            contribution_type=d.get("contribution_type", "other"),
            implementation_notes=d.get("implementation_notes", ""),
            impact_score=d.get("impact_score", 0.0),
            resulting_artifact=d.get("resulting_artifact", ""),
            created_at=d.get("created_at", time.time()),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Seed investigations (pre-loaded on first run)
# ──────────────────────────────────────────────────────────────────────────────

def build_seed_investigations() -> list[InvestigationStream]:
    """
    Three pre-defined investigations to seed the system on first run.
    Blockers are high-quality (specificity pre-validated).
    Start in "queued" status — user activates up to MAX_ACTIVE_INVESTIGATIONS.
    """
    inv1 = InvestigationStream.new(
        title="EarthOS — Contradiction-Driven Inquiry Architecture",
        description=(
            "Building a persistent AI research system that identifies internal "
            "contradictions in its own belief state, generates specific evidence "
            "needs to resolve them, and actively searches for researchers who can "
            "satisfy those needs. Core challenge: how to represent and prioritize "
            "open investigations across long timescales without context collapse."
        ),
        base_priority=0.92,
        status="queued",
        related_topics=["cognitive architectures", "active inference", "continual learning",
                        "research agents", "memory systems"],
    )
    inv1.blockers = [
        ResearchBlocker.new(
            description=(
                "Empirical evidence from existing long-horizon agent systems showing "
                "how memory architecture choice (episodic vs. semantic vs. working) "
                "affects investigation persistence and coherence over months-long timescales"
            ),
            blocker_type="domain_knowledge",
            severity=0.85,
            specificity_score=0.82,
        ),
        ResearchBlocker.new(
            description=(
                "Validated benchmark for measuring cognitive coherence in continual "
                "learning agents across investigation cycles — not task performance, "
                "but consistency of belief updates when new contradicting evidence arrives"
            ),
            blocker_type="methodology",
            severity=0.80,
            specificity_score=0.85,
        ),
    ]

    inv2 = InvestigationStream.new(
        title="Persistent Investigation Streams in Research Agents",
        description=(
            "Developing the architecture for AI agents that maintain active research "
            "investigations across sessions, updating evidence maps and contradiction "
            "registers as new information arrives. Key open question: what data structure "
            "best represents an 'open investigation' that can be updated incrementally "
            "without catastrophic interference?"
        ),
        base_priority=0.85,
        status="queued",
        related_topics=["long-horizon planning", "memory architectures", "active research agents",
                        "knowledge graphs", "continual learning"],
    )
    inv2.blockers = [
        ResearchBlocker.new(
            description=(
                "Survey or comparative study of long-horizon agent architectures "
                "showing which memory representations scale to year-long investigations "
                "without catastrophic interference or unbounded memory growth"
            ),
            blocker_type="domain_knowledge",
            severity=0.80,
            specificity_score=0.78,
        ),
    ]

    inv3 = InvestigationStream.new(
        title="TARS — Scientific ML for Exoplanet Discovery",
        description=(
            "Applying differentiable simulation and active inquiry methods to astrophysics "
            "data pipelines for automated hypothesis generation and validation in exoplanet "
            "research. Key question: can ML methods match or exceed established spectroscopic "
            "methods for exoplanet characterization, and what evaluation protocol would "
            "make that comparison credible?"
        ),
        base_priority=0.75,
        status="queued",
        related_topics=["exoplanet science", "scientific ML", "astrophysics", "active inference",
                        "differentiable simulation"],
    )
    inv3.blockers = [
        ResearchBlocker.new(
            description=(
                "Access to validated exoplanet transit light-curve datasets with ground-truth "
                "labels for benchmarking ML classification performance against established "
                "spectroscopic characterization methods on the same observational data"
            ),
            blocker_type="dataset",
            severity=0.88,
            specificity_score=0.88,
        ),
        ResearchBlocker.new(
            description=(
                "Established evaluation protocol for comparing ML-based and physics-model-based "
                "exoplanet characterization on identical observational data — including "
                "uncertainty quantification and calibration metrics"
            ),
            blocker_type="methodology",
            severity=0.72,
            specificity_score=0.82,
        ),
    ]

    return [inv1, inv2, inv3]
