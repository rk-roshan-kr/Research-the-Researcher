"""
investigation_engine.py — Lifecycle manager for InvestigationStream objects.

Build order (per architecture approval):
    1. activate/queue/update_priorities  — MAX_ACTIVE=5 enforcement
    2. add_blocker/resolve_blocker       — quality-gated blocker management
    3. add_contradiction                 — contradiction + priority escalation
    4. suggest_blockers                  — LLM-assisted (on demand, user approves)
    5. detect_contradictions_from_dossier — hardest, left last

MAX_ACTIVE_INVESTIGATIONS = 5 is hardcoded. Not configurable.
Can be relaxed after the system is proven in practice.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

from academic_pipeline.investigation_schemas import (
    Contradiction,
    InvestigationStream,
    MemoryEvent,
    ProfessorMemory,
    ResearchBlocker,
    MAX_ACTIVE_INVESTIGATIONS,
)
from academic_pipeline.specificity_validator import (
    validate_blocker_specificity,
    format_rejection_message,
)
from academic_pipeline.cache_manager import CacheManager


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class BlockerTooVagueError(ValueError):
    """
    Raised when a ResearchBlocker description fails the specificity gate.
    Contains the rejection message and refinement hints.
    """
    def __init__(self, description: str, refinement_message: str):
        self.description = description
        self.refinement_message = refinement_message
        super().__init__(refinement_message)


class ActiveSlotsFullError(RuntimeError):
    """
    Raised when activate_investigation is called and all 5 active slots are
    occupied by equally- or higher-priority investigations.
    """
    def __init__(self, requested_priority: float, lowest_active_priority: float):
        self.requested_priority = requested_priority
        self.lowest_active_priority = lowest_active_priority
        super().__init__(
            f"All {MAX_ACTIVE_INVESTIGATIONS} active slots are occupied by "
            f"higher-priority investigations "
            f"(lowest active: {lowest_active_priority:.2f}, "
            f"requested: {requested_priority:.2f}). "
            f"Resolve or queue an existing investigation to make room."
        )


# ──────────────────────────────────────────────────────────────────────────────
# InvestigationEngine
# ──────────────────────────────────────────────────────────────────────────────

class InvestigationEngine:
    """
    Manages the full lifecycle of InvestigationStream objects.

    All state is persisted to SQLite via CacheManager.
    In-process objects are not cached — always loaded fresh from DB.
    This keeps the engine stateless across Flask requests and CLI calls.

    Thread-safety: inherited from CacheManager's Lock.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        ollama_model: str = "qwen2.5-coder:7b",
        ollama_host: str = "http://localhost:11434",
    ):
        self._cache = cache_manager
        self._ollama_model = ollama_model
        self._ollama_host = ollama_host
        self._llm = None  # lazy-loaded on first LLM call

    # ── Step 1: Activate / Queue / Priority ───────────────────────────────────

    def create_investigation(
        self,
        title: str,
        description: str,
        base_priority: float,
        source_papers: Optional[list[str]] = None,
        related_topics: Optional[list[str]] = None,
        auto_activate: bool = False,
    ) -> InvestigationStream:
        """
        Creates a new InvestigationStream and persists it.

        Default status is "queued". Set auto_activate=True to attempt
        immediate activation (subject to MAX_ACTIVE enforcement).

        Returns the created investigation (with final status).
        """
        inv = InvestigationStream.new(
            title=title,
            description=description,
            base_priority=base_priority,
            source_papers=source_papers,
            related_topics=related_topics,
            status="queued",
        )
        self._cache.save_investigation(inv.to_dict())
        logger.info(f"Created investigation '{title}' [{inv.id[:8]}] → queued")

        if auto_activate:
            try:
                self.activate_investigation(inv.id)
            except ActiveSlotsFullError as e:
                logger.info(f"Auto-activate failed: {e}. Investigation stays queued.")

        return self._load(inv.id)

    def activate_investigation(self, inv_id: str) -> InvestigationStream:
        """
        Moves a queued investigation to 'active'.

        Enforcement rules (MAX_ACTIVE = 5):
        1. Count current active + blocked investigations.
        2. If count < MAX_ACTIVE: activate immediately.
        3. If count == MAX_ACTIVE:
           a. Find the active/blocked investigation with the lowest effective_priority.
           b. If new investigation's priority > lowest active: demote lowest → queued,
              activate new one.
           c. If new investigation's priority <= lowest: raise ActiveSlotsFullError.

        Raises:
            ValueError: if investigation not found or already active/resolved.
            ActiveSlotsFullError: if slots full and priority too low.
        """
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")
        if inv.status in ("active", "blocked"):
            logger.info(f"Investigation '{inv.title}' is already active.")
            return inv
        if inv.status == "resolved":
            raise ValueError(f"Cannot activate a resolved investigation.")

        active_invs = self._load_all(status_filter=None)
        active = [i for i in active_invs if i.status in ("active", "blocked")]

        if len(active) < MAX_ACTIVE_INVESTIGATIONS:
            inv.status = "active"
            inv.updated_at = time.time()
            self._save(inv)
            logger.info(
                f"Activated '{inv.title}' "
                f"[{len(active) + 1}/{MAX_ACTIVE_INVESTIGATIONS} slots]"
            )
            return inv

        # Slots full — check if we can displace the lowest-priority active
        lowest = min(active, key=lambda i: i.effective_priority)
        if inv.effective_priority > lowest.effective_priority:
            logger.info(
                f"Displacing '{lowest.title}' "
                f"(priority {lowest.effective_priority:.2f}) → queued"
            )
            lowest.status = "queued"
            lowest.updated_at = time.time()
            self._save(lowest)

            inv.status = "active"
            inv.updated_at = time.time()
            self._save(inv)
            logger.info(f"Activated '{inv.title}' (displaced lower-priority investigation)")
            return inv

        raise ActiveSlotsFullError(inv.effective_priority, lowest.effective_priority)

    def queue_investigation(self, inv_id: str) -> InvestigationStream:
        """
        Moves an active/blocked investigation back to 'queued'.
        Frees up an active slot.
        """
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")
        if inv.status == "resolved":
            raise ValueError("Cannot queue a resolved investigation.")
        inv.status = "queued"
        inv.updated_at = time.time()
        self._save(inv)
        logger.info(f"Queued '{inv.title}'")
        return inv

    def resolve_investigation(
        self, inv_id: str, resolution_notes: str = ""
    ) -> InvestigationStream:
        """Marks an investigation as resolved. Frees an active slot."""
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")
        inv.status = "resolved"
        inv.resolved_at = time.time()
        inv.resolution_notes = resolution_notes
        inv.updated_at = time.time()
        self._save(inv)
        logger.info(f"Resolved '{inv.title}'")
        return inv

    def update_priorities(self) -> list[InvestigationStream]:
        """
        Recomputes effective_priority for ALL investigations based on their
        current contradictions (severity × confidence).

        Called automatically after any contradiction add/update.
        May trigger automatic slot re-ordering: if a queued investigation's
        escalated priority exceeds an active one, the swap happens.

        Returns the updated list of all investigations (sorted by priority desc).
        """
        all_invs = self._load_all()

        # Recompute priorities
        for inv in all_invs:
            old_prio = inv.effective_priority
            inv.refresh_priority()
            if abs(inv.effective_priority - old_prio) > 0.001:
                logger.info(
                    f"Priority updated: '{inv.title}' "
                    f"{old_prio:.2f} → {inv.effective_priority:.2f}"
                )
            self._save(inv)

        # Re-enforce MAX_ACTIVE: sort active+queued by priority, assign slots
        active_blocked = sorted(
            [i for i in all_invs if i.status in ("active", "blocked", "queued")],
            key=lambda i: i.effective_priority,
            reverse=True,
        )
        resolved_speculative = [
            i for i in all_invs if i.status in ("resolved", "speculative")
        ]

        slot_count = 0
        for inv in active_blocked:
            if slot_count < MAX_ACTIVE_INVESTIGATIONS:
                if inv.status == "queued":
                    inv.status = "active"
                    inv.updated_at = time.time()
                    self._save(inv)
                    logger.info(f"Auto-promoted '{inv.title}' → active (slot freed)")
                slot_count += 1
            else:
                if inv.status in ("active", "blocked"):
                    inv.status = "queued"
                    inv.updated_at = time.time()
                    self._save(inv)
                    logger.info(f"Auto-demoted '{inv.title}' → queued (slots full)")

        return sorted(
            self._load_all(),
            key=lambda i: i.effective_priority,
            reverse=True,
        )

    # ── Step 2: Blockers ──────────────────────────────────────────────────────

    def add_blocker(
        self,
        inv_id: str,
        description: str,
        blocker_type: str,
        severity: float,
    ) -> ResearchBlocker:
        """
        Validates description specificity, creates the blocker, and persists it.

        Raises:
            BlockerTooVagueError: if specificity_score < 0.40.
            ValueError: if investigation not found.
        """
        result = validate_blocker_specificity(description)
        if not result.is_eligible:
            raise BlockerTooVagueError(
                description,
                format_rejection_message(description, result),
            )

        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")

        blocker = ResearchBlocker.new(
            description=description,
            blocker_type=blocker_type,
            severity=severity,
            specificity_score=result.score,
            # Embedding computed lazily when CollaborationScorer first uses this blocker
            embedding=None,
        )
        inv.blockers.append(blocker)

        # A newly-added blocker may flip a speculative/active → blocked
        if inv.status == "active" and all(b.resolved for b in inv.blockers):
            pass  # all resolved, stays active
        elif inv.status in ("active", "speculative") and not blocker.resolved:
            inv.status = "blocked"  # has at least one unresolved blocker

        inv.updated_at = time.time()
        self._save(inv)
        logger.info(
            f"Added blocker to '{inv.title}': "
            f"'{description[:60]}...' "
            f"(specificity={result.score:.2f})"
        )
        return blocker

    def resolve_blocker(
        self,
        inv_id: str,
        blocker_id: str,
        resolved_by: str,
        evidence_strength: str,
        notes: str = "",
    ) -> tuple[InvestigationStream, ResearchBlocker]:
        """
        NEVER auto-resolves. Always called explicitly by the user.

        Marks a blocker as resolved. Updates investigation status:
          - If all blockers are now resolved → status = "active"
          - Logs event to ResearchMemory (if prof name given)

        evidence_strength must be one of:
            "paper_mention", "pilot_result", "replicated_result", "published_result"
        """
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")

        blocker = next((b for b in inv.blockers if b.id == blocker_id), None)
        if blocker is None:
            raise ValueError(f"Blocker {blocker_id!r} not found in '{inv.title}'.")

        blocker.resolved = True
        blocker.resolved_by = resolved_by
        blocker.resolved_at = time.time()
        blocker.evidence_strength = evidence_strength

        # If all blockers resolved, flip back to active
        if all(b.resolved for b in inv.blockers):
            inv.status = "active"
            logger.info(f"All blockers resolved in '{inv.title}' → active")

        inv.updated_at = time.time()
        self._save(inv)
        logger.info(
            f"Blocker resolved in '{inv.title}' by '{resolved_by}' "
            f"(strength: {evidence_strength})"
        )
        return inv, blocker

    # ── Step 3: Contradictions ────────────────────────────────────────────────

    def add_contradiction(
        self,
        inv_id: str,
        claim_yours: str,
        claim_external: str,
        source_yours: str,
        source_external: str,
        severity: float,
        confidence: float,
        resolution_hypothesis: str = "",
        contradiction_type: str = "empirical",
    ) -> tuple[InvestigationStream, Contradiction]:
        """
        Adds a contradiction and immediately recomputes effective_priority.

        The priority escalation formula (severity × confidence thresholds)
        is applied automatically via refresh_priority().

        If the escalated priority is high enough to displace a lower-priority
        active investigation, update_priorities() handles the slot swap.
        """
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")

        contradiction = Contradiction.new(
            claim_yours=claim_yours,
            claim_external=claim_external,
            source_yours=source_yours,
            source_external=source_external,
            resolution_hypothesis=resolution_hypothesis,
            severity=severity,
            confidence=confidence,
            contradiction_type=contradiction_type,
        )
        inv.contradictions.append(contradiction)

        old_priority = inv.effective_priority
        inv.refresh_priority()
        inv.updated_at = time.time()
        self._save(inv)

        if inv.effective_priority > old_priority + 0.001:
            logger.info(
                f"Contradiction added to '{inv.title}': "
                f"signal={contradiction.effective_signal:.2f} → "
                f"priority {old_priority:.2f} → {inv.effective_priority:.2f}"
            )
            # Trigger global re-ordering in case this investigation should now
            # displace a lower-priority active one
            self.update_priorities()
        else:
            logger.info(
                f"Contradiction added to '{inv.title}': "
                f"signal={contradiction.effective_signal:.2f} (no priority change)"
            )

        return self._load(inv_id), contradiction

    def update_contradiction_confidence(
        self,
        inv_id: str,
        contradiction_id: str,
        new_confidence: float,
        notes: str = "",
    ) -> InvestigationStream:
        """
        Updates confidence on an existing contradiction and recomputes priority.

        Primary use case: evidence acquisition lowers confidence on a contradiction
        (not severity — the underlying tension may still exist, but we're less sure
        our reading of the external paper is correct).
        """
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")

        contradiction = next(
            (c for c in inv.contradictions if c.id == contradiction_id), None
        )
        if contradiction is None:
            raise ValueError(f"Contradiction {contradiction_id!r} not found.")

        old_confidence = contradiction.confidence
        contradiction.confidence = max(0.0, min(1.0, new_confidence))

        inv.refresh_priority()
        inv.updated_at = time.time()
        self._save(inv)
        logger.info(
            f"Contradiction confidence updated: {old_confidence:.2f} → "
            f"{contradiction.confidence:.2f} "
            f"(new signal: {contradiction.effective_signal:.2f})"
        )
        self.update_priorities()
        return self._load(inv_id)

    # ── Step 4: LLM-Assisted Blocker Suggestion ───────────────────────────────

    def suggest_blockers(
        self,
        inv_id: str,
        self_model: Optional[dict] = None,
    ) -> list[dict]:
        """
        LLM-assisted blocker suggestion. Returns draft blocker dicts.
        The user must approve each one — drafts are NEVER auto-added.

        Each draft dict:
        {
            "description": str,
            "blocker_type": str,
            "severity": float,
            "specificity_score": float,    # pre-validated
            "is_eligible": bool,
            "refinement_hints": list[str], # empty if eligible
        }

        The LLM call is synchronous. Typical duration: ~15s on Qwen2.5:7B.
        """
        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")

        llm = self._get_llm()
        drafts = llm.suggest_blockers_sync(inv, self_model)

        # Run each draft through the specificity validator before returning
        enriched = []
        for draft in drafts:
            desc = draft.get("description", "")
            result = validate_blocker_specificity(desc)
            enriched.append({
                **draft,
                "specificity_score": result.score,
                "is_eligible": result.is_eligible,
                "refinement_hints": result.refinement_hints,
            })

        logger.info(
            f"Suggested {len(enriched)} blockers for '{inv.title}' "
            f"({sum(1 for d in enriched if d['is_eligible'])} eligible)"
        )
        return enriched

    def accept_suggested_blocker(self, inv_id: str, draft: dict) -> ResearchBlocker:
        """
        User accepted a suggested blocker draft. Adds it to the investigation.
        The draft must be eligible (is_eligible=True).

        Raises BlockerTooVagueError if the draft was not eligible.
        """
        if not draft.get("is_eligible", False):
            result = validate_blocker_specificity(draft.get("description", ""))
            raise BlockerTooVagueError(
                draft.get("description", ""),
                format_rejection_message(draft.get("description", ""), result),
            )
        return self.add_blocker(
            inv_id=inv_id,
            description=draft["description"],
            blocker_type=draft.get("blocker_type", "methodology"),
            severity=draft.get("severity", 0.7),
        )

    # ── Step 5: Contradiction Detection from Dossier (hardest — left last) ────

    def detect_contradictions_from_dossier(
        self,
        inv_id: str,
        prof_dossier: dict,
        self_model: dict,
        detection_threshold: float = 0.80,
        use_llm_refinement: bool = True,
    ) -> list[Contradiction]:
        """
        Scans a professor dossier for candidate contradictions with an investigation.

        Step 1 — Deterministic (always runs):
            Embed each of the user's strong_claims from self_model.
            Compare against each professor paper abstract embedding.
            Flag pairs where cosine > detection_threshold as candidates.

        Step 2 — LLM Refinement (optional, only for flagged pairs):
            For each candidate pair, ask the LLM:
            "Are these actually in tension, or just related?"
            Uses ContradictionAnalysis schema.
            Returns Contradiction or None per pair.

        Net result: LLM is called only for genuine candidates.
        No LLM cost for non-overlapping papers.

        Returns: list of newly-detected Contradiction objects (not yet added to inv).
        Call add_contradiction() for each one the user accepts.

        NOTE: This method does NOT auto-add contradictions. It surfaces candidates.
        The user reviews and calls add_contradiction() explicitly.
        """
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "numpy and sentence-transformers required for contradiction detection."
            )
            return []

        inv = self._load(inv_id)
        if inv is None:
            raise ValueError(f"Investigation {inv_id!r} not found.")

        strong_claims = self_model.get("strong_claims", [])
        if not strong_claims:
            logger.info("No strong_claims in self_model. Skipping detection.")
            return []

        # Collect professor paper abstracts
        prof_papers = prof_dossier.get("publications", [])
        abstracts = [p.get("abstract", "") for p in prof_papers if p.get("abstract")]
        titles = [p.get("title", "") for p in prof_papers if p.get("abstract")]
        dois = [p.get("doi", "") for p in prof_papers if p.get("abstract")]

        if not abstracts:
            logger.info(f"No abstracts in dossier for '{prof_dossier.get('name')}'. Skipping.")
            return []

        # Embed claims and abstracts using cached_encode and default model BAAI/bge-large-en-v1.5
        from academic_pipeline.embedding_cache import cached_encode
        claim_embeddings = cached_encode(strong_claims, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
        abstract_embeddings = cached_encode(abstracts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)

        # Cross-similarity: shape (n_claims, n_abstracts)
        sim_matrix = claim_embeddings @ abstract_embeddings.T

        candidates = []
        for i, claim in enumerate(strong_claims):
            for j, abstract in enumerate(abstracts):
                sim = float(sim_matrix[i, j])
                if sim >= detection_threshold:
                    candidates.append({
                        "claim_yours": claim,
                        "source_external": f"{titles[j]} ({dois[j] or 'no doi'})",
                        "abstract_snippet": abstract[:500],
                        "similarity": sim,
                    })

        logger.info(
            f"Detected {len(candidates)} candidate contradictions "
            f"(threshold={detection_threshold}) for '{prof_dossier.get('name')}'"
        )

        if not candidates:
            return []

        if not use_llm_refinement:
            # Return raw candidates as placeholder Contradiction objects (low confidence)
            contradictions = []
            for c in candidates:
                contradictions.append(Contradiction.new(
                    claim_yours=c["claim_yours"],
                    claim_external=c["abstract_snippet"],
                    source_yours="self_model strong_claims",
                    source_external=c["source_external"],
                    resolution_hypothesis="[Pending LLM analysis]",
                    severity=c["similarity"],
                    confidence=0.30,   # low confidence without LLM verification
                    contradiction_type="empirical",
                ))
            return contradictions

        # LLM refinement for each candidate
        llm = self._get_llm()
        confirmed = []
        for candidate in candidates:
            result = llm.analyze_contradiction_sync(
                claim_yours=candidate["claim_yours"],
                claim_external=candidate["abstract_snippet"],
                source_external=candidate["source_external"],
            )
            if result and result.get("is_contradiction", False):
                confirmed.append(Contradiction.new(
                    claim_yours=candidate["claim_yours"],
                    claim_external=result.get("refined_external_claim",
                                              candidate["abstract_snippet"][:200]),
                    source_yours="self_model",
                    source_external=candidate["source_external"],
                    resolution_hypothesis=result.get("resolution_hypothesis", ""),
                    severity=result.get("severity", candidate["similarity"]),
                    confidence=result.get("confidence", 0.60),
                    contradiction_type=result.get("contradiction_type", "empirical"),
                ))

        logger.info(
            f"LLM confirmed {len(confirmed)}/{len(candidates)} "
            f"candidate contradictions"
        )
        return confirmed

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_investigation(self, inv_id: str) -> Optional[InvestigationStream]:
        return self._load(inv_id)

    def get_active_investigations(self) -> list[InvestigationStream]:
        """Returns active + blocked investigations, sorted by effective_priority desc."""
        return sorted(
            [i for i in self._load_all() if i.status in ("active", "blocked")],
            key=lambda i: i.effective_priority,
            reverse=True,
        )

    def get_queued_investigations(self) -> list[InvestigationStream]:
        return sorted(
            [i for i in self._load_all() if i.status == "queued"],
            key=lambda i: i.effective_priority,
            reverse=True,
        )

    def get_all_investigations(self) -> list[InvestigationStream]:
        return sorted(
            self._load_all(),
            key=lambda i: i.effective_priority,
            reverse=True,
        )

    def active_slot_count(self) -> int:
        return self._cache.count_active_investigations()

    def get_unresolved_blockers(
        self,
    ) -> list[tuple[InvestigationStream, ResearchBlocker]]:
        """
        Returns (investigation, blocker) pairs where the blocker is unresolved
        and the investigation is active or blocked.
        Sorted by (inv.effective_priority × blocker.effective_priority) desc.
        Primary driver for 'what to search for next'.
        """
        pairs = []
        for inv in self.get_active_investigations():
            for blocker in inv.open_blockers:
                weight = inv.effective_priority * blocker.effective_priority
                pairs.append((weight, inv, blocker))
        pairs.sort(reverse=True, key=lambda t: t[0])
        return [(inv, blocker) for _, inv, blocker in pairs]

    # ── Seed initialization ───────────────────────────────────────────────────

    def seed_investigations(self, force: bool = False) -> list[InvestigationStream]:
        """
        Loads the three pre-built seed investigations on first run.

        Skips if any investigations already exist (unless force=True).
        Activates up to MAX_ACTIVE_INVESTIGATIONS from the seed set.

        Returns the created investigations.
        """
        from academic_pipeline.investigation_schemas import build_seed_investigations

        existing = self._load_all()
        if existing and not force:
            logger.info(
                f"Skipping seed: {len(existing)} investigations already exist."
            )
            return existing

        seeds = build_seed_investigations()
        for inv in seeds:
            self._cache.save_investigation(inv.to_dict())
            logger.info(f"Seeded investigation: '{inv.title}'")

        # Activate as many as we have slots for
        for inv in seeds:
            try:
                self.activate_investigation(inv.id)
            except ActiveSlotsFullError:
                break

        return [self._load(inv.id) for inv in seeds]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, inv_id: str) -> Optional[InvestigationStream]:
        data = self._cache.get_investigation(inv_id)
        return InvestigationStream.from_dict(data) if data else None

    def _load_all(self, status_filter: Optional[str] = None) -> list[InvestigationStream]:
        rows = self._cache.list_investigations(status_filter=status_filter)
        return [InvestigationStream.from_dict(d) for d in rows]

    def _save(self, inv: InvestigationStream):
        self._cache.save_investigation(inv.to_dict())

    def _get_llm(self):
        """Lazy-load the LLM client on first use."""
        if self._llm is None:
            from academic_pipeline.local_llm import LocalLLMClient
            self._llm = LocalLLMClient(
                model=self._ollama_model,
                host=self._ollama_host,
            )
        return self._llm
