"""
tests/test_schemas.py — Unit tests for Week 1 deliverables.

Tests:
    1. Contradiction: effective_signal, priority escalation thresholds
    2. ResearchBlocker: effective_priority, is_eligible_for_enrs, serialization
    3. InvestigationStream: compute_effective_priority, open_blockers, serialization
    4. ProfessorMemory + MemoryEvent: serialization round-trip
    5. SpecificityValidator: rejection, acceptance, edge cases
    6. CacheManager: new tables (self_model, investigation, collab score, professor memory)
    7. Seed investigations: pre-built blockers pass specificity gate
"""

import os
import sys
import time
import unittest
import tempfile

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from academic_pipeline.investigation_schemas import (
    Contradiction,
    ResearchBlocker,
    InvestigationStream,
    ProfessorMemory,
    MemoryEvent,
    MAX_ACTIVE_INVESTIGATIONS,
    build_seed_investigations,
    SPECIFICITY_THRESHOLD,
)
from academic_pipeline.specificity_validator import (
    validate_blocker_specificity,
    format_rejection_message,
)
from academic_pipeline.cache_manager import CacheManager


# ──────────────────────────────────────────────────────────────────────────────
# Contradiction tests
# ──────────────────────────────────────────────────────────────────────────────

class TestContradiction(unittest.TestCase):

    def _make(self, severity: float, confidence: float) -> Contradiction:
        return Contradiction.new(
            claim_yours="Our model learns X online",
            claim_external="Chen et al. say X requires offline training",
            source_yours="EarthOS paper",
            source_external="Chen et al. 2023",
            resolution_hypothesis="May be scale-dependent",
            severity=severity,
            confidence=confidence,
        )

    def test_effective_signal_is_product(self):
        c = self._make(severity=0.9, confidence=0.8)
        self.assertAlmostEqual(c.effective_signal, 0.72, places=5)

    def test_high_severity_low_confidence_no_escalation(self):
        """severity=0.95, confidence=0.30 → signal=0.285 → no escalation"""
        inv = InvestigationStream.new("Test", "Description", base_priority=0.5)
        inv.contradictions = [self._make(0.95, 0.30)]
        inv.refresh_priority()
        self.assertAlmostEqual(inv.effective_priority, 0.5, places=5)

    def test_moderate_confidence_partial_escalation(self):
        """severity=0.9, confidence=0.75 → signal=0.675 → +0.30"""
        inv = InvestigationStream.new("Test", "Description", base_priority=0.5)
        inv.contradictions = [self._make(0.9, 0.75)]
        inv.refresh_priority()
        self.assertAlmostEqual(inv.effective_priority, 0.8, places=5)

    def test_high_confidence_full_escalation(self):
        """severity=0.95, confidence=0.92 → signal=0.874 → +0.50"""
        inv = InvestigationStream.new("Test", "Description", base_priority=0.5)
        inv.contradictions = [self._make(0.95, 0.92)]
        inv.refresh_priority()
        self.assertAlmostEqual(inv.effective_priority, 1.0, places=5)

    def test_priority_capped_at_one(self):
        inv = InvestigationStream.new("Test", "Description", base_priority=0.9)
        inv.contradictions = [self._make(1.0, 1.0)]
        inv.refresh_priority()
        self.assertLessEqual(inv.effective_priority, 1.0)

    def test_resolved_contradictions_ignored(self):
        """Resolved contradictions should not escalate priority."""
        c = self._make(severity=1.0, confidence=1.0)
        c.status = "resolved"
        inv = InvestigationStream.new("Test", "Description", base_priority=0.4)
        inv.contradictions = [c]
        inv.refresh_priority()
        self.assertAlmostEqual(inv.effective_priority, 0.4, places=5)

    def test_serialization_roundtrip(self):
        c = self._make(0.8, 0.7)
        restored = Contradiction.from_dict(c.to_dict())
        self.assertEqual(c.id, restored.id)
        self.assertAlmostEqual(c.severity, restored.severity)
        self.assertAlmostEqual(c.confidence, restored.confidence)
        self.assertEqual(c.status, restored.status)


# ──────────────────────────────────────────────────────────────────────────────
# ResearchBlocker tests
# ──────────────────────────────────────────────────────────────────────────────

class TestResearchBlocker(unittest.TestCase):

    def _make(self, severity=0.8, specificity=0.75) -> ResearchBlocker:
        return ResearchBlocker.new(
            description="Need curriculum dataset for continual learning agents",
            blocker_type="dataset",
            severity=severity,
            specificity_score=specificity,
        )

    def test_effective_priority_is_product(self):
        b = self._make(severity=0.8, specificity=0.75)
        self.assertAlmostEqual(b.effective_priority, 0.60, places=5)

    def test_eligible_when_above_threshold(self):
        b = self._make(specificity=SPECIFICITY_THRESHOLD)
        self.assertTrue(b.is_eligible_for_enrs)

    def test_ineligible_when_below_threshold(self):
        b = self._make(specificity=SPECIFICITY_THRESHOLD - 0.01)
        self.assertFalse(b.is_eligible_for_enrs)

    def test_ineligible_when_resolved(self):
        b = self._make(specificity=0.80)
        b.resolved = True
        b.resolved_by = "Prof. Mathis"
        self.assertFalse(b.is_eligible_for_enrs)

    def test_severity_clamped(self):
        b = self._make(severity=1.5)
        self.assertLessEqual(b.severity, 1.0)

    def test_specificity_clamped(self):
        b = self._make(specificity=-0.2)
        self.assertGreaterEqual(b.specificity_score, 0.0)

    def test_serialization_roundtrip(self):
        b = self._make()
        restored = ResearchBlocker.from_dict(b.to_dict())
        self.assertEqual(b.id, restored.id)
        self.assertAlmostEqual(b.severity, restored.severity)
        self.assertAlmostEqual(b.specificity_score, restored.specificity_score)
        self.assertEqual(b.resolved, restored.resolved)


# ──────────────────────────────────────────────────────────────────────────────
# InvestigationStream tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInvestigationStream(unittest.TestCase):

    def _make_investigation(self, status="active") -> InvestigationStream:
        inv = InvestigationStream.new(
            title="Developmental Cognition",
            description="Testing developmental learning hypotheses.",
            base_priority=0.7,
            status=status,
        )
        return inv

    def _add_good_blocker(self, inv: InvestigationStream):
        b = ResearchBlocker.new(
            description="Good blocker",
            blocker_type="dataset",
            severity=0.8,
            specificity_score=0.75,
        )
        inv.blockers.append(b)
        return b

    def test_no_blockers_not_scoreable(self):
        inv = self._make_investigation("active")
        self.assertFalse(inv.is_scoreable)

    def test_active_with_eligible_blockers_is_scoreable(self):
        inv = self._make_investigation("active")
        self._add_good_blocker(inv)
        self.assertTrue(inv.is_scoreable)

    def test_queued_not_scoreable(self):
        inv = self._make_investigation("queued")
        self._add_good_blocker(inv)
        self.assertFalse(inv.is_scoreable)

    def test_resolved_not_scoreable(self):
        inv = self._make_investigation("resolved")
        self._add_good_blocker(inv)
        self.assertFalse(inv.is_scoreable)

    def test_open_blockers_excludes_resolved(self):
        inv = self._make_investigation("active")
        b1 = self._add_good_blocker(inv)
        b2 = self._add_good_blocker(inv)
        b2.resolved = True
        self.assertEqual(len(inv.open_blockers), 1)
        self.assertEqual(inv.open_blockers[0].id, b1.id)

    def test_open_blockers_excludes_below_threshold(self):
        inv = self._make_investigation("active")
        self._add_good_blocker(inv)  # eligible
        bad = ResearchBlocker.new("vague", "dataset", 0.9, 0.10)  # ineligible
        inv.blockers.append(bad)
        self.assertEqual(len(inv.open_blockers), 1)

    def test_max_active_constant(self):
        self.assertEqual(MAX_ACTIVE_INVESTIGATIONS, 5)

    def test_serialization_roundtrip(self):
        inv = self._make_investigation("blocked")
        self._add_good_blocker(inv)
        inv.contradictions.append(
            Contradiction.new("A", "B", "ours", "theirs", "maybe", 0.8, 0.7)
        )
        restored = InvestigationStream.from_dict(inv.to_dict())
        self.assertEqual(inv.id, restored.id)
        self.assertEqual(inv.title, restored.title)
        self.assertEqual(len(restored.blockers), 1)
        self.assertEqual(len(restored.contradictions), 1)
        self.assertAlmostEqual(
            restored.contradictions[0].confidence, 0.7, places=5
        )


# ──────────────────────────────────────────────────────────────────────────────
# ProfessorMemory tests
# ──────────────────────────────────────────────────────────────────────────────

class TestProfessorMemory(unittest.TestCase):

    def test_prof_id_uses_openalex_when_available(self):
        mem = ProfessorMemory.new("Alex Mathis", "EPFL", openalex_id="A12345")
        self.assertEqual(mem.prof_id, "A12345")

    def test_prof_id_hashes_name_when_no_openalex(self):
        mem = ProfessorMemory.new("Alex Mathis", "EPFL")
        self.assertTrue(mem.prof_id.startswith("name_"))

    def test_add_event_updates_timestamp(self):
        mem = ProfessorMemory.new("Alex Mathis", "EPFL")
        before = mem.last_updated_at
        time.sleep(0.01)
        mem.add_event(MemoryEvent(event_type="dossier_viewed", notes="test"))
        self.assertGreater(mem.last_updated_at, before)

    def test_serialization_roundtrip(self):
        mem = ProfessorMemory.new("Alex Mathis", "EPFL", openalex_id="A12345")
        mem.last_score = 72.4
        mem.status = "contacted"
        mem.add_event(MemoryEvent(event_type="email_sent", notes="sent initial"))
        restored = ProfessorMemory.from_dict(mem.to_dict())
        self.assertEqual(restored.professor_name, "Alex Mathis")
        self.assertAlmostEqual(restored.last_score, 72.4)
        self.assertEqual(restored.status, "contacted")
        self.assertEqual(len(restored.events), 1)
        self.assertEqual(restored.events[0].event_type, "email_sent")


# ──────────────────────────────────────────────────────────────────────────────
# SpecificityValidator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSpecificityValidator(unittest.TestCase):

    def test_vague_one_liner_rejected(self):
        result = validate_blocker_specificity("Need better AI for my research")
        self.assertFalse(result.is_eligible)
        self.assertLess(result.score, SPECIFICITY_THRESHOLD)

    def test_vague_generic_rejected(self):
        result = validate_blocker_specificity("Need developmental learning data")
        self.assertFalse(result.is_eligible)

    def test_good_blocker_accepted(self):
        desc = (
            "Need empirical evidence that curriculum ordering changes representation "
            "structure (not just task performance) in continual learning agents"
        )
        result = validate_blocker_specificity(desc)
        self.assertTrue(result.is_eligible)
        self.assertGreaterEqual(result.score, SPECIFICITY_THRESHOLD)

    def test_strong_blocker_high_score(self):
        desc = (
            "Need validated exoplanet transit light-curve datasets with ground-truth "
            "labels for benchmarking ML classification performance against established "
            "spectroscopic characterization methods on the same observational data"
        )
        result = validate_blocker_specificity(desc)
        self.assertTrue(result.is_eligible)
        self.assertGreaterEqual(result.score, 0.60)

    def test_rejected_blocker_has_hints(self):
        result = validate_blocker_specificity("Improve AI")
        self.assertFalse(result.is_eligible)
        self.assertGreater(len(result.refinement_hints), 0)

    def test_accepted_blocker_empty_hints(self):
        desc = (
            "Need fMRI data collected during motor learning tasks in human subjects "
            "to validate that our computational model of representational change "
            "matches biological learning dynamics when compared against RSA metrics"
        )
        result = validate_blocker_specificity(desc)
        self.assertTrue(result.is_eligible)
        self.assertEqual(len(result.refinement_hints), 0)

    def test_causal_claim_boosts_score(self):
        # Use a sentence that lacks method/domain/measurable signals
        # so the causal bonus is the decisive differentiator.
        desc_without_causal = (
            "Access to information about learning experiments in various systems "
            "done by researchers at institutions studying training procedures"
        )
        desc_with_causal = (
            "Access to information about learning experiments in various systems "
            "done by researchers at institutions showing how training procedures "
            "affect outcomes"
        )
        r1 = validate_blocker_specificity(desc_without_causal)
        r2 = validate_blocker_specificity(desc_with_causal)
        # The causal sentence should score strictly higher due to "showing" + "affect"
        self.assertGreater(r2.score, r1.score)


    def test_score_clamped_between_zero_and_one(self):
        for desc in [
            "bad",
            "Need empirical evidence that curriculum ordering changes representation "
            "structure compared to non-curriculum training in continual learning agents "
            "using RSA metrics on benchmark datasets against state-of-the-art baselines",
        ]:
            result = validate_blocker_specificity(desc)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_format_rejection_message_contains_hints(self):
        desc = "Need data"
        result = validate_blocker_specificity(desc)
        msg = format_rejection_message(desc, result)
        self.assertIn("rejected", msg)
        self.assertIn("To improve it:", msg)


# ──────────────────────────────────────────────────────────────────────────────
# CacheManager — new table tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheManagerNewTables(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir)

    def test_self_model_save_and_load(self):
        data = {
            "topic_embedding": [0.1, 0.2, 0.3],
            "paper_count": 5,
            "built_at": time.time(),
            "model_hash": "abc123",   # required by save_self_model
        }
        self.cache.save_self_model(data)
        loaded = self.cache.get_self_model()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["paper_count"], 5)

    def test_self_model_hash_match_returns_data(self):
        data = {"paper_count": 3, "built_at": time.time(), "model_hash": "xyz999"}
        self.cache.save_self_model(data)
        loaded = self.cache.get_self_model(expected_hash="xyz999")
        self.assertIsNotNone(loaded)

    def test_self_model_hash_mismatch_returns_none(self):
        """If corpus changed (different hash), treat as cache miss → rebuild."""
        data = {"paper_count": 3, "built_at": time.time(), "model_hash": "xyz999"}
        self.cache.save_self_model(data)
        loaded = self.cache.get_self_model(expected_hash="different_hash")
        self.assertIsNone(loaded)

    def test_self_model_requires_model_hash(self):
        """save_self_model must raise if model_hash is missing."""
        with self.assertRaises(ValueError):
            self.cache.save_self_model({"paper_count": 1})

    def test_collaboration_score_save_and_load(self):
        key = "abc123_A12345"
        data = {"total_score": 72.4, "professor_name": "Mathis"}
        self.cache.save_collaboration_score(key, data)
        loaded = self.cache.get_collaboration_score(key)
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded["total_score"], 72.4)

    def test_collaboration_score_wrong_key_returns_none(self):
        self.assertIsNone(self.cache.get_collaboration_score("nonexistent"))

    def test_investigation_save_and_load(self):
        inv = InvestigationStream.new("Test Inv", "A description.", 0.7, status="active")
        self.cache.save_investigation(inv.to_dict())
        loaded = self.cache.get_investigation(inv.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Test Inv")

    def test_investigation_list_by_status(self):
        inv1 = InvestigationStream.new("Active", "...", 0.8, status="active")
        inv2 = InvestigationStream.new("Queued", "...", 0.6, status="queued")
        self.cache.save_investigation(inv1.to_dict())
        self.cache.save_investigation(inv2.to_dict())
        active = self.cache.list_investigations(status_filter="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["title"], "Active")

    def test_investigation_list_sorted_by_priority(self):
        inv_lo = InvestigationStream.new("Low", "...", 0.3, status="active")
        inv_hi = InvestigationStream.new("High", "...", 0.9, status="active")
        self.cache.save_investigation(inv_lo.to_dict())
        self.cache.save_investigation(inv_hi.to_dict())
        all_invs = self.cache.list_investigations()
        self.assertEqual(all_invs[0]["title"], "High")

    def test_count_active_investigations(self):
        for title, status in [("A", "active"), ("B", "blocked"), ("C", "queued")]:
            inv = InvestigationStream.new(title, "...", 0.5, status=status)
            self.cache.save_investigation(inv.to_dict())
        count = self.cache.count_active_investigations()
        self.assertEqual(count, 2)  # active + blocked

    def test_professor_memory_save_and_load(self):
        mem = ProfessorMemory.new("Alex Mathis", "EPFL", openalex_id="A99999")
        mem.last_score = 78.5
        self.cache.save_professor_memory(mem.prof_id, mem.to_dict())
        loaded = self.cache.get_professor_memory("A99999")
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded["last_score"], 78.5)

    def test_professor_memory_not_found(self):
        self.assertIsNone(self.cache.get_professor_memory("nonexistent"))

    def test_stats_includes_new_tables(self):
        stats = self.cache.stats()
        self.assertIn("self_model", stats)
        self.assertIn("investigation_streams", stats)
        self.assertIn("collaboration_scores", stats)
        self.assertIn("professor_memory", stats)


# ──────────────────────────────────────────────────────────────────────────────
# Seed investigation quality tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSeedInvestigations(unittest.TestCase):
    """All seed investigation blockers must pass the specificity gate."""

    def test_seed_investigations_created(self):
        seeds = build_seed_investigations()
        self.assertEqual(len(seeds), 3)

    def test_all_seed_blockers_pass_specificity(self):
        seeds = build_seed_investigations()
        for inv in seeds:
            for blocker in inv.blockers:
                result = validate_blocker_specificity(blocker.description)
                self.assertTrue(
                    result.is_eligible,
                    msg=(
                        f"Seed blocker failed specificity gate\n"
                        f"  Investigation: {inv.title}\n"
                        f"  Blocker: {blocker.description[:80]}...\n"
                        f"  Score: {result.score:.2f}\n"
                        f"  Hints: {result.refinement_hints}"
                    ),
                )

    def test_seed_investigations_have_valid_priorities(self):
        seeds = build_seed_investigations()
        for inv in seeds:
            self.assertGreaterEqual(inv.base_priority, 0.0)
            self.assertLessEqual(inv.base_priority, 1.0)

    def test_seed_investigations_are_serializable(self):
        seeds = build_seed_investigations()
        for inv in seeds:
            d = inv.to_dict()
            restored = InvestigationStream.from_dict(d)
            self.assertEqual(inv.id, restored.id)
            self.assertEqual(len(inv.blockers), len(restored.blockers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
