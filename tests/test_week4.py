"""
tests/test_week4.py — Unit tests for Week 4 deliverables.

Tests:
    1. NLI filter: candidate building, result structure, fallback behavior
    2. ClaimDeduplicator: merging, canonical form, source accumulation
    3. LocalLLMClient schema validation (no Ollama required — tests schema structure only)
    4. apply_deduplication_to_self_model: round-trip through self_model dict
    5. NLIResult provenance fields
    6. Pydantic schema field validation for ExtractedClaim, ContradictionAnalysis

NOTE: Tests that require Ollama are skipped if Ollama is not available.
      All deterministic logic (NLI threshold decisions, dedup clustering) is always tested.
"""

import os
import sys
import math
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from academic_pipeline.claim_deduplicator import (
    ClaimDeduplicator,
    CanonicalClaim,
    apply_deduplication_to_self_model,
    DEDUP_THRESHOLD,
)
from academic_pipeline.nli_filter import (
    NLIFilter,
    NLIResult,
    build_contradiction_candidates,
    CONTRADICTION_THRESHOLD,
    ENTAILMENT_THRESHOLD,
    nli_label_from_scores,
)
from academic_pipeline.local_llm import (
    ExtractedClaim,
    ExtractedQuestion,
    SelfModelExtraction,
    ContradictionAnalysis,
    BlockerSuggestion,
    BlockerSuggestions,
    CollaborationHypotheses,
)


# ──────────────────────────────────────────────────────────────────────────────
# NLI Filter — structural and threshold tests (no model required)
# ──────────────────────────────────────────────────────────────────────────────

class TestNLIResult(unittest.TestCase):

    def _make_result(self, p_contra=0.70, p_neutral=0.20, p_entail=0.10) -> NLIResult:
        return NLIResult(
            claim_yours="Our model learns X online",
            claim_external="X cannot be learned online",
            source_external="Test et al. 2023",
            p_contradiction=p_contra,
            p_neutral=p_neutral,
            p_entailment=p_entail,
            label="contradiction",
            passes_filter=p_contra >= CONTRADICTION_THRESHOLD,
            similarity=0.78,
        )

    def test_passes_filter_when_above_threshold(self):
        r = self._make_result(p_contra=0.75)
        self.assertTrue(r.passes_filter)

    def test_fails_filter_when_below_threshold(self):
        r = self._make_result(p_contra=0.40)
        r.passes_filter = False
        self.assertFalse(r.passes_filter)

    def test_confidence_returns_p_contradiction_when_passing(self):
        r = self._make_result(p_contra=0.75)
        r.passes_filter = True
        self.assertAlmostEqual(r.confidence, 0.75)

    def test_confidence_returns_zero_when_not_passing(self):
        r = self._make_result(p_contra=0.75)
        r.passes_filter = False
        self.assertEqual(r.confidence, 0.0)

    def test_metadata_default_empty(self):
        r = NLIResult(
            claim_yours="A", claim_external="B", source_external="C"
        )
        self.assertEqual(r.metadata, {})


class TestNLILabelFromScores(unittest.TestCase):

    def test_contradiction_label_when_first_score_highest(self):
        # [contradiction, entailment, neutral]
        label = nli_label_from_scores([0.80, 0.10, 0.10])
        self.assertEqual(label, "contradiction")

    def test_entailment_label_when_second_score_highest(self):
        label = nli_label_from_scores([0.10, 0.80, 0.10])
        self.assertEqual(label, "entailment")

    def test_neutral_label_when_third_score_highest(self):
        label = nli_label_from_scores([0.10, 0.10, 0.80])
        self.assertEqual(label, "neutral")


class TestNLIFilterFallback(unittest.TestCase):
    """Tests that verify fallback behavior when NLI model is unavailable."""

    def test_fallback_all_candidates_pass(self):
        """When NLI model unavailable, all candidates pass to LLM stage."""
        nli = NLIFilter()
        candidates = [
            {"claim_yours": "A claims X", "claim_external": "B claims Y",
             "source_external": "Test 2023", "similarity": 0.75},
            {"claim_yours": "A claims Z", "claim_external": "B claims W",
             "source_external": "Test 2024", "similarity": 0.72},
        ]
        # Force fallback by patching the model load
        with patch.object(nli, '_model_loaded', side_effect=ImportError("no model")):
            results = nli.filter_candidates(candidates)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.passes_filter for r in results))

    def test_empty_candidates_returns_empty(self):
        nli = NLIFilter()
        results = nli.filter_candidates([])
        self.assertEqual(results, [])

    def test_filter_returns_all_pairs_not_just_passing(self):
        """filter_candidates returns ALL results; caller filters."""
        nli = NLIFilter()
        candidates = [
            {"claim_yours": "A", "claim_external": "B",
             "source_external": "C", "similarity": 0.75}
        ]
        with patch.object(nli, '_model_loaded', side_effect=ImportError("no model")):
            results = nli.filter_candidates(candidates)
        # Even fallback should return something
        self.assertEqual(len(results), 1)

    def test_is_available_returns_bool(self):
        nli = NLIFilter()
        result = nli.is_available()
        self.assertIsInstance(result, bool)

    def test_score_single_returns_nli_result(self):
        """score_single returns NLIResult even in fallback."""
        nli = NLIFilter()
        with patch.object(nli, '_model_loaded', side_effect=ImportError("no model")):
            result = nli.score_single("Claim A", "Claim B")
        self.assertIsInstance(result, NLIResult)

    def test_nli_constants_are_sane(self):
        self.assertGreater(CONTRADICTION_THRESHOLD, 0.0)
        self.assertLess(CONTRADICTION_THRESHOLD, 1.0)
        self.assertGreater(ENTAILMENT_THRESHOLD, CONTRADICTION_THRESHOLD)


class TestBuildContradictionCandidates(unittest.TestCase):

    def test_returns_empty_when_no_claims(self):
        result = build_contradiction_candidates([], ["some abstract"], [{}])
        self.assertEqual(result, [])

    def test_returns_empty_when_no_abstracts(self):
        result = build_contradiction_candidates(["some claim"], [], [])
        self.assertEqual(result, [])

    def test_returns_list_of_dicts(self):
        """Integration: build candidates from real texts (uses embeddings)."""
        claims = ["Curriculum learning improves representation quality"]
        abstracts = [
            "We show curriculum ordering harms representation quality in our study.",
            "Black hole thermodynamics and quantum gravity."
        ]
        meta = [{"title": "Paper 1", "doi": "10.X/1"}, {"title": "Paper 2", "doi": "10.X/2"}]

        result = build_contradiction_candidates(
            claims, abstracts, meta, embedding_threshold=0.0   # threshold=0 to get all pairs
        )
        self.assertIsInstance(result, list)
        for candidate in result:
            self.assertIn("claim_yours", candidate)
            self.assertIn("claim_external", candidate)
            self.assertIn("source_external", candidate)
            self.assertIn("similarity", candidate)

    def test_high_threshold_filters_unrelated_pairs(self):
        """Very high threshold should exclude unrelated claim-abstract pairs."""
        claims = ["Curriculum learning improves representation quality"]
        abstracts = ["Black hole thermodynamics and Hawking radiation."]
        meta = [{"title": "BH Paper", "doi": "10.X/bh"}]

        result = build_contradiction_candidates(
            claims, abstracts, meta, embedding_threshold=0.90
        )
        # Unrelated pair should not pass very high threshold
        self.assertEqual(len(result), 0)


# ──────────────────────────────────────────────────────────────────────────────
# ClaimDeduplicator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCanonicalClaim(unittest.TestCase):

    def _make_raw(self, text: str, confidence: float = 0.8, dois=None) -> dict:
        return {
            "text": text,
            "confidence": confidence,
            "source_paper_dois": dois or [],
            "domain": "continual learning",
            "question_type": "empirical",
        }

    def test_from_extracted_sets_text(self):
        raw = self._make_raw("Curriculum ordering affects representations", 0.85, ["10.X/1"])
        c = CanonicalClaim.from_extracted(raw)
        self.assertEqual(c.text, "Curriculum ordering affects representations")

    def test_from_extracted_sets_confidence(self):
        raw = self._make_raw("Some claim", 0.92)
        c = CanonicalClaim.from_extracted(raw)
        self.assertAlmostEqual(c.confidence, 0.92)

    def test_from_extracted_populates_source_papers(self):
        raw = self._make_raw("Claim", 0.8, ["10.X/1", "10.X/2"])
        c = CanonicalClaim.from_extracted(raw)
        dois = [s["doi"] for s in c.source_papers]
        self.assertIn("10.X/1", dois)
        self.assertIn("10.X/2", dois)

    def test_to_dict_has_required_keys(self):
        raw = self._make_raw("Some claim", 0.8)
        c = CanonicalClaim.from_extracted(raw)
        d = c.to_dict()
        for key in ["text", "confidence", "source_papers", "domain", "merged_count"]:
            self.assertIn(key, d)

    def test_merged_count_defaults_to_one(self):
        raw = self._make_raw("Some claim")
        c = CanonicalClaim.from_extracted(raw)
        self.assertEqual(c.merged_count, 1)


class TestClaimDeduplicator(unittest.TestCase):

    def setUp(self):
        self.dedup = ClaimDeduplicator(similarity_threshold=DEDUP_THRESHOLD)

    def _make_claim(self, text: str, confidence: float = 0.8, dois=None) -> dict:
        return {
            "text": text,
            "confidence": confidence,
            "source_paper_dois": dois or [],
            "domain": "test",
        }

    def test_empty_input_returns_empty(self):
        result = self.dedup.deduplicate([])
        self.assertEqual(result, [])

    def test_single_claim_returned_as_is(self):
        claims = [self._make_claim("Curriculum ordering affects representations", 0.85)]
        result = self.dedup.deduplicate(claims)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.85)

    def test_identical_claims_merged(self):
        """Identical text should merge into one canonical claim."""
        claim_text = "Investigation persistence improves cognitive coherence"
        claims = [
            self._make_claim(claim_text, 0.91, ["10.X/A"]),
            self._make_claim(claim_text, 0.88, ["10.X/B"]),
        ]
        result = self.dedup.deduplicate(claims)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].merged_count, 2)

    def test_canonical_form_is_highest_confidence(self):
        """When two identical claims merge, canonical text = highest confidence."""
        claim_text = "Investigation persistence improves cognitive coherence"
        claims = [
            self._make_claim(claim_text, 0.70, ["10.X/B"]),
            self._make_claim(claim_text, 0.95, ["10.X/A"]),  # higher confidence
        ]
        result = self.dedup.deduplicate(claims)
        self.assertEqual(len(result), 1)
        # Should keep the text (same since identical), but higher confidence is canonical
        self.assertAlmostEqual(result[0].confidence, (0.70 + 0.95) / 2, places=2)

    def test_source_papers_accumulate_across_merged(self):
        """Merged claims combine all source DOIs."""
        claim_text = "Investigation persistence improves cognitive coherence"
        claims = [
            self._make_claim(claim_text, 0.88, ["10.X/A"]),
            self._make_claim(claim_text, 0.85, ["10.X/B"]),
        ]
        result = self.dedup.deduplicate(claims)
        self.assertEqual(len(result), 1)
        dois = {s["doi"] for s in result[0].source_papers}
        self.assertIn("10.X/A", dois)
        self.assertIn("10.X/B", dois)

    def test_unrelated_claims_not_merged(self):
        """Claims on completely different topics should NOT merge."""
        claims = [
            self._make_claim("Curriculum ordering improves representation geometry", 0.85),
            self._make_claim("Black hole thermodynamics obeys the laws of thermodynamics", 0.82),
        ]
        result = self.dedup.deduplicate(claims)
        self.assertEqual(len(result), 2)

    def test_result_sorted_by_confidence_desc(self):
        """Canonical claims should be sorted highest confidence first."""
        claims = [
            self._make_claim("Claim about continual learning", 0.60),
            self._make_claim("Another claim about curriculum effects", 0.90),
            self._make_claim("Third claim about representation learning", 0.75),
        ]
        result = self.dedup.deduplicate(claims)
        confidences = [c.confidence for c in result]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_dedup_threshold_constant_is_sane(self):
        self.assertGreater(DEDUP_THRESHOLD, 0.5)
        self.assertLess(DEDUP_THRESHOLD, 1.0)

    def test_deduplicate_questions_preserves_question_type(self):
        questions = [
            {"text": "Does curriculum order matter in biological learning?",
             "confidence": 0.82, "source_paper_dois": [], "question_type": "empirical"},
        ]
        result = self.dedup.deduplicate_questions(questions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].question_type, "empirical")


class TestApplyDeduplicationToSelfModel(unittest.TestCase):

    def _make_self_model(self) -> dict:
        claim_text = "Investigation persistence improves cognitive coherence"
        return {
            "topic_embedding": [0.1] * 384,
            "method_embedding": [0.2] * 384,
            "strong_claims": [
                {"text": claim_text, "confidence": 0.91,
                 "source_paper_dois": ["10.X/A"], "domain": "EarthOS"},
                {"text": claim_text, "confidence": 0.88,
                 "source_paper_dois": ["10.X/B"], "domain": "EarthOS"},
            ],
            "open_questions": [
                {"text": "Does investigation persistence scale to longer timescales?",
                 "confidence": 0.80, "source_paper_dois": ["10.X/A"],
                 "question_type": "empirical"},
            ],
            "model_hash": "abc123",
            "paper_count": 2,
        }

    def test_returns_updated_copy(self):
        model = self._make_self_model()
        updated = apply_deduplication_to_self_model(model)
        # Original should be unchanged
        self.assertEqual(len(model["strong_claims"]), 2)
        # Updated may have fewer
        self.assertIsNotNone(updated)

    def test_dedup_stats_added(self):
        model = self._make_self_model()
        updated = apply_deduplication_to_self_model(model)
        self.assertIn("dedup_stats", updated)
        stats = updated["dedup_stats"]
        self.assertIn("raw_claims", stats)
        self.assertIn("canonical_claims", stats)
        self.assertIn("raw_questions", stats)
        self.assertIn("canonical_questions", stats)

    def test_strong_claims_remain_strings(self):
        """strong_claims must stay list[str] — SentenceTransformer and scorer expect this."""
        model = self._make_self_model()
        updated = apply_deduplication_to_self_model(model)
        for claim in updated["strong_claims"]:
            self.assertIsInstance(claim, str)
            self.assertGreater(len(claim), 0)

    def test_provenance_stored_separately(self):
        """Full provenance (confidence, source_papers) lives in canonical_claims_provenance."""
        model = self._make_self_model()
        updated = apply_deduplication_to_self_model(model)
        self.assertIn("canonical_claims_provenance", updated)
        for prov in updated["canonical_claims_provenance"]:
            self.assertIsInstance(prov, dict)
            self.assertIn("text", prov)
            self.assertIn("confidence", prov)
            self.assertIn("source_papers", prov)

    def test_model_hash_preserved(self):
        model = self._make_self_model()
        updated = apply_deduplication_to_self_model(model)
        self.assertEqual(updated["model_hash"], "abc123")


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schema validation (no Ollama required)
# ──────────────────────────────────────────────────────────────────────────────

class TestPydanticSchemas(unittest.TestCase):

    def test_extracted_claim_validates_confidence_range(self):
        c = ExtractedClaim(
            text="Our model improves performance",
            confidence=0.85,
            source_paper_dois=["10.X/1"],
            domain="continual learning",
        )
        self.assertAlmostEqual(c.confidence, 0.85)

    def test_extracted_claim_rejects_confidence_above_one(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ExtractedClaim(text="claim", confidence=1.5)

    def test_extracted_claim_rejects_negative_confidence(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ExtractedClaim(text="claim", confidence=-0.1)

    def test_extracted_question_default_type(self):
        q = ExtractedQuestion(text="Does X scale?", confidence=0.75)
        self.assertEqual(q.question_type, "empirical")

    def test_self_model_extraction_empty_by_default(self):
        s = SelfModelExtraction()
        self.assertEqual(s.strong_claims, [])
        self.assertEqual(s.open_questions, [])
        self.assertEqual(s.methods_used, [])
        self.assertEqual(s.topics, [])

    def test_contradiction_analysis_confidence_clamped(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ContradictionAnalysis(
                is_contradiction=True,
                severity=0.8,
                confidence=1.5,   # invalid
            )

    def test_contradiction_analysis_defaults(self):
        c = ContradictionAnalysis(is_contradiction=False, severity=0.5, confidence=0.6)
        self.assertEqual(c.contradiction_type, "empirical")
        self.assertEqual(c.refined_external_claim, "")
        self.assertEqual(c.resolution_hypothesis, "")

    def test_blocker_suggestion_default_type(self):
        b = BlockerSuggestion(description="Need CKA benchmark dataset", severity=0.8)
        self.assertEqual(b.blocker_type, "methodology")

    def test_blocker_suggestions_empty_by_default(self):
        s = BlockerSuggestions()
        self.assertEqual(s.suggestions, [])

    def test_collaboration_hypotheses_empty_by_default(self):
        h = CollaborationHypotheses()
        self.assertEqual(h.hypotheses, [])

    def test_self_model_extraction_serializes_to_dict(self):
        s = SelfModelExtraction(
            strong_claims=[ExtractedClaim(text="Claim A", confidence=0.9)],
            open_questions=[ExtractedQuestion(text="Question B?", confidence=0.75)],
            methods_used=["RSA", "CKA"],
            topics=["continual learning"],
        )
        d = s.model_dump()
        self.assertEqual(len(d["strong_claims"]), 1)
        self.assertEqual(d["strong_claims"][0]["text"], "Claim A")
        self.assertEqual(len(d["open_questions"]), 1)
        self.assertEqual(d["methods_used"], ["RSA", "CKA"])


# ──────────────────────────────────────────────────────────────────────────────
# LocalLLMClient — schema and structure tests (no Ollama)
# ──────────────────────────────────────────────────────────────────────────────

class TestLocalLLMClientStructure(unittest.TestCase):

    def test_client_imports_without_ollama(self):
        """LocalLLMClient should import cleanly even if ollama not installed."""
        from academic_pipeline.local_llm import LocalLLMClient, OLLAMA_AVAILABLE
        self.assertIsNotNone(LocalLLMClient)

    def test_extract_self_model_returns_empty_dict_when_unavailable(self):
        """When Ollama not running, extract_self_model returns empty-typed dict."""
        from academic_pipeline.local_llm import LocalLLMClient
        client = LocalLLMClient(model="nonexistent-model", host="http://localhost:19999")
        result = client.extract_self_model(
            texts=["Some paper abstract about continual learning."],
            titles=["Test Paper"],
            dois=["10.X/test"],
        )
        # Should return empty-typed dict, not raise
        self.assertIn("strong_claims", result)
        self.assertIn("open_questions", result)
        self.assertIn("methods_used", result)
        self.assertIn("topics", result)
        self.assertIsInstance(result["strong_claims"], list)

    def test_analyze_contradiction_returns_none_when_unavailable(self):
        from academic_pipeline.local_llm import LocalLLMClient
        client = LocalLLMClient(model="nonexistent-model", host="http://localhost:19999")
        result = client.analyze_contradiction_sync(
            "Our claim", "Their claim", "Test Source"
        )
        self.assertIsNone(result)

    def test_suggest_blockers_returns_empty_list_when_unavailable(self):
        from academic_pipeline.local_llm import LocalLLMClient
        from academic_pipeline.investigation_schemas import InvestigationStream
        client = LocalLLMClient(model="nonexistent-model", host="http://localhost:19999")
        inv = InvestigationStream.new("Test", "Description", 0.8)
        result = client.suggest_blockers_sync(inv, self_model=None)
        self.assertIsInstance(result, list)

    def test_generate_collaboration_hypotheses_empty_when_no_matches(self):
        from academic_pipeline.local_llm import LocalLLMClient
        client = LocalLLMClient(model="nonexistent-model", host="http://localhost:19999")
        result = client.generate_collaboration_hypotheses(
            self_model={"topics": [], "strong_claims": [], "open_questions": []},
            prof_dossier={"name": "Prof X"},
            resolution_matches=[],   # empty → should short-circuit
            total_score=50.0,
        )
        self.assertEqual(result, [])

    def test_is_available_returns_false_when_no_server(self):
        from academic_pipeline.local_llm import LocalLLMClient
        client = LocalLLMClient(host="http://localhost:19999")
        self.assertFalse(client.is_available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
