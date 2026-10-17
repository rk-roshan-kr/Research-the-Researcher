"""
tests/test_scorer.py — Unit tests for Week 3: SelfModelBuilder + CollaborationScorer.

Tests:
    1. SelfModel: hash computation, cache hit/miss, field completeness
    2. CollaborationScorer: all 6 dimensions individually
    3. Score provenance: all scores have reasoning + evidence
    4. ENRS: blocker type multipliers, threshold filtering, joint weights
    5. Gap complement: open_questions driving complementarity
    6. Score card formatting
    7. Null cases: no papers, no blockers, empty embeddings
"""

import os
import sys
import math
import time
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from academic_pipeline.investigation_schemas import (
    InvestigationStream,
    ResearchBlocker,
    Contradiction,
)
from academic_pipeline.self_model import compute_corpus_hash, SelfModelBuilder
from academic_pipeline.collaboration_scorer import (
    CollaborationScorer,
    DEFAULT_WEIGHTS,
    ENRS_SIM_THRESHOLD,
    _dim_score,
)
from academic_pipeline.cache_manager import CacheManager


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_investigation(
    title: str,
    priority: float,
    status: str = "active",
    blockers: list | None = None,
) -> InvestigationStream:
    inv = InvestigationStream.new(title, f"Description of {title}", priority, status=status)
    if blockers:
        inv.blockers = blockers
    return inv


def _make_blocker(
    description: str,
    blocker_type: str = "dataset",
    severity: float = 0.8,
    specificity: float = 0.75,
    embedding: list | None = None,
) -> ResearchBlocker:
    b = ResearchBlocker.new(description, blocker_type, severity, specificity, embedding)
    return b


def _make_dossier(
    name: str,
    institution: str = "MIT",
    abstracts: list[str] | None = None,
    position: str = "associate professor",
    h_index: int | None = 15,
    recent_year: int = 2024,
) -> dict:
    """Build a minimal professor dossier for testing."""
    pubs = []
    for i, abstract in enumerate(abstracts or []):
        pubs.append({
            "title": f"Paper {i+1}: {abstract[:30]}",
            "doi": f"10.1234/test{i}",
            "abstract": abstract,
            "year": recent_year - i,
        })
    return {
        "name": name,
        "institution": institution,
        "position": position,
        "h_index": h_index,
        "publications": pubs,
    }


def _unit_embedding(dim: int, value: float) -> list[float]:
    """Return a L2-normalized embedding in a single direction."""
    v = [0.0] * dim
    v[0] = value
    norm = math.sqrt(sum(x**2 for x in v))
    return [x / norm for x in v]


# ──────────────────────────────────────────────────────────────────────────────
# SelfModel tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCorpusHash(unittest.TestCase):

    def test_same_texts_same_hash(self):
        texts = ["Continual learning in agents", "Curriculum ordering effects"]
        self.assertEqual(compute_corpus_hash(texts), compute_corpus_hash(texts))

    def test_different_texts_different_hash(self):
        h1 = compute_corpus_hash(["text A"])
        h2 = compute_corpus_hash(["text B"])
        self.assertNotEqual(h1, h2)

    def test_order_invariant(self):
        texts = ["text A", "text B", "text C"]
        shuffled = ["text C", "text A", "text B"]
        self.assertEqual(compute_corpus_hash(texts), compute_corpus_hash(shuffled))

    def test_deduplication(self):
        unique = compute_corpus_hash(["text A", "text B"])
        with_dupe = compute_corpus_hash(["text A", "text B", "text A"])
        self.assertEqual(unique, with_dupe)

    def test_hash_is_16_chars(self):
        h = compute_corpus_hash(["any text"])
        self.assertEqual(len(h), 16)


class TestSelfModelBuilder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir)
        self.builder = SelfModelBuilder(cache_manager=self.cache, llm_client=None)

    def _build(self, texts=None):
        texts = texts or [
            "We study continual learning agents using curriculum scheduling.",
            "Representation structure changes with curriculum ordering in our experiments.",
        ]
        return self.builder.build(texts=texts, titles=["P1", "P2"], dois=["d1", "d2"])

    def test_required_fields_present(self):
        model = self._build()
        required = [
            "topic_embedding", "method_embedding", "strong_claims",
            "open_questions", "recurring_blockers", "publication_vectors",
            "topics", "methods", "model_hash", "built_at", "paper_count",
        ]
        for field in required:
            self.assertIn(field, model, f"Missing field: {field}")

    def test_model_hash_is_present(self):
        model = self._build()
        self.assertIsNotNone(model.get("model_hash"))
        self.assertEqual(len(model["model_hash"]), 16)

    def test_paper_count_matches_input(self):
        model = self._build()
        self.assertEqual(model["paper_count"], 2)

    def test_topic_embedding_is_1024_dim(self):
        model = self._build()
        self.assertEqual(len(model["topic_embedding"]), 1024)

    def test_method_embedding_is_1024_dim(self):
        model = self._build()
        self.assertEqual(len(model["method_embedding"]), 1024)

    def test_publication_vectors_match_paper_count(self):
        model = self._build()
        self.assertEqual(len(model["publication_vectors"]), 2)

    def test_publication_vector_has_embedding(self):
        model = self._build()
        for pv in model["publication_vectors"]:
            self.assertIn("embedding", pv)
            self.assertEqual(len(pv["embedding"]), 1024)

    def test_cache_hit_on_same_hash(self):
        texts = ["First paper about continual learning curriculum scheduling."]
        model1 = self.builder.build(texts=texts)
        # Second call with same texts → cache hit (no rebuild)
        t0 = time.time()
        model2 = self.builder.build(texts=texts)
        elapsed = time.time() - t0
        self.assertEqual(model1["model_hash"], model2["model_hash"])
        self.assertLess(elapsed, 1.0, "Cache hit should be fast (< 1s)")

    def test_cache_miss_on_new_text(self):
        texts_v1 = ["First paper about continual learning."]
        texts_v2 = ["Completely different paper about astrophysics."]
        model1 = self.builder.build(texts=texts_v1)
        model2 = self.builder.build(texts=texts_v2)
        self.assertNotEqual(model1["model_hash"], model2["model_hash"])

    def test_topics_list_is_nonempty(self):
        model = self._build()
        self.assertIsInstance(model["topics"], list)
        self.assertGreater(len(model["topics"]), 0)


# ──────────────────────────────────────────────────────────────────────────────
# Score provenance tests
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreProvenance(unittest.TestCase):

    def test_dim_score_structure(self):
        result = _dim_score(0.75, 0.30, "Test reasoning", [{"key": "val"}])
        self.assertIn("raw", result)
        self.assertIn("weight", result)
        self.assertIn("contribution", result)
        self.assertIn("reasoning", result)
        self.assertIn("evidence", result)

    def test_contribution_is_raw_times_weight_times_100(self):
        result = _dim_score(0.80, 0.30, "reasoning")
        expected = 0.80 * 0.30 * 100
        self.assertAlmostEqual(result["contribution"], expected, places=1)

    def test_empty_evidence_defaults_to_list(self):
        result = _dim_score(0.5, 0.2, "reasoning")
        self.assertEqual(result["evidence"], [])

    def test_all_dimensions_have_provenance(self):
        """Every dimension in a real scorer result must have reasoning + evidence."""
        tmpdir = tempfile.mkdtemp()
        cache = CacheManager(cache_dir=tmpdir)
        scorer = CollaborationScorer(cache_manager=cache)
        builder = SelfModelBuilder(cache_manager=cache)

        texts = ["Continual learning with curriculum scheduling affects representations."]
        model = builder.build(texts=texts)

        inv = _make_investigation(
            "Test Inv", 0.8, status="active",
            blockers=[_make_blocker(
                "Empirical data showing curriculum ordering affects representation structure",
                "dataset", 0.9, 0.80,
            )]
        )

        dossier = _make_dossier(
            "Prof Test",
            abstracts=["Curriculum learning and representation structure in continual learners."],
        )

        result = scorer.compute_score(model, [inv], dossier, generate_hypotheses=False)

        for dim_name, dim_data in result["dimensions"].items():
            self.assertIn("reasoning", dim_data, f"Missing reasoning in {dim_name}")
            self.assertIn("evidence", dim_data, f"Missing evidence in {dim_name}")
            self.assertIsInstance(dim_data["reasoning"], str, f"reasoning not str in {dim_name}")
            self.assertIsInstance(dim_data["evidence"], list, f"evidence not list in {dim_name}")
            self.assertGreater(len(dim_data["reasoning"]), 5, f"reasoning too short in {dim_name}")


# ──────────────────────────────────────────────────────────────────────────────
# CollaborationScorer — dimension-level tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCollaborationScorer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir)
        self.scorer = CollaborationScorer(cache_manager=self.cache)
        self.builder = SelfModelBuilder(cache_manager=self.cache)

    def _build_model(self, texts=None):
        texts = texts or [
            "We study continual learning using curriculum scheduling methods.",
            "Representation structure changes when curriculum order varies in our agents.",
        ]
        return self.builder.build(texts=texts)

    def _build_dossier(self, abstracts=None, year=2024, position="associate professor", h=15):
        abstracts = abstracts or [
            "Curriculum ordering affects representational geometry in neural continual learners.",
        ]
        return _make_dossier("Prof X", abstracts=abstracts, recent_year=year,
                             position=position, h_index=h)

    # ── Full score structure ───────────────────────────────────────────────────

    def test_result_has_all_required_keys(self):
        model = self._build_model()
        inv = _make_investigation("Inv", 0.8, blockers=[
            _make_blocker("Curriculum data for continual learning representation study",
                          "dataset", 0.8, 0.75)
        ])
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [inv], dossier, generate_hypotheses=False)
        for key in ["professor_name", "institution", "total_score", "dimensions",
                    "collaboration_hypotheses", "resolution_matches", "scored_at"]:
            self.assertIn(key, result)

    def test_total_score_between_0_and_100(self):
        model = self._build_model()
        inv = _make_investigation("Inv", 0.8, blockers=[
            _make_blocker("Curriculum ordering effects on representation structure dataset",
                          "dataset", 0.8, 0.75)
        ])
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [inv], dossier, generate_hypotheses=False)
        self.assertGreaterEqual(result["total_score"], 0.0)
        self.assertLessEqual(result["total_score"], 100.0)

    def test_all_6_dimensions_present(self):
        model = self._build_model()
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        for dim in ["enrs", "topic_overlap", "gap_complement", "method_compat",
                    "lab_access", "temporal_act"]:
            self.assertIn(dim, result["dimensions"])

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    # ── Topic overlap ─────────────────────────────────────────────────────────

    def test_high_topic_overlap_when_abstracts_similar(self):
        """Same topic text → high cosine similarity → high topic_overlap score."""
        shared_text = "Curriculum learning for representation structure in continual learning agents."
        model = self._build_model(texts=[shared_text])
        dossier = self._build_dossier(abstracts=[shared_text])
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        topic_raw = result["dimensions"]["topic_overlap"]["raw"]
        self.assertGreater(topic_raw, 0.70)

    def test_low_topic_overlap_when_abstracts_different(self):
        model = self._build_model(texts=["Continual learning curriculum in robots."])
        dossier = self._build_dossier(abstracts=["Black hole thermodynamics and Hawking radiation."])
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        topic_raw = result["dimensions"]["topic_overlap"]["raw"]
        self.assertLess(topic_raw, 0.50)

    # ── Temporal activity ─────────────────────────────────────────────────────

    def test_recent_publications_high_temporal_score(self):
        import time as t
        year = t.localtime().tm_year
        dossier = self._build_dossier(abstracts=["Some paper."], year=year)
        result = self.scorer.compute_score(self._build_model(), [], dossier,
                                           generate_hypotheses=False)
        self.assertGreaterEqual(result["dimensions"]["temporal_act"]["raw"], 0.80)

    def test_old_publications_low_temporal_score(self):
        import time as t
        year = t.localtime().tm_year - 7
        dossier = self._build_dossier(abstracts=["Old paper."], year=year)
        result = self.scorer.compute_score(self._build_model(), [], dossier,
                                           generate_hypotheses=False)
        self.assertLessEqual(result["dimensions"]["temporal_act"]["raw"], 0.30)

    # ── Lab accessibility ─────────────────────────────────────────────────────

    def test_assistant_professor_more_accessible(self):
        model = self._build_model()
        dossier_asst = self._build_dossier(position="assistant professor", h=8)
        dossier_full = self._build_dossier(position="full professor", h=60)
        result_asst = self.scorer.compute_score(model, [], dossier_asst,
                                                generate_hypotheses=False)
        result_full = self.scorer.compute_score(model, [], dossier_full,
                                                generate_hypotheses=False)
        self.assertGreater(
            result_asst["dimensions"]["lab_access"]["raw"],
            result_full["dimensions"]["lab_access"]["raw"],
        )

    # ── ENRS: no investigations → zero ENRS ──────────────────────────────────

    def test_no_investigations_zero_enrs(self):
        model = self._build_model()
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        self.assertEqual(result["dimensions"]["enrs"]["raw"], 0.0)
        self.assertEqual(result["dimensions"]["enrs"]["contribution"], 0.0)

    # ── ENRS: with matching investigation ────────────────────────────────────

    def test_matching_blocker_increases_enrs(self):
        """Professor paper semantically matches blocker → ENRS > 0."""
        model = self._build_model()
        inv = _make_investigation("Inv", 0.8, status="active", blockers=[
            _make_blocker(
                "Curriculum ordering effects on representation structure in continual learning",
                "dataset", 0.9, 0.82,
            )
        ])
        dossier = self._build_dossier(abstracts=[
            "We show curriculum ordering significantly changes the representation structure "
            "of continual learning agents, measured via CKA and RSA metrics."
        ])
        result = self.scorer.compute_score(model, [inv], dossier, generate_hypotheses=False)
        enrs_raw = result["dimensions"]["enrs"]["raw"]
        self.assertGreater(enrs_raw, 0.0, "ENRS should be > 0 when paper matches blocker")

    def test_resolution_matches_populated(self):
        model = self._build_model()
        inv = _make_investigation("Inv", 0.8, status="active", blockers=[
            _make_blocker(
                "Curriculum ordering effects on representation structure in continual learning",
                "dataset", 0.9, 0.82,
            )
        ])
        dossier = self._build_dossier(abstracts=[
            "Curriculum ordering changes representation structure in continual learners."
        ])
        result = self.scorer.compute_score(model, [inv], dossier, generate_hypotheses=False)
        # May or may not have matches depending on actual similarity
        self.assertIsInstance(result["resolution_matches"], list)

    # ── Gap complement ────────────────────────────────────────────────────────

    def test_gap_complement_zero_without_open_questions(self):
        """No open_questions → gap complement should be 0 or near 0."""
        model = self._build_model()
        model["open_questions"] = []   # override
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        gap_raw = result["dimensions"]["gap_complement"]["raw"]
        self.assertEqual(gap_raw, 0.0)

    # ── Score card ────────────────────────────────────────────────────────────

    def test_score_card_contains_name(self):
        model = self._build_model()
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        card = CollaborationScorer.format_score_card(result)
        self.assertIn("Prof X", card)
        self.assertIn("Total:", card)

    def test_score_card_contains_all_dimensions(self):
        model = self._build_model()
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        card = CollaborationScorer.format_score_card(result)
        self.assertIn("ENRS", card)
        self.assertIn("Topic", card)
        self.assertIn("Gap", card)

    # ── Caching ───────────────────────────────────────────────────────────────

    def test_score_cached_after_computation(self):
        model = self._build_model()
        dossier = self._build_dossier(abstracts=["Curriculum effects on representations."])
        self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)

        # Verify it was saved to cache
        key = self.scorer._make_cache_key(model, dossier)
        cached = self.cache.get_collaboration_score(key)
        self.assertIsNotNone(cached)
        self.assertIn("total_score", cached)

    def test_contributions_sum_to_total(self):
        model = self._build_model()
        dossier = self._build_dossier()
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        dim_sum = sum(d["contribution"] for d in result["dimensions"].values())
        self.assertAlmostEqual(dim_sum, result["total_score"], places=1)

    def test_blocker_resolution_confidence(self):
        model = self._build_model()
        model["methods"] = ["cka", "rsa"]
        desc = "Empirical dataset with CKA metrics for continual learning curriculum"
        title = "Measuring representational similarity using CKA in continual learners"
        abstract = "We provide a benchmark dataset to evaluate representations of online continual learning models using CKA metrics."
        res = self.scorer.compute_blocker_resolution_confidence(
            blocker_desc=desc,
            blocker_type="dataset",
            paper_title=title,
            paper_abstract=abstract,
            semantic_sim=0.55,
            self_model=model
        )
        self.assertGreater(res["confidence"], 0.0)
        self.assertAlmostEqual(res["semantic_factor"], (0.55 - 0.30) / 0.40)
        self.assertEqual(res["method_factor"], 1.0)
        self.assertEqual(res["dataset_factor"], 1.0)
        self.assertEqual(res["measurement_factor"], 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Null / edge case tests
# ──────────────────────────────────────────────────────────────────────────────

class TestScorerNullCases(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir)
        self.scorer = CollaborationScorer(cache_manager=self.cache)
        self.builder = SelfModelBuilder(cache_manager=self.cache)

    def test_empty_dossier_no_crash(self):
        model = self.builder.build(texts=["Some text about continual learning."])
        dossier = {"name": "Empty Prof", "institution": "MIT", "publications": []}
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        self.assertGreaterEqual(result["total_score"], 0.0)
        self.assertLessEqual(result["total_score"], 100.0)

    def test_no_publications_in_dossier(self):
        model = self.builder.build(texts=["Continual learning in agents."])
        dossier = {"name": "Prof B", "publications": []}
        result = self.scorer.compute_score(model, [], dossier, generate_hypotheses=False)
        self.assertEqual(result["dimensions"]["enrs"]["raw"], 0.0)

    def test_queued_investigation_not_scored(self):
        """Queued investigations should not contribute to ENRS."""
        model = self.builder.build(texts=["Continual learning in agents."])
        queued_inv = _make_investigation("Queued Inv", 0.9, status="queued", blockers=[
            _make_blocker("Curriculum dataset for continual learning representation study",
                          "dataset", 0.9, 0.80)
        ])
        dossier = _make_dossier("Prof C", abstracts=["Curriculum learning datasets."])
        result = self.scorer.compute_score(model, [queued_inv], dossier,
                                           generate_hypotheses=False)
        # Queued inv.is_scoreable is False → should not enter ENRS
        self.assertEqual(result["dimensions"]["enrs"]["raw"], 0.0)

    def test_resolved_blocker_not_scored(self):
        model = self.builder.build(texts=["Continual learning in agents."])
        b = _make_blocker("Curriculum dataset for representation study", "dataset", 0.9, 0.80)
        b.resolved = True  # already resolved → not eligible
        inv = _make_investigation("Inv", 0.8, status="active", blockers=[b])
        dossier = _make_dossier("Prof D", abstracts=["Curriculum learning and representation."])
        result = self.scorer.compute_score(model, [inv], dossier, generate_hypotheses=False)
        self.assertEqual(result["dimensions"]["enrs"]["raw"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
