"""
nli_filter.py — Natural Language Inference gate for contradiction detection.

Inserts between embedding-based candidate selection and LLM verification:

    Claims × Professor Papers
         ↓ cosine similarity
    Candidate Pairs (sim >= threshold)
         ↓ NLI (this file)
    Confirmed Contradiction Candidates
         ↓ LLM analysis (local_llm.py)
    ContradictionAnalysis objects

This eliminates the dominant weakness in Week 2's detect_contradictions_from_dossier:
    - Embedding similarity = "same topic"
    - NLI "contradiction" label = "opposing claims"

These are categorically different problems. Embedding similarity alone will flag:
    "Curriculum learning helps"
    "Curriculum learning helps more than expected"
as a contradiction candidate. NLI will correctly label this as ENTAILMENT.

Model choice: cross-encoder/nli-deberta-v3-base
    - 86M parameters (~350MB download)
    - Best-in-class on MNLI/SNLI for a locally runnable model
    - Better than sentence-transformers NLI because it uses cross-encoding
      (both texts processed jointly), not independent embeddings
    - ~100-200ms per pair on CPU

Alternatives (larger, more accurate):
    - cross-encoder/nli-deberta-v3-large  (400M params)
    - facebook/bart-large-mnli            (400M params)

Labels: "contradiction", "neutral", "entailment"

Design decisions:
    - Model is lazy-loaded on first call and cached process-wide.
    - Batch scoring: all pairs scored together for efficiency.
    - Returns NLIResult with label, scores for all three classes, and pair metadata.
    - Threshold is configurable: default CONTRADICTION_THRESHOLD = 0.60
      (lower than typical 0.80 because we want to pass genuine ambiguous cases
       to the LLM for final judgment, not discard them prematurely).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NLI_MODEL_NAME        = "cross-encoder/nli-deberta-v3-large"
CONTRADICTION_THRESHOLD = 0.60   # pass to LLM if P(contradiction) >= this
ENTAILMENT_THRESHOLD    = 0.75   # skip pair if P(entailment) >= this (not a contradiction)

# ── Lazy model cache ──────────────────────────────────────────────────────────

_NLI_MODEL = None

def _get_nli_model():
    """Lazy-load the NLI cross-encoder. Warm on first call (~3s), cached after."""
    global _NLI_MODEL
    if _NLI_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading NLI model: {NLI_MODEL_NAME}")
            _NLI_MODEL = CrossEncoder(NLI_MODEL_NAME)
            logger.info("NLI model loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for NLI filtering. "
                "Run: pip install sentence-transformers"
            )
    return _NLI_MODEL


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class NLIResult:
    """Result for a single (claim_yours, claim_external) pair."""
    claim_yours:    str
    claim_external: str
    source_external: str

    # Probabilities for each class (sum to ~1.0)
    p_contradiction: float = 0.0
    p_neutral:       float = 0.0
    p_entailment:    float = 0.0

    # Derived verdict
    label: str = "neutral"   # "contradiction" | "neutral" | "entailment"
    passes_filter: bool = False  # True if P(contradiction) >= CONTRADICTION_THRESHOLD

    # Optional: source metadata for tracing
    similarity: float = 0.0   # embedding similarity that originally flagged this pair
    metadata: dict = field(default_factory=dict)

    # Quality flags — populated when NLI is unavailable or a scoring error occurs.
    # Consumers (collaboration_scorer, investigation_engine) should surface these.
    # Known flags:
    #   "nli_unavailable"   — NLI model not installed; fallback to LLM-only
    #   "nli_scoring_error" — NLI ran but threw an exception mid-batch
    quality_flags: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """P(contradiction) when passes_filter is True, else 0.0."""
        return self.p_contradiction if self.passes_filter else 0.0

    @property
    def is_degraded(self) -> bool:
        """True if this result was produced under degraded (non-NLI) conditions."""
        return len(self.quality_flags) > 0


def nli_label_from_scores(scores: list[float]) -> str:
    """
    CrossEncoder NLI models return scores in order: [contradiction, entailment, neutral]
    Map to label string.
    """
    # DeBERTa-v3-base order: contradiction=0, entailment=1, neutral=2
    labels = ["contradiction", "entailment", "neutral"]
    max_idx = max(range(len(scores)), key=lambda i: scores[i])
    return labels[max_idx]


# ── NLI Filter ────────────────────────────────────────────────────────────────

class NLIFilter:
    """
    Filters embedding-based contradiction candidates using NLI.

    Usage:
        nli = NLIFilter()
        results = nli.filter_candidates(candidates)
        confirmed = [r for r in results if r.passes_filter]
    """

    def __init__(
        self,
        contradiction_threshold: float = CONTRADICTION_THRESHOLD,
        entailment_threshold:    float = ENTAILMENT_THRESHOLD,
        model_name: str = NLI_MODEL_NAME,
    ):
        self.contradiction_threshold = contradiction_threshold
        self.entailment_threshold    = entailment_threshold
        self.model_name = model_name
        self._model = None  # lazy

    def _model_loaded(self):
        if self._model is None:
            self._model = _get_nli_model()
        return self._model

    def filter_candidates(
        self,
        candidates: list[dict],
        batch_size: int = 32,
    ) -> list[NLIResult]:
        """
        Score a list of candidate contradiction pairs with NLI.

        Each candidate dict must have:
            claim_yours:     str
            claim_external:  str   (or abstract_snippet)
            source_external: str
            similarity:      float (from embedding stage)

        Returns NLIResult list (all pairs, not just passing ones).
        Caller filters with: [r for r in results if r.passes_filter]
        """
        if not candidates:
            return []

        try:
            model = self._model_loaded()
        except ImportError as e:
            logger.error(f"NLI model unavailable: {e}")
            # Graceful degradation: treat all candidates as passing
            return self._fallback_no_nli(candidates)

        # Build (premise, hypothesis) pairs for the cross-encoder
        # Convention: premise = external claim, hypothesis = your claim
        # "Does the external paper's claim contradict your claim?"
        # Optimization: Dual-encoder pre-filtering. Only send pairs with similarity >= 0.50 to the cross-encoder.
        pairs = []
        cross_encoder_indices = []
        results = [None] * len(candidates)

        for i, c in enumerate(candidates):
            sim = c.get("similarity", 1.0)
            if sim < 0.50:
                # Bypassed: mark immediately as neutral/not contradiction
                results[i] = NLIResult(
                    claim_yours=c.get("claim_yours", ""),
                    claim_external=c.get("claim_external") or c.get("abstract_snippet", ""),
                    source_external=c.get("source_external", ""),
                    p_contradiction=0.0,
                    p_neutral=1.0,
                    p_entailment=0.0,
                    label="neutral",
                    passes_filter=False,
                    similarity=sim,
                    metadata={k: v for k, v in c.items()
                              if k not in ("claim_yours", "claim_external", "abstract_snippet",
                                           "source_external", "similarity")},
                )
            else:
                premise    = c.get("claim_external") or c.get("abstract_snippet", "")
                hypothesis = c.get("claim_yours", "")
                pairs.append((premise[:512], hypothesis[:512]))
                cross_encoder_indices.append(i)

        if not pairs:
            logger.info("All NLI candidate pairs filtered out by dual-encoder similarity pre-filter.")
            return results

        logger.info(f"NLI scoring {len(pairs)}/{len(candidates)} candidate pairs (batch_size={batch_size})")

        # CrossEncoder.predict returns shape (N, 3) for NLI models
        try:
            import numpy as np
            raw_scores = model.predict(pairs, batch_size=batch_size)

            # Apply softmax to convert logits to probabilities
            def softmax(x):
                e = np.exp(x - np.max(x))
                return e / e.sum()

            for idx, (c_idx, scores) in enumerate(zip(cross_encoder_indices, raw_scores)):
                candidate = candidates[c_idx]
                probs = softmax(scores)
                # DeBERTa-v3-base NLI label order: contradiction=0, entailment=1, neutral=2
                p_contradiction = float(probs[0])
                p_entailment    = float(probs[1])
                p_neutral       = float(probs[2])

                label = nli_label_from_scores(probs.tolist())
                passes = (
                    p_contradiction >= self.contradiction_threshold
                    and p_entailment < self.entailment_threshold
                )

                results[c_idx] = NLIResult(
                    claim_yours=candidate.get("claim_yours", ""),
                    claim_external=candidate.get("claim_external") or candidate.get("abstract_snippet", ""),
                    source_external=candidate.get("source_external", ""),
                    p_contradiction=p_contradiction,
                    p_neutral=p_neutral,
                    p_entailment=p_entailment,
                    label=label,
                    passes_filter=passes,
                    similarity=candidate.get("similarity", 0.0),
                    metadata={k: v for k, v in candidate.items()
                              if k not in ("claim_yours", "claim_external", "abstract_snippet",
                                           "source_external", "similarity")},
                )

            n_passing = sum(1 for r in results if r.passes_filter)
            logger.info(
                f"NLI filter: {n_passing}/{len(results)} pairs passed "
                f"(threshold={self.contradiction_threshold})"
            )
            return results

        except Exception as e:
            logger.error(f"NLI scoring failed: {e}")
            return self._fallback_no_nli(candidates, flag="nli_scoring_error")

    def _fallback_no_nli(self, candidates: list[dict], flag: str = "nli_unavailable") -> list[NLIResult]:
        """
        Fallback when NLI model is unavailable or throws.
        Marks all candidates as passing with low confidence (0.40).
        Sets quality_flags=[flag] on every result so consumers can detect degradation.
        The LLM stage will still verify - this is not silent failure, just visible fallback.
        """
        logger.warning(
            f"NLI filter unavailable ({flag}). All embedding candidates will pass to LLM stage. "
            "Install sentence-transformers to enable NLI filtering."
        )
        return [
            NLIResult(
                claim_yours=c.get("claim_yours", ""),
                claim_external=c.get("claim_external") or c.get("abstract_snippet", ""),
                source_external=c.get("source_external", ""),
                p_contradiction=0.40,
                p_neutral=0.40,
                p_entailment=0.20,
                label="neutral",        # conservative label
                passes_filter=True,    # pass to LLM for final judgment
                similarity=c.get("similarity", 0.0),
                quality_flags=[flag],  # VISIBLE degradation signal
            )
            for c in candidates
        ]

    def score_single(self, claim_yours: str, claim_external: str) -> NLIResult:
        """
        Score a single pair. Convenience method for interactive use.
        """
        result = self.filter_candidates([{
            "claim_yours": claim_yours,
            "claim_external": claim_external,
            "source_external": "direct input",
            "similarity": 1.0,
        }])
        return result[0] if result else NLIResult(
            claim_yours=claim_yours,
            claim_external=claim_external,
            source_external="direct input",
        )

    def is_available(self) -> bool:
        """Returns True if the NLI model can be loaded."""
        try:
            from sentence_transformers import CrossEncoder
            return True
        except ImportError:
            return False

    def find_contradiction_candidates(
        self,
        my_claims: list[dict],
        their_claims: list[dict],
        embedding_threshold: float = 0.40,
    ) -> list[tuple[dict, dict]]:
        """
        Batched contradiction matching:
        1. Encodes claim texts.
        2. Filters by cosine similarity (> embedding_threshold).
        3. Runs DeBERTa NLI on candidate pairs to detect contradiction probability.
        4. Returns list of tuples (my_claim_dict, their_claim_dict) that pass the NLI threshold.
        """
        if not my_claims or not their_claims:
            return []

        from academic_pipeline.embedding_cache import cached_encode
        import numpy as np

        my_texts = [c["claim"] for c in my_claims]
        their_texts = [c["claim"] for c in their_claims]

        my_embs = cached_encode(my_texts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
        their_embs = cached_encode(their_texts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)

        sim_matrix = my_embs @ their_embs.T

        candidates = []
        for i, my_c in enumerate(my_claims):
            for j, their_c in enumerate(their_claims):
                sim = float(sim_matrix[i, j])
                if sim >= embedding_threshold:
                    candidates.append((my_c, their_c, sim))

        if not candidates:
            return []

        nli_pairs = [(item[1]["claim"][:512], item[0]["claim"][:512]) for item in candidates]
        
        try:
            model = self._model_loaded()
            raw_scores = model.predict(nli_pairs, batch_size=32)
            scores_arr = np.asarray(raw_scores)
            
            if np.max(scores_arr) > 1.0 or np.min(scores_arr) < 0.0:
                exp_scores = np.exp(scores_arr - np.max(scores_arr, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                probs = scores_arr

            p_contradictions = probs[:, 0]
            
            passing_pairs = []
            for idx, p_c in enumerate(p_contradictions):
                if p_c >= self.contradiction_threshold:
                    my_c, their_c, sim = candidates[idx]
                    passing_pairs.append((my_c, their_c))
            return passing_pairs
        except Exception as e:
            logger.error(f"NLI batch scoring failed in find_contradiction_candidates: {e}")
            return [(item[0], item[1]) for item in candidates]


# ── Updated detect_contradictions pipeline ────────────────────────────────────

def build_contradiction_candidates(
    strong_claims: list[str],
    paper_abstracts: list[str],
    paper_meta: list[dict],
    embedding_threshold: float = 0.70,
) -> list[dict]:
    """
    Step 1 of the contradiction pipeline: embedding-based candidate selection.

    Returns candidate dicts ready for NLI filtering:
    {
        claim_yours: str,
        claim_external: str,    # abstract snippet
        source_external: str,   # "Paper Title (doi)"
        similarity: float,
    }
    """
    if not strong_claims or not paper_abstracts:
        return []

    from academic_pipeline.embedding_cache import cached_encode

    claim_embs   = cached_encode(strong_claims,   model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
    abstract_embs= cached_encode(paper_abstracts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)

    # (n_claims, n_abstracts)
    sim_matrix = claim_embs @ abstract_embs.T

    candidates = []
    import numpy as np
    for i, claim in enumerate(strong_claims):
        for j, abstract in enumerate(paper_abstracts):
            sim = float(sim_matrix[i, j])
            if sim >= embedding_threshold:
                meta = paper_meta[j] if j < len(paper_meta) else {}
                title = meta.get("title", f"Paper {j+1}")
                doi   = meta.get("doi",   "")
                candidates.append({
                    "claim_yours":    claim,
                    "claim_external": abstract[:600],
                    "source_external": f"{title} ({doi})" if doi else title,
                    "similarity":     sim,
                    "paper_meta":     meta,
                })

    logger.info(
        f"Embedding stage: {len(candidates)} candidates "
        f"from {len(strong_claims)} claims × {len(paper_abstracts)} papers "
        f"(threshold={embedding_threshold})"
    )
    return candidates
