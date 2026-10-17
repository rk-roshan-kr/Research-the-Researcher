"""
claim_deduplicator.py — Canonical claim tracking with source paper provenance.

Problem:
    If EarthOS papers repeatedly state the same claim:
        Paper A: "Investigation persistence improves coherence"
        Paper B: "Persistent investigations maintain coherence over time"
        Paper C: "Investigation persistence is necessary for coherence"

    Naive extraction produces 3 separate claims.

    With deduplication:
        canonical_claim: "Investigation persistence improves cognitive coherence"
            ├─ source: Paper A (doi: 10.X/A, confidence: 0.91)
            ├─ source: Paper B (doi: 10.X/B, confidence: 0.88)
            └─ source: Paper C (doi: 10.X/C, confidence: 0.85)

    The canonical form is the claim with the highest confidence.
    Evidence accumulates across papers.

Design decisions:
    - Cosine similarity threshold: 0.85 (claims must be very similar to merge)
      Lower than 0.90 to catch paraphrases, higher than 0.80 to avoid over-merging.
    - Canonical form: the claim with highest confidence (not an LLM rewrite)
    - Source papers: accumulate all DOIs from merged claims
    - Confidence: weighted average of all merged claim confidences
    - Order: within each cluster, sort by confidence descending
    - Works for both strong_claims and open_questions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.85   # cosine similarity above which two claims are "the same"


# ── Canonical claim ───────────────────────────────────────────────────────────

@dataclass
class CanonicalClaim:
    """
    A deduplicated, canonical claim or question with accumulated source provenance.

    text:            The canonical phrasing (highest-confidence version)
    confidence:      Weighted average of all source confidences
    source_papers:   All DOIs + per-source confidence that assert this claim
    domain:          From the highest-confidence source
    merged_count:    How many raw claims were merged into this canonical form
    embedding:       384-dim embedding (for downstream similarity use)
    """
    text: str
    confidence: float
    source_papers: list[dict]   # [{doi, confidence, title}, ...]
    domain: str = ""
    merged_count: int = 1
    embedding: Optional[list[float]] = None

    # For questions only
    question_type: str = "empirical"

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "confidence":    round(self.confidence, 4),
            "source_papers": self.source_papers,
            "domain":        self.domain,
            "merged_count":  self.merged_count,
            "question_type": self.question_type,
        }

    @classmethod
    def from_extracted(cls, raw: dict) -> "CanonicalClaim":
        """Create from a raw ExtractedClaim/ExtractedQuestion dict."""
        source_papers = []
        for doi in raw.get("source_paper_dois", []):
            if doi and doi != "N/A":
                source_papers.append({"doi": doi, "confidence": raw.get("confidence", 0.7)})

        return cls(
            text=raw.get("text", ""),
            confidence=raw.get("confidence", 0.7),
            source_papers=source_papers,
            domain=raw.get("domain", ""),
            question_type=raw.get("question_type", "empirical"),
        )


# ── ClaimDeduplicator ─────────────────────────────────────────────────────────

class ClaimDeduplicator:
    """
    Deduplicates a list of extracted claims/questions into canonical forms.

    Works for both strong_claims and open_questions — same algorithm,
    just different input fields.
    """

    def __init__(self, similarity_threshold: float = DEDUP_THRESHOLD):
        self.threshold = similarity_threshold

    def deduplicate(self, raw_claims: list[dict]) -> list[CanonicalClaim]:
        """
        Deduplicate a list of raw extracted claim/question dicts.

        Algorithm:
            1. Embed all claim texts.
            2. Build similarity matrix.
            3. Greedy clustering: claim i starts a new cluster unless it has
               similarity >= threshold to an already-assigned cluster representative.
            4. Within each cluster, the canonical form is the highest-confidence claim.
            5. Source papers and confidences accumulate across all merged claims.

        Returns canonical claims sorted by confidence descending.
        """
        if not raw_claims:
            return []

        texts = [c.get("text", "") for c in raw_claims]
        valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not valid:
            return []

        indices, valid_texts = zip(*valid)
        indices = list(indices)

        try:
            import numpy as np
            from academic_pipeline.embedding_cache import cached_encode
            embs = cached_encode(list(valid_texts), model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
            sim_matrix = embs @ embs.T
        except Exception as e:
            logger.warning(f"Deduplication embedding failed: {e}. Returning raw claims.")
            return [CanonicalClaim.from_extracted(c) for c in raw_claims]

        # Greedy clustering
        assigned: dict[int, int] = {}   # claim_index → cluster_rep_index
        clusters: dict[int, list[int]] = {}  # rep_index → [member indices]

        for pos, idx in enumerate(indices):
            merged = False
            for rep_idx, members in clusters.items():
                rep_pos = indices.index(rep_idx)
                if float(sim_matrix[pos, rep_pos]) >= self.threshold:
                    members.append(idx)
                    assigned[idx] = rep_idx
                    merged = True
                    break
            if not merged:
                clusters[idx] = [idx]
                assigned[idx] = idx

        # Build canonical claims
        canonical: list[CanonicalClaim] = []
        for rep_idx, member_indices in clusters.items():
            cluster_raws = [raw_claims[i] for i in member_indices]

            # Canonical form = highest-confidence claim in cluster
            best = max(cluster_raws, key=lambda c: c.get("confidence", 0.0))

            # Accumulate sources from all members
            source_papers: list[dict] = []
            seen_dois = set()
            for raw in cluster_raws:
                for doi in raw.get("source_paper_dois", []):
                    if doi and doi != "N/A" and doi not in seen_dois:
                        source_papers.append({
                            "doi": doi,
                            "confidence": raw.get("confidence", 0.7),
                            "title": "",   # populated later if needed
                        })
                        seen_dois.add(doi)

            # Weighted average confidence
            confidences = [c.get("confidence", 0.7) for c in cluster_raws]
            avg_confidence = sum(confidences) / len(confidences)

            # Embedding = centroid of cluster members (for downstream use)
            rep_pos = indices.index(rep_idx)
            member_positions = [indices.index(m) for m in member_indices if m in indices]
            centroid = embs[member_positions].mean(axis=0).tolist()

            canonical.append(CanonicalClaim(
                text=best.get("text", ""),
                confidence=avg_confidence,
                source_papers=source_papers,
                domain=best.get("domain", ""),
                merged_count=len(member_indices),
                embedding=centroid,
                question_type=best.get("question_type", "empirical"),
            ))

            if len(member_indices) > 1:
                logger.info(
                    f"Merged {len(member_indices)} claims → '{best.get('text', '')[:60]}'"
                )

        result = sorted(canonical, key=lambda c: c.confidence, reverse=True)
        logger.info(
            f"Deduplication: {len(raw_claims)} raw claims → {len(result)} canonical "
            f"(threshold={self.threshold})"
        )
        return result

    def deduplicate_questions(self, raw_questions: list[dict]) -> list[CanonicalClaim]:
        """Same as deduplicate but preserves question_type field."""
        return self.deduplicate(raw_questions)


# ── Convenience: update self_model with deduplication ────────────────────────

def apply_deduplication_to_self_model(
    self_model: dict,
    similarity_threshold: float = DEDUP_THRESHOLD,
) -> dict:
    """
    Apply claim deduplication to a self_model dict (returns updated copy).

    IMPORTANT: strong_claims and open_questions remain as list[str] so that
    all downstream consumers (SentenceTransformer, investigation_engine,
    collaboration_scorer, nli_filter) continue to work without changes.

    Full provenance (confidence, source_papers, merged_count) is stored in:
        canonical_claims_provenance:   list[dict]
        canonical_questions_provenance: list[dict]

    This follows the "don't break existing interfaces" rule.
    """
    deduplicator = ClaimDeduplicator(similarity_threshold=similarity_threshold)
    updated = dict(self_model)

    raw_claims    = self_model.get("strong_claims", [])
    raw_questions = self_model.get("open_questions", [])

    # Normalise: raw items can be str (from LLM extraction) or dict (already deduped)
    def _to_dict(item) -> dict:
        if isinstance(item, str):
            return {"text": item, "confidence": 0.7, "source_paper_dois": [], "domain": ""}
        return item

    raw_claim_dicts    = [_to_dict(c) for c in raw_claims]
    raw_question_dicts = [_to_dict(q) for q in raw_questions]

    canonical_claims    = deduplicator.deduplicate(raw_claim_dicts)
    canonical_questions = deduplicator.deduplicate_questions(raw_question_dicts)

    # ── Keep existing fields as plain text strings (backward compatible) ───────
    updated["strong_claims"]  = [c.text for c in canonical_claims]
    updated["open_questions"] = [q.text for q in canonical_questions]

    # ── Full provenance in separate fields for consumers that need it ──────────
    updated["canonical_claims_provenance"]    = [c.to_dict() for c in canonical_claims]
    updated["canonical_questions_provenance"] = [q.to_dict() for q in canonical_questions]

    updated["dedup_stats"] = {
        "raw_claims":            len(raw_claims),
        "canonical_claims":      len(canonical_claims),
        "raw_questions":         len(raw_questions),
        "canonical_questions":   len(canonical_questions),
        "similarity_threshold":  similarity_threshold,
    }

    logger.info(
        f"Dedup: {len(raw_claims)} claims → {len(canonical_claims)} canonical, "
        f"{len(raw_questions)} questions → {len(canonical_questions)} canonical"
    )
    return updated
