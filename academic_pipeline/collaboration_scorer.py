# -*- coding: utf-8 -*-
"""
collaboration_scorer.py - Computes collaboration potential between the user and a professor.

Every score is individually inspectable with full provenance:
    {
        "score": 21.4,
        "reasoning": "Prof has paper matching blocker on curriculum datasets (sim=0.88)",
        "evidence": [{"paper_title": "...", "doi": "...", "similarity": 0.88, ...}]
    }

This design is mandatory because:
    - Six months later you'll ask "Why did Prof X rank #1?"
    - With 200+ professors in memory, rankings must be auditable.
    - Debugging bad scores without provenance is impossible.

Weight distribution (approved):
    ENRS (Evidence Need Resolution):  30%  <- dominant signal, blocker-based
    Topic Overlap:                     20%
    Gap Complement:                    20%
    Method Compatibility:              15%
    Lab Accessibility:                 10%
    Temporal Activity:                  5%
    Total:                            100%

All 6 dimensions are deterministic (embeddings + rules).
LLM is called AFTER scoring to generate hypotheses, not DURING.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from academic_pipeline.investigation_schemas import (
    InvestigationStream,
    ResearchBlocker,
    BLOCKER_TYPE_MULTIPLIERS,
    SPECIFICITY_THRESHOLD,
)
from academic_pipeline.embedding_cache import cached_encode

# ?? Score weights (sum = 1.0) ?????????????????????????????????????????????????

DEFAULT_WEIGHTS = {
    "enrs":            0.30,  # Evidence Need Resolution Score
    "topic_overlap":   0.20,
    "gap_complement":  0.20,
    "method_compat":   0.15,
    "lab_access":      0.10,
    "temporal_act":    0.05,
}

# Similarity threshold below which a paper is ignored in ENRS
ENRS_SIM_THRESHOLD = 0.30


# ?? Score provenance dataclass ????????????????????????????????????????????????

def _dim_score(
    raw: float,
    weight: float,
    reasoning: str,
    evidence: Optional[list[dict]] = None,
) -> dict:
    """
    Wraps a single dimension score with its weight, contribution, and provenance.

    raw:      0.0-1.0, the raw dimension score
    weight:   fraction of total (e.g. 0.30 for ENRS)
    reasoning: one-sentence explanation of how the score was computed
    evidence: list of supporting evidence dicts (papers, blockers, etc.)
    """
    return {
        "raw": round(raw, 4),
        "weight": weight,
        "contribution": round(raw * weight * 100, 2),  # points out of 100
        "reasoning": reasoning,
        "evidence": evidence or [],
    }


# ?? CollaborationScorer ???????????????????????????????????????????????????????

class CollaborationScorer:
    """
    Computes a collaboration score for (self_model, investigations, prof_dossier).

    All scoring is deterministic (embeddings + rules).
    LLM is invoked only after scoring, for hypothesis generation.
    """

    def __init__(
        self,
        cache_manager=None,
        llm_client=None,
        weights: Optional[dict] = None,
    ):
        self._cache = cache_manager
        self._llm = llm_client
        self._weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        # Normalize weights to sum to 1.0
        total = sum(self._weights.values())
        self._weights = {k: v / total for k, v in self._weights.items()}

    def adjust_weights_from_crm(self):
        """
        Queries the cache for historical professor memories and outcomes
        and dynamically adjusts dimension weights to prioritize dimensions
        that correlate with successful outcomes (e.g., high reply rates, active threads).
        """
        if not self._cache:
            return

        try:
            memories = self._cache.list_professor_memories()
            if not memories:
                return

            total_replies = 0
            access_success_scores = []

            for mem in memories:
                replies = mem.get("reply_count", 0) or 0
                status = mem.get("status", "")
                is_success = replies > 0 or status in ("in_conversation", "resolved")

                if is_success:
                    total_replies += 1
                    h_index = mem.get("h_index", 30) or 30
                    access_est = 1.0 if h_index < 20 else (0.5 if h_index < 40 else 0.2)
                    access_success_scores.append(access_est)

            if total_replies > 0:
                avg_access = float(np.mean(access_success_scores)) if access_success_scores else 0.5
                new_weights = dict(DEFAULT_WEIGHTS)
                
                # Dynamic learning bias:
                if avg_access > 0.60:
                    # Assist/Associate PIs reply more: increase lab_access weight
                    new_weights["lab_access"] += 0.05
                    new_weights["topic_overlap"] -= 0.05
                elif avg_access < 0.40:
                    # Elite PIs are responding! Increase ENRS and gap complement weight
                    new_weights["enrs"] += 0.05
                    new_weights["gap_complement"] += 0.05
                    new_weights["lab_access"] -= 0.10
                
                # Normalize
                total = sum(new_weights.values())
                self._weights = {k: v / total for k, v in new_weights.items()}
                logger.info(f"Dynamic weights adjusted based on CRM feedback: {self._weights}")
        except Exception as e:
            logger.warning(f"Failed to adjust weights from CRM: {e}")

    def compute_score(
        self,
        self_model: dict,
        investigations: list[InvestigationStream],
        prof_dossier: dict,
        generate_hypotheses: bool = True,
    ) -> dict:
        """
        Compute the full collaboration score.

        Returns a fully-provenant score dict:
        {
            "professor_name": str,
            "institution": str,
            "total_score": float,           # 0-100
            "dimensions": {
                "enrs":           {raw, weight, contribution, reasoning, evidence},
                "topic_overlap":  {...},
                "gap_complement": {...},
                "method_compat":  {...},
                "lab_access":     {...},
                "temporal_act":   {...},
            },
            "collaboration_hypotheses": list[str],   # [] if generate_hypotheses=False
            "resolution_matches": list[dict],         # which blockers match which papers
            "scored_at": float,
        }
        """
        # Dynamically adjust weight vectors based on CRM history (Layer 10: Outcome Learning)
        self.adjust_weights_from_crm()

        from academic_pipeline.self_model import get_embedder
        embedder = get_embedder()
        prof_name = prof_dossier.get("name", "Unknown")
        institution = prof_dossier.get("institution", "")

        # Build professor embedding vectors (cached per call)
        prof_vec = self._build_professor_vector(prof_dossier, embedder)

        # Only score active/blocked investigations with eligible blockers
        scoreable = [inv for inv in investigations if inv.is_scoreable]

        # ?? Compute all 6 dimensions ??????????????????????????????????????????
        enrs_score, enrs_evidence, resolution_matches = self._score_enrs(
            scoreable, prof_vec, self_model
        )
        topic_score, topic_evidence = self._score_topic_overlap(self_model, prof_vec)
        gap_score, gap_evidence = self._score_gap_complement(self_model, prof_vec)
        method_score, method_evidence = self._score_method_compat(self_model, prof_vec)
        access_score, access_evidence = self._score_lab_accessibility(prof_dossier)
        temporal_score, temporal_evidence = self._score_temporal_activity(prof_dossier)

        w = self._weights
        dimensions = {
            "enrs": _dim_score(
                enrs_score, w["enrs"],
                self._enrs_reasoning(enrs_score, resolution_matches),
                enrs_evidence,
            ),
            "topic_overlap": _dim_score(
                topic_score, w["topic_overlap"],
                self._topic_reasoning(topic_score),
                topic_evidence,
            ),
            "gap_complement": _dim_score(
                gap_score, w["gap_complement"],
                self._gap_reasoning(gap_score, gap_evidence),
                gap_evidence,
            ),
            "method_compat": _dim_score(
                method_score, w["method_compat"],
                self._method_reasoning(method_score),
                method_evidence,
            ),
            "lab_access": _dim_score(
                access_score, w["lab_access"],
                self._access_reasoning(access_score, prof_dossier),
                access_evidence,
            ),
            "temporal_act": _dim_score(
                temporal_score, w["temporal_act"],
                self._temporal_reasoning(temporal_score, prof_dossier),
                temporal_evidence,
            ),
        }

        total = sum(d["contribution"] for d in dimensions.values())

        # ?? LLM hypothesis generation (post-scoring, not during) ??????????????
        hypotheses: list[str] = []
        if generate_hypotheses and self._llm is not None and resolution_matches:
            try:
                hypotheses = self._llm.generate_collaboration_hypotheses(
                    self_model=self_model,
                    prof_dossier=prof_dossier,
                    resolution_matches=resolution_matches,
                    total_score=total,
                )
            except Exception as e:
                logger.warning(f"Hypothesis generation failed: {e}")

        result = {
            "professor_name": prof_name,
            "institution": institution,
            "total_score": round(total, 2),
            "dimensions": dimensions,
            "collaboration_hypotheses": hypotheses,
            "resolution_matches": resolution_matches,
            "paper_embeddings": prof_vec.get("paper_embeddings", []),
            "paper_meta": prof_vec.get("paper_meta", []),
            "scored_at": time.time(),
        }

        # Cache the score if cache_manager available
        if self._cache is not None:
            cache_key = self._make_cache_key(self_model, prof_dossier)
            self._cache.save_collaboration_score(cache_key, result)

        return result

    # ?? Dimension 1: ENRS - Evidence Need Resolution Score (30%) ?????????????

    def compute_blocker_resolution_confidence(
        self,
        blocker_desc: str,
        blocker_type: str,
        paper_title: str,
        paper_abstract: str,
        semantic_sim: float,
        self_model: dict,
    ) -> dict:
        """
        Compute the dynamic Blocker Resolution Confidence.
        Returns a dictionary with the composite confidence score and individual factor details.
        """
        # 1. Semantic alignment (normalized in active range [0.30, 0.70])
        if semantic_sim <= 0.30:
            semantic_factor = 0.0
        elif semantic_sim >= 0.70:
            semantic_factor = 1.0
        else:
            semantic_factor = (semantic_sim - 0.30) / 0.40

        # Prepare lowercase text for keyword matching
        paper_text = (paper_title + " " + paper_abstract).lower()
        blocker_text_lower = blocker_desc.lower()

        # Helper to check token match
        import re
        def contains_word(text, word):
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            return bool(re.search(pattern, text))

        # 2. Method Overlap
        method_factor = 0.0
        
        # User methods
        user_methods = self_model.get("methods", []) or self_model.get("methods_used", []) or []
        user_method_match = False
        for m in user_methods:
            if m.strip() and contains_word(paper_text, m.strip()):
                user_method_match = True
                break

        # Check overlap of blocker description words with paper (excluding generic stop words)
        stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", 
            "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", 
            "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", 
            "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", 
            "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", 
            "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", 
            "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", 
            "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", 
            "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", 
            "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", 
            "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
            "wasnt", "we", "wed", "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", 
            "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", 
            "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "the", "we", "our",
            "us", "need", "needs", "want", "wants", "require", "requires", "blocking", "blocker"
        }
        
        # Specific high-value method words/acronyms to look for
        key_method_terms = ["cka", "rsa", "cebra", "representation", "continual", "curriculum", "latent", "dynamical", "dynamics"]
        
        blocker_words = re.findall(r'\b\w+\b', blocker_text_lower)
        blocker_method_words = [w for w in blocker_words if w not in stopwords and (len(w) > 3 or w in key_method_terms)]
        
        blocker_method_match = False
        for w in blocker_method_words:
            if contains_word(paper_text, w):
                blocker_method_match = True
                break

        if user_method_match and blocker_method_match:
            method_factor = 1.0
        elif user_method_match or blocker_method_match:
            method_factor = 0.8
        else:
            general_methods = ["method", "algorithm", "framework", "architecture", "model", "approach", "network", "system", "pipeline", "optimization"]
            if any(contains_word(paper_text, gm) for gm in general_methods):
                method_factor = 0.4
            else:
                method_factor = 0.0

        # 3. Dataset Compatibility
        dataset_keywords = ["dataset", "data", "corpus", "benchmark", "cohort", "database", "recordings", "measurements", "subjects", "samples"]
        has_dataset_need = (blocker_type == "dataset") or any(contains_word(blocker_text_lower, dk) for dk in dataset_keywords)
        
        if has_dataset_need:
            if any(contains_word(paper_text, dk) for dk in dataset_keywords):
                dataset_factor = 1.0
            else:
                dataset_factor = 0.0
        else:
            if any(contains_word(paper_text, dk) for dk in dataset_keywords):
                dataset_factor = 0.8
            else:
                dataset_factor = 1.0

        # 4. Measurement/Metric Fit
        measurement_keywords = ["cka", "rsa", "cebra", "similarity", "metric", "measure", "correlation", "distance", "coherence", "metrology", "evaluation", "score", "performance", "accuracy", "error", "loss"]
        blocker_meas_words = [w for w in blocker_words if w in measurement_keywords]
        
        meas_overlap = False
        for mw in blocker_meas_words:
            if contains_word(paper_text, mw):
                meas_overlap = True
                break
                
        if meas_overlap:
            measurement_factor = 1.0
        else:
            if any(contains_word(paper_text, mk) for mk in measurement_keywords):
                measurement_factor = 0.7
            else:
                measurement_factor = 0.0

        raw_weights = {
            "semantic": 0.30,
            "method": 0.25,
            "dataset": 0.25 if blocker_type == "dataset" else 0.00,
            "measurement": 0.20 if blocker_type == "methodology" else 0.10,
        }
        total_w = sum(raw_weights.values())
        norm_weights = {k: v / total_w for k, v in raw_weights.items()}

        confidence = (
            norm_weights["semantic"] * semantic_factor +
            norm_weights["method"] * method_factor +
            norm_weights["dataset"] * dataset_factor +
            norm_weights["measurement"] * measurement_factor
        )

        return {
            "confidence": round(confidence, 4),
            "semantic_factor": round(semantic_factor, 4),
            "method_factor": round(method_factor, 4),
            "dataset_factor": round(dataset_factor, 4),
            "measurement_factor": round(measurement_factor, 4),
            "weights": {k: round(v, 4) for k, v in norm_weights.items()}
        }

    def compute_blocker_resolution_probability(
        self,
        blocker: ResearchBlocker,
        paper_meta: list[dict],
        paper_embs: np.ndarray,
        blocker_emb: np.ndarray,
        program_claims: list[dict],
        program_methods: list[str],
        program_datasets: list[str],
        program_metrics: list[str],
    ) -> tuple[float, str, dict]:
        """
        Compute the probability (0.0 to 1.0) that the program resolves the blocker,
        incorporating semantic similarity and structured constraints.
        """
        # Calculate semantic similarities between paper embeddings and blocker embedding
        sims = (paper_embs @ blocker_emb).astype(float)
        max_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
        max_sim = float(sims[max_idx]) if max_idx >= 0 else 0.0

        if max_sim < ENRS_SIM_THRESHOLD:
            return 0.0, f"Below semantic threshold ({ENRS_SIM_THRESHOLD}), probability is 0.0", {
                "max_similarity": max_sim,
                "best_paper_idx": max_idx,
                "semantic_factor": 0.0,
                "method_factor": 0.0,
                "dataset_factor": 0.0,
                "measurement_factor": 0.0
            }

        # Linear mapping of max_sim from [0.30, 0.70] to [0.10, 0.90]
        semantic_factor = 0.10 + 0.80 * max(0.0, min(1.0, (max_sim - 0.30) / 0.40))
        prob = semantic_factor

        # Standard check helper for substring matching
        def is_match(target: str, items: list[str]) -> bool:
            if not target or target.upper() in ("N/A", "NONE", ""):
                return False
            target_lower = target.lower().strip()
            for item in items:
                if not item:
                    continue
                item_lower = item.lower().strip()
                if target_lower in item_lower or item_lower in target_lower:
                    return True
            return False

        # Gather paper texts for fallback search
        paper_texts = [
            (p.get("title", "") + " " + p.get("abstract", "")).lower()
            for p in paper_meta
        ]

        updates = []
        method_factor = 0.5
        dataset_factor = 0.5
        measurement_factor = 0.5

        # 1. Method overlap
        method_required = blocker.required_method not in ("N/A", "NONE", "")
        if method_required:
            claim_methods = [c.get("method", "") for c in program_claims]
            all_program_methods = program_methods + claim_methods
            
            # Semantic method similarity
            try:
                method_emb = cached_encode(
                    [blocker.required_method], model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True
                )[0]
                method_sims = paper_embs @ method_emb
                max_method_sim = float(np.max(method_sims)) if len(method_sims) > 0 else 0.0
            except Exception:
                max_method_sim = 0.0

            if max_method_sim >= 0.50 or is_match(blocker.required_method, all_program_methods) or any(blocker.required_method.lower() in pt for pt in paper_texts):
                prob = prob + (1.0 - prob) * 0.5
                method_factor = 1.0
                updates.append(f"Matched required method '{blocker.required_method}' (sem={max_method_sim:.2f})")
            else:
                prob = prob * 0.5
                method_factor = 0.0
                updates.append(f"Missing required method '{blocker.required_method}' (sem={max_method_sim:.2f})")

        # 2. Dataset compatibility
        dataset_required = blocker.required_dataset not in ("N/A", "NONE", "")
        if dataset_required:
            claim_datasets = [c.get("dataset", "") for c in program_claims]
            all_program_datasets = program_datasets + claim_datasets
            
            # Semantic dataset similarity
            try:
                dataset_emb = cached_encode(
                    [blocker.required_dataset], model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True
                )[0]
                dataset_sims = paper_embs @ dataset_emb
                max_dataset_sim = float(np.max(dataset_sims)) if len(dataset_sims) > 0 else 0.0
            except Exception:
                max_dataset_sim = 0.0

            if max_dataset_sim >= 0.50 or is_match(blocker.required_dataset, all_program_datasets) or any(blocker.required_dataset.lower() in pt for pt in paper_texts):
                prob = prob + (1.0 - prob) * 0.5
                dataset_factor = 1.0
                updates.append(f"Matched required dataset '{blocker.required_dataset}' (sem={max_dataset_sim:.2f})")
            else:
                prob = prob * 0.5
                dataset_factor = 0.0
                updates.append(f"Missing required dataset '{blocker.required_dataset}' (sem={max_dataset_sim:.2f})")

        # 3. Measurement/Metric fit
        metric_required = blocker.required_metric not in ("N/A", "NONE", "")
        if metric_required:
            claim_metrics = [c.get("metric", "") for c in program_claims]
            all_program_metrics = program_metrics + claim_metrics
            
            # Semantic metric similarity
            try:
                metric_emb = cached_encode(
                    [blocker.required_metric], model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True
                )[0]
                metric_sims = paper_embs @ metric_emb
                max_metric_sim = float(np.max(metric_sims)) if len(metric_sims) > 0 else 0.0
            except Exception:
                max_metric_sim = 0.0

            if max_metric_sim >= 0.50 or is_match(blocker.required_metric, all_program_metrics) or any(blocker.required_metric.lower() in pt for pt in paper_texts):
                prob = prob + (1.0 - prob) * 0.5
                measurement_factor = 1.0
                updates.append(f"Matched required metric '{blocker.required_metric}' (sem={max_metric_sim:.2f})")
            else:
                prob = prob * 0.5
                measurement_factor = 0.0
                updates.append(f"Missing required metric '{blocker.required_metric}' (sem={max_metric_sim:.2f})")

        prob = max(0.0, min(1.0, prob))
        if not updates:
            rationale = f"Semantic overlap probability: {prob:.2f} (no explicit constraints)."
        else:
            rationale = f"Probability={prob:.2f} based on constraints: {'; '.join(updates)}."

        return prob, rationale, {
            "max_similarity": max_sim,
            "best_paper_idx": max_idx,
            "semantic_factor": semantic_factor,
            "method_factor": method_factor,
            "dataset_factor": dataset_factor,
            "measurement_factor": measurement_factor
        }

    def _score_enrs(
        self,
        investigations: list[InvestigationStream],
        prof_vec: dict,
        self_model: dict,
    ) -> tuple[float, list[dict], list[dict]]:
        """
        ENRS = weighted average of best-blocker resolution probability across investigations.
        """
        paper_embeddings = prof_vec.get("paper_embeddings")
        if paper_embeddings is None or len(paper_embeddings) == 0 or not investigations:
            return 0.0, [{"note": "No professor papers with embeddings available"}], []

        paper_embs = np.array(paper_embeddings)  # (N_papers, 384)
        paper_meta = prof_vec.get("paper_meta", [])  # [{title, doi, year, abstract}, ...]

        program_claims = prof_vec.get("program_claims", [])
        program_methods = prof_vec.get("program_methods", [])
        program_datasets = prof_vec.get("program_datasets", [])
        program_metrics = prof_vec.get("program_metrics", [])

        numerator = 0.0
        denominator = 0.0
        evidence: list[dict] = []
        resolution_matches: list[dict] = []

        for inv in investigations:
            inv_weight = inv.effective_priority
            for blocker in inv.open_blockers:
                blocker_weight = blocker.effective_priority
                joint_weight = inv_weight * blocker_weight
                denominator += joint_weight

                if blocker.embedding is None:
                    emb = cached_encode(
                        [blocker.description], model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True
                    )[0]
                    blocker.embedding = emb.tolist()
                else:
                    emb = np.array(blocker.embedding)

                # Compute probability
                prob, rationale, details = self.compute_blocker_resolution_probability(
                    blocker=blocker,
                    paper_meta=paper_meta,
                    paper_embs=paper_embs,
                    blocker_emb=emb,
                    program_claims=program_claims,
                    program_methods=program_methods,
                    program_datasets=program_datasets,
                    program_metrics=program_metrics,
                )

                best_paper_idx = details["best_paper_idx"]
                if best_paper_idx >= 0:
                    type_mult = BLOCKER_TYPE_MULTIPLIERS.get(blocker.blocker_type, 1.0)
                    resolved_contribution = min(1.0, prob * type_mult)
                    numerator += joint_weight * resolved_contribution

                    best_paper = paper_meta[best_paper_idx] if best_paper_idx < len(paper_meta) else {}
                    match_record = {
                        "investigation_title": inv.title,
                        "investigation_id": inv.id,
                        "blocker_id": blocker.id,
                        "blocker_description": blocker.description,
                        "blocker_type": blocker.blocker_type,
                        "blocker_severity": blocker.severity,
                        "blocker_specificity": blocker.specificity_score,
                        "paper_title": best_paper.get("title", ""),
                        "paper_doi": best_paper.get("doi", ""),
                        "paper_year": best_paper.get("year"),
                        "similarity": round(details["max_similarity"], 4),
                        "confidence": prob,
                        "semantic_factor": details["semantic_factor"],
                        "method_factor": details["method_factor"],
                        "dataset_factor": details["dataset_factor"],
                        "measurement_factor": details["measurement_factor"],
                        "type_multiplier": type_mult,
                        "adjusted_similarity": round(resolved_contribution, 4),
                        "rationale": rationale,
                    }
                    evidence.append(match_record)
                    resolution_matches.append(match_record)
                else:
                    max_sim = details["max_similarity"]
                    evidence.append({
                        "blocker_description": blocker.description[:80],
                        "best_similarity": round(max_sim, 4),
                        "note": f"Below threshold ({ENRS_SIM_THRESHOLD}), no resolution",
                        "rationale": rationale,
                    })

        score = numerator / denominator if denominator > 0 else 0.0
        return min(1.0, score), evidence, resolution_matches

    # ?? Dimension 2: Topic Overlap (20%) ??????????????????????????????????????

    def _score_topic_overlap(
        self, self_model: dict, prof_vec: dict
    ) -> tuple[float, list[dict]]:
        """
        Cosine similarity between user's topic_embedding and professor's topic_centroid.
        """
        user_emb = np.array(self_model.get("topic_embedding", []))
        prof_emb = np.array(prof_vec.get("topic_centroid", []))

        if len(user_emb) == 0 or len(prof_emb) == 0 or len(user_emb) != len(prof_emb):
            return 0.0, [{"note": "Missing embeddings"}]

        sim = float(np.dot(user_emb, prof_emb))  # both normalized
        sim = max(0.0, min(1.0, sim))

        return sim, [{"cosine_similarity": round(sim, 4), "note": "topic centroid vs. professor centroid"}]

    # ?? Dimension 3: Gap Complement (20%) ????????????????????????????????????

    def _score_gap_complement(
        self, self_model: dict, prof_vec: dict
    ) -> tuple[float, list[dict]]:
        """
        How well the professor's research complements (rather than overlaps) the user's.

        Uses open_questions from self_model as the "gap signal" -
        if the professor's papers address questions the user hasn't answered,
        that's complementarity, not just similarity.

        Score = avg max-similarity of each open_question against prof papers.
        """
        open_questions_raw = self_model.get("open_questions", [])
        paper_embeddings = prof_vec.get("paper_embeddings")
        paper_meta = prof_vec.get("paper_meta", [])

        if not open_questions_raw or paper_embeddings is None or len(paper_embeddings) == 0:
            return 0.0, [{"note": "No open_questions or professor papers"}]

        # Extract question strings if they are dicts
        open_questions = []
        for q in open_questions_raw:
            if isinstance(q, dict):
                open_questions.append(q.get("text") or q.get("question") or "")
            else:
                open_questions.append(str(q))

        question_embs = cached_encode(open_questions, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
        paper_embs = np.array(paper_embeddings)

        # (n_questions, n_papers)
        sim_matrix = question_embs @ paper_embs.T

        evidence: list[dict] = []
        scores: list[float] = []

        for i, question in enumerate(open_questions):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_idx])
            scores.append(best_sim)
            best_paper = paper_meta[best_idx] if best_idx < len(paper_meta) else {}
            evidence.append({
                "open_question": question,
                "best_matching_paper": best_paper.get("title", ""),
                "similarity": round(best_sim, 4),
            })

        score = float(np.mean(scores)) if scores else 0.0
        return max(0.0, min(1.0, score)), evidence

    # ?? Dimension 4: Method Compatibility (15%) ???????????????????????????????

    def _score_method_compat(
        self, self_model: dict, prof_vec: dict
    ) -> tuple[float, list[dict]]:
        """
        Cosine similarity between user's method_embedding and professor's method_centroid.
        High = they use similar approaches ? easier collaboration.
        """
        user_meth = np.array(self_model.get("method_embedding", []))
        prof_meth = np.array(prof_vec.get("method_centroid", prof_vec.get("topic_centroid", [])))

        if len(user_meth) == 0 or len(prof_meth) == 0:
            return 0.0, [{"note": "Missing method embeddings"}]

        sim = float(np.dot(user_meth, prof_meth))
        sim = max(0.0, min(1.0, sim))

        user_methods = self_model.get("methods", [])
        prof_methods = prof_vec.get("methods", [])
        shared = [m for m in user_methods if m.lower() in {p.lower() for p in prof_methods}]

        return sim, [{
            "method_similarity": round(sim, 4),
            "user_methods": user_methods[:5],
            "prof_methods": prof_methods[:5],
            "shared_methods": shared,
        }]

    # ?? Dimension 5: Lab Accessibility (10%) ??????????????????????????????????

    def _score_lab_accessibility(
        self, prof_dossier: dict
    ) -> tuple[float, list[dict]]:
        """
        Computes lab accessibility deterministically using explicit signals:
        - career_stage: assistant (+0.25), associate (+0.15), full (+0.00), emeritus (-0.30), unknown (+0.05)
        - h_index: <15 (+0.15), 15-30 (+0.05), 30-50 (+0.00), >50 (-0.20)
        - last_publication_year: recent (<= 1 year ago: +0.10), old (>= 5 years ago: -0.20)
        - student_supervision_signal: scaled contribution (up to +0.15)
        """
        score = 0.5  # neutral default
        evidence: list[dict] = []

        # 1. h_index
        h_index = prof_dossier.get("h_index")
        if h_index is None:
            h_index = prof_dossier.get("citationCount") or 0
        if not isinstance(h_index, int):
            try:
                h_index = int(h_index)
            except (ValueError, TypeError):
                h_index = 0

        # 2. career_stage
        career_stage = prof_dossier.get("career_stage")
        if not career_stage or career_stage == "unknown":
            position = (prof_dossier.get("position") or prof_dossier.get("title") or "").lower()
            if "assistant" in position:
                career_stage = "assistant"
            elif "associate" in position:
                career_stage = "associate"
            elif "emeritus" in position or "retired" in position:
                career_stage = "emeritus"
            elif "professor" in position or "director" in position or "head" in position:
                career_stage = "full"
            else:
                career_stage = "unknown"

        # 3. last_publication_year
        last_pub_year = prof_dossier.get("last_publication_year")
        if last_pub_year is None:
            pubs = prof_dossier.get("publications", [])
            years = [
                p.get("year")
                for p in pubs
                if isinstance(p.get("year"), int) and p["year"] > 1990
            ]
            last_pub_year = max(years) if years else None

        # 4. student_supervision_signal
        student_sig = prof_dossier.get("student_supervision_signal")
        if student_sig is None:
            lab_desc = (prof_dossier.get("lab_description") or "").lower()
            student_keywords = ["phd student", "graduate student", "doctoral student", "postdoc", "supervised"]
            matches = sum(1 for kw in student_keywords if kw in lab_desc)
            student_sig = min(1.0, matches * 0.25)

        # Career stage contribution
        if career_stage == "assistant":
            score += 0.25
            evidence.append({"signal": f"career_stage={career_stage}", "delta": 0.25})
        elif career_stage == "associate":
            score += 0.15
            evidence.append({"signal": f"career_stage={career_stage}", "delta": 0.15})
        elif career_stage == "full":
            score += 0.00
            evidence.append({"signal": f"career_stage={career_stage}", "delta": 0.0})
        elif career_stage == "emeritus":
            score -= 0.30
            evidence.append({"signal": f"career_stage={career_stage}", "delta": -0.30})
        else:
            score += 0.05
            evidence.append({"signal": f"career_stage=unknown", "delta": 0.05})

        # h-index contribution
        if h_index < 15:
            score += 0.15
            evidence.append({"signal": f"h_index={h_index} (<15)", "delta": 0.15})
        elif h_index <= 30:
            score += 0.05
            evidence.append({"signal": f"h_index={h_index} (15-30)", "delta": 0.05})
        elif h_index <= 50:
            score += 0.0
            evidence.append({"signal": f"h_index={h_index} (30-50)", "delta": 0.0})
        else:
            score -= 0.20
            evidence.append({"signal": f"h_index={h_index} (>50)", "delta": -0.20})

        # last_publication_year contribution
        if last_pub_year is not None:
            current_year = time.localtime().tm_year
            gap = current_year - last_pub_year
            if gap <= 1:
                score += 0.10
                evidence.append({"signal": f"last_publication_year={last_pub_year} (recent)", "delta": 0.10})
            elif gap >= 5:
                score -= 0.20
                evidence.append({"signal": f"last_publication_year={last_pub_year} (old)", "delta": -0.20})
            else:
                evidence.append({"signal": f"last_publication_year={last_pub_year}", "delta": 0.0})
        else:
            evidence.append({"signal": "last_publication_year=None", "delta": 0.0})

        # student_supervision_signal contribution
        supervision_delta = student_sig * 0.15
        score += supervision_delta
        evidence.append({"signal": f"student_supervision_signal={student_sig:.2f}", "delta": round(supervision_delta, 2)})

        score = max(0.0, min(1.0, score))
        return score, evidence

    # ?? Dimension 6: Temporal Activity (5%) ???????????????????????????????????

    def _score_temporal_activity(
        self, prof_dossier: dict
    ) -> tuple[float, list[dict]]:
        """
        How recently has this professor been actively publishing?

        Uses year of most recent paper from the dossier.
        Score:
            published within last year:  1.0
            2 years ago:                 0.80
            3 years ago:                 0.60
            4 years ago:                 0.40
            5+ years ago:                0.20
        """
        current_year = time.localtime().tm_year
        pubs = prof_dossier.get("publications", [])

        if not pubs:
            return 0.30, [{"note": "No publications found"}]

        years = [
            p.get("year")
            for p in pubs
            if isinstance(p.get("year"), int) and p["year"] > 1990
        ]
        if not years:
            return 0.30, [{"note": "No year data in publications"}]

        most_recent = max(years)
        gap = current_year - most_recent

        if gap <= 1:
            score = 1.0
        elif gap <= 2:
            score = 0.80
        elif gap <= 3:
            score = 0.60
        elif gap <= 4:
            score = 0.40
        else:
            score = 0.20

        return score, [{
            "most_recent_paper_year": most_recent,
            "years_since_last_paper": gap,
            "score": score,
        }]

    # ?? Professor vector extraction ???????????????????????????????????????????

    def _build_professor_vector(self, prof_dossier: dict, embedder) -> dict:
        """
        Extract embedding vectors from an existing profiling pipeline dossier.

        No new API calls. No LLM. Uses existing publication data.
        Computes and returns:
            topic_centroid:   mean of all paper abstract embeddings
            method_centroid:  same (no explicit method extraction yet)
            paper_embeddings: list of per-paper embeddings
            paper_meta:       list of {title, doi, year} dicts
            methods:          empty list (requires LLM to populate properly)
        """
        pubs = prof_dossier.get("publications", [])
        abstracts = []
        meta = []

        for p in pubs:
            abstract = (p.get("abstract") or "").strip()
            if abstract:
                abstracts.append(abstract)
                meta.append({
                    "title": p.get("title", ""),
                    "doi": p.get("doi", ""),
                    "year": p.get("year"),
                    "abstract": abstract,
                })

        if not abstracts:
            return {
                "topic_centroid": [],
                "method_centroid": [],
                "paper_embeddings": [],
                "paper_meta": [],
                "methods": [],
                "program_claims": prof_dossier.get("program_claims", []),
                "program_methods": prof_dossier.get("program_methods", []),
                "program_datasets": prof_dossier.get("program_datasets", []),
                "program_metrics": prof_dossier.get("program_metrics", []),
                "program_title": prof_dossier.get("program_title", ""),
                "program_summary": prof_dossier.get("program_summary", ""),
            }

        paper_embs = cached_encode(abstracts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
        centroid = paper_embs.mean(axis=0)

        return {
            "topic_centroid": centroid.tolist(),
            "method_centroid": centroid.tolist(),  # same until method extraction added
            "paper_embeddings": paper_embs.tolist(),
            "paper_meta": meta,
            "methods": [],
            "program_claims": prof_dossier.get("program_claims", []),
            "program_methods": prof_dossier.get("program_methods", []),
            "program_datasets": prof_dossier.get("program_datasets", []),
            "program_metrics": prof_dossier.get("program_metrics", []),
            "program_title": prof_dossier.get("program_title", ""),
            "program_summary": prof_dossier.get("program_summary", ""),
        }

    # ?? Reasoning generators (called post-scoring for provenance) ?????????????

    def _enrs_reasoning(self, score: float, matches: list[dict]) -> str:
        if not matches:
            return "No professor papers matched any unresolved blockers above threshold."
        top = max(matches, key=lambda m: m.get("similarity", 0))
        return (
            f"Best match: '{top.get('paper_title', 'unknown')[:60]}' "
            f"(sim={top.get('similarity', 0):.2f}) resolves blocker: "
            f"'{top.get('blocker_description', '')[:60]}'"
        )

    def _topic_reasoning(self, score: float) -> str:
        if score > 0.75:
            return f"Strong topic overlap (cosine={score:.2f}). Closely related fields."
        elif score > 0.50:
            return f"Moderate topic overlap (cosine={score:.2f}). Adjacent fields."
        else:
            return f"Low topic overlap (cosine={score:.2f}). Distant fields."

    def _gap_reasoning(self, score: float, evidence: list[dict]) -> str:
        if not evidence:
            return "Gap complement not computed (no open questions)."
        top_q = max(evidence, key=lambda e: e.get("similarity", 0))
        return (
            f"Open question '{top_q.get('open_question', '')[:50]}' "
            f"best addressed by '{top_q.get('best_matching_paper', '')[:50]}' "
            f"(sim={top_q.get('similarity', 0):.2f})"
        )

    def _method_reasoning(self, score: float) -> str:
        if score > 0.70:
            return f"High method overlap (cosine={score:.2f}). Similar toolkit ? easier collaboration."
        elif score > 0.40:
            return f"Partial method overlap (cosine={score:.2f}). Some shared tools."
        else:
            return f"Low method overlap (cosine={score:.2f}). Complementary toolkits."

    def _access_reasoning(self, score: float, prof_dossier: dict) -> str:
        pos = prof_dossier.get("position") or prof_dossier.get("title") or "unknown position"
        return f"Lab accessibility score={score:.2f} for {pos}."

    def _temporal_reasoning(self, score: float, prof_dossier: dict) -> str:
        pubs = prof_dossier.get("publications", [])
        years = [p.get("year") for p in pubs if isinstance(p.get("year"), int)]
        if years:
            return f"Most recent publication: {max(years)}. Activity score={score:.2f}."
        return f"No publication year data. Activity score={score:.2f}."

    # ?? Utilities ??????????????????????????????????????????????????????????????

    def _make_cache_key(self, self_model: dict, prof_dossier: dict) -> str:
        """Stable cache key: self_model hash + professor openalex ID."""
        model_hash = self_model.get("model_hash", "no_hash")
        prof_id = (
            prof_dossier.get("openalex_id")
            or prof_dossier.get("id")
            or prof_dossier.get("name", "unknown").replace(" ", "_")
        )
        return f"{model_hash}_{prof_id}"

    @staticmethod
    def format_score_card(result: dict) -> str:
        """
        Human-readable single-professor score summary for CLI/logging.

        Example output:
            Prof. Alexander Mathis  EPFL                     79.4 / 100
            ?????????????????????????????????????????????????????????????
            ENRS            ????????????????????????  21.4 pts  (30%)
            Topic Overlap   ????????????????????????  14.0 pts  (20%)
            Gap Complement  ????????????????????????  14.4 pts  (20%)
            Method Compat   ????????????????????????   9.8 pts  (15%)
            Lab Access      ????????????????????????   6.2 pts  (10%)
            Temporal Act.   ????????????????????????   4.0 pts   (5%)
        """
        name = result.get("professor_name", "")
        inst = result.get("institution", "")
        total = result.get("total_score", 0.0)
        dims = result.get("dimensions", {})

        lines = [
            f"{name}  {inst}",
            f"Total: {total:.1f} / 100",
            "?" * 50,
        ]

        DIM_LABELS = {
            "enrs":           "ENRS             ",
            "topic_overlap":  "Topic Overlap    ",
            "gap_complement": "Gap Complement   ",
            "method_compat":  "Method Compat    ",
            "lab_access":     "Lab Access       ",
            "temporal_act":   "Temporal Act.    ",
        }

        for key, label in DIM_LABELS.items():
            if key not in dims:
                continue
            d = dims[key]
            raw = d.get("raw", 0.0)
            contrib = d.get("contribution", 0.0)
            pct = d.get("weight", 0.0) * 100
            bar_filled = int(raw * 24)
            bar = "?" * bar_filled + "?" * (24 - bar_filled)
            lines.append(f"  {label} {bar}  {contrib:5.1f} pts  ({pct:.0f}%)")

        lines.append("?" * 50)

        matches = result.get("resolution_matches", [])
        if matches:
            lines.append("Why this professor:")
            for m in matches[:3]:
                lines.append(
                    f"  * Blocker: '{m.get('blocker_description', '')[:50]}...'"
                )
                lines.append(
                    f"    ? Paper: '{m.get('paper_title', '')[:50]}...' "
                    f"(sim={m.get('similarity', 0):.2f})"
                )

        hypotheses = result.get("collaboration_hypotheses", [])
        if hypotheses:
            lines.append("")
            lines.append("Collaboration hypotheses:")
            for h in hypotheses:
                lines.append(f"  * {h}")

        return "\n".join(lines)
