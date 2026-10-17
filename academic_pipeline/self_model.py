"""
self_model.py — Builds the user's Research Identity Vector.

SelfModel is the representation of YOUR research, distilled into a form
that can be matched against professors. It is NOT a biography — it is a
structured signal of what you claim, what you don't know, what keeps
blocking you, and what methods you use.

Key design decisions:
    - model_hash is computed from the input corpus (not time-based)
      so cache invalidation happens automatically when you publish a new paper.
    - All embeddings are 384-dim (all-MiniLM-L6-v2, same as ENRS).
      Cosine similarity works across all dimensions.
    - strong_claims + open_questions are the most valuable fields —
      they're what make ENRS and contradiction detection meaningful.
    - recurring_blockers tracks patterns across investigations over time.

Output schema:
    {
        "topic_embedding":       list[float],  # centroid of all paper topic embs
        "method_embedding":      list[float],  # centroid of all method embs
        "strong_claims":         list[str],    # things you assert in your papers
        "open_questions":        list[str],    # things you don't know yet
        "recurring_blockers":    list[str],    # blocker types that keep appearing
        "publication_vectors":   list[dict],   # per-paper: {title, doi, embedding}
        "topics":                list[str],    # extracted topic keywords
        "methods":               list[str],    # extracted method keywords
        "model_hash":            str,          # sha256 of input corpus
        "built_at":              float,
        "paper_count":           int,
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


_EMBEDDERS: dict[str, Any] = {}

def get_embedder_by_name(model_name: str) -> Any:
    """Lazy-load a specific SentenceTransformer model. Process-cached."""
    global _EMBEDDERS
    if model_name not in _EMBEDDERS:
        try:
            from sentence_transformers import SentenceTransformer
            try:
                _EMBEDDERS[model_name] = SentenceTransformer(
                    model_name, local_files_only=True
                )
            except Exception:
                _EMBEDDERS[model_name] = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer loaded ({model_name})")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _EMBEDDERS[model_name]


def get_embedder():
    """Lazy-load default SentenceTransformer. Warm on first call (~2s), cached after."""
    return get_embedder_by_name("BAAI/bge-large-en-v1.5")


def compute_corpus_hash(texts: list[str]) -> str:
    """
    SHA-256 of the sorted, deduplicated input texts.
    Sorted so that paper ordering doesn't change the hash.
    Used for cache invalidation — if hash changes, rebuild self-model.
    """
    combined = "\n---\n".join(sorted(set(t.strip() for t in texts if t.strip())))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# ── LLM extraction schemas ─────────────────────────────────────────────────────
# These define what the LLM extracts from the paper corpus.
# Pydantic-style dicts (compatible with local_llm.py Ollama structured output).

SELF_MODEL_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "strong_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific factual or theoretical claims the researcher makes "
                "across their papers. Must be precise and assertoric "
                "(e.g. 'Curriculum ordering improves representation quality "
                "in online continual learners'). Not goals or questions."
            ),
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Genuine open questions the researcher has not yet answered. "
                "Must be specific (e.g. 'Does curriculum order matter equally "
                "in biological and artificial learning systems?'). "
                "Not rhetorical questions."
            ),
        },
        "methods_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific methods, frameworks, or techniques the researcher uses.",
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Core research topics (3–8 keywords).",
        },
    },
    "required": ["strong_claims", "open_questions", "methods_used", "topics"],
}


# ── SelfModelBuilder ──────────────────────────────────────────────────────────

class SelfModelBuilder:
    """
    Builds the Research Identity Vector from a corpus of papers/texts.

    Usage:
        builder = SelfModelBuilder(cache_manager=cache, llm_client=llm)
        model = builder.build(
            texts=["abstract 1...", "abstract 2..."],
            dois=["10.1234/abc", "10.1234/def"],
            titles=["Paper 1", "Paper 2"],
            investigation_engine=engine,  # optional: extract recurring_blockers
        )
    """

    def __init__(
        self,
        cache_manager,
        llm_client=None,   # LocalLLMClient; optional (skips claim/question extraction)
    ):
        self._cache = cache_manager
        self._llm = llm_client

    def build(
        self,
        texts: list[str],        # paper abstracts (required)
        titles: Optional[list[str]] = None,
        dois: Optional[list[str]] = None,
        investigation_engine=None,  # to extract recurring_blockers
        force_rebuild: bool = False,
    ) -> dict:
        """
        Builds or returns cached self-model.

        Cache hit conditions:
            model_hash matches AND force_rebuild is False

        Returns the self-model dict (always includes model_hash).
        """
        titles = titles or [""] * len(texts)
        dois = dois or [""] * len(texts)

        model_hash = compute_corpus_hash(texts)

        # Check cache first
        if not force_rebuild:
            cached = self._cache.get_self_model(expected_hash=model_hash)
            if cached and cached.get("embedding_model") == "BAAI/bge-large-en-v1.5":
                logger.info(f"Self-model cache hit (hash={model_hash})")
                return cached

        logger.info(f"Building self-model from {len(texts)} texts (hash={model_hash})")

        embedder = get_embedder()

        # ── Step 1: Embed all texts ───────────────────────────────────────────
        import numpy as np
        from academic_pipeline.embedding_cache import cached_encode
        text_embeddings = cached_encode(texts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)  # (N, 1024)

        # Topic centroid = mean of all paper embeddings
        topic_embedding = text_embeddings.mean(axis=0).tolist()

        # ── Step 2: Per-paper publication vectors ─────────────────────────────
        publication_vectors = []
        for i, (text, title, doi) in enumerate(zip(texts, titles, dois)):
            publication_vectors.append({
                "title": title,
                "doi": doi,
                "embedding": text_embeddings[i].tolist(),
                "abstract_snippet": text[:300],
            })

        # ── Step 3: LLM extraction (claims, questions, methods, topics) ───────
        strong_claims: list[dict] = []
        structured_claims: list[dict] = []
        open_questions: list[dict] = []
        methods_used: list[str] = []
        topics: list[str] = []

        if self._llm is not None:
            try:
                import uuid
                for text, title, doi in zip(texts, titles, dois):
                    extracted = self._llm.extract_claims_and_evidence_sync(title, text)
                    
                    # Process extracted claims
                    for c_detail in extracted.get("claims", []):
                        claim_text = c_detail.get("claim", "")
                        if not claim_text:
                            continue
                        
                        # Backward-compatible list of dicts for strong_claims
                        strong_claims.append({
                            "text": claim_text,
                            "confidence": c_detail.get("confidence", 1.0),
                            "source_paper_dois": [doi] if doi else []
                        })
                        
                        # New first-class structured claims
                        structured_claims.append({
                            "id": str(uuid.uuid4()),
                            "claim": claim_text,
                            "method": c_detail.get("method", "N/A"),
                            "dataset": c_detail.get("dataset", "N/A"),
                            "metric": c_detail.get("metric", "N/A"),
                            "confidence": c_detail.get("confidence", 1.0),
                            "paper_doi": doi or ""
                        })
                        
                    # Process open questions
                    for q_text in extracted.get("open_questions", []):
                        if q_text:
                            open_questions.append({
                                "text": q_text,
                                "confidence": 1.0,
                                "source_paper_dois": [doi] if doi else []
                            })
                            
                    # Collect methods
                    for m in extracted.get("methods", []):
                        if m and m.lower() != "n/a":
                            methods_used.append(m)

                # Heuristic topic extraction
                topics = self._extract_topics_heuristic(texts)
                logger.info(
                    f"LLM extracted: {len(strong_claims)} claims, "
                    f"{len(open_questions)} questions, "
                    f"{len(methods_used)} methods"
                )
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}. Using embedding-only model.")
                topics = self._extract_topics_heuristic(texts)
        else:
            logger.info("No LLM client — using embedding-only self-model.")
            topics = self._extract_topics_heuristic(texts)

        # ── Step 4: Method embedding (embed methods_used if available) ────────
        if methods_used:
            from academic_pipeline.embedding_cache import cached_encode
            method_embeddings = cached_encode(methods_used, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
            method_embedding = method_embeddings.mean(axis=0).tolist()
        else:
            method_embedding = topic_embedding  # fallback: same as topic

        # ── Step 5: Recurring blockers from active investigations ─────────────
        recurring_blockers: list[str] = []
        if investigation_engine is not None:
            try:
                unresolved = investigation_engine.get_unresolved_blockers()
                blocker_types: dict[str, int] = {}
                for _, blocker in unresolved:
                    blocker_types[blocker.blocker_type] = (
                        blocker_types.get(blocker.blocker_type, 0) + 1
                    )
                # Report types that appear more than once
                recurring_blockers = [
                    f"{btype} ({count}x)"
                    for btype, count in sorted(
                        blocker_types.items(), key=lambda x: -x[1]
                    )
                    if count > 1
                ]
            except Exception as e:
                logger.warning(f"Could not extract recurring blockers: {e}")

        # ── Step 6: Assemble and cache ────────────────────────────────────────

        # Clean topics: strip JSON artifacts, deduplicate, cap at 20
        _seen_t: set[str] = set()
        _clean: list[str] = []
        for _t in topics:
            _t = str(_t).strip().strip("[],'\"")
            if len(_t) >= 3 and _t not in {("],"), "[", "]", "..."}:
                _k = _t.lower()
                if _k not in _seen_t:
                    _seen_t.add(_k)
                    _clean.append(_t)
        topics = _clean[:20]

        model = {
            "topic_embedding": topic_embedding,
            "method_embedding": method_embedding,
            "strong_claims": strong_claims,
            "structured_claims": structured_claims,
            "open_questions": open_questions,
            "recurring_blockers": recurring_blockers,
            "publication_vectors": publication_vectors,
            "topics": topics,
            "methods": methods_used,
            "model_hash": model_hash,
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "built_at": time.time(),
            "paper_count": len(texts),
        }

        # Save ours claims to SQL program graph database
        try:
            from academic_pipeline.research_graph import ResearchProgramGraph, ClaimNode, ResearchProgramNode
            graph = ResearchProgramGraph(cache_dir=self._cache.cache_dir)
            ours_prog_id = "prog_ours"
            ours_prog = ResearchProgramNode(
                id=ours_prog_id,
                title="Our Research Program",
                summary="Core research topics: " + ", ".join(topics[:5]),
                university="Ours",
                professor_name="Ours",
                last_refresh=time.time()
            )
            graph.save_program(ours_prog)
            
            for c in structured_claims:
                claim_node = ClaimNode(
                    id=c["id"],
                    text=c["claim"],
                    type="ours",
                    confidence=c.get("confidence", 1.0),
                    paper_doi=c.get("paper_doi", ""),
                    method=c.get("method", "N/A"),
                    dataset=c.get("dataset", "N/A"),
                    metric=c.get("metric", "N/A"),
                    program_id=ours_prog_id
                )
                graph.save_claim(claim_node)
            logger.info(f"Saved {len(structured_claims)} ours claims to program graph database")
        except Exception as e:
            logger.warning(f"Could not save ours claims to program graph database: {e}")

        self._cache.save_self_model(model)
        logger.info(f"Self-model built and cached (hash={model_hash})")
        return model

    def _extract_topics_heuristic(self, texts: list[str]) -> list[str]:
        """
        Fallback topic extraction when no LLM is available.
        Uses TF-IDF-style scoring: favours words that are frequent in the
        corpus but not so common that they are essentially stopwords.
        Returns top 12 domain-meaningful content words.
        """
        import re
        import math
        from collections import Counter

        # Broad stopword set: generic English + document-structure words that
        # dominate research-report corpora without carrying domain signal.
        STOPWORDS = {
            # English function words
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "this", "that", "these",
            "those", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "we", "our", "us", "it", "its",
            "their", "they", "as", "which", "such", "not", "also", "both",
            "can", "into", "than", "through", "using", "based", "show",
            "shows", "propose", "result", "results", "paper", "work",
            # Document-structure words common in research reports / markdown
            "text", "phase", "test", "tests", "report", "note", "notes",
            "section", "chapter", "figure", "table", "item", "list",
            "code", "data", "file", "type", "each", "more", "most",
            "some", "when", "where", "what", "then", "here", "there",
            "been", "only", "used", "uses", "used", "well", "high",
            "new", "old", "full", "true", "false", "none", "self",
            "path", "name", "time", "line", "next", "last", "same",
            "other", "any", "all", "one", "two", "three", "four", "five",
            "level", "step", "part", "case", "term", "form", "word",
            "point", "value", "place", "long", "short", "good", "best",
            "make", "made", "take", "taken", "given", "like", "just",
            "while", "after", "before", "during", "without", "within",
            "between", "across", "over", "under", "above", "below",
            "from", "about", "again", "even", "very", "still", "much",
            "many", "rather", "current", "currently", "previous", "following",
            "however", "therefore", "thus", "hence", "since", "because",
            "approach", "method", "methods", "technique", "techniques",
            "function", "class", "object", "node", "graph", "field",
            "structure", "process", "operation", "implementation",
        }

        # Per-document word sets for IDF
        doc_word_sets: list[set] = []
        all_words: list[str] = []

        for text in texts:
            doc_words = set()
            for word in re.findall(r"[a-zA-Z]{5,}", text.lower()):
                if word not in STOPWORDS:
                    all_words.append(word)
                    doc_words.add(word)
            doc_word_sets.append(doc_words)

        tf = Counter(all_words)
        n_docs = max(len(doc_word_sets), 1)

        # IDF: log(N / df) — penalises words that appear in almost every doc
        df: dict[str, int] = Counter()
        for dws in doc_word_sets:
            for w in dws:
                df[w] += 1

        scored = {}
        for word, freq in tf.items():
            idf = math.log(n_docs / (1 + df.get(word, 0)))
            scored[word] = freq * idf

        # Return top-12 by TF-IDF score
        return [w for w, _ in sorted(scored.items(), key=lambda x: -x[1])[:12]]


# ── Convenience: build from existing pipeline dossier ────────────────────────

def build_self_model_from_dossiers(
    dossiers: list[dict],
    cache_manager,
    llm_client=None,
    investigation_engine=None,
) -> dict:
    """
    Build self-model from a list of professor-style dossier dicts
    (same format as the existing pipeline output) treated as YOUR papers.

    This bridges the existing profiling pipeline with the new self-model.
    """
    texts, titles, dois = [], [], []
    for d in dossiers:
        for pub in d.get("publications", []):
            abstract = pub.get("abstract", "")
            if abstract:
                texts.append(abstract)
                titles.append(pub.get("title", ""))
                dois.append(pub.get("doi", ""))

    if not texts:
        raise ValueError(
            "No abstracts found in dossiers. "
            "Run the profiling pipeline with abstract fetching enabled."
        )

    builder = SelfModelBuilder(cache_manager=cache_manager, llm_client=llm_client)
    return builder.build(
        texts=texts,
        titles=titles,
        dois=dois,
        investigation_engine=investigation_engine,
    )
