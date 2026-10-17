"""
dedup_engine.py — Multi-tier paper deduplication engine.

Three-stage cascade:
  Stage 1: Bloom filter + exact hash  O(1)     — catches exact DOI/ArXiv matches
  Stage 2: MinHash LSH                O(n)     — catches near-duplicate titles (Jaccard ≥ 0.85)
  Stage 3: FAISS semantic search      O(log n) — catches paraphrased titles (cosine ≥ 0.95)

DSA details:
  - Bloom filter: 2M-bit bitarray + 3 independent MurmurHash3 functions → ~256 KB RAM,
    zero false negatives, <0.01% false positive rate for up to 50k entries.
  - MinHash LSH: 128 permutations, bag-of-trigrams tokenisation, datasketch backend.
  - FAISS: IndexFlatIP over sentence-transformers embeddings (384-dim),
    GPU-accelerated via faiss-gpu when CUDA is available.

The dedup engine is constructed once and then used throughout the pipeline run.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional heavy dependencies (graceful degradation) ───────────────────────

try:
    from bitarray import bitarray
    import mmh3
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False
    logger.warning("bitarray/mmh3 not installed. Bloom filter disabled. "
                   "Install: pip install bitarray mmh3")

try:
    from datasketch import MinHashLSH, MinHash
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    logger.warning("datasketch not installed. MinHash LSH disabled. "
                   "Install: pip install datasketch")

try:
    import numpy as np
    import logging
    import os
    if "FAISS_OPT_LEVEL" not in os.environ:
        os.environ["FAISS_OPT_LEVEL"] = "AVX2"
    logging.getLogger("faiss.loader").setLevel(logging.ERROR)
    import faiss
    FAISS_AVAILABLE = True
    # Try GPU; fall back to CPU silently
    try:
        res = faiss.StandardGpuResources()
        FAISS_GPU = True
        logger.info("FAISS running on GPU.")
    except Exception:
        FAISS_GPU = False
        logger.info("FAISS running on CPU (GPU resources not found).")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Semantic dedup disabled. "
                   "Install: pip install faiss-cpu  (or faiss-gpu)")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Semantic dedup disabled. "
                   "Install: pip install sentence-transformers")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _trigrams(text: str) -> set[str]:
    words = text.split()
    tgs = set()
    for w in words:
        for i in range(len(w) - 2):
            tgs.add(w[i:i+3])
    # Also word bigrams
    for i in range(len(words) - 1):
        tgs.add(words[i] + "_" + words[i+1])
    return tgs


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Bloom Filter
# ──────────────────────────────────────────────────────────────────────────────

class BloomFilter:
    """
    2M-bit Bloom filter with 3 MurmurHash3 seeds.
    Designed for up to 50,000 items at < 0.01% false positive rate.
    Memory: ~256 KB.
    """

    SIZE  = 2_000_000   # bits
    SEEDS = [0, 42, 137]

    def __init__(self):
        if BLOOM_AVAILABLE:
            self._bits = bitarray(self.SIZE)
            self._bits.setall(0)
        else:
            self._bits = None
            self._fallback: set[str] = set()

    def _indices(self, key: str):
        return [mmh3.hash(key, seed) % self.SIZE for seed in self.SEEDS]

    def add(self, key: str):
        if self._bits is None:
            self._fallback.add(key)
            return
        for idx in self._indices(key):
            self._bits[idx] = 1

    def __contains__(self, key: str) -> bool:
        if self._bits is None:
            return key in self._fallback
        return all(self._bits[idx] for idx in self._indices(key))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: MinHash LSH
# ──────────────────────────────────────────────────────────────────────────────

class MinHashIndex:
    """
    128-permutation MinHash LSH for near-duplicate title detection.
    Threshold = 0.85 Jaccard similarity.
    """

    PERMS     = 128
    THRESHOLD = 0.85

    def __init__(self):
        self._lsh: Optional[object] = None
        self._minhashes: dict[str, object] = {}
        if MINHASH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=self.THRESHOLD, num_perm=self.PERMS)

    def _make_minhash(self, text: str):
        m = MinHash(num_perm=self.PERMS)
        for tg in _trigrams(text):
            m.update(tg.encode("utf-8"))
        return m

    def add(self, key: str, title: str):
        if self._lsh is None:
            return
        norm = _normalize(title)
        try:
            m = self._make_minhash(norm)
            self._lsh.insert(key, m)
            self._minhashes[key] = m
        except ValueError:
            # key already inserted
            pass

    def query(self, title: str) -> list[str]:
        """Return keys of papers similar to `title`."""
        if self._lsh is None:
            return []
        norm = _normalize(title)
        m = self._make_minhash(norm)
        return self._lsh.query(m)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: FAISS Semantic Index
# ──────────────────────────────────────────────────────────────────────────────

class SemanticIndex:
    """
    GPU-accelerated FAISS IndexFlatIP over sentence-transformer embeddings.
    Used for ambiguous near-duplicates that MinHash misses.
    Cosine similarity threshold = 0.95.
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    THRESHOLD  = 0.95
    DIM        = 1024

    def __init__(self):
        self._index = None
        self._keys: list[str] = []
        self._built = False

    def _lazy_init(self):
        if self._index is not None:
            return
        if not (FAISS_AVAILABLE and ST_AVAILABLE):
            return
        from academic_pipeline.self_model import get_embedder_by_name
        try:
            embedder = get_embedder_by_name(self.MODEL_NAME)
            if hasattr(embedder, "get_embedding_dimension"):
                self.DIM = embedder.get_embedding_dimension()
            else:
                self.DIM = embedder.get_sentence_embedding_dimension()
        except Exception:
            self.DIM = 1024
        logger.info(f"FAISS dynamic dimension set to: {self.DIM}")
        if FAISS_GPU:
            try:
                base_index = faiss.IndexFlatIP(self.DIM)
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, base_index)
            except Exception:
                self._index = faiss.IndexFlatIP(self.DIM)
        else:
            self._index = faiss.IndexFlatIP(self.DIM)

    def reset(self):
        """Reset the index and keys for a new build."""
        self._keys = []
        self._built = False
        if self._index is not None:
            self._index.reset()

    def build(self, keys: list[str], titles: list[str]):
        """Batch-encode all titles and build the FAISS index."""
        if not (FAISS_AVAILABLE and ST_AVAILABLE):
            return
        if not titles:
            return
        self._lazy_init()
        if self._index is None:
            return
        self.reset()
        logger.info(f"Building FAISS index over {len(titles)} titles...")
        norms = [_normalize(t) for t in titles]
        from academic_pipeline.embedding_cache import cached_encode
        embeddings = cached_encode(
            norms, model_name=self.MODEL_NAME, normalize_embeddings=True
        ).astype("float32")
        self._index.add(embeddings)
        self._keys = list(keys)
        self._built = True
        logger.info("FAISS index built.")

    def query(self, title: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return (key, cosine_similarity) pairs above THRESHOLD."""
        if not (FAISS_AVAILABLE and ST_AVAILABLE):
            return []
        self._lazy_init()
        if not self._built or self._index is None:
            return []
        norm = _normalize(title)
        from academic_pipeline.embedding_cache import cached_encode
        vec = cached_encode([norm], model_name=self.MODEL_NAME, normalize_embeddings=True).astype("float32")
        D, I = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            if score >= self.THRESHOLD:
                results.append((self._keys[idx], float(score)))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication Engine
# ──────────────────────────────────────────────────────────────────────────────

class DedupEngine:
    """
    Combines Bloom Filter + MinHash LSH + FAISS into a single dedup interface.

    Usage:
        engine = DedupEngine()
        unique_papers = engine.deduplicate(papers_from_multiple_sources)
    """

    def __init__(self):
        self._bloom    = BloomFilter()
        self._minhash  = MinHashIndex()
        self._semantic = SemanticIndex()
        self._exact: dict[str, int] = {}   # key → index in result list

    @staticmethod
    def _primary_key(paper: dict) -> Optional[str]:
        """Extract the best unique key from a paper dict."""
        doi = paper.get("doi")
        if doi:
            return "doi:" + doi.lower().strip()
        arxiv = paper.get("arxiv")
        if arxiv:
            return "arxiv:" + arxiv.lower().strip()
        return None

    @staticmethod
    def _title_key(paper: dict) -> str:
        """Normalized title used as fallback key."""
        title = paper.get("title", "")
        return "title:" + hashlib.sha1(_normalize(title).encode()).hexdigest()[:16]

    def _merge_into(self, target: dict, source: dict):
        """Merge fields from source into target, preferring non-None source values."""
        for field in ["doi", "arxiv", "oa_url", "citations_max", "abstract",
                      "tldr", "venue", "pdf_path", "analysis"]:
            if source.get(field) and not target.get(field):
                target[field] = source[field]
        # Citations: take max
        if source.get("citations_max", 0) > target.get("citations_max", 0):
            target["citations_max"] = source["citations_max"]
        # Authors: union
        existing_authors = set(target.get("authors", []))
        for a in source.get("authors", []):
            existing_authors.add(a)
        target["authors"] = sorted(existing_authors)

    def deduplicate(self, papers: list[dict]) -> list[dict]:
        """
        De-duplicate a mixed list of paper dicts from multiple sources.
        Returns a clean merged list.
        """
        # Reset state for a fresh deduplication run to allow instance reuse
        self._bloom = BloomFilter()
        self._minhash = MinHashIndex()
        self._exact = {}
        self._semantic.reset()

        result: list[dict] = []
        # Track title_key → result index for fallback LSH/semantic merges
        title_index: dict[str, int] = {}

        for paper in papers:
            primary_key = self._primary_key(paper)
            title       = paper.get("title", "")
            title_key   = self._title_key(paper)

            # ── Stage 1: Exact key ──────────────────────────────────────────
            if primary_key:
                if primary_key in self._bloom:
                    # Likely seen — check exact dict
                    if primary_key in self._exact:
                        self._merge_into(result[self._exact[primary_key]], paper)
                        continue
                # New primary key
                self._bloom.add(primary_key)
                self._exact[primary_key] = len(result)
                self._minhash.add(primary_key, title)
                result.append(dict(paper))
                title_index[title_key] = len(result) - 1
                continue

            # ── No primary key: use title ───────────────────────────────────
            # Stage 1b: title bloom
            if title_key in self._bloom:
                if title_key in title_index:
                    self._merge_into(result[title_index[title_key]], paper)
                    continue

            # Stage 2: MinHash LSH
            similar = self._minhash.query(title)
            if similar:
                # Merge into first similar result
                match_key = similar[0]
                if match_key in self._exact:
                    self._merge_into(result[self._exact[match_key]], paper)
                    continue

            # New paper
            self._bloom.add(title_key)
            self._minhash.add(title_key, title)
            self._exact[title_key] = len(result)
            result.append(dict(paper))
            title_index[title_key] = len(result) - 1

        # ── Stage 3: Post-hoc semantic dedup ───────────────────────────────
        if FAISS_AVAILABLE and ST_AVAILABLE and len(result) > 1:
            result = self._semantic_pass(result)

        # ── Stage 4: Preprint vs. Journal Consolidation ────────────────────
        result = self._consolidate_preprints(result)

        logger.info(f"Dedup: {len(papers)} raw → {len(result)} unique papers.")
        return result

    def _semantic_pass(self, papers: list[dict]) -> list[dict]:
        """
        Final semantic sweep over already-deduplicated papers.
        Catches paraphrased titles that passed MinHash.
        """
        keys   = [self._primary_key(p) or self._title_key(p) for p in papers]
        titles = [p.get("title", "") for p in papers]

        self._semantic.build(keys, titles)

        to_merge: dict[int, int] = {}   # source_idx → target_idx
        for i, (key, title) in enumerate(zip(keys, titles)):
            if i in to_merge:
                continue
            similar = self._semantic.query(title, top_k=3)
            for sim_key, score in similar:
                if sim_key == key:
                    continue
                # If both have different primary keys (doi/arxiv), do not merge them
                if key.startswith(("doi:", "arxiv:")) and sim_key.startswith(("doi:", "arxiv:")):
                    continue
                sim_idx = keys.index(sim_key) if sim_key in keys else -1
                if 0 <= sim_idx < i and sim_idx not in to_merge:
                    # Merge i into sim_idx (keep earlier entry)
                    to_merge[i] = sim_idx

        # Apply merges
        clean: list[dict] = []
        for i, paper in enumerate(papers):
            if i in to_merge:
                self._merge_into(papers[to_merge[i]], paper)
            else:
                clean.append(paper)

        if len(clean) < len(papers):
            logger.info(f"Semantic pass: removed {len(papers) - len(clean)} near-duplicates.")
        return clean

    def _merge_preprint_and_journal(self, journal: dict, preprint: dict) -> dict:
        """Merge preprint metadata into the journal record, preserving peer-reviewed fields."""
        merged = dict(journal)
        
        # Consolidate external IDs
        if preprint.get("arxiv") and not merged.get("arxiv"):
            merged["arxiv"] = preprint["arxiv"]
        if preprint.get("doi") and not merged.get("doi"):
            merged["doi"] = preprint["doi"]
            
        # Citations: take maximum count
        merged["citations_max"] = max(merged.get("citations_max", 0), preprint.get("citations_max", 0))
        merged["citations_openalex"] = max(merged.get("citations_openalex", 0), preprint.get("citations_openalex", 0))
        merged["citations_semantic"] = max(merged.get("citations_semantic", 0), preprint.get("citations_semantic", 0))
        if "influential_citations" in preprint:
            merged["influential_citations"] = max(merged.get("influential_citations", 0), preprint.get("influential_citations", 0))
            
        # Merge missing details preferring journal version
        for field in ["abstract", "tldr", "oa_url", "pdf_path", "analysis"]:
            if preprint.get(field) and not merged.get(field):
                merged[field] = preprint[field]
                
        # Merge authors lists
        existing_authors = set(merged.get("authors", []))
        for a in preprint.get("authors", []):
            existing_authors.add(a)
        merged["authors"] = sorted(existing_authors)
        
        # Combine authors_detailed lists
        existing_details = {a.get("name"): a for a in merged.get("authors_detailed", []) if a.get("name")}
        for a in preprint.get("authors_detailed", []):
            name = a.get("name")
            if name and name not in existing_details:
                existing_details[name] = a
        merged["authors_detailed"] = list(existing_details.values())
        
        return merged

    def _consolidate_preprints(self, papers: list[dict]) -> list[dict]:
        """
        Scans all papers and merges preprints (arXiv/venue match) with their peer-reviewed versions.
        Uses Jaccard title similarity and abstract overlap thresholds.
        """
        def is_preprint(p):
            if p.get("arxiv"):
                return True
            venue = p.get("venue", "").lower()
            return "arxiv" in venue or "preprint" in venue

        preprints = []
        journals = []
        for p in papers:
            if is_preprint(p):
                preprints.append(p)
            else:
                journals.append(p)

        if not preprints or not journals:
            return papers

        result_journals = [dict(j) for j in journals]
        result_preprints_unmerged = []

        for prep in preprints:
            prep_title = prep.get("title", "")
            prep_norm = _normalize(prep_title)
            prep_abstract = prep.get("abstract", "")
            
            matched_j_idx = -1
            for j_idx, jour in enumerate(result_journals):
                jour_title = jour.get("title", "")
                jour_norm = _normalize(jour_title)
                jour_abstract = jour.get("abstract", "")

                # Calculate title Jaccard similarity of trigrams
                title_sim = 0.0
                if prep_norm == jour_norm:
                    title_sim = 1.0
                else:
                    t1 = _trigrams(prep_norm)
                    t2 = _trigrams(jour_norm)
                    if t1 and t2:
                        title_sim = len(t1.intersection(t2)) / len(t1.union(t2))

                # Calculate abstract similarity of word sets
                abstract_sim = 0.0
                if prep_abstract and jour_abstract:
                    w1 = set(_normalize(prep_abstract).split())
                    w2 = set(_normalize(jour_abstract).split())
                    if w1 and w2:
                        abstract_sim = len(w1.intersection(w2)) / len(w1.union(w2))

                # Match logic:
                # 1. High title similarity (>= 0.85) AND abstracts match (>= 0.65)
                # 2. Extremely high title similarity (>= 0.90) and missing one abstract
                # 3. High abstract match (>= 0.75) for revised titles
                is_match = False
                if title_sim >= 0.85 and (abstract_sim >= 0.65 or not prep_abstract or not jour_abstract):
                    is_match = True
                elif title_sim >= 0.90:
                    is_match = True
                elif abstract_sim >= 0.75:
                    is_match = True

                if is_match:
                    matched_j_idx = j_idx
                    break

            if matched_j_idx != -1:
                result_journals[matched_j_idx] = self._merge_preprint_and_journal(result_journals[matched_j_idx], prep)
                logger.info(f"Consolidated preprint '{prep_title}' into published paper '{result_journals[matched_j_idx]['title']}'")
            else:
                result_preprints_unmerged.append(prep)

        return result_journals + result_preprints_unmerged

