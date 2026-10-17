"""
run_collaboration_analysis.py — End-to-end collaboration discovery pipeline.

Single command that runs the complete workflow:

    Research Corpus (abstracts)
          ↓
    Self Model (identity vector, claims, questions, methods)
          ↓
    Investigations (active blockers, contradictions)
          ↓
    Professor Dossiers (from existing profiling pipeline or cache)
          ↓
    CollaborationScorer (6 dimensions, ENRS, full provenance)
          ↓
    Ranked Output: Top professors × Top blocker matches × Contradictions

Usage:
    python run_collaboration_analysis.py --help
    python run_collaboration_analysis.py --corpus outputs/my_papers.json
    python run_collaboration_analysis.py --corpus outputs/my_papers.json --top-n 20
    python run_collaboration_analysis.py --corpus outputs/my_papers.json --no-llm

Output:
    Terminal: formatted score cards with provenance
    JSON:     outputs/collaboration_analysis_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Reconfigure stdout/stderr to handle UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Bootstrap path ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.self_model import SelfModelBuilder
from academic_pipeline.claim_deduplicator import apply_deduplication_to_self_model
from academic_pipeline.collaboration_scorer import CollaborationScorer
from academic_pipeline.investigation_engine import InvestigationEngine
from academic_pipeline.local_llm import LocalLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_collaboration_analysis")


# ── Print helpers ──────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _header(title: str) -> None:
    _sep("═")
    print(f"  {title}")
    _sep("═")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_corpus(path: str) -> tuple[list[str], list[str], list[str]]:
    """
    Load paper corpus from JSON. Accepts two formats:
    1. List of dossier dicts:  [{name, publications: [{abstract, title, doi}]}]
    2. Flat list of papers:    [{abstract, title, doi}]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts, titles, dois = [], [], []
    if not (isinstance(data, list) and data):
        return texts, titles, dois

    first = data[0]
    if isinstance(first, dict) and "publications" in first:
        # Format 1: dossier list
        for dossier in data:
            for pub in dossier.get("publications", []):
                abstract = (pub.get("abstract") or "").strip()
                if abstract:
                    texts.append(abstract)
                    titles.append(pub.get("title", ""))
                    dois.append(pub.get("doi", ""))
    elif isinstance(first, dict) and "abstract" in first:
        # Format 2: flat paper list
        for pub in data:
            abstract = (pub.get("abstract") or "").strip()
            if abstract:
                texts.append(abstract)
                titles.append(pub.get("title", ""))
                dois.append(pub.get("doi", ""))

    logger.info(f"Loaded {len(texts)} abstracts from {path}")
    return texts, titles, dois


def _load_dossiers_from_cache(cache: CacheManager, names: list[str] | None = None) -> list[dict]:
    """
    Load professor dossiers using list_professor_memories + get_resolved_profile.
    If names is None, loads all cached profiles.
    """
    all_dossiers = []
    known_memories = cache.list_professor_memories()
    known_names = [m.get("name", "") for m in known_memories if m.get("name")]

    if names:
        name_set = {n.lower() for n in names}
        known_names = [n for n in known_names if n.lower() in name_set]

    for name in known_names:
        profile = cache.get_resolved_profile(name)
        if profile:
            all_dossiers.append(profile)

    logger.info(f"Loaded {len(all_dossiers)} dossiers from cache")
    return all_dossiers


def _load_dossiers_from_file(path: str) -> list[dict]:
    """Load professor dossiers directly from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data


# ── Seed investigations ────────────────────────────────────────────────────────

def _seed_default_investigations(engine: InvestigationEngine) -> None:
    """
    Create 3 concrete EarthOS/TARS investigations with specific blockers so
    the ENRS dimension has real targets to match professor papers against.
    Only runs if no active investigations exist yet.
    """
    active = engine.get_active_investigations()
    if active:
        logger.info(f"Skipping seed: {len(active)} investigations already exist.")
        return

    logger.info("Seeding 3 default investigations...")

    # ── Investigation 1: EarthOS Curriculum Ordering ───────────────────────────
    inv1 = engine.create_investigation(
        title="Does curriculum ordering affect long-term cognitive coherence in EarthOS?",
        description=(
            "EarthOS exhibits structured investigation sequences. "
            "Whether the order in which investigations are activated affects "
            "the coherence and utility of later investigations is unknown."
        ),
        base_priority=0.90,
        auto_activate=True,
    )
    engine.add_blocker(
        inv_id=inv1.id,
        description=(
            "Empirical dataset comparing representation geometry under different "
            "curriculum orderings in biological or artificial continual learners "
            "(CKA/RSA measurements, minimum 3 conditions)"
        ),
        blocker_type="dataset",
        severity=0.88,
    )
    engine.add_blocker(
        inv_id=inv1.id,
        description=(
            "Validated methodology for measuring coherence in open-ended "
            "investigation sequences beyond standard episodic memory benchmarks"
        ),
        blocker_type="methodology",
        severity=0.75,
    )

    # ── Investigation 2: TARS Non-monotonic Learning ───────────────────────────
    inv2 = engine.create_investigation(
        title="Why does TARS prediction error not decrease monotonically during training?",
        description=(
            "TARS training shows non-monotonic prediction error curves. "
            "The mechanism behind these plateaus and sudden drops is unknown."
        ),
        base_priority=0.80,
        auto_activate=True,
    )
    engine.add_blocker(
        inv_id=inv2.id,
        description=(
            "Theoretical framework explaining non-monotonic learning curves in "
            "active inference agents under partial observability"
        ),
        blocker_type="domain_expertise",
        severity=0.85,
    )

    # ── Investigation 3: Blocker Specificity Calibration ───────────────────────
    inv3 = engine.create_investigation(
        title="What specificity threshold separates useful from useless research blockers?",
        description=(
            "The Research-the-Researcher system uses a specificity validator. "
            "The threshold (0.60) was chosen heuristically. "
            "Empirical validation against real collaboration outcomes is needed."
        ),
        base_priority=0.65,
        auto_activate=True,
    )
    engine.add_blocker(
        inv_id=inv3.id,
        description=(
            "Labeled dataset of research blockers rated by domain experts on "
            "specificity and eventual resolution outcome (n >= 100 blockers)"
        ),
        blocker_type="dataset",
        severity=0.70,
    )

    active_now = engine.get_active_investigations()
    total_blockers = sum(len(i.blockers) for i in active_now)
    print(f"  Seeded 3 investigations, {len(active_now)} active, {total_blockers} blockers total.")


# ── Core analysis ──────────────────────────────────────────────────────────────

def run_collaboration_analysis(
    corpus_texts: list[str],
    corpus_titles: list[str],
    corpus_dois: list[str],
    professor_dossiers: list[dict],
    cache_dir: str = "outputs",
    ollama_model: str = "qwen2.5-coder:7b",
    ollama_host: str = "http://localhost:11434",
    top_n: int = 20,
    generate_hypotheses: bool = True,
    force_rebuild_self_model: bool = False,
    dedup_claims: bool = True,
) -> dict:
    """
    Core analysis function. Returns a result dict:
    {
        self_model, ranked_professors, top_blocker_matches,
        top_contradictions, quality_flags, analysis_metadata
    }
    """
    t0 = time.time()
    quality_flags: list[str] = []
    cache = CacheManager(cache_dir=cache_dir)

    # ── Step 1: LLM + NLI availability checks ─────────────────────────────────
    llm = LocalLLMClient(model=ollama_model, host=ollama_host)
    if not llm.is_available():
        logger.warning(
            f"Ollama not available at {ollama_host}. "
            "Running in embedding-only mode (no claim extraction, no hypotheses)."
        )
        quality_flags.append("llm_unavailable")
        llm = None
        generate_hypotheses = False

    # NLI filter availability (independent of LLM — uses a local cross-encoder)
    try:
        from academic_pipeline.nli_filter import NLIFilter as _NLIFilter
        if not _NLIFilter().is_available():
            quality_flags.append("nli_unavailable")
            logger.warning("NLI cross-encoder not available. Contradiction detection will use embedding-only fallback.")
    except Exception:
        quality_flags.append("nli_unavailable")

    # ── Step 2: Build Self Model ───────────────────────────────────────────────
    print()
    _header("STEP 1 / 4  —  Building Research Identity (Self Model)")

    builder = SelfModelBuilder(cache_manager=cache, llm_client=llm if generate_hypotheses else None)
    self_model = builder.build(
        texts=corpus_texts,
        titles=corpus_titles,
        dois=corpus_dois,
        force_rebuild=force_rebuild_self_model,
    )

    if dedup_claims and (self_model.get("strong_claims") or self_model.get("open_questions")):
        try:
            self_model = apply_deduplication_to_self_model(self_model)
            stats = self_model.get("dedup_stats", {})
            print(f"  Claims:    {stats.get('raw_claims', 0)} raw → {stats.get('canonical_claims', 0)} canonical")
            print(f"  Questions: {stats.get('raw_questions', 0)} raw → {stats.get('canonical_questions', 0)} canonical")
        except Exception as e:
            logger.warning(f"Claim deduplication failed ({e}). Using raw claims.")
            quality_flags.append("dedup_fallback")

    # ── Self-model confidence check ────────────────────────────────────────────
    paper_count = self_model.get("paper_count", 0)
    topics      = self_model.get("topics", [])
    if paper_count < 5:
        quality_flags.append(f"low_confidence_self_model:only_{paper_count}_papers")
        logger.warning(
            f"Self-model built from only {paper_count} paper(s). "
            "Scores may be unreliable. Provide more papers via --corpus for better results."
        )
    elif not topics:
        quality_flags.append("low_confidence_self_model:no_topics_extracted")
        logger.warning("No topics extracted from corpus. Topic overlap and gap complement scores will be 0.")

    print(f"  Topics:    {', '.join(self_model.get('topics', [])[:6]) or '(none extracted)'}")
    print(f"  Papers:    {self_model.get('paper_count', 0)}")
    print(f"  Hash:      {self_model.get('model_hash', 'N/A')}")
    print()

    # ── Step 3: Load Investigation State ──────────────────────────────────────
    _header("STEP 2 / 4  —  Loading Investigation State")

    engine = InvestigationEngine(cache_manager=cache)
    active_investigations = engine.get_active_investigations()
    unresolved_blocker_pairs = engine.get_unresolved_blockers()

    print(f"  Active investigations:  {len(active_investigations)}")
    print(f"  Unresolved blockers:    {len(unresolved_blocker_pairs)}")

    scoreable = [inv for inv in active_investigations if inv.is_scoreable]
    if not scoreable:
        logger.warning(
            "No scoreable investigations (active investigations with unresolved blockers). "
            "ENRS will be 0 for all professors. Add investigations via --seed-investigations."
        )
        quality_flags.append("no_scoreable_investigations")
    else:
        print()
        print("  Scoreable investigations:")
        for inv in scoreable:
            n = len(inv.open_blockers)
            print(f"    • {inv.title[:58]}  ({n} blocker{'s' if n != 1 else ''})")
    print()

    # ── Step 4: Score Professors ───────────────────────────────────────────────
    _header("STEP 3 / 4  —  Scoring Professors")

    if not professor_dossiers:
        logger.error("No professor dossiers provided. Cannot score.")
        return {
            "self_model": self_model,
            "ranked_professors": [],
            "top_blocker_matches": [],
            "top_contradictions": [],
            "quality_flags": quality_flags + ["no_professor_dossiers"],
            "analysis_metadata": {"elapsed_s": round(time.time() - t0, 1)},
        }

    scorer = CollaborationScorer(cache_manager=cache)
    scored: list[dict] = []

    print(f"  Scoring {len(professor_dossiers)} professor(s)...")
    for i, dossier in enumerate(professor_dossiers):
        name = dossier.get("name", f"Professor {i+1}")
        try:
            result = scorer.compute_score(
                self_model=self_model,
                investigations=scoreable,
                prof_dossier=dossier,
                generate_hypotheses=generate_hypotheses,
            )
            scored.append(result)
            print(f"  [{i+1:3d}/{len(professor_dossiers)}]  {name[:40]:40s}  {result['total_score']:5.1f}/100")
        except Exception as e:
            logger.error(f"Scoring failed for '{name}': {e}")
            quality_flags.append(f"scoring_error:{name}")

    scored.sort(key=lambda r: r.get("total_score", 0.0), reverse=True)
    top_results = scored[:top_n]

    # ── Step 5: Global top blocker resolution matches ─────────────────────────
    all_matches: list[dict] = []
    for r in scored:
        for m in r.get("resolution_matches", []):
            all_matches.append({**m, "professor_name": r.get("professor_name", "")})
    top_matches = sorted(all_matches, key=lambda m: m.get("similarity", 0.0), reverse=True)[:10]

    # ── Step 6: Unresolved contradictions ─────────────────────────────────────
    all_contradictions = []
    for inv in engine.get_active_investigations():
        for c in inv.contradictions:
            if not c.resolved:
                all_contradictions.append(c)
    top_contradictions = sorted(all_contradictions, key=lambda c: c.effective_signal, reverse=True)[:10]

    # ── Step 7: Print Results ─────────────────────────────────────────────────
    _header("STEP 4 / 4  —  Results")

    print()
    print(f"  {'Rank':<5} {'Professor':<35} {'Institution':<20} {'Score':>7}")
    _sep()
    for rank, r in enumerate(top_results, 1):
        name = r.get("professor_name", "")[:34]
        inst = r.get("institution", "")[:19]
        score = r.get("total_score", 0.0)
        enrs_contribution = r.get("dimensions", {}).get("enrs", {}).get("contribution", 0.0)
        marker = "★" if enrs_contribution > 10.0 else " "
        print(f"  {rank:<5} {name:<35} {inst:<20} {score:>6.1f}  {marker}")

    print()
    print("  ★ = strong ENRS match (professor directly addresses a blocker)")
    print()

    if top_matches:
        _sep()
        print("  Top Blocker Resolutions:")
        for m in top_matches[:5]:
            print(f"  • [{m.get('professor_name','')[:25]}]  "
                  f"Blocker: '{m.get('blocker_description','')[:45]}'")
            print(f"    → Paper: '{m.get('paper_title','')[:55]}'  "
                  f"(sim={m.get('similarity',0):.2f})")
        print()

    if top_contradictions:
        _sep()
        print("  Top Unresolved Contradictions (by effective signal):")
        for c in top_contradictions[:5]:
            claim_preview = getattr(c, "claim_yours", getattr(c, "description", ""))[:68]
            print(f"  • signal={c.effective_signal:.2f}  sev={c.severity:.2f}  conf={c.confidence:.2f}")
            print(f"    '{claim_preview}'")
        print()

    if quality_flags:
        _sep()
        print("  Quality Flags (system degradation signals):")
        for flag in quality_flags:
            print(f"  ⚠  {flag}")
        print()

    elapsed = time.time() - t0

    # ── Step 8: Save JSON output ───────────────────────────────────────────────
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"collaboration_analysis_{timestamp}.json"

    payload = {
        "self_model": {k: v for k, v in self_model.items()
                       if k not in ("topic_embedding", "method_embedding", "publication_vectors")},
        "ranked_professors": top_results,
        "top_blocker_matches": top_matches,
        "top_contradictions": [
            {
                "claim_yours": getattr(c, "claim_yours", ""),
                "claim_external": getattr(c, "claim_external", ""),
                "severity": c.severity,
                "confidence": c.confidence,
                "effective_signal": c.effective_signal,
                "source_external": getattr(c, "source_external", ""),
                "resolved": c.resolved,
            }
            for c in top_contradictions
        ],
        "quality_flags": quality_flags,
        "analysis_metadata": {
            "elapsed_s": round(elapsed, 1),
            "professors_scored": len(scored),
            "top_n": top_n,
            "model_hash": self_model.get("model_hash"),
            "paper_count": self_model.get("paper_count"),
            "scoreable_investigations": len(scoreable),
            "unresolved_blockers": len(unresolved_blocker_pairs),
            "timestamp": timestamp,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    _sep("═")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    _sep("═")

    return payload


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research Collaboration Discovery — end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score all cached professors against your papers
  python run_collaboration_analysis.py --corpus outputs/my_papers.json

  # Score from a specific dossier file, top 10
  python run_collaboration_analysis.py \\
      --corpus outputs/my_papers.json \\
      --dossiers outputs/epfl_professors.json \\
      --top-n 10

  # Seed 3 EarthOS/TARS investigations then run analysis
  python run_collaboration_analysis.py --seed-investigations

  # Embedding-only mode (no Ollama required)
  python run_collaboration_analysis.py --corpus outputs/my_papers.json --no-llm

  # Force self-model rebuild (after adding new papers)
  python run_collaboration_analysis.py --corpus outputs/my_papers.json --force-rebuild
""",
    )
    p.add_argument("--corpus", default=None,
                   help="Path to JSON file with your paper abstracts.")
    p.add_argument("--dossiers", default=None,
                   help="Path to JSON file with professor dossiers. "
                        "If omitted, loads all from cache.")
    p.add_argument("--cache-dir", default="outputs",
                   help="Cache directory (default: outputs)")
    p.add_argument("--top-n", type=int, default=20,
                   help="Top N professors to display (default: 20)")
    p.add_argument("--ollama-model", default="qwen2.5-coder:7b",
                   help="Ollama model name (default: qwen2.5-coder:7b)")
    p.add_argument("--ollama-host", default="http://localhost:11434",
                   help="Ollama host URL (default: http://localhost:11434)")
    p.add_argument("--no-llm", action="store_true",
                   help="Embedding-only mode — no claim extraction, no hypotheses.")
    p.add_argument("--no-hypotheses", action="store_true",
                   help="Skip hypothesis generation (scores only)")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild self-model even if cached")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip claim deduplication")
    p.add_argument("--seed-investigations", action="store_true",
                   help="Seed 3 default EarthOS/TARS investigations if none exist")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load corpus ────────────────────────────────────────────────────────────
    if args.corpus:
        corpus_texts, corpus_titles, corpus_dois = _load_corpus(args.corpus)
    else:
        logger.warning(
            "No --corpus provided. Using 2 synthetic abstracts for demonstration. "
            "Pass --corpus path/to/papers.json for real results."
        )
        corpus_texts = [
            "We study the effect of investigation ordering on long-term coherence in "
            "EarthOS, a persistent cognitive architecture. Our experiments show that "
            "curriculum scheduling of investigation activation improves downstream "
            "representational quality as measured by CKA.",
            "TARS (Temporal Active Reasoning System) exhibits non-monotonic prediction "
            "error during active inference training. We hypothesize that phase transitions "
            "in the generative model explain observed plateaus. Curriculum learning "
            "principles may apply to reduce variance.",
        ]
        corpus_titles = ["EarthOS Investigation Ordering", "TARS Prediction Error Analysis"]
        corpus_dois   = ["10.X/earthos-1", "10.X/tars-1"]

    if not corpus_texts:
        logger.error("No abstracts found in corpus. Aborting.")
        sys.exit(1)

    # ── Load professor dossiers ────────────────────────────────────────────────
    cache = CacheManager(cache_dir=args.cache_dir)

    if args.dossiers:
        professor_dossiers = _load_dossiers_from_file(args.dossiers)
    else:
        professor_dossiers = _load_dossiers_from_cache(cache)
        if not professor_dossiers:
            logger.warning(
                "No professor dossiers found in cache. "
                "Run the profiling pipeline first, or pass --dossiers path/to/profs.json"
            )

    # ── Seed investigations if requested ──────────────────────────────────────
    if args.seed_investigations:
        engine = InvestigationEngine(cache_manager=cache)
        _seed_default_investigations(engine)

    # ── Run analysis ───────────────────────────────────────────────────────────
    run_collaboration_analysis(
        corpus_texts=corpus_texts,
        corpus_titles=corpus_titles,
        corpus_dois=corpus_dois,
        professor_dossiers=professor_dossiers,
        cache_dir=args.cache_dir,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        top_n=args.top_n,
        generate_hypotheses=not args.no_hypotheses and not args.no_llm,
        force_rebuild_self_model=args.force_rebuild,
        dedup_claims=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
