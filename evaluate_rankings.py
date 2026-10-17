"""
evaluate_rankings.py — Compare system rankings against manual judgments.

Workflow:
    1. You manually assign relevance scores (1–5) to professors you know.
    2. This script scores the same professors using the collaboration pipeline.
    3. It computes Kendall's τ and prints a disagreement table showing which
       dimension caused each rank gap.

Manual ranking file format (evals/manual_rankings.yaml):
    - name: "Karl Friston"
      manual_score: 5
      notes: "Active inference pioneer, directly relevant to TARS"
    - name: "Yoshua Bengio"
      manual_score: 3
      notes: "Representation learning, but not active inference"
    - name: "Jiajun Wu"
      manual_score: 1
      notes: "Computer vision, no EarthOS relevance"

Usage:
    # Create manual rankings file, then:
    python evaluate_rankings.py \\
        --corpus outputs/self_corpus.json \\
        --dossiers outputs/epfl_professors.json \\
        --manual evals/manual_rankings.yaml

    # Embedding-only mode (fast)
    python evaluate_rankings.py \\
        --corpus outputs/self_corpus.json \\
        --dossiers outputs/epfl_professors.json \\
        --manual evals/manual_rankings.yaml --no-llm

Output:
    Terminal:  rank correlation + disagreement table
    JSON:      evals/evaluation_results_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.self_model import SelfModelBuilder
from academic_pipeline.claim_deduplicator import apply_deduplication_to_self_model
from academic_pipeline.collaboration_scorer import CollaborationScorer
from academic_pipeline.investigation_engine import InvestigationEngine
from academic_pipeline.local_llm import LocalLLMClient

logging.basicConfig(
    level=logging.WARNING,  # quiet by default during eval
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_rankings")


# ── Load manual rankings ───────────────────────────────────────────────────────

def _load_manual_rankings(path: str) -> list[dict]:
    """Load YAML or JSON manual rankings file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manual rankings file not found: {path}")

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or []
        except ImportError:
            # Fallback: parse simple YAML manually
            return _parse_simple_yaml(p.read_text(encoding="utf-8"))

    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _parse_simple_yaml(text: str) -> list[dict]:
    """Minimal YAML list parser for the simple ranking format."""
    import re
    items = []
    current: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- name:"):
            if current:
                items.append(current)
            current = {"name": line.split(":", 1)[1].strip().strip('"')}
        elif line.startswith("manual_score:") and current:
            current["manual_score"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("notes:") and current:
            current["notes"] = line.split(":", 1)[1].strip().strip('"')
    if current:
        items.append(current)
    return items


# ── Rank correlation ───────────────────────────────────────────────────────────

def _kendall_tau(x: list[float], y: list[float]) -> float:
    """Compute Kendall's τ-b between two ranking lists (concordant - discordant)."""
    n = len(x)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = x[i] - x[j]
            sy = y[i] - y[j]
            if sx * sy > 0:
                concordant += 1
            elif sx * sy < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman's ρ (rank correlation)."""
    n = len(x)
    if n < 2:
        return 0.0
    # Convert to ranks
    def _rank(v):
        sorted_v = sorted(enumerate(v), key=lambda t: t[1], reverse=True)
        ranks = [0.0] * n
        for rank, (idx, _) in enumerate(sorted_v):
            ranks[idx] = rank + 1.0
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


# ── Dimension explanation ──────────────────────────────────────────────────────

def _biggest_dimension_gap(result: dict) -> tuple[str, float]:
    """Return the dimension name and contribution that most explains a high/low score."""
    dims = result.get("dimensions", {})
    if not dims:
        return ("unknown", 0.0)
    biggest = max(dims.items(), key=lambda kv: kv[1].get("contribution", 0.0))
    return biggest[0], biggest[1].get("contribution", 0.0)


def _format_dimensions(result: dict) -> str:
    """Compact dimension breakdown for disagreement table."""
    dims = result.get("dimensions", {})
    parts = []
    for k, v in dims.items():
        contrib = v.get("contribution", 0.0)
        if contrib > 0.5:
            parts.append(f"{k}={contrib:.1f}")
    return "  ".join(parts) if parts else "all~0"


def _tier_accuracy(aligned: list[dict]) -> dict:
    """
    Group researchers by manual tier (1-5) and compute mean/std of system scores.

    Returns a dict:  {tier_int: {"mean": float, "std": float, "n": int, "names": list}}

    Why this matters:
        Kendall's τ can look OK while the system still fails to separate tiers.
        Example:  τ = 0.55  but  tier-5 avg = 48,  tier-1 avg = 41  → useless.
        Tier accuracy reveals this immediately.
    """
    import math
    buckets: dict[int, list] = {}
    for a in aligned:
        tier = int(round(a["manual_score"]))
        buckets.setdefault(tier, []).append(a)

    result = {}
    for tier in sorted(buckets.keys(), reverse=True):
        scores = [a["system_score"] for a in buckets[tier]]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0.0
        result[tier] = {
            "mean":  round(mean, 1),
            "std":   round(math.sqrt(variance), 1),
            "n":     len(scores),
            "names": [a["name"] for a in buckets[tier]],
        }
    return result


def _print_tier_accuracy(tier_data: dict, W: int = 72) -> None:
    """Print tier accuracy table with an ASCII bar chart."""
    print("  TIER ACCURACY  (does system score separate manual tiers?):")
    print("-" * W)
    print(f"  {'Manual Tier':<14} {'Avg System':>10}  {'Std':>5}  {'N':>3}  {'Bar (0─100)':<32}")
    print("-" * W)

    # Compute expected direction signal
    tiers_sorted = sorted(tier_data.keys(), reverse=True)
    prev_mean = None
    direction_ok = True

    for tier in tiers_sorted:
        info = tier_data[tier]
        mean = info["mean"]
        std  = info["std"]
        n    = info["n"]

        # Bar: scale 0-100 into 30 chars
        bar_len = max(1, int(mean * 30 / 100))
        bar = "█" * bar_len

        # Direction check
        if prev_mean is not None and mean > prev_mean:
            direction_ok = False
            marker = "  ⚠"
        else:
            marker = ""

        label = f"Tier {tier}  (manual={tier})"
        print(f"  {label:<14} {mean:>10.1f}  {std:>5.1f}  {n:>3}  {bar:<32}{marker}")
        prev_mean = mean

    print()

    # Separation signal: difference between highest and lowest tier means
    if len(tiers_sorted) >= 2:
        top = tier_data[tiers_sorted[0]]["mean"]
        bot = tier_data[tiers_sorted[-1]]["mean"]
        separation = top - bot
        if separation >= 40:
            verdict = "✓ Strong separation"
        elif separation >= 20:
            verdict = "~ Moderate separation"
        else:
            verdict = "✗ Weak separation — system may not distinguish relevance tiers"
        print(f"  Tier separation (T5 − T1 avg): {separation:+.1f} points  →  {verdict}")
        if not direction_ok:
            print("  ⚠  Non-monotonic: a lower-tier had higher avg system score than a higher tier.")
    print()


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_evaluation(
    corpus_texts: list[str],
    corpus_titles: list[str],
    corpus_dois: list[str],
    manual_rankings: list[dict],
    dossier_file: str | None,
    cache_dir: str = "outputs",
    ollama_model: str = "qwen2.5-coder:7b",
    ollama_host: str = "http://localhost:11434",
    use_llm: bool = False,
) -> dict:
    t0 = time.time()
    cache = CacheManager(cache_dir=cache_dir)

    # Build self model
    llm = LocalLLMClient(model=ollama_model, host=ollama_host) if use_llm else None
    if use_llm and not llm.is_available():
        print("  ⚠  Ollama unavailable — running embedding-only")
        llm = None

    builder = SelfModelBuilder(cache_manager=cache, llm_client=llm)
    self_model = builder.build(texts=corpus_texts, titles=corpus_titles, dois=corpus_dois)
    if self_model.get("strong_claims") or self_model.get("open_questions"):
        try:
            self_model = apply_deduplication_to_self_model(self_model)
        except Exception:
            pass

    # Load investigations
    engine = InvestigationEngine(cache_manager=cache)
    scoreable = [i for i in engine.get_active_investigations() if i.is_scoreable]

    # Load professor dossiers, filtered to only the manually-ranked ones
    manual_names = {r["name"].lower() for r in manual_rankings}

    if dossier_file:
        with open(dossier_file, encoding="utf-8") as f:
            all_dossiers = json.load(f)
        if isinstance(all_dossiers, dict):
            all_dossiers = [all_dossiers]
    else:
        # Try to load from cache
        all_dossiers = []
        for mem in cache.list_professor_memories():
            name = mem.get("name", "")
            if name.lower() in manual_names:
                prof = cache.get_resolved_profile(name)
                if prof:
                    all_dossiers.append(prof)

    # Filter dossiers to only manually-ranked ones
    ranked_dossiers = [
        d for d in all_dossiers
        if d.get("name", "").lower() in manual_names
    ]

    if not ranked_dossiers:
        print("\n  ⚠  No dossiers found for any manually-ranked professor.")
        print("     Check that names match exactly between --manual and --dossiers.\n")
        return {}

    # Score
    scorer = CollaborationScorer(cache_manager=cache)
    scored: list[dict] = []
    print(f"\n  Scoring {len(ranked_dossiers)} manually-ranked professor(s)...\n")
    for dossier in ranked_dossiers:
        name = dossier.get("name", "?")
        try:
            result = scorer.compute_score(
                self_model=self_model,
                investigations=scoreable,
                prof_dossier=dossier,
                generate_hypotheses=False,
            )
            scored.append(result)
        except Exception as e:
            print(f"  ⚠  Scoring failed for '{name}': {e}")

    if not scored:
        print("  No professors scored successfully.")
        return {}

    # Align manual scores to system scores
    system_by_name = {r.get("professor_name", "").lower(): r for r in scored}
    aligned: list[dict] = []
    for manual in manual_rankings:
        name = manual["name"]
        sys_result = system_by_name.get(name.lower())
        if sys_result is None:
            continue
        aligned.append({
            "name":         name,
            "manual_score": float(manual.get("manual_score", 0)),
            "system_score": sys_result.get("total_score", 0.0),
            "notes":        manual.get("notes", ""),
            "dimensions":   sys_result.get("dimensions", {}),
        })

    if len(aligned) < 2:
        print(f"  ⚠  Only {len(aligned)} professor(s) matched. Need at least 2 for correlation.")
        return {}

    # Correlation
    manual_scores = [a["manual_score"] for a in aligned]
    system_scores = [a["system_score"] for a in aligned]
    tau = _kendall_tau(manual_scores, system_scores)
    rho = _spearman_rho(manual_scores, system_scores)

    # Disagreements — sorted by absolute rank gap
    for a in aligned:
        a["gap"] = abs(a["manual_score"] - a["system_score"] / 20.0)  # normalise system /100 to /5

    disagreements = sorted(aligned, key=lambda a: a["gap"], reverse=True)

    # Tier accuracy
    tier_data = _tier_accuracy(aligned)

    # ── Print results ─────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print("  EVALUATION RESULTS")
    print("=" * W)
    print(f"  Professors evaluated:  {len(aligned)}")
    print(f"  Kendall's τ:           {tau:+.3f}  (1.0 = perfect agreement, -1.0 = perfect inversion)")
    print(f"  Spearman's ρ:          {rho:+.3f}")
    print()

    if tau >= 0.6:
        print("  ✓ Strong agreement — system rankings correlate well with your judgment")
    elif tau >= 0.3:
        print("  ~ Moderate agreement — worth examining disagreements")
    else:
        print("  ✗ Weak agreement — review disagreements to understand systematic gaps")
    print()

    print(f"  {'Professor':<28} {'Manual':>6}  {'System':>7}  {'Gap':>5}  Top driver")
    print("-" * W)
    for a in aligned:
        dim, contrib = _biggest_dimension_gap({"dimensions": a["dimensions"]})
        gap_arrow = "▲" if a["system_score"] / 20.0 > a["manual_score"] else "▼"
        print(f"  {a['name'][:27]:<28} {a['manual_score']:>6.1f}  "
              f"{a['system_score']:>6.1f}/100  {a['gap']:>4.2f}{gap_arrow}  "
              f"{dim}={contrib:.1f}")
    print()

    print("  TOP DISAGREEMENTS (investigate these first):")
    print("-" * W)
    for a in disagreements[:5]:
        direction = "overrated" if a["system_score"] / 20.0 > a["manual_score"] else "underrated"
        print(f"\n  {a['name']}  →  system {direction} by {a['gap']:.2f}")
        if a["notes"]:
            print(f"    Manual note: \"{a['notes']}\"")
        dim_str = _format_dimensions({"dimensions": a["dimensions"]})
        print(f"    System: {dim_str}")

    print()
    print("=" * W)

    # ── Save results ─────────────────────────────────────────────────────────
    output_dir = Path("evals")
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"evaluation_results_{timestamp}.json"

    payload = {
        "correlation": {"kendall_tau": tau, "spearman_rho": rho},
        "tier_accuracy": tier_data,
        "professors":  aligned,
        "metadata": {
            "n_evaluated":    len(aligned),
            "n_total_manual": len(manual_rankings),
            "elapsed_s":      round(time.time() - t0, 1),
            "model_hash":     self_model.get("model_hash"),
            "timestamp":      timestamp,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to: {output_path}\n")

    return payload


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare system rankings against manual judgments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Manual rankings YAML format (save to evals/manual_rankings.yaml):
  - name: "Karl Friston"
    manual_score: 5
    notes: "Active inference pioneer, directly relevant to TARS"
  - name: "Yoshua Bengio"
    manual_score: 3
    notes: "Deep learning, indirect relevance"
  - name: "Jiajun Wu"
    manual_score: 1
    notes: "Computer vision, minimal EarthOS relevance"

Scoring guide:
  5 = perfect fit — work directly addresses an active investigation
  4 = strong fit — significant topic/method overlap
  3 = moderate — related area, some overlap
  2 = weak — adjacent field, limited connection
  1 = no fit — different research domain entirely
""",
    )
    p.add_argument("--corpus",  default=None,
                   help="Path to JSON file with your paper abstracts (from export_self_corpus.py)")
    p.add_argument("--dossiers", default=None,
                   help="Path to JSON file with professor dossiers")
    p.add_argument("--manual",  default="evals/manual_rankings.yaml",
                   help="Path to manual rankings YAML/JSON file (default: evals/manual_rankings.yaml)")
    p.add_argument("--cache-dir", default="outputs",
                   help="Cache directory (default: outputs)")
    p.add_argument("--ollama-model", default="qwen2.5-coder:7b")
    p.add_argument("--ollama-host",  default="http://localhost:11434")
    p.add_argument("--no-llm", action="store_true",
                   help="Embedding-only scoring (fast, reproducible)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load corpus
    if args.corpus:
        with open(args.corpus, encoding="utf-8") as f:
            papers = json.load(f)
        corpus_texts  = [p.get("abstract", "") for p in papers if p.get("abstract")]
        corpus_titles = [p.get("title", "") for p in papers if p.get("abstract")]
        corpus_dois   = [p.get("doi", "") for p in papers if p.get("abstract")]
    else:
        logger.warning("No --corpus provided. Using 2 synthetic EarthOS abstracts.")
        corpus_texts  = [
            "EarthOS is a persistent cognitive architecture that maintains structured "
            "investigation sequences across time. We show that curriculum ordering of "
            "investigation activation improves representational coherence as measured by CKA.",
            "TARS exhibits non-monotonic prediction error during active inference training. "
            "Phase transitions in the generative model explain observed plateaus.",
        ]
        corpus_titles = ["EarthOS Investigation Ordering", "TARS Prediction Error"]
        corpus_dois   = ["10.X/earthos-1", "10.X/tars-1"]

    # Load manual rankings
    try:
        manual_rankings = _load_manual_rankings(args.manual)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nCreate evals/manual_rankings.yaml with your manual scores first.")
        print("See --help for the file format.\n")
        sys.exit(1)

    if not manual_rankings:
        print("Manual rankings file is empty. Add at least 5 professors for meaningful correlation.")
        sys.exit(1)

    print(f"\n  Manual rankings loaded: {len(manual_rankings)} professors")

    run_evaluation(
        corpus_texts=corpus_texts,
        corpus_titles=corpus_titles,
        corpus_dois=corpus_dois,
        manual_rankings=manual_rankings,
        dossier_file=args.dossiers,
        cache_dir=args.cache_dir,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
