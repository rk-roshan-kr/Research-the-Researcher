"""
batch_profile_researchers.py — Profile all researchers in evals/manual_rankings.yaml
and save a combined dossier JSON ready for evaluate_rankings.py.

This is the bridge between the manual evaluation set and the scoring engine.

Usage:
    # Profile all 25 researchers (15-30 min depending on API rate limits)
    python batch_profile_researchers.py --manual evals/manual_rankings.yaml

    # Profile only a specific tier (e.g. prestige traps first, as a quick sanity check)
    python batch_profile_researchers.py --manual evals/manual_rankings.yaml --tier 1

    # Resume: skip already-cached researchers
    python batch_profile_researchers.py --manual evals/manual_rankings.yaml --resume

    # Limit papers per researcher (faster, less accurate)
    python batch_profile_researchers.py --manual evals/manual_rankings.yaml --limit-papers 5

Output:
    outputs/eval_dossiers.json          — all successfully profiled researchers
    outputs/eval_profile_report.json    — per-researcher status (success/failure/skipped)

Then run evaluation:
    python evaluate_rankings.py \\
        --corpus outputs/self_corpus.json \\
        --dossiers outputs/eval_dossiers.json \\
        --manual evals/manual_rankings.yaml --no-llm

Recommended profiling order (--tier flag):
    1. python batch_profile_researchers.py --tier 1  (prestige traps: Wu, Li, Ng)
       → verify these score LOW before investing in full run
    2. python batch_profile_researchers.py --tier 5  (positive controls: Friston, O'Reilly, Eliasmith)
       → verify these score HIGH — ENRS must fire
    3. python batch_profile_researchers.py           (all remaining)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from academic_pipeline.pipeline import AcademicPipeline
from academic_pipeline.cache_manager import CacheManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_profile")

# Suppress noisy sub-loggers during batch runs
for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── YAML loader ───────────────────────────────────────────────────────────────

def _load_manual_rankings(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manual rankings file not found: {path}")
    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or []
    except ImportError:
        return _parse_simple_yaml(p.read_text(encoding="utf-8"))


def _parse_simple_yaml(text: str) -> list[dict]:
    items, current = [], {}
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


# ── Batch profiler ────────────────────────────────────────────────────────────

def batch_profile(
    researchers: list[dict],
    cache_dir: str = "outputs",
    limit_papers: int = 10,
    resume: bool = True,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Profile each researcher. Returns (dossiers, report).

    dossiers: list of successfully resolved profile dicts
    report:   per-researcher status records
    """
    pipeline = AcademicPipeline(
        cache_dir=cache_dir,
        ollama_model="qwen3:14b",
    )
    cache = CacheManager(cache_dir=cache_dir)

    dossiers: list[dict] = []
    report:   list[dict] = []

    n = len(researchers)
    for idx, entry in enumerate(researchers, 1):
        name = entry["name"]
        tier = int(round(entry.get("manual_score", 0)))
        notes = entry.get("notes", "")

        print(f"\n  [{idx:2d}/{n}]  {name}  (manual tier {tier})")
        if notes:
            print(f"         note: {notes[:80]}")

        # Resume check — skip if already cached
        if resume:
            existing = cache.get_resolved_profile(name)
            if existing and existing.get("publications"):
                pub_count = len(existing.get("publications", []))
                print(f"         ✓  Cached ({pub_count} publications) — skipping")
                dossiers.append(existing)
                report.append({
                    "name": name, "tier": tier,
                    "status": "cached", "pub_count": pub_count,
                })
                continue

        if dry_run:
            print(f"         [dry-run] would call run_pipeline('{name}', limit_papers={limit_papers})")
            report.append({"name": name, "tier": tier, "status": "dry_run"})
            continue

        t0 = time.time()
        try:
            result = pipeline.run_pipeline(
                query=name,
                limit_papers=limit_papers,
                deep_run=False,
                graph_depth=1,
                max_coauthors=10,
            )
            elapsed = round(time.time() - t0, 1)
            pub_count = len(result.get("publications", []))
            print(f"         ✓  Profiled in {elapsed}s — {pub_count} publications")
            dossiers.append(result)
            report.append({
                "name": name, "tier": tier,
                "status": "success",
                "pub_count": pub_count,
                "elapsed_s": elapsed,
            })

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            logger.error(f"Failed to profile '{name}' ({elapsed}s): {e}")
            report.append({
                "name": name, "tier": tier,
                "status": "failed",
                "error": str(e),
                "elapsed_s": elapsed,
            })

    return dossiers, report


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_report(report: list[dict], dossiers: list[dict], output_path: str) -> None:
    W = 68
    print()
    print("=" * W)
    print("  BATCH PROFILE REPORT")
    print("=" * W)

    by_status: dict[str, list] = {}
    for r in report:
        by_status.setdefault(r["status"], []).append(r)

    success  = len(by_status.get("success",  []))
    cached   = len(by_status.get("cached",   []))
    failed   = len(by_status.get("failed",   []))
    dry_run  = len(by_status.get("dry_run",  []))

    print(f"  Total:    {len(report)}")
    print(f"  Success:  {success}")
    print(f"  Cached:   {cached}")
    print(f"  Failed:   {failed}")
    if dry_run:
        print(f"  Dry-run:  {dry_run}")
    print()

    if by_status.get("failed"):
        print("  FAILURES (re-run or add --skip-failed):")
        for r in by_status["failed"]:
            print(f"    • {r['name']}: {r.get('error','?')[:60]}")
        print()

    print(f"  Dossiers saved to: {output_path}")
    print()
    print("  Next step:")
    print("    python evaluate_rankings.py \\")
    print(f"        --corpus outputs/self_corpus.json \\")
    print(f"        --dossiers {output_path} \\")
    print("        --manual evals/manual_rankings.yaml --no-llm")
    print()
    print("=" * W)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch profile researchers from evals/manual_rankings.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended run order (sanity checks before full batch):

  # 1. Prestige traps — should score LOW
  python batch_profile_researchers.py --tier 1

  # 2. Positive controls — should score HIGH (ENRS must fire)
  python batch_profile_researchers.py --tier 5

  # 3. Full set
  python batch_profile_researchers.py --resume

  # 4. Evaluate
  python evaluate_rankings.py \\
      --corpus outputs/self_corpus.json \\
      --dossiers outputs/eval_dossiers.json \\
      --manual evals/manual_rankings.yaml --no-llm
""",
    )
    p.add_argument("--manual", default="evals/manual_rankings.yaml",
                   help="Manual rankings YAML file (default: evals/manual_rankings.yaml)")
    p.add_argument("--tier", type=int, default=None,
                   help="Only profile researchers of this manual tier (1-5)")
    p.add_argument("--cache-dir", default="outputs",
                   help="Cache directory (default: outputs)")
    p.add_argument("--limit-papers", type=int, default=10,
                   help="Papers to analyze per researcher (default: 10, lower=faster)")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-cached researchers (default: True)")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-profile even if already cached")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be profiled without making API calls")
    p.add_argument("--output", default="outputs/eval_dossiers.json",
                   help="Output dossier file (default: outputs/eval_dossiers.json)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load manual rankings
    try:
        all_researchers = _load_manual_rankings(args.manual)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Filter by tier if requested
    researchers = all_researchers
    if args.tier is not None:
        researchers = [r for r in all_researchers if int(round(r.get("manual_score", 0))) == args.tier]
        if not researchers:
            print(f"No researchers found with manual_score={args.tier}")
            sys.exit(1)
        print(f"\n  Filtering to tier {args.tier}: {len(researchers)} researchers")

    print(f"\n  Loaded {len(researchers)} researcher(s) to profile")
    print(f"  Cache dir:    {args.cache_dir}")
    print(f"  Limit papers: {args.limit_papers}")
    print(f"  Resume:       {not args.no_resume}")
    if args.dry_run:
        print("  Mode:         DRY RUN (no API calls)\n")

    # Confirm before long batch
    if not args.dry_run and len(researchers) > 5:
        est_min = len(researchers) * 1.5  # rough 1.5 min/researcher
        print(f"\n  Estimated time: ~{est_min:.0f} min (may vary by API rate limits)")
        confirm = input("  Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            sys.exit(0)

    # Run
    dossiers, report = batch_profile(
        researchers=researchers,
        cache_dir=args.cache_dir,
        limit_papers=args.limit_papers,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        # Merge with any existing eval_dossiers.json (so partial runs accumulate)
        output_path = Path(args.output)
        existing: list[dict] = []
        if output_path.exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    existing = [existing]
            except Exception:
                existing = []

        # Merge: new results overwrite by name, preserving existing for unmodified
        existing_by_name = {d.get("name", "").lower(): d for d in existing}
        for d in dossiers:
            existing_by_name[d.get("name", "").lower()] = d
        merged = list(existing_by_name.values())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False, default=str)

        # Save report
        report_path = Path(args.cache_dir) / "eval_profile_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        _print_report(report, dossiers, str(output_path))


if __name__ == "__main__":
    main()
