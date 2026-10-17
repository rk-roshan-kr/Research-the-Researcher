"""
export_self_corpus.py — Export your own papers from Semantic Scholar or OpenAlex
into the JSON format expected by run_collaboration_analysis.py --corpus.

Usage:
    # By Semantic Scholar author ID
    python export_self_corpus.py --s2-author-id 2109419
    
    # By author name (searches Semantic Scholar)
    python export_self_corpus.py --author-name "Your Name" --institution "Your University"
    
    # By OpenAlex author ID
    python export_self_corpus.py --openalex-id A12345678

Output:
    outputs/self_corpus.json   — list of {abstract, title, doi, year, citations}

Then run:
    python run_collaboration_analysis.py --corpus outputs/self_corpus.json --no-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_self_corpus")


# ── Semantic Scholar ───────────────────────────────────────────────────────────

def _fetch_s2_by_id(author_id: str, limit: int = 100) -> list[dict]:
    """Fetch papers for a Semantic Scholar author ID."""
    import requests

    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
    params = {
        "fields": "title,abstract,year,externalIds,citationCount",
        "limit": min(limit, 1000),
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Semantic Scholar API error: {e}")
        return []

    papers = []
    for p in data.get("data", []):
        abstract = (p.get("abstract") or "").strip()
        if not abstract:
            continue
        papers.append({
            "title":     p.get("title", ""),
            "abstract":  abstract,
            "doi":       (p.get("externalIds") or {}).get("DOI", ""),
            "year":      p.get("year"),
            "citations": p.get("citationCount", 0),
            "source":    "semantic_scholar",
        })

    logger.info(f"Fetched {len(papers)} papers with abstracts from Semantic Scholar (author_id={author_id})")
    return papers


def _search_s2_author(name: str, institution: str = "") -> str | None:
    """Search Semantic Scholar for an author by name. Returns the best-match author ID."""
    import requests

    url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {"query": f"{name} {institution}".strip(), "fields": "name,affiliations,paperCount"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Author search error: {e}")
        return None

    candidates = data.get("data", [])
    if not candidates:
        logger.warning(f"No Semantic Scholar authors found for '{name}'")
        return None

    # Print candidates so user can verify
    print(f"\n  Found {len(candidates)} candidate(s) on Semantic Scholar:\n")
    for i, c in enumerate(candidates[:5]):
        affils = ", ".join(a.get("name", "") for a in (c.get("affiliations") or [])[:2])
        print(f"  [{i+1}] {c.get('name','?')}  |  {affils or 'unknown affil'}  |  {c.get('paperCount',0)} papers")
        print(f"       ID: {c.get('authorId','?')}")

    print()
    choice = input("  Enter number to select (or press Enter to use #1): ").strip()
    idx = (int(choice) - 1) if choice.isdigit() and 1 <= int(choice) <= len(candidates) else 0
    selected = candidates[idx]
    print(f"  → Selected: {selected.get('name')} (ID={selected.get('authorId')})\n")
    return selected.get("authorId")


# ── OpenAlex ──────────────────────────────────────────────────────────────────

def _fetch_openalex_by_id(openalex_id: str, limit: int = 100) -> list[dict]:
    """Fetch papers for an OpenAlex author ID (e.g. A12345678)."""
    import requests

    url = "https://api.openalex.org/works"
    params = {
        "filter":   f"author.id:{openalex_id}",
        "select":   "title,abstract_inverted_index,doi,publication_year,cited_by_count",
        "per-page": min(limit, 200),
        "mailto":   os.environ.get("OPENALEX_EMAIL", "researcher@example.com"),
    }
    try:
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"OpenAlex API error: {e}")
        return []

    papers = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        if not abstract:
            continue
        doi_raw = w.get("doi", "") or ""
        doi = doi_raw.replace("https://doi.org/", "").strip()
        papers.append({
            "title":     w.get("title", ""),
            "abstract":  abstract,
            "doi":       doi,
            "year":      w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "source":    "openalex",
        })

    logger.info(f"Fetched {len(papers)} papers with abstracts from OpenAlex (id={openalex_id})")
    return papers


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract string from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


# ── Output ────────────────────────────────────────────────────────────────────

def _save_corpus(papers: list[dict], output_path: str) -> None:
    """Save corpus to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(papers)} papers to {output_path}")


def _print_summary(papers: list[dict]) -> None:
    """Print a brief summary of the exported corpus."""
    if not papers:
        return
    years = [p["year"] for p in papers if p.get("year")]
    cites = [p["citations"] for p in papers if p.get("citations", 0) > 0]
    print(f"\n  ─────────────────────────────────────────")
    print(f"  Corpus summary: {len(papers)} papers with abstracts")
    if years:
        print(f"  Year range:     {min(years)} – {max(years)}")
    if cites:
        avg = sum(cites) / len(cites)
        print(f"  Citations:      avg {avg:.0f}  max {max(cites)}")
    print(f"  ─────────────────────────────────────────")
    print(f"\n  Next step:")
    print(f"    python run_collaboration_analysis.py \\")
    print(f"        --corpus outputs/self_corpus.json \\")
    print(f"        --no-llm --seed-investigations\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export your papers from Semantic Scholar / OpenAlex for use with run_collaboration_analysis.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # By Semantic Scholar author ID (fastest, most reliable)
  python export_self_corpus.py --s2-author-id 2109419

  # Search by name (interactive candidate selection)
  python export_self_corpus.py --author-name "Ada Lovelace" --institution "Cambridge"

  # By OpenAlex author ID
  python export_self_corpus.py --openalex-id A12345678

  # Limit to most recent 30 papers
  python export_self_corpus.py --s2-author-id 2109419 --limit 30
""",
    )
    p.add_argument("--s2-author-id", default=None,
                   help="Semantic Scholar author ID (e.g. 2109419)")
    p.add_argument("--author-name", default=None,
                   help="Author name to search on Semantic Scholar")
    p.add_argument("--institution", default="",
                   help="Institution hint to refine --author-name search")
    p.add_argument("--openalex-id", default=None,
                   help="OpenAlex author ID (e.g. A12345678)")
    p.add_argument("--limit", type=int, default=100,
                   help="Max papers to fetch (default: 100)")
    p.add_argument("--output", default="outputs/self_corpus.json",
                   help="Output path (default: outputs/self_corpus.json)")
    p.add_argument("--min-citations", type=int, default=0,
                   help="Filter out papers with fewer than N citations (default: 0 = keep all)")
    p.add_argument("--min-year", type=int, default=None,
                   help="Only include papers from this year onwards")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    papers: list[dict] = []

    if args.s2_author_id:
        papers = _fetch_s2_by_id(args.s2_author_id, limit=args.limit)

    elif args.author_name:
        author_id = _search_s2_author(args.author_name, args.institution)
        if author_id:
            papers = _fetch_s2_by_id(author_id, limit=args.limit)
        else:
            logger.error("No author selected. Aborting.")
            sys.exit(1)

    elif args.openalex_id:
        papers = _fetch_openalex_by_id(args.openalex_id, limit=args.limit)

    else:
        print("Error: provide one of --s2-author-id, --author-name, or --openalex-id")
        print("Run with --help for usage examples.")
        sys.exit(1)

    if not papers:
        logger.error("No papers with abstracts found. Check the author ID or name.")
        sys.exit(1)

    # Apply filters
    if args.min_citations > 0:
        before = len(papers)
        papers = [p for p in papers if p.get("citations", 0) >= args.min_citations]
        logger.info(f"Citation filter ({args.min_citations}+): {before} → {len(papers)} papers")

    if args.min_year:
        before = len(papers)
        papers = [p for p in papers if (p.get("year") or 0) >= args.min_year]
        logger.info(f"Year filter ({args.min_year}+): {before} → {len(papers)} papers")

    _save_corpus(papers, args.output)
    _print_summary(papers)


if __name__ == "__main__":
    main()
