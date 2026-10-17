# Lead Scraper: High-Quality Lead Generation & Email Collection Subproject

This subproject implements a targeted, high-fidelity lead generation pipeline for academic researchers, industry AI practitioners, and newsletter creators.

## Directory Structure

```text
lead_scraper/
├── README.md             # Documentation
├── __init__.py           # Package initializer
├── core.py               # Crawler and heuristic email scraper
├── openalex.py           # OpenAlex API search and metadata fetching
├── searcher.py           # Search helper (directories, conferences, companies)
├── guesser.py            # Email guesser and MX DNS validator
├── database.py           # Lead database with schema and scoring
└── cli.py                # Console interface powered by rich
```

## Lead Data Schema

Each lead is saved in a unified SQLite database containing:

*   `Name`: Full name of the researcher or lead.
*   `Email`: Extracted/verified email address.
*   `Institution`: University, lab, or company.
*   `Position`: Title or academic rank (e.g. PhD, Assistant Professor, CTO).
*   `Research Area`: Primary focus area (e.g. Continual Learning, World Models).
*   `Faculty URL`: Link to their homepage, faculty page, or directory.
*   `Recent Paper`: Title of their most recent/relevant publication.
*   `Why Selected`: Explanation of interest alignment.
*   `Interest Score`: Weighted score based on criteria (0-100).
*   `Campaign`: Name of the outreach batch.
*   `Email Sent`: Timestamp or boolean tracking.
*   `Opened`: Boolean.
*   `Clicked`: Boolean.
*   `Purchased`: Boolean.

## CLI Usage

Run the scraper via:

```bash
python -m lead_scraper.cli [command] [options]
```

### Commands

1. **`search`**: Discover leads through search queries (e.g. university directories, company staff pages).
2. **`openalex`**: Retrieve authors matching specific concepts (e.g. "Continual Learning") and populate metadata.
3. **`guess`**: Generate email permutations for leads using name + domain patterns, running MX verification.
4. **`scrape`**: Visit websites directly to crawl for contact details (handling obfuscated email formats).
5. **`export`**: Export leads matching specific filters or scores to CSV format.
