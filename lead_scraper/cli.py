import os
import sys
import argparse
import asyncio
import logging
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from lead_scraper.database import LeadDatabase
from lead_scraper.core import WebEmailScraper
from lead_scraper.openalex import LeadOpenAlexSearcher
from lead_scraper.searcher import LeadSearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lead_scraper.log", encoding="utf-8"),
        logging.NullHandler()
    ]
)

console = Console()

def display_banner():
    banner = Text.assemble(
        ("\nEarthos Relationship & Outreach CRM Engine\n", "bold bright_yellow"),
        ("Sourcing, scoring, and tracking researchers and buyers\n", "italic cyan"),
        ("Relying entirely on verified, publicly listed institutional contacts\n", "dim white")
    )
    console.print(Panel(banner, border_style="yellow", expand=False))

def handle_scrape(args):
    """Direct URL scraping."""
    scraper = WebEmailScraper()
    console.print(f"[bold cyan]Scraping URL:[/bold cyan] {args.url}")
    
    with console.status("[bold green]Crawling and extracting emails...") as status:
        loop = asyncio.get_event_loop()
        records = loop.run_until_complete(scraper.scrape_url(args.url, deep_crawl=not args.no_deep))
    
    if records:
        console.print(f"\n[bold green]Found {len(records)} verified email address(es):[/bold green]")
        table = Table(title="Scraped Contacts Metadata")
        table.add_column("Email Address", style="bold white")
        table.add_column("Source Type", style="cyan")
        table.add_column("Permission", style="green")
        table.add_column("Confidence", style="yellow")
        for r in sorted(records, key=lambda x: x["confidence"], reverse=True):
            table.add_row(
                r["email"],
                r["email_source_type"],
                r["contact_permission"],
                str(r["confidence"])
            )
        console.print(table)
    else:
        console.print("\n[bold red]No email addresses found on this page.[/bold red]")

def handle_openalex(args):
    """OpenAlex lead generation command."""
    db = LeadDatabase()
    searcher = LeadOpenAlexSearcher()
    
    console.print(Panel(f"Searching OpenAlex for topic: [bold green]'{args.concept}'[/bold green]", border_style="green"))
    
    with console.status("[bold yellow]Fetching authors and papers...") as status:
        raw_leads = searcher.search_leads_by_concept(args.concept, limit=args.limit, use_fallback=not args.no_fallback)
    
    if not raw_leads:
        console.print("[bold red]No authors found on OpenAlex for this topic.[/bold red]")
        return

    console.print(f"Retrieved [bold green]{len(raw_leads)}[/bold green] potential leads. Calculating CRM scores...\n")
    
    table = Table(title="OpenAlex Sourced Contacts")
    table.add_column("Name", style="bold white")
    table.add_column("Tier", style="magenta")
    table.add_column("Email", style="green")
    table.add_column("Conf.", style="yellow")
    table.add_column("Rel.", style="blue")
    table.add_column("Pers %", style="cyan")
    table.add_column("Inst %", style="cyan")
    table.add_column("Infl.", style="yellow")
    
    added_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Saving leads...", total=len(raw_leads))
        
        for lead in raw_leads:
            # Set Campaign name
            lead["campaign"] = args.campaign or f"Earthos: {args.concept}"
            
            # Save to Database (scoring models run automatically inside add_lead)
            db.add_lead(lead)
            added_count += 1
            
            # Re-read to get computed fields
            db_leads = db.get_leads(campaign=lead["campaign"])
            saved_lead = next((l for l in db_leads if l["name"] == lead["name"]), lead)
            
            email_val = saved_lead.get("email") or "[dim]N/A[/dim]"
            
            table.add_row(
                saved_lead["name"],
                saved_lead.get("tier", "PhD"),
                email_val,
                str(saved_lead.get("confidence", 0)),
                str(saved_lead.get("research_relevance", 0)),
                str(saved_lead.get("personal_purchase_probability", 0)),
                str(saved_lead.get("institutional_adoption_probability", 0)),
                str(saved_lead.get("influence_score", 0))
            )
            progress.update(task, advance=1)
            
    console.print(table)
    console.print(f"\n[bold green]Successfully processed and saved {added_count} leads to the database.[/bold green]")

def handle_search(args):
    """Web-search based lead discovery command."""
    db = LeadDatabase()
    searcher = LeadSearcher()
    scraper = WebEmailScraper()
    
    console.print(Panel(f"Running search for directory: [bold green]'{args.query}'[/bold green]", border_style="cyan"))
    
    with console.status("[bold yellow]Searching the web...") as status:
        results = searcher.search_client.search(args.query, num_results=args.limit)
        
    if not results:
        console.print("[bold red]No search results returned.[/bold red]")
        return
        
    console.print(f"Found [bold green]{len(results)}[/bold green] matching directory/faculty page links. Crawling for emails...\n")
    
    table = Table(title="Scraped Search Leads")
    table.add_column("Title / Page", style="bold white")
    table.add_column("URL", style="cyan")
    table.add_column("Emails Found", style="green")
    
    loop = asyncio.get_event_loop()
    
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        
        # Async scrape (returns list of dicts with email + metadata)
        email_records = loop.run_until_complete(scraper.scrape_url(url, deep_crawl=args.deep_crawl))
        
        emails_list = [rec["email"] for rec in email_records]
        email_str = ", ".join(emails_list) if emails_list else "[dim]None found[/dim]"
        table.add_row(
            title[:40] + "..." if len(title) > 40 else title,
            url[:50] + "...",
            email_str
        )
        
        # Save any found emails to the database
        for rec in email_records:
            name_guess = rec["email"].split("@")[0].replace(".", " ").title()
            lead = {
                "name": name_guess,
                "email": rec["email"],
                "institution": urllib.parse.urlparse(url).netloc,
                "position": "Faculty" if "faculty" in url.lower() or "faculty" in title.lower() else "Researcher",
                "research_area": args.query,
                "faculty_url": url,
                "why_selected": f"Scraped from page: {title}",
                "campaign": args.campaign,
                
                # Email Auditing
                "email_source_url": rec["email_source_url"],
                "email_source_type": rec["email_source_type"],
                "confidence": rec["confidence"],
                "contact_permission": rec["contact_permission"]
            }
            db.add_lead(lead)

    console.print(table)
    console.print("\n[bold green]Sourced leads saved to SQLite database.[/bold green]")

def handle_export(args):
    """Export leads from DB to CSV."""
    db = LeadDatabase()
    success = db.export_to_csv(args.output, campaign=args.campaign, min_relevance=args.min_relevance)
    if success:
        console.print(f"[bold green]Successfully exported leads to CSV:[/bold green] {args.output}")
    else:
        console.print("[bold red]Failed to export leads. Make sure the database contains leads matching criteria.[/bold red]")

def handle_import(args):
    """Import leads from CSV to DB."""
    db = LeadDatabase()
    count = db.import_from_csv(args.input, campaign=args.campaign)
    console.print(f"[bold green]Import complete. Imported {count} leads into the database.[/bold green]")

def main():
    parser = argparse.ArgumentParser(
        description="Lead Scraper & Relationship CRM CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    
    # 1. Scrape Command
    scrape_parser = subparsers.add_parser("scrape", help="Extract emails directly from a URL")
    scrape_parser.add_argument("--url", type=str, required=True, help="URL to crawl and scrape")
    scrape_parser.add_argument("--no-deep", action="store_true", help="Disable crawling internal links")
    
    # 2. OpenAlex Command
    openalex_parser = subparsers.add_parser("openalex", help="Search OpenAlex for authors by topic/concept")
    openalex_parser.add_argument("--concept", type=str, required=True, help="Concept/topic to search (e.g. 'Continual Learning')")
    openalex_parser.add_argument("--limit", type=int, default=20, help="Number of leads to retrieve")
    openalex_parser.add_argument("--campaign", type=str, help="Target campaign tag (e.g. 'EarthOS Whitepaper')")
    openalex_parser.add_argument("--no-fallback", action="store_true", help="Disable web search fallback to speed up harvesting")
    
    # 3. Search Command
    search_parser = subparsers.add_parser("search", help="Find leads through general web directory search")
    search_parser.add_argument("--query", type=str, required=True, help="Search query (e.g. 'MIT AI professors')")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search result URLs to process")
    search_parser.add_argument("--deep-crawl", action="store_true", help="Follow internal links (about, contact, staff) on search result pages")
    search_parser.add_argument("--campaign", type=str, default="EarthOS: General", help="Outreach campaign name")
    
    # 4. Export Command
    export_parser = subparsers.add_parser("export", help="Export leads database to CSV")
    export_parser.add_argument("--output", type=str, required=True, help="CSV output filepath")
    export_parser.add_argument("--campaign", type=str, help="Filter export by campaign")
    export_parser.add_argument("--min-relevance", type=int, default=0, help="Minimum Research Relevance filter (0-100)")
    
    # 5. Import Command
    import_parser = subparsers.add_parser("import", help="Import leads from CSV")
    import_parser.add_argument("--input", type=str, required=True, help="CSV input filepath")
    import_parser.add_argument("--campaign", type=str, default="imported", help="Campaign label for imported leads")

    display_banner()
    
    args = parser.parse_args()
    
    if args.command == "scrape":
        handle_scrape(args)
    elif args.command == "openalex":
        handle_openalex(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "export":
        handle_export(args)
    elif args.command == "import":
        handle_import(args)

if __name__ == "__main__":
    main()
