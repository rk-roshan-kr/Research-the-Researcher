import os
import sys
import argparse
import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from academic_pipeline.pipeline import AcademicPipeline
from academic_pipeline.report_generator import ReportGenerator

# Configure Console
console = Console()

class CustomProgress(Progress):
    def __init__(self, *args, **kwargs):
        self.show_details = False  # Set BEFORE parent init to avoid AttributeError during init callbacks
        super().__init__(*args, **kwargs)

    def get_renderable(self):
        progress_table = super().get_renderable()
        if not self.show_details:
            return progress_table

        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text
        from academic_pipeline.local_llm import get_llm_activity_log

        logs = get_llm_activity_log(max_lines=12)
        
        # Color coding or formatting for logs
        formatted_logs = Text()
        if not logs:
            formatted_logs.append("No LLM activity recorded yet. (LLMs analyze concurrently: abstracts=4, full-texts=1)", style="dim")
        else:
            for log in logs:
                if "Tool Call" in log or "search_web" in log:
                    formatted_logs.append(log + "\n", style="bold yellow")
                elif "failed" in log or "Error" in log or "Attempt" in log:
                    formatted_logs.append(log + "\n", style="bold red")
                elif "complete" in log or "Hit" in log:
                    formatted_logs.append(log + "\n", style="bold green")
                elif "Analyzing" in log or "Calling" in log:
                    formatted_logs.append(log + "\n", style="bold cyan")
                else:
                    formatted_logs.append(log + "\n", style="white")

        panel = Panel(
            formatted_logs,
            title="[bold yellow]LLM Live Processing Logs (Press Ctrl+O or O to collapse)[/bold yellow]",
            border_style="yellow",
            padding=(0, 1)
        )
        return Group(progress_table, panel)

def key_listener(progress_instance, stop_event):
    import os
    import sys
    import time
    if os.name != 'nt' or not sys.stdin.isatty():
        return
    import msvcrt
    while not stop_event.is_set():
        try:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                # Check for Ctrl+O (\x0f) or o/O
                if ch in (b'\x0f', b'o', b'O'):
                    progress_instance.show_details = not progress_instance.show_details
                    progress_instance.refresh()
        except Exception:
            pass
        time.sleep(0.1)


def setup_logging(verbose=False):
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
        ]
    )
    # Silence third-party logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)


def display_banner():
    """Display an eye-catching ASCII/text banner in the console."""
    banner_text = Text.assemble(
        ("Academic Profile Collection Pipeline\n", "bold violet"),
        ("Gather & analyze academic papers and social intelligence.\n\n", "italic cyan"),
        ("[Tip] Hardware Performance Tip:\n", "bold green"),
        ("To fully utilize high-end GPUs (e.g. RTX 5070 Ti), launch Ollama with:\n", "dim"),
        ("  $env:OLLAMA_NUM_PARALLEL=8\n", "bold yellow"),
        ("and run the pipeline with: ", "dim"),
        ("--max-workers-abstract 8 --max-workers-fulltext 2", "bold cyan")
    )
    console.print(Panel(banner_text, border_style="violet", expand=False))

def interactive_disambiguation(candidates):
    """
    Present a rich table of candidate authors to the user and let them select the correct one.
    Supports "none of these" which lets the user refine the search or paste a direct ID/URL.
    Returns:
      - a candidate dict if the user selects one
      - {"__action__": "retry", "query": new_query} if the user wants to search again
      - None if the user gives up
    """
    while True:
        console.print("\n[bold yellow]Multiple matching researchers found. Please select one:[/bold yellow]")

        table = Table(show_header=True, header_style="bold violet")
        table.add_column("No.", style="dim", width=5)
        table.add_column("Source", style="cyan", width=14)
        table.add_column("Name", style="bold white", width=18)
        table.add_column("Affiliation", style="green", max_width=44)
        table.add_column("Works", justify="right", style="magenta", width=7)
        table.add_column("Cites", justify="right", style="magenta", width=8)
        table.add_column("H", justify="right", style="yellow", width=4)
        table.add_column("ORCID", style="blue", width=22)

        for idx, cand in enumerate(candidates):
            table.add_row(
                str(idx + 1),
                cand["source"],
                cand["name"],
                cand["affiliation"] or "N/A",
                f"{cand['papers_count']:,}",
                f"{cand['citations_count']:,}",
                str(cand["h_index"]),
                cand["orcid"] or "N/A"
            )
        # Always add a "none" row
        table.add_row(
            "0",
            "-",
            "[italic dim]Not in this list - search differently[/italic dim]",
            "", "", "", "", ""
        )
        console.print(table)

        valid_choices = [str(i + 1) for i in range(len(candidates))] + ["0"]
        choice = Prompt.ask(
            "Enter the number of the correct profile (0 = search differently)",
            choices=valid_choices,
            default="1"
        )

        if choice == "0":
            console.print(
                "\n[bold cyan]Enter a new search term.[/bold cyan]\n"
                "  * Refine the name, e.g. [italic]Jiajun Wu Stanford[/italic]\n"
                "  * Or paste a direct identifier:\n"
                "    - ORCID:          [italic]0000-0002-XXXX-XXXX[/italic]\n"
                "    - OpenAlex URL:   [italic]https://openalex.org/A12345[/italic]\n"
                "    - Semantic Scholar URL: [italic]https://www.semanticscholar.org/author/Name/12345[/italic]\n"
                "    - Google Scholar URL:   [italic]https://scholar.google.com/citations?user=XXXXX[/italic]\n"
            )
            new_query = Prompt.ask("New search / identifier").strip()
            if new_query:
                return {"__action__": "retry", "query": new_query}
            console.print("[red]Empty input — keeping current results.[/red]")
            continue

        selected_idx = int(choice) - 1
        selected = candidates[selected_idx]
        console.print(f"[green]Selected: {selected['name']} ({selected['affiliation'] or 'No Affiliation'})[/green]\n")
        return selected

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    parser = argparse.ArgumentParser(
        description="Fetch and compile academic profile data + social intelligence."
    )
    
    # Input queries
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", "-q", help="General identifier (Name, ORCID, Google Scholar URL, S2 ID, etc.)")
    input_group.add_argument("--name", "-n", help="Researcher name")
    input_group.add_argument("--orcid", "-o", help="ORCID iD (e.g. 0000-0002-1825-0097)")
    input_group.add_argument("--scholar", "-s", help="Google Scholar profile URL or User ID")
    input_group.add_argument("--openalex", help="OpenAlex ID (e.g., A5003442464)")
    input_group.add_argument("--semantic-scholar", help="Semantic Scholar Author ID")
    input_group.add_argument("--batch-file", "-bf", help="Path to a text file containing researcher names/identifiers (one per line)")
    input_group.add_argument("--batch", "-b", nargs="+", help="Space-separated or comma-separated list of researcher names/identifiers")
    input_group.add_argument("--institution", "-inst", help="Institution/University name or OpenAlex ID (e.g., 'Stanford University' or 'I97018004')")
    
    # Configuration
    parser.add_argument("--output-dir", "-out", default="outputs", help="Output directory for reports (default: 'outputs')")
    parser.add_argument("--limit-papers", "-l", type=int, default=10, help="Limit number of papers to download and analyze (default: 10)")
    parser.add_argument("--deep-run", action="store_true", help="Analyze ALL publications of the researcher (warning: takes a long time!)")
    parser.add_argument("--lite-scan", action="store_true", help="Skip PDF downloads entirely and use abstracts only (fast initial screening mode)")
    parser.add_argument("--graph-depth", type=int, default=1, choices=[1, 2],
                        help="Depth of collaboration tree: 1=direct co-authors, 2=co-authors of co-authors (default: 1)")
    parser.add_argument("--max-coauthors", type=int, default=30,
                        help="Max number of co-author nodes to enrich in the graph (default: 30)")
    parser.add_argument("--auto-select", action="store_true", help="Skip interactive selection and pick the top match automatically")
    parser.add_argument("--department", "-dept", help="Department name for institution mode (e.g., 'Computer Science')")
    parser.add_argument("--limit-professors", type=int, default=50, help="Maximum number of professors to process in department mode (default: 50)")
    parser.add_argument("--resume", action="store_true", help="Resume batch execution from previously saved state if available")

    # Local LLM settings (no API key needed)
    parser.add_argument("--ollama-model", default="qwen2.5-coder:7b",
                        help="Ollama model tag to use for paper/social analysis (default: qwen2.5-coder:7b)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--searxng-url", default="http://localhost:8080",
                        help="SearxNG instance URL for local search (default: http://localhost:8080)")
    parser.add_argument("--max-workers-abstract", type=int, default=4,
                        help="Max concurrent analyses for abstract-only papers (default: 4, scale up to 8-12 for high-end GPUs like RTX 5070 Ti)")
    parser.add_argument("--max-workers-fulltext", type=int, default=1,
                        help="Max concurrent analyses for full-text PDF papers (default: 1, scale up to 2-3 for high-end GPUs like RTX 5070 Ti)")

    # Optional academic DB keys (free tiers work fine)
    parser.add_argument("--gemini-key", help="[DEPRECATED] Gemini API Key — no longer required")
    parser.add_argument("--openalex-key", help="OpenAlex API Key")
    parser.add_argument("--openalex-email", help="Email for OpenAlex polite pool")
    parser.add_argument("--s2-key", help="Semantic Scholar API Key")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug verbose logging")
    
    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.verbose)

    # Warn if old Gemini key is passed
    if getattr(args, 'gemini_key', None) or os.environ.get('GEMINI_API_KEY'):
        console.print("[yellow]Note: GEMINI_API_KEY is no longer required. "
                      "All analysis now runs locally via Ollama.[/yellow]")

    display_banner()
    
    # Check if we are running in batch mode
    is_batch_mode = bool(args.batch_file or args.batch or args.institution)
    
    # Determine the query string (for single mode)
    query = ""
    if not is_batch_mode:
        if args.query:
            query = args.query
        elif args.name:
            query = args.name
        elif args.orcid:
            # Pre-format to make it explicitly clean
            query = f"orcid:{args.orcid}" if not args.orcid.startswith("http") else args.orcid
        elif args.scholar:
            query = args.scholar
        elif args.openalex:
            query = args.openalex
        elif args.semantic_scholar:
            query = args.semantic_scholar
        
    # Configure callback for interactive disambiguation
    # If stdout/stdin is not a tty (like in automated tests or script redirect), force auto-select
    is_interactive = sys.stdin.isatty() and not args.auto_select and not is_batch_mode
    callback = interactive_disambiguation if is_interactive else None

    # Resolve output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Pipeline
    try:
        pipeline = AcademicPipeline(
            gemini_api_key=args.gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            openalex_api_key=args.openalex_key,
            openalex_email=args.openalex_email,
            s2_api_key=args.s2_key,
            cache_dir=output_dir,
            download_dir=os.path.join(output_dir, "pdfs"),
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            searxng_url=args.searxng_url,
            max_workers_abstract=args.max_workers_abstract,
            max_workers_fulltext=args.max_workers_fulltext,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize pipeline: {e}[/bold red]")
        return 1

    import atexit
    from academic_pipeline.tracker import ProcessTracker
    cmd_str = "python run.py " + " ".join(sys.argv[1:])
    tracker = ProcessTracker(task_id="cli-run", command=cmd_str)
    tracker.start()
    atexit.register(tracker.stop)

    if is_batch_mode:
        import hashlib
        import json
        import time
        import threading
        
        batch_queries = []
        prof_names = {}
        batch_title = "Consolidated Academic Profiles Batch Report"

        if args.batch_file:
            if not os.path.exists(args.batch_file):
                console.print(f"[bold red]Error: Batch file '{args.batch_file}' does not exist.[/bold red]")
                return 1
            with open(args.batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        batch_queries.append(line)
            batch_title = f"Batch Report: {os.path.basename(args.batch_file)}"
        elif args.batch:
            for item in args.batch:
                if "," in item:
                    batch_queries.extend([q.strip() for q in item.split(",") if q.strip()])
                else:
                    batch_queries.append(item.strip())
            batch_title = "Batch Query Run"
        elif args.institution:
            if not args.department:
                console.print("[bold red]Error: --department / -dept is required when running in university department mode (--institution).[/bold red]")
                return 1
            try:
                console.print(f"[bold cyan]Resolving professors for department '{args.department}' at '{args.institution}'...[/bold cyan]")
                professors = pipeline.resolve_department_professors(
                    args.institution,
                    args.department,
                    limit_professors=args.limit_professors
                )
                if not professors:
                    console.print("[bold red]Error: No professors found for this institution and department.[/bold red]")
                    return 1
                batch_queries = [p["openalex_id"] for p in professors]
                prof_names = {p["openalex_id"]: p["name"] for p in professors}
                batch_title = f"Department of {args.department} at {professors[0]['affiliation']}"
            except Exception as e:
                console.print(f"[bold red]Failed to resolve department: {e}[/bold red]")
                return 1

        if not batch_queries:
            console.print("[bold red]Error: No target queries resolved for batch processing.[/bold red]")
            return 1

        console.print(f"[bold green]Resolved {len(batch_queries)} researchers to process.[/bold green]")

        # Unique state tracking per batch signature
        param_str = ",".join(sorted(batch_queries))
        query_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:12]
        state_filename = f"batch_state_{query_hash}.json"
        state_path = os.path.join(output_dir, state_filename)

        state = {
            "query_hash": query_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processed": {},
            "pending": batch_queries.copy()
        }

        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    saved_state = json.load(f)
                    if saved_state.get("query_hash") == query_hash:
                        state = saved_state
                        console.print(f"[bold green]Resuming batch from saved state. Completed: {len(state['processed'])}, Pending: {len(state['pending'])}[/bold green]")
                    else:
                        console.print("[yellow]Existing batch state file found, but queries list has changed. Starting a new batch run.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Could not load batch state file: {e}. Starting fresh.[/yellow]")

        interrupted = False
        for q in list(state["pending"]):
            if interrupted:
                break
                
            display_name = prof_names.get(q, q)
            console.print(f"\n[bold violet]========================================[/bold violet]")
            console.print(f"[bold violet]Processing target {len(state['processed'])+1} of {len(batch_queries)}: {display_name}[/bold violet]")
            console.print(f"[bold violet]========================================[/bold violet]\n")

            try:
                from rich.progress import TimeElapsedColumn, TaskProgressColumn, TimeRemainingColumn
                stop_event = threading.Event()
                
                with CustomProgress(
                    SpinnerColumn(spinner_name="dots", style="bold violet"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=40, style="dim", complete_style="bold violet"),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    
                    main_task = progress.add_task("[violet]Compiling academic dossier...", total=4)
                    
                    dossier = pipeline.run_pipeline(
                        query=q,
                        limit_papers=args.limit_papers,
                        deep_run=args.deep_run,
                        graph_depth=args.graph_depth,
                        max_coauthors=args.max_coauthors,
                        lite_scan=args.lite_scan,
                        interactive_callback=None,
                        progress=progress
                    )
                    
                    progress.advance(main_task)
                    
                    clean_name = "".join(c for c in dossier["name"] if c.isalnum() or c in (" ", "_")).replace(" ", "_").lower()
                    researcher_dir = os.path.join(output_dir, clean_name)
                    os.makedirs(researcher_dir, exist_ok=True)
                    
                    # JSON
                    json_filename = f"{clean_name}_profile.json"
                    json_path = os.path.join(researcher_dir, json_filename)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(dossier, f, indent=2, ensure_ascii=False)
                    progress.advance(main_task)
                    
                    # MD
                    md_filename = f"{clean_name}_report.md"
                    md_path = os.path.join(researcher_dir, md_filename)
                    generator = ReportGenerator()
                    generator.generate_markdown(dossier, md_path)
                    progress.advance(main_task)
                    
                    # HTML
                    html_filename = f"{clean_name}_report.html"
                    html_path = os.path.join(researcher_dir, html_filename)
                    generator.generate_html(dossier, html_path)
                    progress.advance(main_task)
                    
                    state["processed"][q] = {
                        "name": dossier["name"],
                        "query": q,
                        "status": "success",
                        "error": None,
                        "report_dir": clean_name,
                        "report_html": f"{clean_name}/{html_filename}",
                        "report_md": f"{clean_name}/{md_filename}",
                        "metrics": {
                            "h_index": dossier["metrics"]["h_index"],
                            "citations": dossier["metrics"]["citations"],
                            "publications": dossier["metrics"]["publications"]
                        },
                        "interests": dossier["interests"],
                        "affiliation": dossier["affiliation"]
                    }
                    progress.update(main_task, description="[green]Pipeline execution complete!")
                    
            except KeyboardInterrupt:
                console.print("\n[bold red]KeyboardInterrupt received! Saving progress state and aborting batch cleanly...[/bold red]")
                interrupted = True
                break
            except Exception as e:
                console.print(f"\n[bold red]Failed processing researcher {display_name}: {e}[/bold red]")
                state["processed"][q] = {
                    "name": display_name,
                    "query": q,
                    "status": "failed",
                    "error": str(e),
                    "report_dir": None,
                    "report_html": None,
                    "report_md": None,
                    "metrics": None,
                    "interests": None,
                    "affiliation": None
                }
                
            if q in state["pending"]:
                state["pending"].remove(q)
                
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
            except Exception as se:
                logger.error(f"Failed to write state file: {se}")

        # Final Compilation & Reporting
        results = list(state["processed"].values())
        successful_results = [r for r in results if r["status"] == "success"]
        total_citations = sum(r["metrics"]["citations"] for r in successful_results if r["metrics"] and r["metrics"].get("citations"))
        max_h_index = max((r["metrics"]["h_index"] for r in successful_results if r["metrics"] and r["metrics"].get("h_index")), default=0)
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_targets": len(batch_queries),
            "successful_count": len(successful_results),
            "failed_count": len(results) - len(successful_results),
            "total_citations": total_citations,
            "max_h_index": max_h_index
        }

        generator = ReportGenerator()
        batch_html_path = os.path.join(output_dir, "batch_report.html")
        batch_md_path = os.path.join(output_dir, "batch_report.md")
        
        generator.generate_batch_html(batch_title, results, summary, batch_html_path)
        generator.generate_batch_markdown(batch_title, results, summary, batch_md_path)

        console.print(f"\n[bold green]Batch run complete! Master report and dashboards generated.[/bold green]")
        
        table_output = Table(show_header=True, header_style="bold cyan")
        table_output.add_column("Researcher", style="bold")
        table_output.add_column("Affiliation")
        table_output.add_column("Status")
        table_output.add_column("H-Index", justify="right")
        table_output.add_column("Citations", justify="right")
        table_output.add_column("Reports")
        
        for r in results:
            status_text = "[green]Success[/green]" if r["status"] == "success" else "[red]Failed[/red]"
            if r["status"] == "success":
                h_idx = str(r["metrics"]["h_index"])
                cites = f"{r['metrics']['citations']:,}"
                report_link = f"[link=file:///{os.path.abspath(os.path.join(output_dir, r['report_html'])).replace('\\', '/')}]HTML[/link] | [link=file:///{os.path.abspath(os.path.join(output_dir, r['report_md'])).replace('\\', '/')}]MD[/link]"
            else:
                h_idx = "N/A"
                cites = "N/A"
                report_link = f"[dim]N/A ({r['error']})[/dim]"
            table_output.add_row(r["name"], r["affiliation"] or "N/A", status_text, h_idx, cites, report_link)
            
        console.print(table_output)
        
        abs_batch_html = os.path.abspath(batch_html_path)
        abs_batch_md = os.path.abspath(batch_md_path)
        console.print(f"\n[bold]Master Batch Reports:[/bold]")
        console.print(f"  • Interactive HTML Dashboard: [link=file:///{abs_batch_html.replace('\\', '/')}]file:///{abs_batch_html}[/link]")
        console.print(f"  • Markdown Summary Dashboard: [link=file:///{abs_batch_md.replace('\\', '/')}]file:///{abs_batch_md}[/link]")
        
        # Clean up batch state file if completed fully
        if not state["pending"] and os.path.exists(state_path):
            try:
                os.remove(state_path)
            except Exception:
                pass
                
        return 0

    else:
        # Single profile collection (Original Flow)
        dossier = None
        try:
            from rich.progress import TimeElapsedColumn, TaskProgressColumn, TimeRemainingColumn
            import threading
            stop_event = threading.Event()
            
            with CustomProgress(
                SpinnerColumn(spinner_name="dots", style="bold violet"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40, style="dim", complete_style="bold violet"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                # Start key listener thread on Windows if interactive
                listener_thread = None
                if os.name == 'nt' and sys.stdin.isatty():
                    listener_thread = threading.Thread(target=key_listener, args=(progress, stop_event), daemon=True)
                    listener_thread.start()
                    console.print("[dim]Live LLM activity panel: Press [bold yellow]Ctrl+O[/bold yellow] or [bold yellow]O[/bold yellow] to toggle LLM processing details.[/dim]\n")
                
                main_task = progress.add_task("[violet]Compiling academic dossier...", total=4)
                
                # Step 1: Execute the backend pipeline
                if is_interactive:
                    progress.stop()
                    
                try:
                    dossier = pipeline.run_pipeline(
                        query=query,
                        limit_papers=args.limit_papers,
                        deep_run=args.deep_run,
                        graph_depth=args.graph_depth,
                        max_coauthors=args.max_coauthors,
                        lite_scan=args.lite_scan,
                        interactive_callback=callback,
                        progress=progress
                    )
                finally:
                    # Always signal key listener thread to stop
                    stop_event.set()
                    if listener_thread:
                        listener_thread.join(timeout=1.0)
                
                if is_interactive:
                    progress.start()
                    
                progress.advance(main_task)
                
                # Step 2: Write raw JSON
                clean_name = "".join(c for c in dossier["name"] if c.isalnum() or c in (" ", "_")).replace(" ", "_").lower()
                researcher_dir = os.path.join(output_dir, clean_name)
                os.makedirs(researcher_dir, exist_ok=True)
                
                import json
                json_path = os.path.join(researcher_dir, f"{clean_name}_profile.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(dossier, f, indent=2, ensure_ascii=False)
                progress.advance(main_task)
                
                # Step 3: Render Markdown Report
                md_path = os.path.join(researcher_dir, f"{clean_name}_report.md")
                generator = ReportGenerator()
                generator.generate_markdown(dossier, md_path)
                progress.advance(main_task)
                
                # Step 4: Render HTML Report
                html_path = os.path.join(researcher_dir, f"{clean_name}_report.html")
                generator.generate_html(dossier, html_path)
                progress.advance(main_task)
    
                # Done!
                progress.update(main_task, description="[green]Pipeline execution complete!")
                
            # Display Success Output
            console.print(f"\n[bold green]Success! Dossier generated![/bold green]")
            
            table_output = Table(show_header=True, header_style="bold cyan")
            table_output.add_column("Report Format", style="bold")
            table_output.add_column("Output File Path")
            
            # Absolute paths for ease of clicking
            abs_json = os.path.abspath(json_path)
            abs_md = os.path.abspath(md_path)
            abs_html = os.path.abspath(html_path)
            
            table_output.add_row("Raw JSON Profile", f"[link=file:///{abs_json.replace('\\', '/')}]file:///{abs_json}[/link]")
            table_output.add_row("Markdown Summary", f"[link=file:///{abs_md.replace('\\', '/')}]file:///{abs_md}[/link]")
            table_output.add_row("Interactive HTML", f"[link=file:///{abs_html.replace('\\', '/')}]file:///{abs_html}[/link]")
            
            console.print(table_output)
            
            # Print a small stats dashboard
            console.print(f"\n[bold]Profile Summary for {dossier['name']}:[/bold]")
            console.print(f"  • Affiliation:  [cyan]{dossier['affiliation'] or 'N/A'}[/cyan]")
            console.print(f"  • H-Index:      [yellow]{dossier['metrics']['h_index']}[/yellow]")
            console.print(f"  • Citations:    [magenta]{dossier['metrics']['citations']:,}[/magenta]")
            console.print(f"  • Publications: [blue]{dossier['metrics']['publications']}[/blue]")
            
            analyzed_count = sum(1 for p in dossier["publications"] if p.get("analysis"))
            console.print(f"  • Deep Analyzed Publications: [violet]{analyzed_count} / {len(dossier['publications'])}[/violet]")
    
            graph = dossier.get("collaboration_graph", {})
            console.print(f"  • Collaboration Graph Nodes:  [cyan]{len(graph.get('nodes', []))}[/cyan] (researchers)")
            console.print(f"  • Collaboration Graph Edges:  [cyan]{len(graph.get('edges', []))}[/cyan] (collaboration links)")
            
            return 0
            
        except Exception as e:
            console.print(f"\n[bold red]Pipeline failed: {e}[/bold red]")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

if __name__ == "__main__":
    sys.exit(main())
