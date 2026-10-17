import os
import shutil
import sqlite3
import logging
from academic_pipeline.pipeline import AcademicPipeline
from academic_pipeline.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DB = "outputs/pipeline_cache.db"

def clear_incorrect_caches():
    """Clear name resolution and search caches for Alexander Mathis to force a correct lookup."""
    if os.path.exists(CACHE_DB):
        logger.info(f"Connecting to cache database: {CACHE_DB}")
        try:
            conn = sqlite3.connect(CACHE_DB)
            cursor = conn.cursor()
            
            # Delete Alexander Mathis resolved profile
            cursor.execute("DELETE FROM resolved_profiles WHERE key LIKE ?", ("%alexander_mathis%",))
            # Delete Alexander Mathis search results
            cursor.execute("DELETE FROM search_results WHERE key LIKE ?", ("%alexander_mathis%",))
            # Delete blank abstract entries to force a fresh lookup
            cursor.execute("DELETE FROM paper_abstracts WHERE data LIKE ?", ('%"abstract": ""%',))
            
            conn.commit()
            logger.info("Successfully cleared cache entries for Alexander Mathis and blank abstracts.")
            conn.close()
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            
    # Delete the old incorrect output directory for Alexander Mathis (the Zoology one)
    incorrect_dir = "outputs/alexander_mathis"
    if os.path.exists(incorrect_dir):
        logger.info(f"Removing incorrect Zoology output folder: {incorrect_dir}")
        try:
            shutil.rmtree(incorrect_dir)
            logger.info("Removed Zoology folder successfully.")
        except Exception as e:
            logger.error(f"Error removing Zoology folder: {e}")

def run_update():
    # Initialize the pipeline
    pipeline = AcademicPipeline(
        cache_dir="outputs",
        download_dir="outputs/pdfs",
        ollama_model="qwen2.5-coder:7b",
        max_workers_abstract=4,
        max_workers_fulltext=1
    )
    
    # List of profiles to rebuild/update
    # We use exact OpenAlex IDs where possible to guarantee exact matches, especially for Mathis
    researchers = [
        {"name": "Martin Schrimpf", "query": "A5002008882"},
        {"name": "Alexander Mathis", "query": "A5013861606"}, # Correct EPFL ID
        {"name": "Manoj Singla", "query": "Manoj Singla"},
        {"name": "Donald W. Brown", "query": "Donald W. Brown"},
        {"name": "Alexander W. Astin", "query": "A5078467902"},
        {"name": "Jiajun Wu", "query": "Jiajun Wu"},
        {"name": "Garima Thakur", "query": "Garima Thakur"}
    ]
    
    generator = ReportGenerator()
    results = []
    
    for r_info in researchers:
        name = r_info["name"]
        query = r_info["query"]
        logger.info(f"\n========================================")
        logger.info(f"Updating profile: {name} (Query: {query})")
        logger.info(f"========================================")
        
        try:
            # Execute the pipeline (uses caches for LLM and Social, but fetches/caches abstracts)
            dossier = pipeline.run_pipeline(
                query=query,
                limit_papers=5,
                max_coauthors=0,
                deep_run=False,
                interactive_callback=None
            )
            
            clean_name = "".join(c for c in dossier["name"] if c.isalnum() or c in (" ", "_")).replace(" ", "_").lower()
            researcher_dir = os.path.join("outputs", clean_name)
            os.makedirs(researcher_dir, exist_ok=True)
            
            # Save raw JSON profile
            json_path = os.path.join(researcher_dir, f"{clean_name}_profile.json")
            import json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(dossier, f, indent=2, ensure_ascii=False)
                
            # Generate MD report
            md_path = os.path.join(researcher_dir, f"{clean_name}_report.md")
            generator.generate_markdown(dossier, md_path)
            
            # Generate HTML report
            html_path = os.path.join(researcher_dir, f"{clean_name}_report.html")
            generator.generate_html(dossier, html_path)
            
            # Record result for batch summary
            results.append({
                "name": dossier["name"],
                "query": query,
                "status": "success",
                "error": None,
                "report_dir": clean_name,
                "report_html": f"{clean_name}/{clean_name}_report.html",
                "report_md": f"{clean_name}/{clean_name}_report.md",
                "metrics": {
                    "h_index": dossier["metrics"]["h_index"],
                    "citations": dossier["metrics"]["citations"],
                    "publications": dossier["metrics"]["publications"]
                },
                "interests": dossier["interests"],
                "affiliation": dossier["affiliation"]
            })
            logger.info(f"Successfully updated profile and reports for: {dossier['name']}")
        except Exception as e:
            logger.error(f"Failed to update profile for {name}: {e}")
            results.append({
                "name": name,
                "query": query,
                "status": "failed",
                "error": str(e)
            })

    # Compile a master batch report summary of all update actions
    import time
    successful = [r for r in results if r["status"] == "success"]
    total_citations = sum(r["metrics"]["citations"] for r in successful if r["metrics"])
    max_h_index = max((r["metrics"]["h_index"] for r in successful if r["metrics"]), default=0)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_targets": len(researchers),
        "successful_count": len(successful),
        "failed_count": len(results) - len(successful),
        "total_citations": total_citations,
        "max_h_index": max_h_index
    }
    
    batch_html_path = "outputs/batch_report.html"
    batch_md_path = "outputs/batch_report.md"
    
    generator.generate_batch_html("Master Research Profiles Update Dashboard", results, summary, batch_html_path)
    generator.generate_batch_markdown("Master Research Profiles Update Dashboard", results, summary, batch_md_path)
    logger.info("Successfully updated Master Batch Report dashboards.")

if __name__ == "__main__":
    from academic_pipeline.tracker import ProcessTracker
    tracker = ProcessTracker(task_id="update_db", command="python update_db.py")
    tracker.start()
    try:
        clear_incorrect_caches()
        run_update()
    finally:
        tracker.stop()

