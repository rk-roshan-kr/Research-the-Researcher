#!/usr/bin/env python3
"""
discover_university.py — CLI tool to run the University Discovery Engine.
"""

import os
import sys
import codecs
import argparse
import logging
import asyncio

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError with box-drawing characters
if sys.platform.startswith("win"):
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Setup basic logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logger = logging.getLogger("discover_university")

# Add parent path to import academic_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_engine import InvestigationEngine
from academic_pipeline.university_discovery import UniversityDiscoveryEngine
from academic_pipeline.local_llm import LocalLLMClient

async def run_pipeline(args):
    universities = [u.strip() for u in args.university.split(",") if u.strip()]
    if not universities:
        logger.error("No universities specified.")
        sys.exit(1)

    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    logger.info("Initializing components...")
    cache_manager = CacheManager(cache_dir=cache_dir)
    investigation_engine = InvestigationEngine(cache_manager=cache_manager)
    llm_client = LocalLLMClient(model=args.model)
    
    engine = UniversityDiscoveryEngine(
        cache_manager=cache_manager,
        investigation_engine=investigation_engine,
        llm_client=llm_client
    )
    
    # Rate limit: MAX_CONCURRENT_UNIVERSITIES = 3
    sem_univ = asyncio.Semaphore(3)
    
    async def run_one_university(univ):
        async with sem_univ:
            logger.info(f"Starting discovery run for university: {univ}")
            report_path = await engine.discover_async(
                university_name=univ,
                mode=args.mode,
                limit=args.limit,
                model_name=args.model
            )
            return univ, report_path

    logger.info(f"Spawning parallel runs for universities: {universities}")
    tasks = [run_one_university(u) for u in universities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 60)
    print("ALL UNIVERSITY DISCOVERY RUNS COMPLETED")
    success = True
    for univ, res in zip(universities, results):
        if isinstance(res, Exception):
            logger.error(f"Discovery run failed for {univ}: {res}", exc_info=res)
            print(f"- {univ:<15} : FAILED ({type(res).__name__}: {res})")
            success = False
        else:
            univ_name, path = res
            print(f"- {univ_name:<15} : {path}")
    print("=" * 60 + "\n")
    if not success:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Discover potential research collaborators at a specific university."
    )
    parser.add_argument(
        "--university",
        type=str,
        required=True,
        help="Comma-separated list of universities (e.g. 'EPFL,ETH,MIT')."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["research", "internship"],
        default="internship",
        help="Discovery mode: 'research' (Mode A) or 'internship' (Mode B)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of candidates to score (limit 20)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Local LLM model to use for tension question generation."
    )
    
    args = parser.parse_args()
    
    if args.limit > 20:
        logger.warning("Limit capped at 20. Only top 20 candidates will be scored and listed.")
        args.limit = 20
        
    asyncio.run(run_pipeline(args))

if __name__ == "__main__":
    main()
