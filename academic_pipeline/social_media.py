"""
social_media.py — Local social media profiler.

Delegates entirely to AsyncSocialProfiler (async_fetcher.py).
This module exists for backward compatibility so the pipeline import tree
doesn't need to change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from academic_pipeline.async_fetcher import AsyncSocialProfiler
from academic_pipeline.local_llm import LocalLLMClient
from academic_pipeline.local_search import LocalSearchClient

logger = logging.getLogger(__name__)


class SocialMediaProfiler:
    """
    Backward-compatible wrapper around AsyncSocialProfiler.
    Replaces the previous Gemini-Search-Grounding-based implementation.
    """

    def __init__(self, api_key=None,   # kept for backward-compat; ignored
                 ollama_model: str = "qwen3:14b",
                 ollama_host:  str = "http://localhost:11434",
                 searxng_url:  str = "http://localhost:8080",
                 cache_manager = None):

        self._llm    = LocalLLMClient(model=ollama_model, host=ollama_host, cache_manager=cache_manager)
        self._search = LocalSearchClient(searxng_url=searxng_url, cache_manager=cache_manager)
        self._profiler = AsyncSocialProfiler(
            search_client=self._search,
            llm_client=self._llm,
            max_pages=6,
        )

    def profile_researcher(self, name: str,
                           institution: str = "",
                           interests: list[str] = None,
                           progress=None,
                           task_id=None) -> dict:
        """
        Synchronous entry point.  Runs the async profiler in an event loop.
        """
        async def _run():
            return await self._profiler.profile_researcher(
                name, institution, interests=interests, progress=progress, task_id=task_id
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, _run()).result(timeout=120)
            else:
                return asyncio.run(_run())
        except Exception as e:
            logger.error(f"Social profiling failed for '{name}': {e}")
            return {
                "summary":           f"Social profiling failed for {name}: {e}",
                "linkedin_url":      None,
                "twitter_url":       None,
                "personal_homepage": None,
                "linkedin_summary":  "",
                "twitter_summary":   "",
                "key_topics":        [],
                "tone":              "unknown",
                "recent_news_and_events": "",
            }
