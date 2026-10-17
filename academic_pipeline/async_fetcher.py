"""
async_fetcher.py — Async web fetcher and social media profiler.

Replaces: social_media.py (which used Gemini Search Grounding).

Architecture:
  1. LocalSearchClient finds profile URLs (LinkedIn, Twitter/X, homepage)
  2. AsyncBFSCrawler fetches those pages via crawl4ai (Playwright-backed) or
     plain aiohttp for simple pages
  3. LocalLLMClient (Ollama / Qwen3) summarises the scraped text into a
     structured SocialProfile dict

Key DSA / system design:
  - BFS URL frontier: asyncio.PriorityQueue — most-relevant URLs crawled first
  - URL deduplication: seen_urls set (O(1) membership test)
  - Robots.txt cache: one RobotFileParser per domain loaded lazily
  - Per-domain token bucket: prevents hitting any single domain too fast
  - Playwright-backed rendering via crawl4ai handles SPAs / JS-heavy pages
  - aiohttp + connection pool for static pages (much faster than Playwright)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import aiohttp

from academic_pipeline.local_search import LocalSearchClient, TokenBucket

logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("crawl4ai not installed. JS-rendered page crawling disabled. "
                   "Install: pip install crawl4ai && playwright install chromium")


# ──────────────────────────────────────────────────────────────────────────────
# Robots.txt Cache
# ──────────────────────────────────────────────────────────────────────────────

class RobotCache:
    """Lazily loads and caches robots.txt per domain."""

    def __init__(self):
        self._cache: dict[str, RobotFileParser] = {}

    def can_fetch(self, url: str,
                  user_agent: str = "*") -> bool:
        """Return True if robots.txt allows fetching this URL."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            rp = RobotFileParser()
            rp.set_url(f"{domain}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If we can't read robots.txt, assume allowed
                rp.parse(["User-agent: *", "Allow: /"])
            self._cache[domain] = rp
        return self._cache[domain].can_fetch(user_agent, url)


# ──────────────────────────────────────────────────────────────────────────────
# Async BFS Crawler
# ──────────────────────────────────────────────────────────────────────────────

class AsyncBFSCrawler:
    """
    Priority-queue BFS crawler.

    Priority scoring: lower number = crawled first.
      - LinkedIn /in/ profiles   → priority 0
      - Twitter/X profiles       → priority 1
      - Personal homepages       → priority 2
      - General pages            → priority 5

    URL deduplication uses a seen_urls set (O(1)).
    Per-domain TokenBucket prevents rate-limit errors.
    """

    PRIORITY_MAP = {
        "linkedin.com/in":     0,
        "twitter.com":         1,
        "x.com":               1,
        ".edu":                2,
        ".ac.uk":              2,
        "github.com":          3,
        "researchgate.net":    4,
    }
    DEFAULT_PRIORITY = 5

    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/124.0.0.0 Safari/537.36")

    def __init__(self, max_pages: int = 8, timeout: int = 15):
        self.max_pages   = max_pages
        self.timeout     = timeout
        self._robots     = RobotCache()
        self._buckets: dict[str, TokenBucket] = {}
        self._seen: set[str] = set()

    def _priority(self, url: str) -> int:
        for pattern, pri in self.PRIORITY_MAP.items():
            if pattern in url:
                return pri
        return self.DEFAULT_PRIORITY

    def _domain_bucket(self, url: str) -> TokenBucket:
        domain = urlparse(url).netloc
        if domain not in self._buckets:
            # Conservative: 1 request / 2 seconds per domain
            self._buckets[domain] = TokenBucket(rate=1.0, period=2.0)
        return self._buckets[domain]

    async def _fetch_static(self, session: aiohttp.ClientSession,
                            url: str) -> str:
        """Fetch a static page with aiohttp (fast, no JS)."""
        headers = {"User-Agent": self.UA, "Accept-Language": "en-US,en;q=0.9"}
        async with session.get(url, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=self.timeout),
                               allow_redirects=True,
                               ssl=False) as resp:
            if resp.status != 200:
                return ""
            ctype = resp.headers.get("Content-Type", "")
            if "text" not in ctype and "html" not in ctype:
                return ""
            raw = await resp.text(errors="replace")
            # Strip tags cheaply
            clean = re.sub(r"<[^>]+>", " ", raw)
            clean = re.sub(r"\s+", " ", clean).strip()
            return clean[:20000]

    async def _fetch_js(self, url: str) -> str:
        """Fetch a JS-rendered page via crawl4ai / Playwright."""
        if not CRAWL4AI_AVAILABLE:
            return ""
        try:
            async with AsyncWebCrawler(headless=True,
                                       verbose=False) as crawler:
                config = CrawlerRunConfig(
                    wait_until="domcontentloaded",
                    page_timeout=self.timeout * 1000,
                    user_agent=self.UA,
                )
                result = await crawler.arun(url=url, config=config)
                text = result.markdown or result.extracted_content or ""
                return text[:20000]
        except Exception as e:
            logger.debug(f"crawl4ai fetch failed for {url}: {e}")
            return ""

    def _needs_js(self, url: str) -> bool:
        js_domains = ["linkedin.com", "twitter.com", "x.com"]
        return any(d in url for d in js_domains)

    async def crawl(self, seed_urls: list[str], progress=None, task_id=None) -> dict[str, str]:
        """
        BFS from seed_urls. Returns {url: extracted_text}.
        """
        # Build priority queue
        frontier: asyncio.PriorityQueue = asyncio.PriorityQueue()
        for url in seed_urls:
            if url and url not in self._seen:
                frontier.put_nowait((self._priority(url), url))

        results: dict[str, str] = {}
        pages_crawled = 0

        connector = aiohttp.TCPConnector(limit=8, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            while not frontier.empty() and pages_crawled < self.max_pages:
                _, url = await frontier.get()

                if url in self._seen:
                    continue
                self._seen.add(url)

                # Robots check
                if not self._robots.can_fetch(url):
                    logger.debug(f"robots.txt disallows: {url}")
                    continue

                # Rate limit
                bucket = self._domain_bucket(url)
                await bucket.acquire()

                logger.info(f"Crawling: {url}")
                try:
                    if self._needs_js(url):
                        text = await self._fetch_js(url)
                    else:
                        text = await self._fetch_static(session, url)
                except Exception as e:
                    logger.debug(f"Fetch error for {url}: {e}")
                    text = ""

                if text:
                    results[url] = text
                    pages_crawled += 1
                    if progress and task_id is not None:
                        progress.advance(task_id)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Social Media Profiler (local)
# ──────────────────────────────────────────────────────────────────────────────

class AsyncSocialProfiler:
    """
    Fully local social media intelligence.

    Steps:
      1. LocalSearchClient → discovers LinkedIn / Twitter / homepage URLs
      2. AsyncBFSCrawler   → fetches + extracts text from those pages
      3. LocalLLMClient    → Ollama/Qwen3 summarises into SocialProfile JSON
    """

    def __init__(self, search_client: LocalSearchClient,
                 llm_client,   # LocalLLMClient
                 max_pages: int = 6):
        self._search  = search_client
        self._llm     = llm_client
        self._crawler = AsyncBFSCrawler(max_pages=max_pages)

    async def profile_researcher(self, name: str,
                                 institution: str = "",
                                 interests: list[str] = None,
                                 progress=None,
                                 task_id=None) -> dict:
        """
        Build a social profile for a researcher. Fully local, zero API keys.
        Returns a SocialProfile-compatible dict.
        """
        logger.info(f"Starting local social profiling for: {name}")

        # Step 1: Discover URLs
        urls = self._search.find_researcher_urls(name, institution, interests=interests)
        logger.info(f"Discovered URLs: {urls}")

        # Step 2: Crawl those URLs
        crawled_data = await self._crawler.crawl(urls, progress=progress, task_id=task_id)

        seed_urls = [u for u in urls.values() if u]
        if not seed_urls:
            logger.warning(f"No URLs found for '{name}'. "
                           "Social profile will be empty.")
            return self._empty_profile(name)

        # Step 2: Crawl pages
        crawled = await self._crawler.crawl(seed_urls)

        if not crawled:
            logger.warning(f"All crawl attempts failed for '{name}'.")
            return self._empty_profile(name)

        # Step 3: Summarise with Ollama
        raw_content = {
            "researcher_name": name,
            "institution": institution,
            "discovered_urls": urls,
            "scraped_pages": {
                url: text[:8000] for url, text in crawled.items()
            },
        }

        try:
            profile = await self._llm.summarize_social(name, raw_content)
            # Inject discovered URLs if LLM missed them
            if urls.get("linkedin") and not profile.get("linkedin_url"):
                profile["linkedin_url"] = urls["linkedin"]
            if urls.get("twitter") and not profile.get("twitter_url"):
                profile["twitter_url"] = urls["twitter"]
            if urls.get("homepage") and not profile.get("personal_homepage"):
                profile["personal_homepage"] = urls["homepage"]
            return profile

        except Exception as e:
            logger.error(f"LLM social summarisation failed: {e}")
            # Return a basic profile with raw URL data
            return {
                "summary": f"Online presence data collected for {name}. "
                           f"LLM summarisation unavailable.",
                "linkedin_url":     urls.get("linkedin"),
                "twitter_url":      urls.get("twitter"),
                "personal_homepage": urls.get("homepage"),
                "linkedin_summary": "",
                "twitter_summary":  "",
                "key_topics":       [],
                "tone":             "unknown",
                "recent_news_and_events": "",
            }

    @staticmethod
    def _empty_profile(name: str) -> dict:
        return {
            "summary":           f"No public online presence found for {name}.",
            "linkedin_url":      None,
            "twitter_url":       None,
            "personal_homepage": None,
            "linkedin_summary":  "",
            "twitter_summary":   "",
            "key_topics":        [],
            "tone":              "unknown",
            "recent_news_and_events": "",
        }
