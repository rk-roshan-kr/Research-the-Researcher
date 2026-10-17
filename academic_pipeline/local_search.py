"""
local_search.py — Self-hosted SearxNG client with DuckDuckGo fallback.

Primary:  SearxNG Docker instance at http://localhost:8080
          (zero API keys, aggregates Google/Bing/DDG/Scholar)
Fallback: DuckDuckGo HTML scrape via httpx + BeautifulSoup.

Design:
  - LRU cache with 24h TTL prevents re-querying the same terms on re-runs.
  - Token bucket rate limiter (per domain) enforces polite crawling.
  - Results normalised into a consistent schema regardless of source.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode, urlparse

import requests

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Token Bucket Rate Limiter (per-domain)
# ──────────────────────────────────────────────────────────────────────────────

class TokenBucket:
    """
    Classic token bucket: rate-limits to `rate` requests per `period` seconds.
    Thread-safe using a simple lock.
    """

    def __init__(self, rate: float = 2.0, period: float = 1.0):
        self.rate    = rate    # tokens added per period
        self.period  = period
        self.tokens  = rate    # start full
        self.updated = time.monotonic()
        self._lock   = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now    = time.monotonic()
            delta  = (now - self.updated) / self.period
            self.tokens = min(self.rate, self.tokens + delta * self.rate)
            self.updated = now
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate * self.period
                await asyncio.sleep(wait)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# TTL Cache
# ──────────────────────────────────────────────────────────────────────────────

class TTLCache:
    """Simple dict-backed cache with per-entry TTL (24h default)."""

    def __init__(self, ttl_hours: float = 24.0, maxsize: int = 2000):
        self._store: dict[str, tuple[object, datetime]] = {}
        self._ttl   = timedelta(hours=ttl_hours)
        self._maxsize = maxsize

    def _key(self, query: str, **kwargs) -> str:
        raw = query + str(sorted(kwargs.items()))
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[list]:
        k = self._key(query, **kwargs)
        entry = self._store.get(k)
        if entry is None:
            return None
        val, ts = entry
        if datetime.utcnow() - ts > self._ttl:
            del self._store[k]
            return None
        return val

    def set(self, query: str, value: list, **kwargs):
        if len(self._store) >= self._maxsize:
            # Evict oldest entry
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        k = self._key(query, **kwargs)
        self._store[k] = (value, datetime.utcnow())


# ──────────────────────────────────────────────────────────────────────────────
# Search Result Schema
# ──────────────────────────────────────────────────────────────────────────────

def _make_result(title: str, url: str, snippet: str = "",
                 source: str = "unknown") -> dict:
    return {"title": title, "url": url, "snippet": snippet, "source": source}


# ──────────────────────────────────────────────────────────────────────────────
# SearxNG Client
# ──────────────────────────────────────────────────────────────────────────────

class SearxNGClient:
    """Queries a self-hosted SearxNG instance for JSON results."""

    def __init__(self, base_url: str = "http://localhost:8080",
                 timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._bucket  = TokenBucket(rate=5.0, period=1.0)   # 5 req/sec max
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/search",
                             params={"q": "test", "format": "json"},
                             timeout=3)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        if not self._available:
            logger.info("SearxNG not reachable — will use DuckDuckGo fallback. "
                        "To enable SearxNG: cd searxng && docker compose up -d")
        return self._available

    def search(self, query: str, num_results: int = 10,
               categories: str = "general") -> list[dict]:
        """Synchronous search (run in executor for async use)."""
        params = {
            "q":        query,
            "format":   "json",
            "categories": categories,
        }
        try:
            r = requests.get(f"{self.base_url}/search",
                             params=params,
                             timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data.get("results", [])[:num_results]:
                results.append(_make_result(
                    title   = item.get("title", ""),
                    url     = item.get("url", ""),
                    snippet = item.get("content", ""),
                    source  = "searxng",
                ))
            return results
        except Exception as e:
            logger.debug(f"SearxNG search failed for '{query}': {e}")
            return []


# ──────────────────────────────────────────────────────────────────────────────
# DuckDuckGo Fallback
# ──────────────────────────────────────────────────────────────────────────────

class DDGFallback:
    """
    Lightweight DuckDuckGo HTML scraper.
    Only activated when SearxNG is unavailable.
    Respects DDG's robots.txt (no login wall, no JS needed for basic results).
    """

    SEARCH_URL = "https://html.duckduckgo.com/html/"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._bucket = TokenBucket(rate=1.5, period=1.0)   # gentler rate for DDG
        from threading import Lock
        self._lock = Lock()
        self._last_request_time = 0.0

    def search(self, query: str, num_results: int = 10) -> list[dict]:
        """HTML scrape of DuckDuckGo results page."""
        if not BS4_AVAILABLE:
            logger.warning("beautifulsoup4 not installed. DDG fallback unavailable.")
            return []
        
        with self._lock:
            # Enforce polite spacing of requests
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)
            self._last_request_time = time.monotonic()
            
            try:
                data = {"q": query, "kl": "us-en", "kp": "-1"}
                r = requests.post(
                    self.SEARCH_URL,
                    data=data,
                    headers=self.HEADERS,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                results = []
                for result in soup.select(".result")[:num_results]:
                    title_el   = result.select_one(".result__title a")
                    snippet_el = result.select_one(".result__snippet")
                    if not title_el:
                        continue
                    raw_url = title_el.get("href", "")
                    # DDG wraps URLs in a redirect — extract actual URL
                    if "uddg=" in raw_url:
                        from urllib.parse import parse_qs, urlparse
                        parsed = urlparse(raw_url)
                        actual = parse_qs(parsed.query).get("uddg", [raw_url])[0]
                        raw_url = actual
                    results.append(_make_result(
                        title   = title_el.get_text(strip=True),
                        url     = raw_url,
                        snippet = snippet_el.get_text(strip=True) if snippet_el else "",
                        source  = "duckduckgo",
                    ))
                return results
            except Exception as e:
                logger.debug(f"DDG search failed for '{query}': {e}")
                return []


# ──────────────────────────────────────────────────────────────────────────────
# Public Search Client (auto-selects SearxNG or DDG)
# ──────────────────────────────────────────────────────────────────────────────

class LocalSearchClient:
    """
    Unified search interface: SearxNG → DuckDuckGo fallback.
    All results are cached with a 24h TTL to avoid redundant web requests.
    """

    def __init__(self, searxng_url: str = "http://localhost:8080", cache_manager = None):
        self._searxng = SearxNGClient(base_url=searxng_url)
        self._ddg     = DDGFallback()
        self._cache   = TTLCache(ttl_hours=24, maxsize=3000)
        self._cache_manager = cache_manager

    def search(self, query: str, num_results: int = 10) -> list[dict]:
        """Search with cache. Auto-routes to best available backend."""
        # Try local SQLite cache first
        if self._cache_manager:
            try:
                cached_results = self._cache_manager.get_search_results(query)
                if cached_results is not None:
                    try:
                        from academic_pipeline.local_llm import log_activity
                        log_activity(f"Search Cache Hit: '{query}'")
                    except ImportError:
                        pass
                    return cached_results
            except Exception as e:
                logger.error(f"Failed to read search cache: {e}")

        cached = self._cache.get(query, num=num_results)
        if cached is not None:
            return cached

        if self._searxng.is_available():
            results = self._searxng.search(query, num_results)
        else:
            results = self._ddg.search(query, num_results)
            time.sleep(0.7)   # polite DDG delay

        if results is not None:
            self._cache.set(query, results, num=num_results)
            if self._cache_manager:
                try:
                    self._cache_manager.save_search_results(query, results)
                except Exception as e:
                    logger.error(f"Failed to save search cache: {e}")
        else:
            # Cache empty list if query returned None to prevent spamming next time
            if self._cache_manager:
                try:
                    self._cache_manager.save_search_results(query, [])
                except Exception as e:
                    logger.error(f"Failed to save empty search cache: {e}")

        return results or []

    def find_researcher_urls(self, name: str,
                             institution: str = "",
                             interests: list[str] = None) -> dict[str, Optional[str]]:
        """
        Runs targeted queries to discover researcher's social/web profiles.
        Returns a dict: {linkedin, twitter, homepage}
        Uses multiple queries and a scoring algorithm to rank candidates.
        """
        found: dict[str, Optional[str]] = {
            "linkedin": None,
            "twitter": None,
            "homepage": None
        }

        # --- Helper for scoring candidates ---
        def score_candidate(url: str, title: str, snippet: str, profile_type: str) -> float:
            score = 0.0
            
            # Basic validation
            title_lower = title.lower()
            snippet_lower = snippet.lower()
            url_lower = url.lower()
            
            if profile_type == "linkedin":
                if "linkedin.com/in/" not in url_lower:
                    return -100.0
                # Skip search or dir pages
                if any(x in url_lower for x in ["/dir/", "/pub/dir/", "/search", "/title"]):
                    return -50.0
            elif profile_type == "twitter":
                if "twitter.com/" not in url_lower and "x.com/" not in url_lower:
                    return -100.0
                if any(x in url_lower for x in ["/status/", "/intent/", "/share", "/hashtag/", "/search"]):
                    return -50.0
            
            # 1. Name match
            name_parts = [p.lower() for p in name.split() if len(p) > 2]
            exact_name_lower = name.lower()
            
            # Exact name in title/URL
            if exact_name_lower in title_lower:
                score += 25.0
            if exact_name_lower in url_lower:
                score += 20.0
            
            # Name parts in title
            for part in name_parts:
                if part in title_lower:
                    score += 5.0
                if part in url_lower:
                    score += 5.0
                    
            # 2. Institution Match
            if institution:
                inst_words = [w.lower().strip(",.()&") for w in institution.split()]
                # filter out common words
                stop_words = {"university", "college", "institute", "of", "and", "the", "at", "for", "technology", "science", "research", "lab"}
                inst_keywords = [w for w in inst_words if w and w not in stop_words and len(w) > 2]
                
                inst_match_count = 0
                for kw in inst_keywords:
                    if kw in title_lower or kw in snippet_lower:
                        score += 10.0
                        inst_match_count += 1
                if inst_match_count > 0:
                    score += 10.0  # Bonus for matching any institution keyword
            
            # 3. Interests match
            if interests:
                interests_match_count = 0
                for interest in interests:
                    interest_lower = interest.lower()
                    if interest_lower in title_lower or interest_lower in snippet_lower:
                        score += 5.0
                        interests_match_count += 1
                if interests_match_count > 0:
                    score += 5.0  # Bonus for matching any interest
                    
            return score

        # 1. LinkedIn Search
        linkedin_queries = []
        if institution:
            linkedin_queries.append(f'"{name}" "{institution}" site:linkedin.com/in')
        linkedin_queries.append(f'"{name}" (professor OR PhD OR researcher OR university) site:linkedin.com/in')
        
        linkedin_candidates = []
        seen_urls = set()
        for q in linkedin_queries:
            results = self.search(q, num_results=5)
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    score = score_candidate(url, r.get("title", ""), r.get("snippet", ""), "linkedin")
                    if score >= 0:
                        linkedin_candidates.append((url, score))
        
        if linkedin_candidates:
            linkedin_candidates.sort(key=lambda x: x[1], reverse=True)
            found["linkedin"] = linkedin_candidates[0][0]

        # 2. Twitter/X Search
        twitter_queries = []
        if institution:
            twitter_queries.append(f'"{name}" "{institution}" (site:twitter.com OR site:x.com)')
        twitter_queries.append(f'"{name}" (site:twitter.com OR site:x.com) researcher OR professor OR PhD')
        
        twitter_candidates = []
        seen_urls = set()
        for q in twitter_queries:
            results = self.search(q, num_results=5)
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    score = score_candidate(url, r.get("title", ""), r.get("snippet", ""), "twitter")
                    if score >= 0:
                        twitter_candidates.append((url, score))
                        
        if twitter_candidates:
            twitter_candidates.sort(key=lambda x: x[1], reverse=True)
            found["twitter"] = twitter_candidates[0][0]

        # 3. Homepage Search
        homepage_query = f'"{name}" {institution} homepage OR personal site professor researcher'
        results = self.search(homepage_query, num_results=5)
        for r in results:
            url = r.get("url", "")
            if url:
                if not any(d in url.lower() for d in [
                    "linkedin.com", "twitter.com", "x.com",
                    "scholar.google", "researchgate.net",
                    "semanticscholar.org", "openalex.org",
                ]):
                    found["homepage"] = url
                    break

        return found
