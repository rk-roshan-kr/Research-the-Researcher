"""
pdf_downloader.py — Async PDF download pipeline.

Design:
  - asyncio.PriorityQueue for the download frontier:
      Priority 0: arXiv URLs (most reliable, always open access)
      Priority 1: Open Access URLs from OpenAlex
      Priority 2: DOI resolver (may hit paywall)
      Priority 3: Semantic Scholar PDF link
  - asyncio.Semaphore(8) limits concurrent downloads to 8 (no socket exhaustion)
  - aiohttp with connection pooling and streaming writes (O(1) RAM per download)
  - Chunked 8KB writes to disk — never fully buffers PDF in RAM
  - Retry with exponential backoff on 429 / 5xx responses
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import unicodedata
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024  # 8 KB streaming chunks
MAX_PDF_MB = 50        # Skip suspiciously large files (> 50 MB)


# ──────────────────────────────────────────────────────────────────────────────
# URL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _arxiv_pdf_url(identifier: str) -> Optional[str]:
    """Convert an arXiv ID or abstract URL to a direct PDF URL."""
    if not identifier:
        return None
    # Already a PDF url
    if "/pdf/" in identifier:
        return identifier if identifier.endswith(".pdf") else identifier + ".pdf"
    # Abstract URL
    match = re.search(r"arxiv\.org/abs/([^\s?#]+)", identifier)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    # Raw ID like "2301.12345"
    if re.match(r"\d{4}\.\d{4,5}(v\d+)?$", identifier):
        return f"https://arxiv.org/pdf/{identifier}.pdf"
    return None


def _clean_filename(title: str, max_len: int = 80) -> str:
    """Make a filesystem-safe filename from a paper title."""
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))
    title = re.sub(r"[^\w\s-]", "", title.lower())
    title = re.sub(r"[\s]+", "_", title.strip())
    return title[:max_len] + ".pdf"


# ──────────────────────────────────────────────────────────────────────────────
# Async PDF Downloader
# ──────────────────────────────────────────────────────────────────────────────

class PDFDownloader:
    """
    Async, priority-queue-based PDF downloader.

    Usage (sync entry point for pipeline compatibility):
        downloader = PDFDownloader(download_dir="outputs/pdfs")
        path = downloader.download_paper(title, oa_url, external_ids)

    For batch async use:
        paths = await downloader.batch_download(papers)
    """

    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/124.0.0.0 Safari/537.36")

    def __init__(self, download_dir: str = "outputs/pdfs",
                 max_concurrent: int = 8,
                 timeout: int = 60):
        os.makedirs(download_dir, exist_ok=True)
        self.download_dir   = download_dir
        self.max_concurrent = max_concurrent
        self.timeout        = timeout
        self._semaphore: asyncio.Semaphore | None = None  # lazy: created per event-loop

    # ── URL resolution ────────────────────────────────────────────────────────

    def _candidate_urls(self, oa_url: Optional[str],
                        external_ids: Optional[dict]) -> list[tuple[int, str]]:
        """
        Build a priority-ordered list of (priority, url) candidates.
        Lower priority number = tried first.
        """
        ext = external_ids or {}
        candidates: list[tuple[int, str]] = []

        # Priority 0: arXiv (direct PDF, always open)
        arxiv_id = (ext.get("ArXiv") or ext.get("arxiv") or
                    ext.get("arXiv") or "")
        if arxiv_id:
            url = _arxiv_pdf_url(arxiv_id)
            if url:
                candidates.append((0, url))

        # Priority 1: OA URL from OpenAlex
        if oa_url:
            candidates.append((1, oa_url))

        # Priority 2: DOI → unpaywall redirect
        doi = ext.get("DOI") or ext.get("doi") or ""
        if doi:
            candidates.append((2, f"https://doi.org/{doi}"))
            candidates.append((2, f"https://unpaywall.org/{doi}"))

        # Priority 3: Semantic Scholar PDF link
        s2_pdf = ext.get("s2_pdf_url") or ""
        if s2_pdf:
            candidates.append((3, s2_pdf))

        return sorted(candidates, key=lambda x: x[0])

    # ── Core async downloader ─────────────────────────────────────────────────

    async def _download_one(self, session: aiohttp.ClientSession,
                             url: str, filepath: str,
                             retries: int = 3) -> bool:
        """
        Download `url` to `filepath`.  Returns True on success.
        Streams in 8KB chunks to avoid large RAM usage.
        """
        headers = {
            "User-Agent":    self.UA,
            "Accept":        "application/pdf,*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        # Lazy semaphore — must be created inside the running event loop
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        for attempt in range(1, retries + 1):
            try:
                async with self._semaphore:
                    async with session.get(
                        url, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        allow_redirects=True, ssl=False
                    ) as resp:
                        if resp.status in (429, 503):
                            wait = 2 ** attempt
                            logger.debug(f"Rate limited ({resp.status}). "
                                         f"Waiting {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            return False

                        ctype = resp.headers.get("Content-Type", "")
                        if "pdf" not in ctype and "octet-stream" not in ctype:
                            # Check if redirected to a valid PDF
                            content_start = await resp.content.read(8)
                            if not content_start.startswith(b"%PDF"):
                                return False
                            # Prepend the peeked bytes
                            with open(filepath, "wb") as f:
                                f.write(content_start)
                                async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                                    f.write(chunk)
                            return True

                        # Check size header
                        content_len = int(resp.headers.get("Content-Length", 0))
                        if content_len > MAX_PDF_MB * 1024 * 1024:
                            logger.debug(f"PDF too large ({content_len // 1024 // 1024} MB): {url}")
                            return False

                        # Stream to disk
                        with open(filepath, "wb") as f:
                            total = 0
                            async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                                f.write(chunk)
                                total += len(chunk)
                                if total > MAX_PDF_MB * 1024 * 1024:
                                    logger.debug(f"Aborting oversized download: {url}")
                                    return False
                        return True

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug(f"Download attempt {attempt} failed for {url}: {e}")
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        return False

    async def _try_candidates(self, session: aiohttp.ClientSession,
                               candidates: list[tuple[int, str]],
                               filepath: str) -> Optional[str]:
        """Try candidates in priority order, return filepath on success."""
        for _, url in candidates:
            logger.debug(f"Trying PDF URL: {url}")
            ok = await self._download_one(session, url, filepath)
            if ok:
                # Validate it's actually a PDF
                if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                    with open(filepath, "rb") as f:
                        if f.read(4) == b"%PDF":
                            return filepath
                # Invalid — clean up
                try:
                    os.remove(filepath)
                except OSError:
                    pass
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def download_paper(self, paper_title: str,
                       oa_url: Optional[str] = None,
                       external_ids: Optional[dict] = None) -> Optional[str]:
        """
        Synchronous entry point (runs async loop internally).
        Returns local filepath on success, None on failure.
        """
        filename = _clean_filename(paper_title)
        filepath = os.path.join(self.download_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
            logger.debug(f"PDF already cached: {filepath}")
            return filepath

        candidates = self._candidate_urls(oa_url, external_ids)
        if not candidates:
            logger.debug(f"No download URLs for: {paper_title}")
            return None

        async def _run():
            connector = aiohttp.TCPConnector(limit=self.max_concurrent, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                return await self._try_candidates(session, candidates, filepath)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already inside asyncio (e.g., called from pipeline async context)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, _run())
                    return future.result(timeout=self.timeout + 30)
            else:
                return asyncio.run(_run())
        except Exception as e:
            logger.warning(f"PDF download failed for '{paper_title}': {e}")
            return None

    async def batch_download(self, papers: list[dict], progress=None, task_id=None) -> dict[str, Optional[str]]:
        """
        Download many papers concurrently.
        Returns {paper_title: local_path_or_None}.
        """
        results: dict[str, Optional[str]] = {}
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, ssl=False)
        
        async def _download_and_advance(title, coro):
            try:
                res = await coro
                return title, res
            except Exception as e:
                return title, e
            finally:
                if progress and task_id is not None:
                    progress.advance(task_id)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for p in papers:
                title    = p.get("title", "unknown")
                oa_url   = p.get("oa_url")
                ext_ids  = p.get("external_ids") or {}
                # Normalise arxiv key — pipeline stores lowercase, downloader needs both
                if not ext_ids.get("ArXiv") and p.get("arxiv"):
                    ext_ids = dict(ext_ids)
                    ext_ids["ArXiv"] = p["arxiv"]
                if not ext_ids.get("doi") and p.get("doi"):
                    ext_ids = dict(ext_ids)
                    ext_ids["doi"] = p["doi"]
                filename = _clean_filename(title)
                filepath = os.path.join(self.download_dir, filename)
                candidates = self._candidate_urls(oa_url, ext_ids)

                if not candidates:
                    results[title] = None
                    if progress and task_id is not None:
                        progress.advance(task_id)
                    continue

                coro = self._try_candidates(session, candidates, filepath)
                tasks.append(_download_and_advance(title, coro))

            if tasks:
                completed = await asyncio.gather(*tasks)
                for title, result in completed:
                    if isinstance(result, Exception):
                        logger.debug(f"Batch download error for '{title}': {result}")
                        results[title] = None
                    else:
                        results[title] = result

        return results
