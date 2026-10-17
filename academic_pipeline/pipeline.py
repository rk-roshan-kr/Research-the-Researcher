import os
import asyncio
import logging
import re
import time
import concurrent.futures
from academic_pipeline.openalex import OpenAlexClient
from academic_pipeline.semantic_scholar import SemanticScholarClient
from academic_pipeline.google_scholar import GoogleScholarScraper
from academic_pipeline.resolver import ProfileResolver
from academic_pipeline.pdf_downloader import PDFDownloader
from academic_pipeline.pdf_analyzer import PDFAnalyzer
from academic_pipeline.social_media import SocialMediaProfiler
from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.collaboration_graph import CollaborationGraphBuilder
from academic_pipeline.dedup_engine import DedupEngine
from academic_pipeline.local_search import LocalSearchClient

logger = logging.getLogger(__name__)

# Cache invalidation version — increment when extraction logic changes to force re-analysis
ANALYSIS_VERSION = 4


class AcademicPipeline:
    """
    Orchestrates the entire researcher profiling pipeline:
    1. Resolves identities across systems (OpenAlex, Semantic Scholar, Google Scholar).
    2. Fetches detailed metrics and publication history.
    3. Multi-tier deduplication: Bloom filter → MinHash LSH → FAISS semantic.
    4. Profiles social media via local crawl4ai + SearxNG + Ollama LLM.
    5. Downloads and analyzes PDFs via local Ollama (Qwen3:14B on GPU).
    6. Builds deep collaboration graph: BFS + heapq + Union-Find.
    7. Formats and caches reports (SQLite + LRU cache).
    
    Zero external API keys required after initial setup.
    """
    def __init__(self, gemini_api_key=None, openalex_api_key=None, openalex_email=None,
                 s2_api_key=None, cache_dir="outputs", download_dir="outputs/pdfs",
                 ollama_model="qwen3:14b", ollama_host="http://localhost:11434",
                 searxng_url="http://localhost:8080",
                 max_workers_abstract=4, max_workers_fulltext=1):

        self.max_workers_abstract = max_workers_abstract
        self.max_workers_fulltext = max_workers_fulltext

        self.openalex = OpenAlexClient(api_key=openalex_api_key, email=openalex_email)
        self.s2       = SemanticScholarClient(api_key=s2_api_key)
        self.scholar  = GoogleScholarScraper()

        self.cache      = CacheManager(cache_dir=cache_dir)
        self.search     = LocalSearchClient(searxng_url=searxng_url, cache_manager=self.cache)
        self.resolver   = ProfileResolver(self.openalex, self.s2, self.scholar, self.search)
        self.downloader = PDFDownloader(download_dir=download_dir)
        self.analyzer   = PDFAnalyzer(api_key=gemini_api_key, model_name=ollama_model, ollama_host=ollama_host, cache_manager=self.cache)
        self.profiler   = SocialMediaProfiler(
            api_key=gemini_api_key,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            searxng_url=searxng_url,
            cache_manager=self.cache,
        )
        self.dedup      = DedupEngine()
        self.graph_builder = CollaborationGraphBuilder(
            openalex_client=self.openalex,
            s2_client=self.s2,
            cache_manager=self.cache,
        )

    def normalize_title(self, title):
        """Normalizes a title for deduplication."""
        if not title:
            return ""
        return re.sub(r'[^a-z0-9]', '', title.lower())

    def extract_doi(self, paper):
        """Extracts and normalizes the DOI from a paper object."""
        # OpenAlex uses top-level 'doi' field (e.g. 'https://doi.org/10.1145/3318464.3389700')
        doi = paper.get("doi")
        if doi:
            return doi.split("doi.org/")[-1].strip().lower()
            
        # Semantic Scholar uses nested 'externalIds'
        ext_ids = paper.get("externalIds") or paper.get("external_ids") or {}
        for key, val in ext_ids.items():
            if key.lower() == "doi" and val:
                return val.strip().lower()
        return None

    def extract_arxiv(self, paper):
        """Extracts and normalizes ArXiv ID from a paper object."""
        ext_ids = paper.get("externalIds") or paper.get("external_ids") or {}
        for key, val in ext_ids.items():
            if key.lower() == "arxiv" and val:
                if "/" in val:
                    val = val.split("/")[-1]
                return val.strip().lower()
        return None

    def _normalise_oa(self, oa_works: list) -> list:
        """Convert raw OpenAlex work dicts into the canonical paper schema."""
        out = []
        for work in oa_works:
            doi   = self.extract_doi(work)
            arxiv = self.extract_arxiv(work)
            title = work.get("title", "")
            if not title:
                continue

            authors_detailed = []
            authors = []
            for a in work.get("authorships", []):
                author_info = a.get("author") or {}
                name = author_info.get("display_name", "")
                if name:
                    authors.append(name)
                    insts = a.get("institutions") or []
                    affil = insts[0].get("display_name", "") if insts else ""
                    orcid = author_info.get("orcid") or ""
                    orcid_id = orcid.split("/")[-1] if orcid else None
                    authors_detailed.append({
                        "name": name,
                        "id": author_info.get("id", "").split("/")[-1] if author_info.get("id") else None,
                        "orcid": orcid_id,
                        "affiliation": affil
                    })

            venue = ""
            for loc in work.get("locations", []):
                src = loc.get("source") or {}
                venue = src.get("display_name") or src.get("name") or ""
                if venue:
                    break
            if not venue:
                prim_loc = work.get("primary_location") or {}
                prim_src = prim_loc.get("source") or {}
                venue = prim_src.get("display_name") or ""

            oa_url = (work.get("open_access") or {}).get("oa_url")
            is_oa  = (work.get("open_access") or {}).get("is_oa", False)

            # Reconstruct abstract from inverted index if present
            abstract = None
            inv_index = work.get("abstract_inverted_index")
            if inv_index:
                try:
                    max_pos = max(pos for positions in inv_index.values() for pos in positions)
                    words = [""] * (max_pos + 1)
                    for word, positions in inv_index.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = " ".join(words).strip() or None
                except Exception:
                    pass

            out.append({
                "title":              title,
                "year":               work.get("publication_year"),
                "venue":              venue,
                "authors":            authors,
                "authors_detailed":   authors_detailed,
                "citations_openalex": work.get("cited_by_count", 0),
                "citations_semantic": 0,
                "citations_max":      work.get("cited_by_count", 0),
                "doi":                doi,
                "arxiv":              arxiv,
                "oa_url":             oa_url,
                "is_oa":              is_oa,
                "abstract":           abstract,
                "tldr":               None,
                "pdf_path":           None,
                "analysis":           None,
                "external_ids":       {"doi": doi, "arxiv": arxiv},
            })
        return out

    def _normalise_s2(self, s2_papers: list) -> list:
        """Convert raw Semantic Scholar paper dicts into the canonical paper schema."""
        out = []
        for paper in s2_papers:
            doi   = self.extract_doi(paper)
            arxiv = self.extract_arxiv(paper)
            title = paper.get("title", "")
            if not title:
                continue

            cit_s2 = paper.get("citationCount", 0) or 0
            inf_cit = paper.get("influentialCitationCount", 0) or 0
            abstract = paper.get("abstract")
            tldr_obj = paper.get("tldr")
            tldr = tldr_obj.get("text") if tldr_obj else None
            authors_detailed = []
            authors = []
            for a in (paper.get("authors") or []):
                name = a.get("name")
                if name:
                    authors.append(name)
                    authors_detailed.append({
                        "name": name,
                        "id": a.get("authorId"),
                        "orcid": None,
                        "affiliation": ""
                    })
            
            oa_pdf = paper.get("openAccessPdf")
            oa_url = oa_pdf.get("url") if oa_pdf else None
            is_oa = bool(oa_url)

            out.append({
                "title":              title,
                "year":               paper.get("year"),
                "venue":              paper.get("venue") or "",
                "authors":            authors,
                "authors_detailed":   authors_detailed,
                "citations_openalex": 0,
                "citations_semantic": cit_s2,
                "citations_max":      cit_s2,
                "influential_citations": inf_cit,
                "doi":                doi,
                "arxiv":              arxiv,
                "oa_url":             oa_url,
                "is_oa":              is_oa,
                "abstract":           abstract,
                "tldr":               tldr,
                "pdf_path":           None,
                "analysis":           None,
                "external_ids":       {"doi": doi, "arxiv": arxiv},
            })
        return out

    def _enrich_missing_abstracts(self, publications, progress=None):
        """
        For any paper missing an abstract, attempt to fetch it from Semantic Scholar or OpenAlex.
        Caches the results.
        """
        import requests
        logger.info("Enriching papers with missing abstracts...")
        missing_papers = [p for p in publications if not p.get("abstract")]
        if not missing_papers:
            return

        # Add a progress task if progress bar is active
        task_id = None
        if progress:
            task_id = progress.add_task("[cyan]Fetching missing abstracts...", total=len(missing_papers))

        # Check API keys for rate limits
        has_s2_key = bool(self.s2.api_key)
        delay = 0.1 if has_s2_key else 1.0
        
        s2_rate_limited = False

        for idx, paper in enumerate(missing_papers):
            title = paper.get("title")
            doi = paper.get("doi")
            arxiv = paper.get("arxiv")
            
            # 1. Check cache first
            paper_key = f"abstract_{doi or arxiv or title.replace(' ', '_')}"
            cached_abstract = self.cache.get_paper_abstract(paper_key)
            if cached_abstract is not None:  # Note: can be empty string for cached negative result
                if cached_abstract:
                    paper["abstract"] = cached_abstract
                if progress and task_id is not None:
                    progress.advance(task_id)
                continue

            # 2. Try fetching
            abstract = None
            api_failed = False
            
            # 2a. Try OpenAlex by DOI first (very generous rate limits, no API key required for 10req/s)
            if doi:
                oa_url = f"https://api.openalex.org/works/doi:{doi}"
                params = {}
                if self.openalex.api_key:
                    params["api_key"] = self.openalex.api_key
                elif self.openalex.email:
                    params["mailto"] = self.openalex.email
                try:
                    r = requests.get(oa_url, params=params, timeout=10)
                    if r.status_code == 200:
                        inv_index = r.json().get("abstract_inverted_index")
                        if inv_index:
                            max_pos = max(pos for positions in inv_index.values() for pos in positions)
                            words = [""] * (max_pos + 1)
                            for word, positions in inv_index.items():
                                for pos in positions:
                                    words[pos] = word
                            abstract = " ".join(words).strip() or None
                    elif r.status_code == 429:
                        logger.warning("OpenAlex API rate limited (429) during abstract enrichment. Sleeping...")
                        time.sleep(5)
                        api_failed = True
                except Exception as e:
                    logger.debug(f"Failed fetching abstract from OA by DOI for '{title}': {e}")

            # 2b. Try Semantic Scholar by DOI (fallback if OpenAlex failed or had no abstract)
            if not abstract and doi and not api_failed and not s2_rate_limited:
                s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
                headers = {"User-Agent": "AcademicProfilePipeline/0.1"}
                if self.s2.api_key:
                    headers["x-api-key"] = self.s2.api_key
                
                for attempt in range(3):
                    try:
                        r = requests.get(s2_url, params={"fields": "abstract"}, headers=headers, timeout=10)
                        if r.status_code == 200:
                            abstract = r.json().get("abstract")
                            break
                        elif r.status_code == 429:
                            logger.warning("Semantic Scholar API rate limited (429) during abstract enrichment. Skipping Semantic Scholar for this run.")
                            s2_rate_limited = True
                            api_failed = True
                            break
                        else:
                            # 404 or other non-rate-limit error means we don't retry, it just doesn't exist
                            break
                    except Exception as e:
                        logger.debug(f"Failed fetching abstract from S2 by DOI for '{title}' (attempt {attempt+1}): {e}")
                        if attempt == 2:
                            api_failed = True
                        time.sleep(2)

            # 2c. Try Semantic Scholar Search by Title (fallback)
            if not abstract and not api_failed and not s2_rate_limited:
                s2_search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                headers = {"User-Agent": "AcademicProfilePipeline/0.1"}
                if self.s2.api_key:
                    headers["x-api-key"] = self.s2.api_key
                
                for attempt in range(3):
                    try:
                        r = requests.get(s2_search_url, params={"query": title, "limit": 1, "fields": "abstract"}, headers=headers, timeout=10)
                        if r.status_code == 200:
                            data = r.json().get("data", [])
                            if data:
                                abstract = data[0].get("abstract")
                            break
                        elif r.status_code == 429:
                            logger.warning("Semantic Scholar API rate limited (429) during abstract search. Skipping Semantic Scholar for this run.")
                            s2_rate_limited = True
                            api_failed = True
                            break
                        else:
                            break
                    except Exception as e:
                        logger.debug(f"Failed searching abstract from S2 by Title for '{title}' (attempt {attempt+1}): {e}")
                        if attempt == 2:
                            api_failed = True
                        time.sleep(2)

            if abstract:
                paper["abstract"] = abstract
                # Save to cache
                self.cache.save_paper_abstract(paper_key, abstract)
            elif not api_failed:
                # Cache negative result as empty string to avoid repeated requests only if query did not fail
                self.cache.save_paper_abstract(paper_key, "")

            if progress and task_id is not None:
                progress.advance(task_id)

            # Polite delay
            time.sleep(delay)

        if progress and task_id is not None:
            progress.update(task_id, visible=False)

    def merge_publications(self, oa_works, s2_papers):
        """
        Legacy entry point kept for test compatibility.
        Delegates to the new DedupEngine pipeline.
        """
        raw = self._normalise_oa(oa_works) + self._normalise_s2(s2_papers)
        return self.dedup.deduplicate(raw)

    def filter_topical_coherence(self, publications, target_name):
        """
        Filters out misindexed publication outliers by building a keyword profile of
        the researcher's core topics (from top cited papers) and checking
        if other papers have low vocabulary overlap and share no common co-authors.
        """
        if len(publications) <= 5:
            logger.info("Fewer than 5 publications; skipping topical coherence filtering.")
            return publications

        # Stop words list for cleaning
        stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could",
            "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", "further",
            "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres",
            "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is",
            "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off",
            "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant",
            "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their",
            "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were",
            "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom",
            "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours",
            "yourself", "yourselves"
        }

        def tokenize(text):
            if not text:
                return set()
            words = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
            return {w for w in words if w not in stop_words and len(w) > 2}

        # 1. Identify core publications (top 5 most cited)
        sorted_by_cites = sorted(publications, key=lambda p: p.get("citations_max", 0), reverse=True)
        core_papers = sorted_by_cites[:5]
        
        # Build core keyword profile
        core_vocab = set()
        core_coauthors = set()
        target_name_clean = target_name.lower().replace(" ", "").replace(".", "")

        for p in core_papers:
            # Vocabulary
            core_vocab.update(tokenize(p.get("title", "")))
            core_vocab.update(tokenize(p.get("abstract", "")))
            
            # Co-authors
            for author in p.get("authors", []):
                author_clean = author.strip()
                author_norm = author_clean.lower().replace(" ", "").replace(".", "")
                if author_norm != target_name_clean:
                    core_coauthors.add(author_norm)

        logger.info(f"Built core topical profile for '{target_name}' with {len(core_vocab)} terms and {len(core_coauthors)} co-authors.")

        if not core_vocab:
            return publications

        filtered_publications = []
        outliers_count = 0

        for p in publications:
            # Always keep core publications
            if p in core_papers:
                filtered_publications.append(p)
                continue

            # Check co-author overlap
            paper_coauthors = set()
            for author in p.get("authors", []):
                author_norm = author.strip().lower().replace(" ", "").replace(".", "")
                if author_norm != target_name_clean:
                    paper_coauthors.add(author_norm)

            shares_coauthors = bool(paper_coauthors.intersection(core_coauthors))

            # Compute vocab overlap
            paper_vocab = tokenize(p.get("title", "")) | tokenize(p.get("abstract", ""))
            if not paper_vocab:
                filtered_publications.append(p)
                continue

            overlap = len(paper_vocab.intersection(core_vocab)) / len(paper_vocab)

            # Filtering threshold: vocab overlap < 8% and shares zero co-authors
            if overlap < 0.08 and not shares_coauthors:
                logger.info(f"Topical filter: removed outlier paper '{p.get('title')}' (vocab overlap={overlap:.2f}, co-authors={len(paper_coauthors)})")
                outliers_count += 1
            else:
                filtered_publications.append(p)

        logger.info(f"Topical filter complete: kept {len(filtered_publications)} papers, filtered {outliers_count} outliers.")
        return filtered_publications

    def extract_coauthors(self, publications, target_name):
        """
        Compiles top collaborators based on publication history.
        """
        coauthor_counts = {}
        target_name_normalized = target_name.lower().replace(" ", "").replace(".", "")

        for paper in publications:
            for author in paper.get("authors", []):
                author_clean = author.strip()
                author_norm = author_clean.lower().replace(" ", "").replace(".", "")
                
                # Don't count target researcher
                if author_norm == target_name_normalized:
                    continue
                    
                if author_clean not in coauthor_counts:
                    coauthor_counts[author_clean] = 0
                coauthor_counts[author_clean] += 1
                
        sorted_coauthors = [{"name": name, "papers_count": count} for name, count in coauthor_counts.items()]
        sorted_coauthors.sort(key=lambda x: x["papers_count"], reverse=True)
        return sorted_coauthors[:20] # Top 20 co-authors

    def run_pipeline(self, query, limit_papers=10, deep_run=False,
                     graph_depth=1, max_coauthors=30,
                     interactive_callback=None,
                     progress=None,
                     lite_scan: bool = False):
        """
        Execute the complete analysis pipeline.

        Args:
            query (str): The search identifier (name, orcid, url).
            limit_papers (int): Number of papers to analyze (download & run LLM).
            deep_run (bool): If True, processes all publications (warning: takes a long time).
            graph_depth (int): 1 = direct co-authors only; 2 = co-authors of co-authors.
            max_coauthors (int): Cap on enriched co-author nodes (API cost control).
            interactive_callback (function): Disambiguation handler.
            progress: rich.progress.Progress instance for UI rendering.
            lite_scan (bool): If True, skip all PDF downloads and operate on abstracts only.
                              Significantly faster — ideal for initial screening of many professors.

        Returns:
            dict: Comprehensive profile data including collaboration_graph.
        """
        # 1. Resolve profiles
        logger.info(f"Resolving researcher profile for query: {query}")
        resolved = self.resolver.resolve_researcher(query, interactive_callback)
        
        if not resolved["name"]:
            logger.error("Could not resolve researcher name. Aborting pipeline.")
            raise ValueError("Could not resolve researcher name from input.")

        # Restart progress bar if it was stopped for interactive selection
        if progress and interactive_callback:
            progress.start()

        # Cache check for resolved profile details to avoid redundant API queries
        name = resolved["name"]
        logger.info(f"Resolved name: {name}")
        
        # Folderize outputs: create researcher-specific subdirectories
        clean_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).replace(" ", "_").lower()
        researcher_dir = os.path.join(self.cache.cache_dir, clean_name)
        self.downloader.download_dir = os.path.join(researcher_dir, "pdfs")
        os.makedirs(self.downloader.download_dir, exist_ok=True)
        
        # 2. Fetch publication lists and detailed profile metadata
        oa_works = []
        s2_papers = []
        
        # Fetch from OpenAlex if ID is resolved
        oa_profile = {}
        if resolved["openalex_id"]:
            try:
                logger.info(f"Fetching OpenAlex profile for A-ID: {resolved['openalex_id']}")
                oa_profile = self.openalex.get_author_by_id(resolved["openalex_id"])
                # Get works
                logger.info("Fetching works from OpenAlex...")
                oa_works = self.openalex.get_author_works(resolved["openalex_id"], max_works=200)
            except Exception as e:
                logger.error(f"Failed to fetch OpenAlex works: {e}")

        # Fetch from Semantic Scholar if ID is resolved
        s2_profile = {}
        if resolved["semantic_scholar_id"]:
            try:
                logger.info(f"Fetching Semantic Scholar profile for S-ID: {resolved['semantic_scholar_id']}")
                s2_profile = self.s2.get_author_by_id(resolved["semantic_scholar_id"])
                # Get papers
                logger.info("Fetching papers from Semantic Scholar...")
                s2_papers = self.s2.get_author_papers(resolved["semantic_scholar_id"], max_papers=200)
            except Exception as e:
                logger.error(f"Failed to fetch Semantic Scholar papers: {e}")

        # 3. Merge + deduplicate publications (Bloom → MinHash LSH → FAISS)
        logger.info("Merging and de-duplicating publication lists...")
        raw_combined = self._normalise_oa(oa_works) + self._normalise_s2(s2_papers)
        publications = self.dedup.deduplicate(raw_combined)
        self._enrich_missing_abstracts(publications, progress=progress)
        publications = self.filter_topical_coherence(publications, name)
        
        # 4. Extract metrics
        citation_years = oa_profile.get("counts_by_year", [])
        citation_years.sort(key=lambda x: x.get("year", 0))

        # Core metrics
        metrics = {
            "h_index_openalex": (oa_profile.get("summary_stats") or {}).get("h_index", 0) if oa_profile else 0,
            "h_index_semantic": s2_profile.get("hIndex", 0) if s2_profile else 0,
            "h_index_google": resolved["google_scholar_data"]["metrics"]["h_index"]["all"] if resolved["google_scholar_data"] else 0,
            "citations_openalex": oa_profile.get("cited_by_count", 0) if oa_profile else 0,
            "citations_semantic": s2_profile.get("citationCount", 0) if s2_profile else 0,
            "citations_google": resolved["google_scholar_data"]["metrics"]["citations"]["all"] if resolved["google_scholar_data"] else 0,
            "publications_count_openalex": oa_profile.get("works_count", 0) if oa_profile else 0,
            "publications_count_semantic": s2_profile.get("paperCount", 0) if s2_profile else 0,
            "publications_count_google": len(resolved["google_scholar_data"]["papers"]) if resolved["google_scholar_data"] else 0,
            "i10_index_openalex": (oa_profile.get("summary_stats") or {}).get("i10_index", 0) if oa_profile else 0,
            "i10_index_google": resolved["google_scholar_data"]["metrics"]["i10_index"]["all"] if resolved["google_scholar_data"] else 0,
            "citation_years": citation_years
        }
        
        # Derive primary citation/h-index values for dashboard
        metrics["h_index"] = max(metrics["h_index_openalex"], metrics["h_index_semantic"], metrics["h_index_google"])
        metrics["citations"] = max(metrics["citations_openalex"], metrics["citations_semantic"], metrics["citations_google"])
        metrics["publications"] = max(metrics["publications_count_openalex"], metrics["publications_count_semantic"], metrics["publications_count_google"], len(publications))

        # Extract top collaborators
        coauthors = self.extract_coauthors(publications, name)

        # 5. Profile Social Media (LinkedIn / Twitter) using local search/LLM
        logger.info("Resolving social media profile...")
        social_cache_key = f"social_{resolved['openalex_id'] or resolved['semantic_scholar_id'] or name.replace(' ', '_')}"
        social_profile = self.cache.get_social_analysis(social_cache_key)
        
        if not social_profile:
            soc_task = progress.add_task("[cyan]Crawling Social Profiles...", total=6) if progress else None
            social_profile = self.profiler.profile_researcher(
                name=name, 
                institution=resolved.get("affiliation", ""),
                interests=resolved.get("interests", []),
                progress=progress,
                task_id=soc_task
            )
            self.cache.save_social_analysis(social_cache_key, social_profile)
            if progress and soc_task is not None:
                progress.update(soc_task, completed=6, description="[cyan]Social Profiles Complete")
        else:
            logger.info("Social media profile loaded from cache.")

        # 6. Deep Paper Analysis (Local GPU)
        if deep_run:
            papers_to_analyze = publications
        else:
            papers_to_analyze = publications[:limit_papers]

        logger.info(f"Starting deep analysis of top {len(papers_to_analyze)} papers...")

        # 6a. Concurrently download PDFs (skip entirely in lite_scan mode)
        dl_task = None
        pdf_results = {}

        if lite_scan:
            logger.info("Lite scan mode: skipping PDF downloads, using abstracts only.")
            try:
                from academic_pipeline.local_llm import log_activity
                log_activity("[Lite Scan] PDF download skipped — abstract-only mode")
            except ImportError:
                pass
        else:
            dl_task = progress.add_task("[green]Downloading PDFs...", total=len(papers_to_analyze)) if progress else None

            async def _do_batch_downloads():
                return await self.downloader.batch_download(papers_to_analyze, progress=progress, task_id=dl_task)

            # Run async downloads
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        pdf_results = ex.submit(asyncio.run, _do_batch_downloads()).result()
                else:
                    pdf_results = asyncio.run(_do_batch_downloads())
            except Exception as e:
                logger.error(f"Batch download failed: {e}")
                pdf_results = {}

            if progress and dl_task is not None:
                progress.update(dl_task, completed=len(papers_to_analyze), description="[green]Downloads Complete")

        # 6b. Extract text and analyze (Parallelized natively using asyncio and semaphores)
        analysis_task = progress.add_task("[yellow]Analyzing Papers (Local LLM)...", total=len(papers_to_analyze)) if progress else None

        async def _analyze_single_async(paper):
            try:
                title = paper["title"]
                pdf_path = pdf_results.get(title)
                
                # Check cache first
                doi = paper.get("doi")
                arxiv = paper.get("arxiv")
                paper_key = f"paper_{doi or arxiv or title.replace(' ', '_')}"
                
                loop = asyncio.get_running_loop()
                cached_analysis = await loop.run_in_executor(None, self.cache.get_paper_analysis, paper_key)

                if cached_analysis:
                    # Version gate: if cached analysis is from an older version, invalidate it
                    if cached_analysis.get("analysis_version", 0) == ANALYSIS_VERSION:
                        paper["analysis"] = cached_analysis
                        if pdf_path:
                            paper["pdf_path"] = pdf_path
                        try:
                            from academic_pipeline.local_llm import log_activity
                            log_activity(f"Paper Cache Hit (v{ANALYSIS_VERSION}): '{title}'")
                        except ImportError:
                            pass
                        if progress and analysis_task is not None:
                            progress.advance(analysis_task)
                        return
                    else:
                        cached_ver = cached_analysis.get('analysis_version', 0)
                        logger.info(f"Cache version mismatch for '{title}': cached={cached_ver}, current={ANALYSIS_VERSION}. Re-analyzing.")
                        try:
                            from academic_pipeline.local_llm import log_activity
                            log_activity(f"Cache invalidated (v{cached_ver}→v{ANALYSIS_VERSION}): '{title}'")
                        except ImportError:
                            pass

                text_to_analyze = paper.get("abstract") or ""
                
                if pdf_path:
                    paper["pdf_path"] = pdf_path
                    text = await loop.run_in_executor(None, self.analyzer.extract_text_from_pdf, pdf_path)
                    if text:
                        text_to_analyze = text
                    else:
                        logger.warning(f"Could not extract text from {pdf_path}, falling back to abstract")
                else:
                    logger.warning(f"Could not download PDF for '{title}', falling back to abstract")
                    
                # Prefetch patent search results in Python to speed up LLM analysis
                patent_query = f'"{name}" "{title}" patent'
                try:
                    from academic_pipeline.local_llm import log_activity
                    results = await loop.run_in_executor(None, self.cache.get_search_results, patent_query)
                    if results is None:
                        # Perform search
                        results = await loop.run_in_executor(None, self.profiler._search.search, patent_query, 3)
                        self.cache.save_search_results(patent_query, results)
                    
                    res_text = "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}" for r in results]) if results else "No relevant patent filings found."
                except Exception as e:
                    logger.error(f"Failed to fetch patent search results: {e}")
                    res_text = "No relevant patent filings found."

                # Slice paper text for speed if it exceeds 30,000 characters
                text_len = len(text_to_analyze)
                if text_len > 30000:
                    text_sliced = text_to_analyze[:25000] + "\n\n[... MIDDLE OF PAPER TRUNCATED FOR SPEED ...]\n\n" + text_to_analyze[-5000:]
                    try:
                        log_activity(f"Sliced paper text context from {text_len} to {len(text_sliced)} chars for speed.")
                    except Exception:
                        pass
                else:
                    text_sliced = text_to_analyze

                if text_sliced:
                    # Native async LLM analysis call
                    analysis = await self.analyzer.analyze_paper_text_async(title, text_sliced, patent_results=res_text)
                    # Inject the current analysis version before caching
                    if isinstance(analysis, dict):
                        analysis["analysis_version"] = ANALYSIS_VERSION
                    paper["analysis"] = analysis
                    await loop.run_in_executor(None, self.cache.save_paper_analysis, paper_key, analysis)
                    
                if progress and analysis_task is not None:
                    progress.advance(analysis_task)
            except Exception as e:
                logger.error(f"Uncaught exception in _analyze_single_async for {paper.get('title', 'Unknown')}: {e}")

        # Partition papers into full-text (heavy context) and abstract-only (light context)
        full_text_papers = []
        abstract_papers = []
        for paper in papers_to_analyze:
            title = paper.get("title")
            if pdf_results.get(title):
                full_text_papers.append(paper)
            else:
                abstract_papers.append(paper)

        # Semaphores to throttle parallel executions to prevent VRAM overflow
        sem_abstract = asyncio.Semaphore(self.max_workers_abstract)
        sem_fulltext = asyncio.Semaphore(self.max_workers_fulltext)

        async def _sem_analyze_abstract(paper):
            async with sem_abstract:
                await _analyze_single_async(paper)

        async def _sem_analyze_fulltext(paper):
            async with sem_fulltext:
                await _analyze_single_async(paper)

        async def _run_all_analyses():
            tasks = []
            for paper in abstract_papers:
                tasks.append(_sem_analyze_abstract(paper))
            for paper in full_text_papers:
                tasks.append(_sem_analyze_fulltext(paper))
            await asyncio.gather(*tasks)

        # Execute all async analyses
        if papers_to_analyze:
            logger.info(f"Analyzing {len(papers_to_analyze)} papers natively with semaphores: "
                        f"abstracts concurrent={self.max_workers_abstract}, PDFs concurrent={self.max_workers_fulltext}")
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    ex.submit(asyncio.run, _run_all_analyses()).result()
            else:
                asyncio.run(_run_all_analyses())
            
        if progress and analysis_task is not None:
            progress.update(analysis_task, description="[yellow]Paper Analysis Complete")

        
        # 6c. Holistic Research Synthesis
        logger.info("Synthesizing overarching intellectual profile...")
        synth_task = progress.add_task("[cyan]Synthesizing Holistic Profile...", total=1) if progress else None
        
        # Collect abstracts and analyses
        corpus_for_llm = []
        for p in papers_to_analyze:
            item = {
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "analysis": p.get("analysis")
            }
            corpus_for_llm.append(item)
            
        research_synthesis = self.analyzer.synthesize_profile(name, corpus_for_llm)
        if progress and synth_task is not None:
            progress.update(synth_task, completed=1, description="[cyan]Synthesis Complete")
            
        # 7. Build deep collaboration graph

        logger.info("Building collaboration / connection tree...")
        graph_task = progress.add_task("[magenta]Enriching Graph Nodes...", total=max_coauthors) if progress else None
        
        collaboration_graph = self.graph_builder.build_graph(
            target_name=name,
            target_openalex_id=resolved["openalex_id"] or "ROOT",
            target_metrics=metrics,
            publications=publications,
            depth=graph_depth,
            max_coauthors=max_coauthors,
            progress=progress,
            task_id=graph_task,
        )
        if progress and graph_task is not None:
            progress.update(graph_task, description="[magenta]Graph Built successfully!")

        # Compile final profile
        profile_dossier = {
            "name": name,
            "orcid": resolved["orcid"],
            "openalex_id": resolved["openalex_id"],
            "semantic_scholar_id": resolved["semantic_scholar_id"],
            "google_scholar_id": resolved["google_scholar_id"],
            "google_scholar_data": resolved.get("google_scholar_data"),
            "affiliation": resolved["affiliation"] or (oa_profile.get("last_known_institutions", [{}])[0].get("display_name") if oa_profile and oa_profile.get("last_known_institutions") else None),
            "interests": resolved["interests"] or (resolved["google_scholar_data"]["interests"] if resolved["google_scholar_data"] else []),
            "photo_url": resolved["photo_url"],
            "metrics": metrics,
            "coauthors": coauthors,
            "social_media": social_profile,
            "publications": publications,
            "collaboration_graph": collaboration_graph,
            "research_synthesis": research_synthesis,
        }

        return profile_dossier

    def resolve_department_professors(self, institution_query, department_query, limit_professors=50):
        """
        Resolves the list of researchers associated with an institution and department.
        Returns:
            list[dict]: A list of author objects.
        """
        logger.info(f"Resolving institution: {institution_query}")
        inst_results = self.openalex.search_institutions(institution_query)
        if not inst_results:
            raise ValueError(f"Could not find institution matching: '{institution_query}'")
        
        inst = inst_results[0]
        inst_id = inst.get("id")
        inst_name = inst.get("display_name")
        logger.info(f"Found institution: {inst_name} ({inst_id})")
        
        logger.info(f"Searching works for department: {department_query} at {inst_name}")
        works = self.openalex.get_works_by_institution_and_dept(inst_id, department_query, max_works=300)
        logger.info(f"Retrieved {len(works)} works matching institution and department query")
        
        author_paper_counts = {}
        author_names = {}
        clean_inst_id = inst_id.split("/")[-1].lower()
        dept_query_clean = department_query.lower()
        
        for work in works:
            for authorship in work.get("authorships", []):
                author_info = authorship.get("author") or {}
                author_id = author_info.get("id")
                if not author_id:
                    continue
                
                # Check institution match
                insts = authorship.get("institutions") or []
                inst_match = any(clean_inst_id in i.get("id", "").lower() for i in insts)
                
                # Check department match in raw affiliation strings
                raw_affils = authorship.get("raw_affiliation_strings") or []
                if isinstance(raw_affils, str):
                    raw_affils = [raw_affils]
                
                dept_match = any(dept_query_clean in aff.lower() for aff in raw_affils)
                
                if inst_match and dept_match:
                    author_clean_id = author_id.split("/")[-1]
                    author_paper_counts[author_clean_id] = author_paper_counts.get(author_clean_id, 0) + 1
                    author_names[author_clean_id] = author_info.get("display_name")
        
        if not author_paper_counts:
            logger.warning("No authors matched both the institution and department raw affiliation criteria.")
            return []
            
        logger.info(f"Identified {len(author_paper_counts)} candidate authors. Fetching full profiles in batch...")
        
        author_ids = list(author_paper_counts.keys())
        profiles = self.openalex.get_authors_batch(author_ids)
        
        candidates = []
        for p in profiles:
            aid = p.get("id", "").split("/")[-1]
            if not aid:
                continue
            
            summary_stats = p.get("summary_stats") or {}
            candidates.append({
                "openalex_id": aid,
                "name": p.get("display_name"),
                "citations": p.get("cited_by_count", 0),
                "h_index": summary_stats.get("h_index", 0),
                "works_count": p.get("works_count", 0),
                "dept_paper_count": author_paper_counts.get(aid, 0),
                "orcid": p.get("orcid", "").split("/")[-1] if p.get("orcid") else None,
                "affiliation": inst_name
            })
            
        # Sort candidates by h-index then total citations then department paper count
        candidates.sort(key=lambda x: (x["h_index"], x["citations"], x["dept_paper_count"]), reverse=True)
        
        selected = candidates[:limit_professors]
        logger.info(f"Selected top {len(selected)} professors for analysis (out of {len(candidates)} candidates)")
        return selected
