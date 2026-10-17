"""
pdf_analyzer.py — Local PDF analysis using Ollama (replaces Gemini API).

Extracts text with layout-sorted PyMuPDF (fitz) or pypdf fallback, then sends
it to the local LLM (Qwen3:14B via Ollama) for structured analysis.
Concurrently extracts visual blocks (figures/charts) and describes them locally.
"""

from __future__ import annotations

import asyncio
import logging
import os
import hashlib

from pypdf import PdfReader

# Try importing PyMuPDF for high-performance layout-sorted parsing
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from academic_pipeline.local_llm import LocalLLMClient

logger = logging.getLogger(__name__)


class PDFAnalyzer:
    """
    Extracts text and describes figures/charts from downloaded PDFs,
    then analyses the rich context via a local LLM.
    """

    def __init__(self, api_key=None,         # kept for backward-compat; ignored
                 model_name: str = "qwen3:14b",
                 ollama_host: str = "http://localhost:11434",
                 cache_manager = None):
        # api_key is intentionally ignored — everything is local
        self.cache = cache_manager
        self._llm = LocalLLMClient(model=model_name, host=ollama_host, cache_manager=cache_manager)
        if not self._llm.is_available():
            logger.warning(
                f"Ollama model '{model_name}' not ready. "
                f"Run: ollama pull {model_name}"
            )

    # ── Text & Visual extraction ──────────────────────────────────────────────

    async def _get_and_cache_image_description(self, img_hash: str, image_bytes: bytes) -> str:
        """Fetch chart/figure description from cache, or describe it locally and save to cache."""
        if self.cache:
            try:
                cached = self.cache.get_image_description(img_hash)
                if cached:
                    return cached
            except Exception as e:
                logger.error(f"Failed to fetch image description from cache: {e}")
                
        # Describe locally via Ollama multimodal model (e.g. moondream)
        desc = await self._llm.describe_image_async(image_bytes)
        
        if self.cache and desc:
            try:
                self.cache.save_image_description(img_hash, desc)
            except Exception as e:
                logger.error(f"Failed to save image description to cache: {e}")
                
        return desc

    async def extract_text_from_pdf_async(self, pdf_path: str) -> str:
        """
        Extract text from a PDF, interweaving local multimodal descriptions of figures/charts.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return ""

        # Check if we already have the extracted .md file next to it
        md_path = pdf_path.replace(".pdf", ".md")
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text:
                    logger.info(f"Loaded cached layout-sorted text from: {md_path}")
                    try:
                        from academic_pipeline.local_llm import log_activity
                        log_activity(f"PDF Text Cache Hit: Loaded Markdown from: {os.path.basename(md_path)}")
                    except ImportError:
                        pass
                    return text
            except Exception as e:
                logger.error(f"Failed to read cached Markdown {md_path}: {e}")

        text = ""
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                pages_content = []
                tasks = []
                placeholders = []
                
                for page_num, page in enumerate(doc):
                    page_width = page.rect.width
                    page_height = page.rect.height
                    
                    # 1. Identify visual blocks (raster images & merged drawing paths for charts)
                    visual_rects = []
                    
                    # Raster images
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        x0, y0, x1, y1, content, block_no, block_type = b
                        if block_type == 1:  # Image
                            w = x1 - x0
                            h = y1 - y0
                            if w > 100 and h > 100:
                                visual_rects.append(fitz.Rect(x0, y0, x1, y1))
                                
                    # Vector drawings (charts/plots)
                    drawings = page.get_drawings()
                    if drawings:
                        rects = [d["rect"] for d in drawings if d.get("rect")]
                        valid_rects = [r for r in rects if r.width > 120 and r.height > 120 and r.y0 > 50 and r.y1 < page_height - 50]
                        
                        # Cluster overlapping or near paths into single bounds
                        clusters = []
                        for r in valid_rects:
                            r_norm = fitz.Rect(r)
                            placed = False
                            for idx, c in enumerate(clusters):
                                if r_norm.intersects(fitz.Rect(c).inflate(60, 60)):
                                    clusters[idx] = c | r_norm
                                    placed = True
                                    break
                            if not placed:
                                clusters.append(r_norm)
                                
                        for c in clusters:
                            # Avoid page-scale margins or backgrounds
                            if c.width < page_width * 0.9 and c.height < page_height * 0.9:
                                overlaps = False
                                for v_rect in visual_rects:
                                    intersection = c & v_rect
                                    if intersection.area() / min(c.area(), v_rect.area()) > 0.5:
                                        overlaps = True
                                        break
                                if not overlaps and c.width > 120 and c.height > 120:
                                    visual_rects.append(c)
                                    
                    # 2. Extract visual blocks & setup background description tasks
                    page_placeholders = []
                    for rect_idx, rect in enumerate(visual_rects):
                        try:
                            # Clamp rect to page boundary
                            clamped_rect = rect & page.rect
                            if clamped_rect.is_empty:
                                continue
                                
                            pix = page.get_pixmap(clip=clamped_rect, dpi=150)
                            image_bytes = pix.tobytes("png")
                            
                            img_hash = hashlib.md5(image_bytes).hexdigest()
                            placeholder_str = f"[[VISUAL_ELEMENT_PLACEHOLDER_{page_num}_{rect_idx}_{img_hash}]]"
                            
                            page_placeholders.append((clamped_rect, placeholder_str))
                            
                            tasks.append(self._get_and_cache_image_description(img_hash, image_bytes))
                            placeholders.append((placeholder_str, img_hash))
                        except Exception as e:
                            logger.error(f"Failed to process visual block on page {page_num+1}: {e}")
                            
                    # 3. Interweave text and visual blocks in reading layout order (two-column support)
                    all_blocks = []
                    for b in blocks:
                        x0, y0, x1, y1, content, block_no, block_type = b
                        if block_type == 0 and content.strip():
                            all_blocks.append({
                                "rect": fitz.Rect(x0, y0, x1, y1),
                                "content": content,
                                "type": "text"
                            })
                            
                    for rect, ph_str in page_placeholders:
                        all_blocks.append({
                            "rect": rect,
                            "content": ph_str,
                            "type": "visual"
                        })
                        
                    # Standard dual-column reading layout sort
                    mid_x = page_width / 2
                    def get_layout_sort_key(block):
                        r = block["rect"]
                        center_x = r.x0 + r.width / 2
                        col = 0 if center_x < mid_x else 1
                        # Full-page width blocks are categorized in column 0
                        if r.width > page_width * 0.7:
                            col = 0
                        return (col, r.y0, r.x0)
                        
                    all_blocks.sort(key=get_layout_sort_key)
                    
                    page_text_blocks = []
                    for block in all_blocks:
                        if block["type"] == "text":
                            page_text_blocks.append(block["content"])
                        else:
                            page_text_blocks.append(f"\n\n{block['content']}\n\n")
                            
                    pages_content.append(f"## Page {page_num + 1}\n\n" + "\n".join(page_text_blocks))
                    
                # 4. Resolve descriptions concurrently
                full_raw_text = "\n\n".join(pages_content)
                descriptions = {}
                if tasks:
                    logger.info(f"Running parallel local description for {len(tasks)} figures/charts...")
                    try:
                        from academic_pipeline.local_llm import log_activity
                        log_activity(f"PDF Visuals: analyzing {len(tasks)} charts concurrently...")
                    except ImportError:
                        pass
                        
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for (ph_str, img_hash), desc in zip(placeholders, results):
                        if isinstance(desc, Exception):
                            logger.error(f"Error describing figure {img_hash}: {desc}")
                            descriptions[ph_str] = "[Error analyzing visual element]"
                        else:
                            descriptions[ph_str] = f"\n\n> **[FIGURE / CHART DETECTED IN PDF]**:\n> {desc}\n\n"
                            
                for ph_str, desc in descriptions.items():
                    full_raw_text = full_raw_text.replace(ph_str, desc)
                    
                text = full_raw_text
                logger.debug(f"Extracted {len(text):,} chars using PyMuPDF (layout-visual) from {pdf_path}")
            except Exception as e:
                logger.error(f"PyMuPDF layout-visual extraction failed for {pdf_path}: {e}. Falling back to standard text...")

        # Fallback to text-only layout if PyMuPDF visual parsing failed
        if not text and PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                pages = []
                for page_num, page in enumerate(doc):
                    t = page.get_text("text", sort=True)
                    if t:
                        pages.append(f"## Page {page_num + 1}\n\n{t}")
                text = "\n\n".join(pages)
            except Exception as e:
                logger.error(f"PyMuPDF layout fallback failed: {e}")

        # Fallback to standard pypdf
        if not text:
            try:
                reader = PdfReader(pdf_path)
                pages = []
                for page_num, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        pages.append(f"## Page {page_num + 1}\n\n{t}")
                text = "\n\n".join(pages)
            except Exception as e:
                logger.error(f"PDF extraction fallback failed for {pdf_path}: {e}")

        # Save the extracted text/md to the .md file next to the PDF
        if text:
            try:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"Saved extracted PDF text as Markdown to: {md_path}")
            except Exception as e:
                logger.error(f"Failed to save extracted Markdown to {md_path}: {e}")

        return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Synchronous wrapper for extract_text_from_pdf_async.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(
                        asyncio.run,
                        self.extract_text_from_pdf_async(pdf_path)
                    ).result(timeout=400)
            else:
                return asyncio.run(self.extract_text_from_pdf_async(pdf_path))
        except Exception as e:
            logger.error(f"Sync PDF extraction failed for {pdf_path}: {e}")
            return ""

    # ── Async LLM analysis ───────────────────────────────────────────────────

    async def analyze_paper_text_async(self, paper_title: str, paper_text: str, patent_results: str = None) -> dict:
        """Native async entry point for paper analysis."""
        if not paper_text:
            return self._empty_analysis()
        return await self._llm.analyze_paper(paper_title, paper_text, patent_results=patent_results)

    async def synthesize_profile_async(self, name: str, papers_data: list[dict]) -> dict:
        """Native async entry point for holistic research synthesis."""
        if not papers_data:
            return self._empty_synthesis()
        return await self._llm.synthesize_research_profile(name, papers_data)

    # ── Synchronous wrappers for compatibility ───────────────────────────────

    def analyze_paper_text(self, paper_title: str, paper_text: str, patent_results: str = None) -> dict:
        """
        Synchronous entry point for pipeline/test compatibility.
        """
        if not paper_text:
            return self._empty_analysis()
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    result = ex.submit(
                        asyncio.run,
                        self.analyze_paper_text_async(paper_title, paper_text, patent_results=patent_results)
                    ).result(timeout=360)
                return result
            else:
                return asyncio.run(
                    self.analyze_paper_text_async(paper_title, paper_text, patent_results=patent_results)
                )
        except Exception as e:
            logger.error(f"Sync LLM analysis failed for '{paper_title}': {e}")
            return self._empty_analysis()

    def synthesize_profile(self, name: str, papers_data: list[dict]) -> dict:
        """
        Synchronous entry point for pipeline/test compatibility.
        """
        if not papers_data:
            return self._empty_synthesis()
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    result = ex.submit(
                        asyncio.run,
                        self.synthesize_profile_async(name, papers_data)
                    ).result(timeout=600)
                return result
            else:
                return asyncio.run(
                    self.synthesize_profile_async(name, papers_data)
                )
        except Exception as e:
            logger.error(f"Sync LLM synthesis failed for '{name}': {e}")
            return self._empty_synthesis()

    @staticmethod
    def _empty_analysis() -> dict:
        return {
            "summary":             "Analysis failed or text could not be extracted.",
            "key_contributions":   [],
            "methodology":         "N/A",
            "datasets_used":       [],
            "limitations":         [],
            "future_work":         [],
            "technical_keywords":  [],
            "impact_assessment":   "N/A",
            "outcome":             "N/A",
            "novelty":             "N/A",
            "patent_filed":        False,
            "patent_details":      "N/A",
        }

    @staticmethod
    def _empty_synthesis() -> dict:
        return {
            "core_focus_areas": [],
            "evolution_of_work": "Synthesis failed.",
            "methodological_trends": [],
            "overall_impact": "N/A",
            "key_collaborators_or_institutions": [],
        }
