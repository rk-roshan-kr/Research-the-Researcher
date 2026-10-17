"""
local_llm.py — Fully local LLM client via Ollama.

Replaces the Gemini API for:
  • PDF paper analysis
  • Social media content summarisation
  • Co-author profile synthesis

Design decisions:
  - Uses the official `ollama` Python library (not the OpenAI-compat shim) for
    guaranteed structured JSON output via Pydantic schema enforcement.
  - A single persistent httpx session is reused across all calls (connection pool).
  - Async wrapper around the synchronous ollama library lets us gather() multiple
    analyses concurrently; the GPU handles batching internally via CUDA queuing.
  - Temperature=0 for deterministic, reproducible analysis.
  - Exponential back-off retry on overload / timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from pydantic import BaseModel, Field
from academic_pipeline.local_search import LocalSearchClient

logger = logging.getLogger(__name__)

# Activity logging for interactive Ctrl+O console
from collections import deque
import threading

_activity_log = deque(maxlen=100)
_log_lock = threading.Lock()

def log_activity(msg: str):
    """Log an LLM activity thread-safely with a timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    with _log_lock:
        _activity_log.append(f"[{timestamp}] {msg}")

def get_llm_activity_log(max_lines: int = 15) -> list[str]:
    """Retrieve the recent LLM activity log lines."""
    with _log_lock:
        return list(_activity_log)[-max_lines:]


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas for structured output
# ──────────────────────────────────────────────────────────────────────────────

class ResearchSynthesis(BaseModel):
    """Macro-level holistic synthesis of a researcher's entire corpus."""
    core_focus_areas: list[str] = Field(description="3-5 primary research domains or themes they consistently work on")
    evolution_of_work: str = Field(description="A paragraph explaining how their research interests have evolved over time based on the papers")
    methodological_trends: list[str] = Field(description="Common algorithms, frameworks, or methodologies they prefer to use")
    overall_impact: str = Field(description="A paragraph summarizing their overall impact on the field based on the provided corpus")
    key_collaborators_or_institutions: list[str] = Field(default_factory=list, description="Notable frequent co-authors or institutions mentioned across their work")

class PaperAnalysis(BaseModel):
    """Structured analysis of a single academic paper."""
    summary: str = Field(description="2-3 sentence plain-language summary of the paper")
    key_contributions: list[str] = Field(description="Top 3-5 novel contributions or findings")
    methodology: str = Field(description="Primary research methodology used")
    datasets_used: list[str] = Field(default_factory=list, description="Any datasets mentioned")
    limitations: list[str] = Field(default_factory=list, description="Stated or implied limitations")
    future_work: list[str] = Field(default_factory=list, description="Directions suggested for future work")
    technical_keywords: list[str] = Field(default_factory=list, description="Core technical terms/concepts")
    impact_assessment: str = Field(description="Assessment of the paper's likely impact and novelty")
    outcome: str = Field(default="N/A", description="Detailed explanation of the primary outcome or result")
    novelty: str = Field(default="N/A", description="What was uniquely novel about this research")
    patent_filed: bool = Field(default=False, description="Whether a patent has been filed based on this research (requires web search tool)")
    patent_details: str = Field(default="N/A", description="Details of the patent if found, or 'N/A' if none found")

class SocialProfile(BaseModel):
    """Synthesised social presence of a researcher."""
    linkedin_url: Optional[str] = Field(default=None)
    twitter_url: Optional[str] = Field(default=None)
    personal_homepage: Optional[str] = Field(default=None)
    summary: str = Field(description="Overall public online presence summary")
    linkedin_summary: str = Field(default="", description="LinkedIn activity and professional summary")
    twitter_summary: str = Field(default="", description="Twitter/X posting style and topics")
    key_topics: list[str] = Field(default_factory=list, description="Recurring themes in public posts")
    tone: str = Field(default="professional", description="Communication tone: academic/casual/technical/etc.")
    recent_news_and_events: str = Field(default="", description="Recent mentions in news or conference appearances")

class ResearcherSynthesis(BaseModel):
    """High-level synthesis of a researcher's complete dossier."""
    research_focus: str = Field(description="Core research focus in 2-3 sentences")
    career_stage: str = Field(description="Early/mid/senior career assessment")
    notable_achievements: list[str] = Field(default_factory=list)
    collaboration_style: str = Field(default="", description="How they tend to collaborate based on co-author patterns")
    influence_assessment: str = Field(description="Assessment of influence in their field")
    emerging_themes: list[str] = Field(default_factory=list, description="Recent shifts in research direction")


# ──────────────────────────────────────────────────────────────────────────────
# Week 4: Extraction schemas with confidence + source tracking
# ──────────────────────────────────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    """A specific assertoric claim from a paper, with confidence and provenance."""
    text: str = Field(description="The claim stated assertorically (not as a question or goal)")
    confidence: float = Field(ge=0.0, le=1.0, description="LLM confidence this is a genuine claim (not background or framing)")
    source_paper_dois: list[str] = Field(default_factory=list, description="DOIs of papers making this claim")
    domain: str = Field(default="", description="Research domain or topic area")

class ExtractedQuestion(BaseModel):
    """An open research question the author genuinely has not answered."""
    text: str = Field(description="The question stated concisely")
    confidence: float = Field(ge=0.0, le=1.0, description="LLM confidence this is a genuine open question")
    source_paper_dois: list[str] = Field(default_factory=list, description="DOIs of papers raising this question")
    question_type: str = Field(default="empirical", description="empirical | theoretical | methodological")

class SelfModelExtraction(BaseModel):
    """Structured research identity extracted from a paper corpus."""
    strong_claims: list[ExtractedClaim] = Field(default_factory=list)
    open_questions: list[ExtractedQuestion] = Field(default_factory=list)
    methods_used: list[str] = Field(default_factory=list, description="Specific named methods (NOT vague terms like 'machine learning')")
    topics: list[str] = Field(default_factory=list, description="3-8 core research topic keywords")

class ContradictionAnalysis(BaseModel):
    """LLM analysis of whether two claims are in genuine tension."""
    is_contradiction: bool = Field(description="True only if claims are genuinely opposing, not merely related")
    contradiction_type: str = Field(default="empirical", description="empirical | definitional | scope | methodological")
    refined_external_claim: str = Field(default="", description="More precise restatement of external claim")
    resolution_hypothesis: str = Field(default="", description="What evidence would resolve this contradiction")
    severity: float = Field(ge=0.0, le=1.0, description="0=trivial, 1=fundamental to both research programs")
    confidence: float = Field(ge=0.0, le=1.0, description="LLM certainty that this IS a contradiction")

class BlockerSuggestion(BaseModel):
    """A specific actionable research blocker suggestion."""
    description: str = Field(description="Specific, actionable description of what is needed")
    blocker_type: str = Field(default="methodology", description="dataset | methodology | domain_expertise | collaboration | hardware | funding")
    severity: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", description="Why this blocks the investigation")

class BlockerSuggestions(BaseModel):
    """Container for multiple blocker suggestions."""
    suggestions: list[BlockerSuggestion] = Field(default_factory=list)

class CollaborationHypotheses(BaseModel):
    """Specific collaboration hypotheses based on matched evidence."""
    hypotheses: list[str] = Field(default_factory=list, description="2-3 specific, actionable hypotheses")


class IntellectualQuestionSuggestion(BaseModel):
    """Tension-based question suggestion bridging blockers and paper claims."""
    has_tension: bool = Field(description="True if there is a genuine connection or tension that warrants a question")
    tension_score: float = Field(ge=0.0, le=1.0, description="Confidence/strength of the tension: 0.0=none, 1.0=critical contradiction")
    tension_description: str = Field(description="Detailed explanation of the tension or contrast between blocker assumptions and their findings")
    intellectual_question: str = Field(description="Bridge question (Would your framework still hold under C?)")
    rationale: str = Field(description="Why this question is serious and builds research credibility")
    email_draft: Optional[str] = Field(default=None, description="A complete, professional cold outreach email draft to the professor, greeting them politely, referencing their specific paper, describing how it relates to our blocker, posing the intellectual question, and asking for their feedback or a brief follow-up discussion.")
    self_context: Optional[str] = Field(default=None, description="PRIVATE: Internal reasoning note for the researcher only — which specific claim, blocker ID, or investigation hypothesis motivated this question. This is NEVER sent to the professor.")
    paper_referenced: Optional[str] = Field(default=None, description="The exact title of the professor's paper this question is based on.")


class ResearchProgramSummary(BaseModel):
    """Structured name and summary of a research program."""
    title: str = Field(description="A concise 2-4 word academic title summarizing the research program (e.g. 'Continual Latent Dynamics')")
    summary: str = Field(description="A 1-2 sentence overview of the core goals, methods, and topics of this research program based on the papers.")


class ExtractedClaimDetail(BaseModel):
    """Detailed structural claim mapping assertoric statements to methods, datasets, and metrics."""
    claim: str = Field(description="Precise assertoric claim (e.g. 'Curriculum ordering improves representation geometry')")
    method: str = Field(description="The primary methodology or framework used to prove this claim (e.g. 'CKA', 'RSA', or 'N/A')")
    dataset: str = Field(description="The dataset utilized (e.g. 'CIFAR100', 'biological EEG', or 'N/A')")
    metric: str = Field(description="The exact metric or measurement used (e.g. 'representation similarity', 'MSE', or 'N/A')")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score that this claim is explicitly supported by the text")

class PaperClaimsExtraction(BaseModel):
    """Structured research claims and evidence details extracted from a single paper or abstract."""
    claims: list[ExtractedClaimDetail] = Field(default_factory=list, description="All distinct factual or theoretical claims asserted by the paper")
    open_questions: list[str] = Field(default_factory=list, description="Explicit open questions left unanswered")
    methods: list[str] = Field(default_factory=list, description="Key named methods used in the paper")
    datasets: list[str] = Field(default_factory=list, description="Key named datasets used in the paper")
    metrics: list[str] = Field(default_factory=list, description="Key named metrics/measurements used in the paper")


class BlockerRequirements(BaseModel):
    """Structured methodology, dataset, and metric required to resolve a research blocker."""
    required_method: str = Field(description="The specific method/technique required, or 'N/A' if none")
    required_dataset: str = Field(description="The specific dataset required, or 'N/A' if none")
    required_metric: str = Field(description="The specific metric/measurement required, or 'N/A' if none")


# ──────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ──────────────────────────────────────────────────────────────────────────────

class LocalLLMClient:
    """
    Thin async wrapper around Ollama for structured JSON inference.

    Usage:
        client = LocalLLMClient(model="qwen2.5-coder:7b")
        analysis = await client.analyze_paper("My paper title", full_text, patent_results="...")
    """

    DEFAULT_MODEL = "qwen2.5-coder:7b"
    DEFAULT_HOST  = "http://localhost:11434"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 max_retries: int = 3, timeout: int = 300,
                 cache_manager = None):
        self.model = model
        self.host  = host
        self.max_retries = max_retries
        self.timeout = timeout
        self._search_client = LocalSearchClient(cache_manager=cache_manager)
        self._llm_semaphore_instance = None

        if OLLAMA_AVAILABLE:
            self._client = _ollama.Client(host=host, timeout=timeout)
        else:
            self._client = None
            logger.warning("ollama package not installed. "
                           "Install with: pip install ollama")

    @property
    def llm_semaphore(self):
        """Lazy-loaded asyncio semaphore to limit concurrency."""
        if self._llm_semaphore_instance is None:
            self._llm_semaphore_instance = asyncio.Semaphore(2)
        return self._llm_semaphore_instance

    @property
    def async_client(self):
        """Get or create the AsyncClient bound to the current running event loop."""
        if not OLLAMA_AVAILABLE:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return _ollama.AsyncClient(host=self.host, timeout=self.timeout)
        if not hasattr(self, "_async_clients_map"):
            self._async_clients_map = {}
        if loop not in self._async_clients_map:
            self._async_clients_map[loop] = _ollama.AsyncClient(host=self.host, timeout=self.timeout)
        return self._async_clients_map[loop]

    # ── Low-level call ────────────────────────────────────────────────────────

    def _chat_structured(self, messages: list[dict], schema: type[BaseModel]) -> dict:
        """Synchronous structured call. Returns validated dict or raises."""
        if not OLLAMA_AVAILABLE or self._client is None:
            raise RuntimeError("Ollama is not installed. Run: pip install ollama")

        prompt_len = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = prompt_len // 3 + 2000
        num_ctx = max(2048, min(16384, estimated_tokens))

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    format=schema.model_json_schema(),
                    options={"temperature": 0, "num_ctx": num_ctx},
                )
                raw = response.message.content
                validated = schema.model_validate_json(raw)
                return validated.model_dump()

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt}/{self.max_retries} failed: {e}. "
                               f"Retrying in {wait}s...")
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for LLM call.")
                    raise

    async def _chat_structured_async(self, messages: list[dict],
                                     schema: type[BaseModel]) -> dict:
        """Native async structured call via AsyncClient."""
        if not OLLAMA_AVAILABLE or self.async_client is None:
            raise RuntimeError("Ollama is not installed. Run: pip install ollama")

        prompt_len = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = prompt_len // 3 + 2000
        num_ctx = max(2048, min(16384, estimated_tokens))

        log_activity(f"Calling LLM: {schema.__name__} (ctx={num_ctx} chars)")

        async with self.llm_semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await self.async_client.chat(
                        model=self.model,
                        messages=messages,
                        format=schema.model_json_schema(),
                        options={"temperature": 0, "num_ctx": num_ctx},
                    )
                    raw = response.message.content
                    validated = schema.model_validate_json(raw)
                    return validated.model_dump()

                except Exception as e:
                    wait = 2 ** attempt
                    log_activity(f"  └─ Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    logger.warning(f"Async LLM call attempt {attempt}/{self.max_retries} failed: {e}. "
                                   f"Retrying in {wait}s...")
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"All {self.max_retries} attempts failed for async LLM call.")
                        raise

    # ── Public API ────────────────────────────────────────────────────────────

    async def synthesize_research_profile(self, name: str, corpus_data: list[dict]) -> dict:
        """
        Synthesizes a macro-level intellectual profile across all the researcher's papers.
        """
        system = (
            "You are a distinguished academic historian and researcher profiler. "
            "Given a list of a researcher's publications (titles, abstracts, and summaries), "
            "synthesize a holistic 'Intellectual Profile' that captures their core themes, "
            "evolution, preferred methodologies, and overarching impact."
        )
        
        # Format the corpus data efficiently
        corpus_str = json.dumps(corpus_data, ensure_ascii=False, indent=2)
        # Trim if it's absurdly huge (14B models typically support 32k context, we'll cap at ~50k chars to be safe)
        if len(corpus_str) > 60000:
            corpus_str = corpus_str[:60000] + "\n...[TRUNCATED due to length]"
            
        user = (
            f"Researcher: {name}\n\n"
            f"Corpus Data:\n{corpus_str}\n\n"
            "Produce a structured Holistic Research Synthesis."
        )
        
        log_activity(f"Synthesizing profile for '{name}' ({len(corpus_data)} papers)")
        res = await self._chat_structured_async(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            ResearchSynthesis,
        )
        log_activity(f"Synthesized profile for '{name}'")
        return res

    async def analyze_paper(self, title: str, text: str,
                            patent_results: Optional[str] = None) -> dict:
        """
        Deep analysis of a single academic paper.

        Patent search is now pre-fetched by the pipeline in Python and passed
        in as `patent_results` to eliminate the expensive agentic tool-call
        round-trip, halving the number of LLM calls per paper.
        """
        if not OLLAMA_AVAILABLE or self.async_client is None:
            return {}

        log_activity(f"Analyzing paper: '{title}' (text={len(text)} chars)")

        # If caller provided pre-fetched patent results, inject them; otherwise note absence.
        if patent_results and patent_results.strip():
            patent_section = (
                f"--- PRE-FETCHED PATENT SEARCH RESULTS ---\n{patent_results}\n"
                "--- END PATENT RESULTS ---\n\n"
            )
            patent_instruction = (
                "Use the patent search results above to determine if a patent was filed. "
                "Set patent_filed=true and populate patent_details if evidence is found."
            )
        else:
            patent_section = ""
            patent_instruction = (
                "No patent search results were available. "
                "Set patent_filed=false and patent_details='N/A'."
            )

        system = (
            "You are an expert academic paper analyst. "
            "Given a paper's title, its text, and pre-fetched patent search results, "
            "produce a precise structured analysis. "
            "Be specific, cite concrete details from the text, and do not invent information."
        )
        user = (
            f"Paper Title: {title}\n\n"
            f"{patent_section}"
            f"--- PAPER TEXT (may be truncated) ---\n{text}\n---\n\n"
            f"{patent_instruction}\n\n"
            "Produce the full structured JSON analysis now."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        try:
            log_activity(f"  └─ Single-call structured analysis for: '{title}'...")
            res = await self._chat_structured_async(messages, PaperAnalysis)
            log_activity(f"Paper analysis complete: '{title}'")
            return res
        except Exception as e:
            log_activity(f"  └─ Analysis failed for '{title}': {e}")
            logger.error(f"LLM analysis failed for '{title}': {e}")
            return {}

    async def summarize_social(self, researcher_name: str,
                               raw_content: dict) -> dict:
        """
        Given raw scraped content (linkedin text, twitter text, etc.),
        produce a structured SocialProfile.
        """
        system = (
            "You are an expert at analysing researchers' public online presence. "
            "Given raw scraped content from their profiles, extract a structured summary. "
            "Only report facts present in the provided content — do not speculate."
        )
        content_str = json.dumps(raw_content, ensure_ascii=False, indent=2)[:12000]
        user = (
            f"Researcher: {researcher_name}\n\n"
            f"Scraped content:\n{content_str}\n\n"
            "Produce a structured social profile summary."
        )
        log_activity(f"Summarizing social profiles for '{researcher_name}'")
        res = await self._chat_structured_async(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            SocialProfile,
        )
        log_activity(f"Summarized social profiles for '{researcher_name}'")
        return res

    async def synthesize_researcher(self, name: str, metrics: dict,
                                    publications: list, coauthors: list) -> dict:
        """
        High-level synthesis of a researcher's complete profile.
        Called once at the end of the pipeline to generate the executive summary.
        """
        system = (
            "You are a senior academic hiring committee member. "
            "Given a researcher's metrics, publication list, and collaborators, "
            "provide a concise, honest assessment."
        )
        pub_titles = [p.get("title", "") for p in publications[:30]]
        coauth_names = [c.get("name", "") for c in coauthors[:20]]
        user = (
            f"Researcher: {name}\n"
            f"Metrics: H-Index={metrics.get('h_index',0)}, "
            f"Citations={metrics.get('citations',0)}, "
            f"Publications={metrics.get('publications',0)}\n"
            f"Recent papers: {', '.join(pub_titles[:15])}\n"
            f"Top co-authors: {', '.join(coauth_names[:10])}\n\n"
            "Synthesise this researcher's profile."
        )
        log_activity(f"Synthesizing researcher assessment for '{name}'")
        res = await self._chat_structured_async(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            ResearcherSynthesis,
        )
        log_activity(f"Synthesized researcher assessment for '{name}'")
        return res

    async def describe_image_async(self, image_bytes: bytes) -> str:
        """Describes a figure, table, or chart from a research paper using a local multimodal model via Ollama."""
        if not OLLAMA_AVAILABLE or self.async_client is None:
            return "[Ollama client not available for local figure analysis]"
            
        import base64
        
        # 1. Determine local vision model if not already cached
        if not hasattr(self, "_vision_model") or self._vision_model is None:
            try:
                models = await self.async_client.list()
                installed = [m.model for m in models.models]
                
                # Check priority vision models
                for candidate in ["moondream", "llama3.2-vision", "minicpm-v", "llava"]:
                    matched = next((m for m in installed if candidate in m.lower()), None)
                    if matched:
                        self._vision_model = matched
                        break
                        
                # Fallback to any model containing 'vision' or 'llava'
                if not self._vision_model:
                    self._vision_model = next((m for m in installed if "vision" in m.lower() or "llava" in m.lower() or "minicpm" in m.lower()), None)
                    
            except Exception as e:
                logger.error(f"Failed to query Ollama models list: {e}")
                self._vision_model = None
                
        # If no vision model is installed, print warning and return placeholder
        if not self._vision_model:
            logger.warning("No local vision model (e.g. 'moondream' or 'llama3.2-vision') found in Ollama. Skipping image description.")
            try:
                log_activity("Warning: No local vision model found. Run 'ollama pull moondream' to enable figure/chart descriptions.")
            except Exception:
                pass
            return "[Visual elements present in PDF but skipped because no local vision model is installed in Ollama. Pull 'moondream' to enable.]"
            
        # 2. Convert to base64
        try:
            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = (
                "You are an expert scientific visual analyst. "
                "Describe what is shown in this figure, chart, table, or diagram from an academic paper in detail. "
                "Focus on: "
                "1. The type of visualization (e.g. line graph, bar chart, schema, scatter plot, flowchart). "
                "2. The data shown: labels, axes, units, data points, trends, or relationships. "
                "3. The main scientific takeaway or conclusion from this figure. "
                "Keep the description concise but highly informative (1-2 short paragraphs). Do not use markdown headers."
            )
            
            log_activity(f"Analyzing figure locally using vision model '{self._vision_model}'...")
            
            response = await self.async_client.chat(
                model=self._vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
            )
            
            desc = response.message.content.strip()
            log_activity(f"Figure description complete ({len(desc)} chars)")
            return desc
            
        except Exception as e:
            logger.error(f"Local Ollama image description failed for '{self._vision_model}': {e}")
            return f"[Error analyzing figure locally: {e}]"
            
    def describe_image(self, image_bytes: bytes) -> str:
        """Synchronous wrapper for describe_image_async."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, self.describe_image_async(image_bytes)).result(timeout=180)
            else:
                return asyncio.run(self.describe_image_async(image_bytes))
        except Exception as e:
            logger.error(f"Sync local image description failed: {e}")
            return f"[Error analyzing figure: {e}]"

    def is_available(self) -> bool:
        """Ping Ollama to confirm the server and model are ready."""
        if not OLLAMA_AVAILABLE or self._client is None:
            return False
        try:
            models = self._client.list()
            model_names = [m.model for m in models.models]
            if not any(self.model in m for m in model_names):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Ollama server not reachable at {self.host}: {e}")
            return False

    # ── Week 4: Research Identity Extraction ──────────────────────────────────

    def extract_self_model(
        self,
        texts: list[str],
        titles: Optional[list[str]] = None,
        dois: Optional[list[str]] = None,
        schema: Optional[dict] = None,   # ignored, kept for API compat
    ) -> dict:
        """
        Extract structured research identity from paper abstracts.

        Processes ONE DOCUMENT AT A TIME then aggregates, so the method
        scales to any corpus size without hitting LLM context limits.
        A single 3000-char doc fits comfortably; 50 docs in one shot does not.

        Returns dict matching SelfModelExtraction schema.
        On failure for individual docs, skips that doc and continues.
        """
        titles = titles or [""] * len(texts)
        dois   = dois   or [""] * len(texts)

        system = (
            "You are a research analyst extracting structured information from a single research document.\n"
            "Rules:\n"
            "- strong_claims: assertoric statements the researcher makes as facts or hypotheses "
            "(NOT goals, questions, or vague statements). Be thorough — extract ALL claims you find.\n"
            "- open_questions: genuine research unknowns the researcher has not yet answered "
            "(NOT rhetorical). Include every unresolved question.\n"
            "- methods_used: specific named methods, frameworks, or techniques "
            "(NOT generic terms like 'machine learning' or 'AI').\n"
            "- topics: core research area keywords (3-6 per document).\n"
            "- confidence: how certain you are the item genuinely appears in the text.\n"
            "Be exhaustive, not selective."
        )

        all_claims:    list[dict] = []
        all_questions: list[dict] = []
        all_methods:   list[str]  = []
        all_topics:    list[str]  = []

        log_activity(f"Extracting self-model from {len(texts)} paper(s) (per-document)")

        for i, (text, title, doi) in enumerate(zip(texts, titles, dois)):
            doc_block = (
                f"Title: {title}\n"
                f"DOI: {doi or 'N/A'}\n\n"
                f"{text[:3000]}"
            )
            user = (
                f"Extract all research claims, open questions, methods, and topics "
                f"from this document:\n\n{doc_block}"
            )
            try:
                result = self._chat_structured(
                    [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                    SelfModelExtraction,
                )
                doc_claims    = result.get("strong_claims", [])
                doc_questions = result.get("open_questions", [])
                doc_methods   = result.get("methods_used", [])
                doc_topics    = result.get("topics", [])

                all_claims.extend(doc_claims)
                all_questions.extend(doc_questions)
                all_methods.extend(doc_methods)
                all_topics.extend(doc_topics)

                logger.debug(
                    f"  doc {i+1}/{len(texts)} '{title[:40]}': "
                    f"{len(doc_claims)}C {len(doc_questions)}Q "
                    f"{len(doc_methods)}M {len(doc_topics)}T"
                )
            except Exception as e:
                logger.warning(f"  doc {i+1}/{len(texts)} extraction failed: {e}")
                continue

        # Deduplicate topics (case-insensitive, preserve insertion order)
        seen_t: set[str] = set()
        unique_topics: list[str] = []
        for t in all_topics:
            key = t.strip().lower()
            if key and key not in seen_t:
                seen_t.add(key)
                unique_topics.append(t.strip())

        # Deduplicate methods
        seen_m: set[str] = set()
        unique_methods: list[str] = []
        for m in all_methods:
            key = m.strip().lower()
            if key and key not in seen_m:
                seen_m.add(key)
                unique_methods.append(m.strip())

        log_activity(
            f"Aggregated: {len(all_claims)} claims, {len(all_questions)} questions, "
            f"{len(unique_methods)} methods, {len(unique_topics)} topics"
        )

        return {
            "strong_claims":  all_claims,
            "open_questions": all_questions,
            "methods_used":   unique_methods,
            "topics":         unique_topics,
        }

    async def extract_claims_and_evidence(self, title: str, text: str) -> dict:
        """
        Structured claim, method, dataset, metric, and open question extraction from a single paper or abstract.
        Allows native async calling.
        """
        system = (
            "You are an expert academic research parser.\n"
            "Given a paper's title and its text/abstract, extract:\n"
            "1. claims: A list of precise assertoric claims. For each claim, identify:\n"
            "   - claim: The exact assertoric claim made in the text.\n"
            "   - method: The primary methodology/framework used to prove this specific claim (e.g., 'CKA', 'RSA', 'neural recordings', or 'N/A').\n"
            "   - dataset: The dataset used (e.g., 'CIFAR100', 'mouse behavior', or 'N/A').\n"
            "   - metric: The exact metric measured (e.g., 'representation similarity', 'MSE', 'classification accuracy', or 'N/A').\n"
            "   - confidence: Your confidence that this claim is explicitly supported by the text (0.0 to 1.0).\n"
            "2. open_questions: Explicit unanswered questions.\n"
            "3. methods: Key methods mentioned.\n"
            "4. datasets: Key datasets mentioned.\n"
            "5. metrics: Key metrics/measurements mentioned.\n"
            "Do not invent information. Use 'N/A' for missing elements."
        )
        user = (
            f"Paper Title: {title}\n\n"
            f"Paper Content:\n{text[:3000]}\n\n"
            "Perform deep structured claims and evidence extraction now."
        )
        log_activity(f"Extracting claims & evidence for paper: '{title[:40]}...'")
        try:
            res = await self._chat_structured_async(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                PaperClaimsExtraction
            )
            return res
        except Exception as e:
            logger.error(f"Structured claim extraction failed for '{title}': {e}")
            return {
                "claims": [],
                "open_questions": [],
                "methods": [],
                "datasets": [],
                "metrics": []
            }

    def extract_claims_and_evidence_sync(self, title: str, text: str) -> dict:
        """Synchronous wrapper for extract_claims_and_evidence."""
        system = (
            "You are an expert academic research parser.\n"
            "Given a paper's title and its text/abstract, extract:\n"
            "1. claims: A list of precise assertoric claims. For each claim, identify:\n"
            "   - claim: The exact assertoric claim made in the text.\n"
            "   - method: The primary methodology/framework used to prove this specific claim (e.g., 'CKA', 'RSA', 'neural recordings', or 'N/A').\n"
            "   - dataset: The dataset used (e.g., 'CIFAR100', 'mouse behavior', or 'N/A').\n"
            "   - metric: The exact metric measured (e.g., 'representation similarity', 'MSE', 'classification accuracy', or 'N/A').\n"
            "   - confidence: Your confidence that this claim is explicitly supported by the text (0.0 to 1.0).\n"
            "2. open_questions: Explicit unanswered questions.\n"
            "3. methods: Key methods mentioned.\n"
            "4. datasets: Key datasets mentioned.\n"
            "5. metrics: Key metrics/measurements mentioned.\n"
            "Do not invent information. Use 'N/A' for missing elements."
        )
        user = (
            f"Paper Title: {title}\n\n"
            f"Paper Content:\n{text[:3000]}\n\n"
            "Perform deep structured claims and evidence extraction now."
        )
        try:
            return self._chat_structured(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                PaperClaimsExtraction
            )
        except Exception as e:
            logger.error(f"Synchronous claim extraction failed for '{title}': {e}")
            return {
                "claims": [],
                "open_questions": [],
                "methods": [],
                "datasets": [],
                "metrics": []
            }

    def extract_blocker_requirements_sync(self, description: str) -> dict:
        """Synchronously extract required_method, required_dataset, and required_metric from a blocker description."""
        system = (
            "You are an expert research system parser.\n"
            "Given a blocker's description, extract the specific methodology/technique, dataset, and metric/measurement required to resolve it.\n"
            "If any field is not explicitly specified or cannot be inferred, use 'N/A'. Do not invent information."
        )
        user = f"Blocker Description: {description}"
        try:
            return self._chat_structured(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                BlockerRequirements
            )
        except Exception as e:
            logger.warning(f"Failed to extract blocker requirements: {e}")
            return {
                "required_method": "N/A",
                "required_dataset": "N/A",
                "required_metric": "N/A"
            }

    async def analyze_contradiction_details(self, my_claim: dict, their_claim: dict) -> dict:
        """
        Evaluate NLI-flagged claim pairs for genuine contradiction and generate resolution hypotheses.
        """
        system = (
            "You are analyzing whether two research claims are in genuine contradiction.\n"
            "Only return is_contradiction=true if the claims OPPOSE each other, "
            "not merely if they discuss the same topic or are slightly different.\n"
            "Provide a clear type (empirical | theoretical | definitional | scope), severity, and confidence score."
        )
        user = (
            f"YOUR CLAIM Details:\n"
            f"- Text: \"{my_claim['claim']}\"\n"
            f"- Method: {my_claim.get('method', 'N/A')}\n"
            f"- Dataset: {my_claim.get('dataset', 'N/A')}\n"
            f"- Metric: {my_claim.get('metric', 'N/A')}\n\n"
            f"THEIR CLAIM Details:\n"
            f"- Text: \"{their_claim['claim']}\"\n"
            f"- Method: {their_claim.get('method', 'N/A')}\n"
            f"- Dataset: {their_claim.get('dataset', 'N/A')}\n"
            f"- Metric: {their_claim.get('metric', 'N/A')}\n\n"
            "Analyze if these two are contradictory. Generate a resolution hypothesis if so."
        )
        log_activity(f"Deep contradiction analysis: '{my_claim['claim'][:40]}...' vs '{their_claim['claim'][:40]}...'")
        try:
            res = await self._chat_structured_async(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                ContradictionAnalysis
            )
            return res
        except Exception as e:
            logger.error(f"Failed deep contradiction analysis: {e}")
            return {
                "is_contradiction": False,
                "contradiction_type": "empirical",
                "refined_external_claim": "",
                "resolution_hypothesis": "",
                "severity": 0.0,
                "confidence": 0.0
            }

    def analyze_contradiction_sync(
        self,
        claim_yours: str,
        claim_external: str,
        source_external: str,
    ) -> Optional[dict]:
        """
        Determine whether claim_yours and claim_external are in genuine tension.

        Called AFTER NLI filter confirms 'contradiction' label.
        Returns ContradictionAnalysis dict or None on failure.
        """
        system = (
            "You are analyzing whether two research claims are in genuine contradiction.\n"
            "Only return is_contradiction=true if the claims OPPOSE each other, "
            "not merely if they discuss the same topic."
        )
        user = (
            f"YOUR CLAIM:\n\"{claim_yours}\"\n\n"
            f"EXTERNAL CLAIM (from {source_external}):\n\"{claim_external}\"\n\n"
            "Are these genuinely contradictory? Analyse carefully."
        )

        log_activity(f"Analyzing contradiction: '{claim_yours[:50]}...'")
        try:
            result = self._chat_structured(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                ContradictionAnalysis,
            )
            log_activity(
                f"Contradiction analysis: is_contradiction={result.get('is_contradiction')}, "
                f"confidence={result.get('confidence', 0):.2f}"
            )
            return result
        except Exception as e:
            logger.warning(f"Contradiction analysis failed: {e}")
            return None

    def suggest_blockers_sync(
        self,
        investigation,   # InvestigationStream (avoid circular import)
        self_model: Optional[dict] = None,
    ) -> list[dict]:
        """
        Suggest specific, actionable research blockers for an investigation.
        Returns list of BlockerSuggestion dicts.
        All drafts must be approved by the user before add_blocker() is called.
        """
        existing = "\n".join(
            f"- {b.description}" for b in investigation.blockers if not b.resolved
        ) or "None currently"

        self_context = ""
        if self_model:
            topics = ", ".join(self_model.get("topics", [])[:5])
            questions = "\n".join(
                f"- {q['text'] if isinstance(q, dict) else q}"
                for q in self_model.get("open_questions", [])[:3]
            )
            self_context = f"\nYour research context:\n- Topics: {topics}\n- Open questions:\n{questions}\n"

        system = (
            "You are a research advisor identifying specific blockers for a research investigation.\n"
            "Each suggestion must be concrete and specific enough that a professor's published "
            "work could plausibly address it. No vague blockers."
        )
        user = (
            f"INVESTIGATION: {investigation.title}\n"
            f"DESCRIPTION: {investigation.description}{self_context}\n"
            f"EXISTING BLOCKERS:\n{existing}\n\n"
            "Suggest 2-4 new, specific, actionable blockers not already listed."
        )

        log_activity(f"Suggesting blockers for '{investigation.title}'")
        try:
            result = self._chat_structured(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                BlockerSuggestions,
            )
            suggestions = result.get("suggestions", [])
            log_activity(f"Suggested {len(suggestions)} blockers for '{investigation.title}'")
            return suggestions
        except Exception as e:
            logger.warning(f"Blocker suggestion failed: {e}")
            return []

    def generate_collaboration_hypotheses(
        self,
        self_model: dict,
        prof_dossier: dict,
        resolution_matches: list[dict],
        total_score: float,
    ) -> list[str]:
        """
        Generate specific collaboration hypotheses based on actual blocker-paper matches.
        Called POST-SCORING. Returns list of hypothesis strings.
        """
        if not resolution_matches:
            return []

        prof_name = prof_dossier.get("name", "this professor")
        my_topics = ", ".join(self_model.get("topics", [])[:4])
        my_claims = [
            c["text"] if isinstance(c, dict) else c
            for c in self_model.get("strong_claims", [])[:3]
        ]
        my_questions = [
            q["text"] if isinstance(q, dict) else q
            for q in self_model.get("open_questions", [])[:3]
        ]

        match_summaries = "\n".join(
            f"- Blocker: '{m.get('blocker_description','')[:80]}' "
            f"→ Paper: '{m.get('paper_title','')[:80]}' (sim={m.get('similarity',0):.2f})"
            for m in resolution_matches[:3]
        )
        claims_block   = "\n".join(f"- {c}" for c in my_claims)   or "None extracted yet"
        questions_block= "\n".join(f"- {q}" for q in my_questions) or "None extracted yet"

        system = (
            "You are drafting specific collaboration hypotheses between two researchers.\n"
            "Each hypothesis must explain: (1) concrete artifact produced, "
            "(2) what each party contributes, (3) why collaboration beats solo work."
        )
        user = (
            f"RESEARCHER (you):\n"
            f"- Topics: {my_topics}\n"
            f"- Claims: {claims_block}\n"
            f"- Open questions: {questions_block}\n\n"
            f"COLLABORATOR: {prof_name} @ {prof_dossier.get('institution','')}\n"
            f"Score: {total_score:.1f}/100\n\n"
            f"MATCHED BLOCKERS:\n{match_summaries}\n\n"
            "Generate 2-3 specific, actionable collaboration hypotheses."
        )

        log_activity(f"Generating collaboration hypotheses for '{prof_name}'")
        try:
            result = self._chat_structured(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                CollaborationHypotheses,
            )
            hypotheses = result.get("hypotheses", [])
            log_activity(f"Generated {len(hypotheses)} hypotheses for '{prof_name}'")
            return [str(h) for h in hypotheses if h]
        except Exception as e:
            logger.warning(f"Hypothesis generation failed for '{prof_name}': {e}")
            return []

    def detect_intellectual_tension(
        self,
        investigation_title: str,
        blocker_description: str,
        prof_paper_title: str,
        prof_paper_abstract: str,
        tension_threshold: float = 0.70,
        my_work_context: Optional[str] = None,
    ) -> dict:
        """
        Analyze whether a research blocker/investigation and a professor's paper/claim
        are in genuine tension, and generate a deep peer-level research question.
        Returns a dict conforming to IntellectualQuestionSuggestion or a status dict
        if threshold gate was not met.
        """
        system = (
            "You are an elite academic peer and research advisor reviewing collaboration opportunities.\n"
            "Your task is to identify a genuine theoretical or empirical tension between a researcher's active blocker "
            "and an external professor's paper/claims.\n\n"
            "A genuine tension exists when:\n"
            "1. The claims imply different/competing predictions (e.g., reward-driven learning vs contradiction-driven learning).\n"
            "2. One framework is a special case or scope limit of the other.\n"
            "3. There is a methodological difference or theoretical gap.\n\n"
            "The generated question MUST bridge the two concepts and test boundaries. "
            "Never generate generic student questions (e.g., 'Can you explain X?'). "
            "Instead, generate questions that establish intellectual credibility (e.g., 'Since your paper suggests A "
            "affects B, would you expect the same behavior if C holds instead of D?').\n\n"
            "If has_tension is true, you MUST also generate a personalized cold outreach email draft (email_draft).\n"
            "Strict rules for email_draft:\n"
            "- It must be highly professional, peer-level, and extremely concise (under 100 words, target ~90 words).\n"
            "- Start directly by referencing their specific paper: 'I was reading your paper on [Paper Title].'\n"
            "- Pose the intellectual question immediately: 'One thing I'm unsure about: [Question]'\n"
            "- Describe the context of our investigation/blocker briefly: 'I'm asking because we're studying [Blocker/Investigation context].'\n"
            "- End politely with a peer-level feedback request: 'Curious whether your intuition is yes, no, or \"depends on the data.\"' or similar.\n"
            "- Greet them with 'Dear Professor [LastName],'.\n"
            "- Do NOT include boilerplate, generic greetings, fluff, or marketing introductions (e.g., 'I hope this email finds you well', 'Our studies aim to', 'Both our approaches')."
        )
        my_work_section = ""
        if my_work_context:
            my_work_section = (
                f"\nRESEARCHER'S OWN WORK CONTEXT (use to write a sharper self_context annotation, do NOT include internal details in email_draft):\n"
                f"{my_work_context}\n"
            )

        user = (
            f"RESEARCHER INVESTIGATION:\n"
            f"- Title: {investigation_title}\n"
            f"- Blocker Description: {blocker_description}\n"
            f"{my_work_section}\n"
            f"PROFESSOR'S WORK:\n"
            f"- Paper Title: {prof_paper_title}\n"
            f"- Paper Abstract: {prof_paper_abstract}\n\n"
            "Analyze carefully. Assign a tension_score (0.0 to 1.0) based on how strong and relevant the tension is. "
            "If there is no genuine tension or connection, set has_tension=false and tension_score=0.0.\n"
            f"Also fill paper_referenced with: '{prof_paper_title}'\n"
            "Fill self_context with a concise private note explaining which of your specific claims/hypotheses/blockers motivated this question — be precise and reference EarthOS internal context if applicable."
        )

        log_activity(f"Detecting intellectual tension for: '{prof_paper_title[:50]}'")
        try:
            result = self._chat_structured(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                IntellectualQuestionSuggestion,
            )
            
            # Gating check
            if not result.get("has_tension", False) or result.get("tension_score", 0.0) < tension_threshold:
                log_activity(
                    f"Tension gate did not pass: has_tension={result.get('has_tension')}, "
                    f"score={result.get('tension_score', 0.0):.2f} (threshold={tension_threshold:.2f})"
                )
                return {
                    "has_tension": False,
                    "tension_score": result.get("tension_score", 0.0),
                    "tension_description": result.get("tension_description", "No significant tension found."),
                    "intellectual_question": "",
                    "rationale": "Tension score was below the required threshold to formulate a high-quality question."
                }
                
            log_activity(
                f"Tension gate passed: score={result.get('tension_score'):.2f}, "
                f"question='{result.get('intellectual_question')[:50]}...'"
            )
            # Ensure paper_referenced is always populated
            if not result.get("paper_referenced"):
                result["paper_referenced"] = prof_paper_title
            return result
        except Exception as e:
            logger.warning(f"Intellectual tension detection failed: {e}")
            return {
                "has_tension": False,
                "tension_score": 0.0,
                "tension_description": f"Failed to execute tension detection: {e}",
                "intellectual_question": "",
                "rationale": "LLM inference error."
            }

    async def summarize_research_program(
        self,
        paper_titles_and_abstracts: list[dict],
        key_terms: list[str]
    ) -> dict:
        """
        Summarize a clustered research program using paper titles/abstracts and key TF-IDF keywords.
        """
        system = (
            "You are a senior academic research profiler.\n"
            "Given a list of publications that belong to a single research cluster, and the top key terms extracted via TF-IDF, "
            "synthesize a concise, domain-specific academic title and a 1-2 sentence summary of this research program.\n"
            "Rules:\n"
            "- The title MUST be extremely concise (2-4 words, e.g. 'Continual Representation Learning' or 'Latent Behavioral Geometry').\n"
            "- Anchors: Use the provided key terms to ensure the title is specific and domain-meaningful. Avoid generic names like 'Advanced AI Frameworks' or 'Machine Learning Methods'.\n"
            "- The summary must be a 1-2 sentence description summarizing the core focus, techniques, and goals of these papers."
        )

        papers_block = ""
        for i, p in enumerate(paper_titles_and_abstracts):
            title = p.get("title", "Unknown Title")
            abstract = p.get("abstract", "") or p.get("abstract_snippet", "")
            papers_block += f"{i+1}. Title: {title}\nAbstract: {abstract[:400]}\n\n"

        key_terms_str = ", ".join(key_terms)
        user = (
            f"KEY TERMS FOR ANCHORING:\n{key_terms_str}\n\n"
            f"CLUSTERED PAPERS:\n{papers_block}\n"
            "Produce the structured Research Program Summary now."
        )

        log_activity(f"Summarizing research program of {len(paper_titles_and_abstracts)} papers")
        try:
            res = await self._chat_structured_async(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                ResearchProgramSummary,
            )
            return res
        except Exception as e:
            logger.warning(f"Failed to summarize research program via LLM: {e}")
            fallback_title = " & ".join(t.capitalize() for t in key_terms[:3]) or "Exploratory Research"
            return {
                "title": fallback_title,
                "summary": f"A research program focusing on topics including {', '.join(key_terms[:5])}."
            }
