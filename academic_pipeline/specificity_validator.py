"""
specificity_validator.py — Deterministic quality gate for ResearchBlocker descriptions.

No LLM. No external calls. Pure text heuristics.

A blocker must score >= SPECIFICITY_THRESHOLD (0.40) to enter the ENRS pipeline.
Below threshold: the blocker is returned to the user with a refinement prompt.

Scoring rubric (each signal adds to a [0.0, 1.0] total, capped at 1.0):
    + 0.20  contains a named method, technique, or framework
    + 0.20  contains a named dataset, domain, or experimental system
    + 0.20  contains a measurable property or outcome
    + 0.15  length >= 15 words (specificity correlates with length)
    + 0.10  contains a causal or conditional claim
    + 0.10  contains a comparative element (vs., compared to, against)
    - 0.30  for each vague signal from VAGUE_SIGNALS (min 0.0)

Refinement prompts are generated per-failure-mode so the user knows
exactly what to add.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Signal word lists ─────────────────────────────────────────────────────────

# Patterns that suggest a named method / technique / framework
# Deliberately excludes vague umbrella terms like:
#   'machine learning', 'deep learning', 'neural network', 'AI'
# because those add no specificity — nearly any blocker could contain them.
_METHOD_PATTERNS = re.compile(
    r"\b("
    r"transformer|attention|bert|gpt|llm|diffusion|vae|gan|cnn|rnn|lstm|"
    r"reinforcement learning|ppo|sac|dqn|a3c|"
    r"slam|kalman|particle filter|monte carlo|mcmc|"
    r"active inference|free energy|variational bayes|"
    r"continual learning|catastrophic forgetting|replay|distillation|"
    r"benchmark|evaluation protocol|metric|loss function|regularization|"
    r"curriculum learning|self-paced|scaffolding|"
    r"spectroscopy|transit photometry|radial velocity|"
    r"fmri|eeg|meg|electrophysiology|calcium imaging|"
    r"rsa|cka|representational similarity|"
    r"jaccard|cosine|embedding|vector|latent space|"
    r"zero-shot|few-shot|meta-learning|transfer learning|"
    r"causal inference|do-calculus|structural equation"
    r")\b",
    re.IGNORECASE,
)

# Patterns that suggest a named domain, dataset, or experimental system
_DOMAIN_PATTERNS = re.compile(
    r"\b("
    r"exoplanet|light[\s\-]?curve|transit|stellar|astrophysics|"
    r"motor cortex|prefrontal|hippocampus|cerebellum|"
    r"imagenet|coco|mnist|cifar|openwebtext|pile|"
    r"clinical|patient|cohort|longitudinal|"
    r"developmental|infant|child|animal|rodent|primate|"
    r"dataset|corpus|benchmark|database|repository|"
    r"simulation|environment|gym|atari|mujoco|"
    r"continual|lifelong|sequential|online|offline|"
    r"agent[\s\-]?architecture|long[\s\-]?horizon|memory[\s\-]?representation|"
    r"investigation|episodic|semantic|working memory"
    r")\b",
    re.IGNORECASE,
)

# Patterns that suggest a measurable property or quantifiable outcome
_MEASURABLE_PATTERNS = re.compile(
    r"\b("
    r"representation|structure|convergence|generalization|"
    r"accuracy|precision|recall|f1|auc|roc|"
    r"performance|efficiency|scalability|robustness|"
    r"uncertainty|calibration|confidence interval|"
    r"coherence|consistency|stability|"
    r"ordering|curriculum|schedule|trajectory|"
    r"bias|variance|overfitting|underfitting|"
    r"embedding|latent|feature|dimension|"
    r"memory|interference|forgetting|plasticity|"
    r"activation|gradient|weight|parameter"
    r")\b",
    re.IGNORECASE,
)

# Patterns that suggest a causal or conditional claim
_CAUSAL_PATTERNS = re.compile(
    r"\b("
    r"affects|changes|improves|degrades|causes|prevents|"
    r"leads to|results in|depends on|requires|enables|"
    r"when|if|given|conditional|upon|"
    r"effect of|impact of|role of|influence of|"
    r"showing|demonstrates|indicates|reveals|suggests|shows how"
    r")\b",
    re.IGNORECASE,
)

# Patterns that suggest comparison / contrast
_COMPARATIVE_PATTERNS = re.compile(
    r"\b("
    r"vs\.?|versus|compared to|against|relative to|"
    r"better than|worse than|superior|inferior|"
    r"baseline|state.of.the.art|sota|outperforms|"
    r"not just|beyond|instead of|rather than"
    r")\b",
    re.IGNORECASE,
)

# Vague phrases that reduce specificity — each match deducts 0.30
VAGUE_SIGNALS = {
    "need better",
    "improve",
    "more data",
    "better model",
    "understand",
    "learn about",
    "explore",
    "look into",
    "general",
    "any information",
    "something about",
    "kind of",
    "sort of",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SpecificityResult:
    score:            float
    is_eligible:      bool              # score >= SPECIFICITY_THRESHOLD
    signals_found:    list[str] = field(default_factory=list)
    vague_hits:       list[str] = field(default_factory=list)
    refinement_hints: list[str] = field(default_factory=list)


SPECIFICITY_THRESHOLD = 0.40


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_blocker_specificity(description: str) -> SpecificityResult:
    """
    Compute specificity score for a ResearchBlocker description.

    Returns a SpecificityResult with:
        score:            float in [0.0, 1.0]
        is_eligible:      True if score >= 0.40
        signals_found:    which positive signals were detected
        vague_hits:       which vague phrases were found
        refinement_hints: specific suggestions to improve rejected blockers

    Example inputs and approximate scores:
        "Need better AI"                        → 0.00 (rejected)
        "Need developmental learning data"      → 0.30 (borderline, rejected)
        "Need curriculum dataset for agents"    → 0.45 (accepted, borderline)
        "Need empirical evidence that curriculum ordering changes representation
         structure (not just task performance) in continual learning agents"
                                                → 0.80 (accepted, strong)
    """
    text = description.strip()
    words = text.split()
    text_lower = text.lower()

    score = 0.0
    signals: list[str] = []
    vague_hits: list[str] = []

    # ── Positive signals ──────────────────────────────────────────────────────

    if _METHOD_PATTERNS.search(text):
        score += 0.20
        signals.append("named method/technique detected")

    if _DOMAIN_PATTERNS.search(text):
        score += 0.20
        signals.append("named domain/dataset/system detected")

    if _MEASURABLE_PATTERNS.search(text):
        score += 0.20
        signals.append("measurable property or outcome detected")

    if len(words) >= 15:
        score += 0.15
        signals.append(f"sufficient length ({len(words)} words)")

    if _CAUSAL_PATTERNS.search(text):
        score += 0.10
        signals.append("causal/conditional claim detected")

    if _COMPARATIVE_PATTERNS.search(text):
        score += 0.10
        signals.append("comparative element detected")

    # ── Vague signal deductions ───────────────────────────────────────────────

    for phrase in VAGUE_SIGNALS:
        if phrase in text_lower:
            score -= 0.30
            vague_hits.append(f'"{phrase}"')

    score = max(0.0, min(1.0, round(score, 3)))

    # ── Refinement hints (only when rejected) ─────────────────────────────────

    hints: list[str] = []
    if score < SPECIFICITY_THRESHOLD:
        if not _METHOD_PATTERNS.search(text):
            hints.append(
                "Name a specific method, technique, or framework "
                "(e.g. 'continual learning', 'fMRI', 'representational similarity analysis')"
            )
        if not _DOMAIN_PATTERNS.search(text):
            hints.append(
                "Name a specific domain, dataset, or experimental system "
                "(e.g. 'exoplanet transit datasets', 'motor cortex recordings')"
            )
        if not _MEASURABLE_PATTERNS.search(text):
            hints.append(
                "Include a measurable property or outcome you need "
                "(e.g. 'representation structure', 'convergence guarantee', 'calibration metric')"
            )
        if len(words) < 15:
            hints.append(
                f"Describe what you need in more detail (current: {len(words)} words, "
                f"aim for at least 15)"
            )
        if vague_hits:
            hints.append(
                f"Remove vague phrases: {', '.join(vague_hits)} — replace with "
                f"specific claims about what evidence would look like"
            )

    return SpecificityResult(
        score=score,
        is_eligible=score >= SPECIFICITY_THRESHOLD,
        signals_found=signals,
        vague_hits=vague_hits,
        refinement_hints=hints,
    )


def format_rejection_message(description: str, result: SpecificityResult) -> str:
    """
    Human-readable rejection message shown in the UI when a blocker
    fails the specificity gate.
    """
    lines = [
        f"Blocker rejected (specificity score: {result.score:.2f}, minimum: {SPECIFICITY_THRESHOLD:.2f})",
        "",
        f'Your description: "{description}"',
        "",
        "To improve it:",
    ]
    for hint in result.refinement_hints:
        lines.append(f"  • {hint}")

    lines += [
        "",
        "Example of a strong blocker:",
        "  'Need empirical evidence that curriculum ordering changes representation",
        "   structure (not just task performance) in continual learning agents'",
    ]
    return "\n".join(lines)
