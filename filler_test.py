import spacy
import random
import re
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable, Callable, Any, Sequence

# Load spaCy model (silent fallback) – still available as default
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# =========================
# Filler Phrase Categories
# =========================
PREFIX_ONLY = [
    "to be honest",
    "frankly",
    "in fact",
    "after all",
    "generally speaking",
    "quite honestly",
    "if you think about it"
]

FLEXIBLE = [
    "actually",
    "basically",
    "you know",
    "truthfully"
]

SUFFIX_ONLY = [
    "in a way"
]

FILLER_PHRASES = PREFIX_ONLY + FLEXIBLE + SUFFIX_ONLY

# Argumentative-only fillers (semantic filter)
ARGUMENTATIVE_FILLERS = {"after all", "in fact"}

# Style presets (extended with strict formal)
STYLE_FILTERS: Dict[str, Sequence[str]] = {
    "neutral": FILLER_PHRASES,
    "casual": FLEXIBLE + ["you know", "to be honest", "basically", "frankly", "in a way"],
    # "formal" keeps mild discourse softeners but excludes more casual ones
    "formal": ["in fact", "after all", "generally speaking", "truthfully"],
    # "formal_strict" removes borderline informal phrases
    "formal_strict": ["in fact", "truthfully", "generally speaking"],
}

DISCOURSE_LEAD_INS = {
    "however", "moreover", "therefore", "thus", "meanwhile", "furthermore",
    "nevertheless", "nonetheless", "consequently", "additionally", "instead",
    "nonetheless", "still", "otherwise"
}

CAUSAL_CONTRAST_MARKERS = {
    "because", "since", "therefore", "thus", "hence", "so", "but", "although",
    "though", "however", "yet", "whereas", "while", "consequently", "thereby"
}

WEAK_FUNCTION_WORDS = {"and", "but", "or", "so", "yet"}

ZERO_WIDTH_MARK = "\u200B"  # used for subtle marker if needed

# =========================
# Config Dataclass
# =========================
@dataclass
class FillerConfig:
    ratio: float = 0.2
    style: str = "neutral"
    min_sentence_len: int = 4
    recent_window: int = 4
    repetition_recent_penalty: float = 1.2
    repetition_global_penalty: float = 0.4  # linear mode penalty strength
    repetition_global_alpha: float = 0.55   # exponential decay alpha
    use_exponential: bool = False           # toggles exponential weight formula
    skip_discourse_lead: bool = True
    skip_all_caps: bool = True
    skip_dialogue: bool = True
    skip_quote_dominated: bool = True       # POS/quote aware avoidance
    mode_weights: Optional[Dict[str, float]] = None
    allow_reapply: bool = False
    structured_plan: bool = False           # return (text, plan)
    dry_run: bool = False                   # only plan, no modification
    nlp_pipeline: Any = None                # dependency injection
    logging_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    enforce_semantics: bool = True          # semantic relevance gating
    attach_marker: bool = True              # add hidden marker to output

# =========================
# Utility / Detection Helpers
# =========================
def _is_all_caps(sent: str) -> bool:
    core = sent.strip()
    letters = [c for c in core if c.isalpha()]
    if len(letters) < 3:
        return False
    return all(c.isupper() for c in letters)

def _is_dialogue(sent: str) -> bool:
    s = sent.strip()
    return s.startswith(("\"", "'", "“", "‘"))

def _quote_dominated(sent: str) -> bool:
    total = len(sent)
    if total == 0:
        return False
    quote_chars = sent.count('"') + sent.count("'") + sent.count('“') + sent.count('”') + sent.count('‘') + sent.count('’')
    return (quote_chars / total) > 0.28  # heuristic threshold

def _already_has_filler(sent_text: str) -> bool:
    lowered = sent_text.strip().lower()
    for fp in FILLER_PHRASES:
        if lowered.startswith(fp + ",") or lowered.startswith(fp + " ,"):
            return True
    return False

def _starts_with_discourse_marker(sent_text: str) -> bool:
    first = sent_text.strip().split()[:1]
    return bool(first and first[0].lower().rstrip(",.") in DISCOURSE_LEAD_INS)

def _is_argumentative(span) -> bool:
    for token in span:
        if token.lemma_.lower() in CAUSAL_CONTRAST_MARKERS:
            return True
    return False

# =========================
# Subject / Mid Insertion Logic
# =========================
def _find_subject_subtree_end(span) -> Optional[int]:
    # span is a Span referencing parent doc; use token.i relative indices
    root = None
    for t in span:
        if t.dep_ == "ROOT" or t.head == t:  # safety
            root = t
            break
    if not root:
        return None
    subj = None
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass") and child in span:
            subj = child
            break
    if not subj:
        # fallback after root if inside span
        if root.i + 1 < span.end:
            rel = root.i - span.start + 1
            return rel
        return None
    end_abs = max(tok.i for tok in subj.subtree if tok in span) + 1
    if end_abs < span.end - 1:
        return end_abs - span.start
    return None

def _sanitize_for_prefix(span, original: str) -> str:
    # Skip lowering if first token is PROPN or entity
    if not original:
        return original
    first_token = span[0]
    if first_token.pos_ == 'PROPN' or first_token.ent_type_:
        return original
    if original[0].isalpha():
        return original[0].lower() + original[1:]
    return original

# =========================
# Mode & Filler Selection
# =========================
DEFAULT_MODE_WEIGHTS = {"prefix": 0.5, "mid": 0.3, "suffix": 0.2}

def _choose_mode(rng: random.Random, mode_weights: Dict[str, float]) -> int:
    pw, mw, sw = mode_weights.get("prefix", 0.5), mode_weights.get("mid", 0.3), mode_weights.get("suffix", 0.2)
    total = max(1e-9, pw + mw + sw)
    return rng.choices([0, 1, 2], weights=[pw/total, mw/total, sw/total], k=1)[0]

def _candidate_fillers_for_mode(mode: int) -> List[str]:
    if mode == 0:
        return list(PREFIX_ONLY + FLEXIBLE)
    if mode == 1:
        return list(FLEXIBLE)
    if mode == 2:
        return list(SUFFIX_ONLY + FLEXIBLE)
    return FILLER_PHRASES

# =========================
# Repetition Weighting
# =========================
def _linear_weight(filler: str, global_freq: Dict[str, int], recent: List[str], recent_penalty: float, global_penalty: float) -> float:
    base = 1.0
    gf = global_freq.get(filler, 0)
    if gf:
        base /= (1.0 + global_penalty * gf)
    if filler in recent:
        occurrences = recent.count(filler)
        base /= (1.0 + recent_penalty * occurrences)
    return max(base, 1e-6)

def _exp_weight(filler: str, global_freq: Dict[str, int], recent: List[str], recent_penalty: float, alpha: float) -> float:
    freq = global_freq.get(filler, 0)
    base = math.exp(-alpha * freq)
    if filler in recent:
        occurrences = recent.count(filler)
        base /= (1.0 + recent_penalty * occurrences)
    return max(base, 1e-8)

def _weighted_choice(rng: random.Random, fillers: List[str], weight_fn) -> str:
    scores = [weight_fn(f) for f in fillers]
    total = sum(scores)
    if total <= 0:
        return rng.choice(fillers)
    r = rng.random() * total
    acc = 0.0
    for f, s in zip(fillers, scores):
        acc += s
        if acc >= r:
            return f
    return fillers[-1]

# =========================
# Insertion Realization
# =========================
def _post_subject_index(span) -> Optional[int]:
    idx = _find_subject_subtree_end(span)
    if idx is None:
        return None
    # Skip weak function words moving forward
    j = idx
    while j < len(span) and span[j].lemma_.lower() in WEAK_FUNCTION_WORDS:
        j += 1
    if j < len(span) - 1:
        return j
    return idx

def _insert_prefix(span, filler: str) -> str:
    original = span.text
    filler_form = filler[0].upper() + filler[1:]
    return f"{filler_form}, {_sanitize_for_prefix(span, original)}"

def _insert_mid(span, filler: str) -> Optional[str]:
    rel = _post_subject_index(span)
    if rel is not None and 1 < rel < len(span) - 2:
        before = span[:rel].text.rstrip()
        after = span[rel:].text.lstrip()
        sep = "" if before.endswith(',') else ","
        return f"{before}{sep} {filler}, {after}"
    # Fallback: first comma inside span
    for i, t in enumerate(span):
        if t.text == ',' and i < len(span) - 4:
            before = span[:i+1].text.rstrip()
            after = span[i+1:].text.lstrip()
            return f"{before} {filler}, {after}"
    # Secondary fallback: token with advmod/aux soon after root
    for i, t in enumerate(span):
        if t.dep_ in ("advmod", "aux", "auxpass") and 1 < i < len(span) - 3:
            before = span[:i].text.rstrip()
            after = span[i:].text.lstrip()
            return f"{before}, {filler}, {after}"
    # Final fallback: 1/3 position
    if len(span) >= 6:
        pos = max(2, len(span)//3)
        before = span[:pos].text
        after = span[pos:].text
        return f"{before}, {filler}, {after}"
    return None

def _insert_suffix(span, filler: str) -> str:
    text = span.text.strip()
    # Handle trailing quotes / ellipsis
    match = re.search(r'(.*?)([\.\!\?]+)?(["”’])?$', text)
    if match:
        core = match.group(1).rstrip()
        punct = (match.group(2) or '') + (match.group(3) or '')
    else:
        core, punct = text, ''
    if core.endswith(','):
        core = core.rstrip(', ').rstrip()
    return f"{core}, {filler}{punct}"

def _realize_insertion(span, filler: str, mode: int) -> str:
    if mode == 0:
        return _insert_prefix(span, filler)
    if mode == 1:
        mid_variant = _insert_mid(span, filler)
        if mid_variant:
            return mid_variant
        return _insert_suffix(span, filler)
    return _insert_suffix(span, filler)

# =========================
# Double Application Guard
# =========================
def _already_processed(sentences: List[str]) -> bool:
    count = sum(1 for s in sentences if _already_has_filler(s))
    return (count / max(1, len(sentences))) > 0.4

# =========================
# Semantic gating of fillers
# =========================
def _filter_fillers_by_semantics(fillers: List[str], span, enforce: bool) -> List[str]:
    if not enforce:
        return fillers
    if not fillers:
        return fillers
    if any(f in ARGUMENTATIVE_FILLERS for f in fillers):
        arg_needed = _is_argumentative(span)
        if not arg_needed:
            # remove argumentative-only options
            fillers = [f for f in fillers if f not in ARGUMENTATIVE_FILLERS]
            # (The second condition redundant but keeps clarity)
    return fillers or []

# =========================
# Main Public API
# =========================
def add_fillers(
    text: str,
    config: Optional[FillerConfig] = None,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Any:
    """Insert filler phrases into text with advanced linguistic controls.

    Returns:
        If config.structured_plan True -> (modified_text_or_original, plan_list)
        Else -> modified text (string) or plan repr if dry_run.
    """
    cfg = config or FillerConfig()

    # Use provided rng else seed
    if rng is None:
        rng = random.Random(seed)

    # External pipeline or default
    pipeline = cfg.nlp_pipeline or nlp
    doc = pipeline(text)
    spans = list(doc.sents)
    if not spans:
        return (text, []) if cfg.structured_plan else text

    sentences = [s.text for s in spans]

    if not cfg.allow_reapply and _already_processed(sentences):
        if cfg.structured_plan:
            return (text, [])
        return text

    allowed_fillers = list(STYLE_FILTERS.get(cfg.style, FILLER_PHRASES))
    mode_weights = cfg.mode_weights or DEFAULT_MODE_WEIGHTS

    # Candidate sentence indices with precomputed flags
    candidates: List[int] = []
    for i, span in enumerate(spans):
        stripped = span.text.strip()
        if len(stripped.split()) < cfg.min_sentence_len:
            continue
        if cfg.skip_all_caps and _is_all_caps(stripped):
            continue
        if cfg.skip_dialogue and _is_dialogue(stripped):
            continue
        if cfg.skip_quote_dominated and _quote_dominated(stripped):
            continue
        if _already_has_filler(stripped):
            continue
        if cfg.skip_discourse_lead and _starts_with_discourse_marker(stripped):
            continue
        candidates.append(i)

    if not candidates:
        return (text, []) if cfg.structured_plan else text

    rng.shuffle(candidates)
    target = max(1, int(len(spans) * cfg.ratio))
    chosen = sorted(candidates[:target])
    chosen_set = set(chosen)

    global_freq: Dict[str, int] = {}
    recent: List[str] = []
    plan: List[Dict[str, Any]] = []
    output: List[str] = []

    for idx, span in enumerate(spans):
        original_text = span.text.strip()
        if idx not in chosen_set:
            output.append(original_text)
            continue
        # Decide insertion mode first
        mode = _choose_mode(rng, mode_weights)
        mode_fillers = [f for f in _candidate_fillers_for_mode(mode) if f in allowed_fillers]

        # Semantic gating
        mode_fillers = _filter_fillers_by_semantics(mode_fillers, span, cfg.enforce_semantics)
        if not mode_fillers:
            output.append(original_text)
            continue

        filler = _get_weighted_filler(rng, mode_fillers, global_freq, recent, cfg)
        if not filler:
            output.append(original_text)
            continue
        new_sentence = _realize_insertion(span, filler, mode)
        output.append(new_sentence)

        recent.append(filler)
        global_freq[filler] = global_freq.get(filler, 0) + 1
        entry = {"index": idx, "filler": filler, "mode": mode, "argumentative": _is_argumentative(span)}
        plan.append(entry)
        if cfg.logging_callback:
            try:
                cfg.logging_callback(entry)
            except Exception:
                pass

    result_text = " ".join(output)
    if cfg.attach_marker:
        result_text += ZERO_WIDTH_MARK  # subtle marker

    if cfg.dry_run:
        return plan if cfg.structured_plan else repr(plan)
    if cfg.structured_plan:
        return result_text, plan
    return result_text

# =========================
# Undo / Removal Support
# =========================
# Build alternation once
_FILLER_ALT = "|".join(re.escape(f) for f in FILLER_PHRASES)

FILLER_PATTERN_PREFIX = re.compile(rf"^(?:{_FILLER_ALT})\s*,\s*", re.IGNORECASE)
FILLER_PATTERN_MID = re.compile(rf",\s*(?:{_FILLER_ALT})\s*,\s*", re.IGNORECASE)
# Old (bad) pattern used \z -> invalid in Python
# FILLER_PATTERN_SUFFIX = re.compile(rf",\s*(?:{_FILLER_ALT})($|[\.\!\?]+|\s+\z)", re.IGNORECASE)
# New: use lookahead; do not consume trailing punctuation/end
FILLER_PATTERN_SUFFIX = re.compile(
    rf",\s*(?:{_FILLER_ALT})(?=(?:[\.\!\?]+(?:['\"”’])?\s*$)|\s*$)",
    re.IGNORECASE
)


def remove_fillers(text: str, nlp_pipeline: Any = None) -> str:
    pipeline = nlp_pipeline or nlp
    def _clean_sentence(s: str) -> str:
        s0 = s
        s = FILLER_PATTERN_PREFIX.sub("", s)
        # Mid occurrences – iterate until stable
        while True:
            ns = FILLER_PATTERN_MID.sub(", ", s)
            if ns == s:
                break
            s = ns
        # Suffix (lookahead version) – simple replace
        s = FILLER_PATTERN_SUFFIX.sub("", s)
        s = re.sub(r"\s+,\s+", ", ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s or s0
    try:
        d = pipeline(text)
        sent_texts = [span.text for span in d.sents]
    except Exception:
        sent_texts = re.split(r'(?<=[\.!?])\s+', text)
    cleaned = [_clean_sentence(s) for s in sent_texts if s.strip()]
    return " ".join(cleaned).replace(ZERO_WIDTH_MARK, '')

# =========================
# Self-test / Demonstration
# =========================
if __name__ == "__main__":
    sample = (
        "The complex algorithm finally produced the correct result. "
        "However, the deployment pipeline remained fragile. "
        "INTERNAL METRICS REPORT. "
        "You know the latency distribution flattened after tuning. "
        "Results were promising, but scalability is uncertain. "
        "Because the dataset was small, the variance was high. "
        "It was, frankly, a surprise."
    )

    cfg = FillerConfig(ratio=0.5, style="formal_strict", structured_plan=True, use_exponential=True, enforce_semantics=True, dry_run=False)
    result, plan = add_fillers(sample, config=cfg, seed=42)
    print("PLAN:")
    for p in plan:
        print(p)
    print("\nRESULT:\n", result.replace(ZERO_WIDTH_MARK, ''))
    print("\nUNDO:\n", remove_fillers(result))