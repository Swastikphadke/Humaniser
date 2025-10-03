import spacy
from textblob import TextBlob, Word
import random
import time
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable, Callable, Any, Sequence

# Use larger model with word vectors, fallback to small model
try:
    nlp = spacy.load("en_core_web_md")  # Has word vectors
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback to small model

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
    return s.startswith(("\"", "'", '''"''', "'"))

def _quote_dominated(sent: str) -> bool:
    total = len(sent)
    if total == 0:
        return False
    quote_chars = sent.count('"') + sent.count("'") + sent.count('"') + sent.count('"') + sent.count(''') + sent.count(''')
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

def _get_weighted_filler(
    rng: random.Random,
    fillers: List[str],
    global_freq: Dict[str, int],
    recent: List[str],
    cfg: FillerConfig
) -> Optional[str]:
    
    if not fillers:
        return None
        
    # Select the weighting function based on the config
    if cfg.use_exponential:
        weight_fn = lambda f: _exp_weight(f, global_freq, recent, cfg.repetition_recent_penalty, cfg.repetition_global_alpha)
    else:
        weight_fn = lambda f: _linear_weight(f, global_freq, recent, cfg.repetition_recent_penalty, cfg.repetition_global_penalty)
    
    # Calculate the weight (score) for each candidate filler - FIX: Call the function
    weights = [weight_fn(f) for f in fillers]
    
    # Use random.choices to pick one filler based on the calculated weights
    return rng.choices(population=fillers, weights=weights, k=1)[0]

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
    match = re.search(r'(.*?)([\.\!\?]+)?([" " "])?$', text)
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
# Main Filler API
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
# New: use lookahead; do not consume trailing punctuation/end  
FILLER_PATTERN_SUFFIX = re.compile(
    rf",\s*(?:{_FILLER_ALT})(?=(?:[\.\!\?]+(?:['\"\u201c\u201d])?\s*$)|\s*$)",
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
# EXISTING SYNONYM FUNCTIONS
# =========================

def add_synonums(text, fraction=0.25):
    doc = nlp(text) 
    
    # Step 1: Count frequencies first
    adv_freq = Counter()
    adj_freq = Counter()
    
    for token in doc:
        if token.pos_ == "ADV" and token.text.isalpha():
            adv_freq[token.text.lower()] += 1
        elif token.pos_ == "ADJ" and token.text.isalpha():
            adj_freq[token.text.lower()] += 1
    
    # Step 2: Get top fraction of frequent words
    frequent_advs = set()
    frequent_adjs = set()
    
    if adv_freq:
        top_advs = adv_freq.most_common(max(1, int(len(adv_freq) * fraction)))
        frequent_advs = set(word for word, count in top_advs)
    
    if adj_freq:
        top_adjs = adj_freq.most_common(max(1, int(len(adj_freq) * fraction)))
        frequent_adjs = set(word for word, count in top_adjs)
    
    # Store the best replacement for each token position
    token_replacements = {}
    
    # Calculate max replacements (don't replace everything)
    total_target_words = len([t for t in doc if t.pos_ in ["ADV", "ADJ"]])
    max_replacements = max(1, int(total_target_words * 0.4))  # Replace at most 40%
    replacement_count = 0
    
    # Step 3: Process tokens and find replacements
    for token in doc:
        if replacement_count >= max_replacements:
            break
            
        pos = token.pos_
        word_text = token.text.lower()
        
        # Only process frequent words
        if pos == "ADV" and word_text in frequent_advs:
            replacement = find_best_synonym(token.text, "ADV", use_randomness=True)
            if replacement:
                token_replacements[token.i] = replacement
                replacement_count += 1
                
        elif pos == "ADJ" and word_text in frequent_adjs:
            replacement = find_best_synonym(token.text, "ADJ", use_randomness=True)
            if replacement:
                token_replacements[token.i] = replacement
                replacement_count += 1

    # Reconstruct text with replacements
    new_tokens = []
    for token in doc:
        if token.i in token_replacements:
            replacement = token_replacements[token.i]
            # Preserve capitalization
            if token.text[0].isupper():
                replacement = replacement.capitalize()
            new_tokens.append(replacement + token.whitespace_)
        else:
            new_tokens.append(token.text + token.whitespace_)
    
    return ''.join(new_tokens).strip()

def find_best_synonym(word_text, pos_type, use_randomness=True):
    """Find the best synonym for a word with quality controls."""
    word = Word(word_text)
    blob1 = TextBlob(word_text)
    
    # Define POS tags to check
    if pos_type == "ADV":
        valid_tags = ['RB', 'RBR', 'RBS']
    else:  # ADJ
        valid_tags = ['JJ', 'JJR', 'JJS']
    
    candidates = []
    
    for syn in word.synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            
            # Quality control filters
            if not is_quality_synonym(synonym, word_text):
                continue
                
            blob = TextBlob(synonym)
            synonym_tags = blob.tags
            is_correct_pos = any(tag[1] in valid_tags for tag in synonym_tags)
            
            # RELAXED sentiment comparison - just needs ANY improvement
            polarity_improved = abs(blob.sentiment.polarity) >= abs(blob1.sentiment.polarity) * 0.8
            subjectivity_improved = abs(blob.sentiment.subjectivity) >= abs(blob1.sentiment.subjectivity) * 0.8
            
            if polarity_improved and subjectivity_improved and is_correct_pos:
                score = abs(blob.sentiment.polarity) + abs(blob.sentiment.subjectivity)
                candidates.append((synonym, score))
    
    if not candidates:
        return None
    
    # Sort by score and add randomness
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    if use_randomness and len(candidates) > 1:
        # 70% chance to pick best, 30% chance for variety
        if random.random() < 0.7:
            return candidates[0][0]
        else:
            # Pick from top 3 candidates randomly
            top_candidates = candidates[:min(3, len(candidates))]
            return random.choice(top_candidates)[0]
    else:
        return candidates[0][0]

def is_quality_synonym(synonym, original):
    """Quality control checks for synonyms."""
    # Single word only
    if len(synonym.split()) > 1:
        return False
    
    # Only alphabetic characters
    if not synonym.isalpha():
        return False
    
    # Similar length (not too long/short)
    if len(synonym) > len(original) + 4 or len(synonym) < 2:
        return False
    
    # Not the same word
    if synonym.lower() == original.lower():
        return False
    
    # Avoid very uncommon words (basic check)
    if len(synonym) > 12:  # Very long words are often uncommon
        return False
    
    return True

def enhanced_pronoun_check_from_docs(doc1, doc2):
    """Checks for a valid pronoun reference using pre-parsed spaCy docs."""
    first_word_of_doc2 = doc2[0]

    # Find subject of first sentence
    subject1 = None
    for token in doc1:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject1 = token
            break
    
    if not subject1 or first_word_of_doc2.pos_ != "PRON":
        return False
    
    pronoun = first_word_of_doc2.text.lower()
    
    # Method 1: Use Named Entity Recognition (most reliable)
    if subject1.ent_type_ == "PERSON":
        return pronoun in ["he", "she", "they"]
    elif subject1.ent_type_ in ["ORG", "GPE", "PRODUCT", "EVENT"]:
        return pronoun in ["it", "they"]
    
    # Method 2: Use spaCy's built-in gender/animacy detection
    if subject1.morph:
        animacy = subject1.morph.get("Animacy")
        if animacy and "Anim" in animacy:  # Animate entities
            return pronoun in ["he", "she", "they", "it"]
        elif animacy and "Inan" in animacy:  # Inanimate entities
            return pronoun in ["it", "they"]
    
    # Method 3: Pure grammatical analysis
    if subject1.tag_ in ["NNS", "NNPS"]:  # Plural nouns
        return pronoun in ["they", "it"]
    elif subject1.tag_ in ["NN", "NNP"]:  # Singular nouns
        return pronoun in ["it", "he", "she"]
    
    # Default: Allow "it" for most nouns, be restrictive with gendered pronouns
    if pronoun == "it":
        return subject1.pos_ in ["NOUN", "PROPN"]
    elif pronoun == "they":
        return subject1.tag_ in ["NNS", "NNPS"] or subject1.pos_ in ["NOUN", "PROPN"]
    elif pronoun in ["he", "she"]:
        return subject1.ent_type_ == "PERSON"
    
    return False

def detect_sentence_relationship(doc1, doc2):
    """
    Detects relationship using a refined 'Action -> State' pattern.
    Prevents misinterpreting descriptive adjectives as causal.
    """
    # Check for the "Action -> State" grammatical pattern
    has_action_verb = any(t.pos_ == "VERB" and t.dep_ == "ROOT" and t.lemma_ not in ["be", "have", "seem"] for t in doc1)
    is_state_description = False
    state_adjective = None
    if len(doc2) > 1 and doc2[1].lemma_ == "be":
        adj_token = next((t for t in doc2 if t.pos_ == "ADJ"), None)
        if adj_token:
            is_state_description = True
            state_adjective = adj_token

    if not (has_action_verb and is_state_description):
        if doc1.has_vector and doc2.has_vector and doc1.similarity(doc2) > 0.65:
            return "strong_continuation"
        return "continuation"

    # If the pattern matches, perform the semantic check
    subject_noun = next((t for t in doc1 if t.dep_ in ("nsubj", "nsubjpass")), None)
    action_verb = next((t for t in doc1 if t.dep_ == "ROOT"), None)

    if not (subject_noun and action_verb and state_adjective and 
            subject_noun.has_vector and action_verb.has_vector and state_adjective.has_vector):
        return "continuation"

    sim_adj_to_verb = state_adjective.similarity(action_verb)
    sim_adj_to_subj = state_adjective.similarity(subject_noun)
    
    # If the adjective is much more related to the verb, it's descriptive, not causal.
    if sim_adj_to_verb > sim_adj_to_subj + 0.15:
        return "continuation"
    else:
        return "causal"

def get_smart_connector_from_docs(doc1, doc2):
    """Get connector using the improved relationship logic."""
    relationship = detect_sentence_relationship(doc1, doc2)
    
    if relationship == "causal":
        connectors = [" because", " as"]
        weights = [70, 30]  # Prefer "because" for clarity
        return random.choices(connectors, weights=weights)[0]
    
    # For continuation relationships
    has_action_verb = any(t.pos_ == "VERB" and t.dep_ == "ROOT" and t.lemma_ not in ["be", "have", "seem"] for t in doc1)
    
    if has_action_verb:
        connectors = [", and", ";"]
        weights = [70, 30]
    else:
        connectors = [", and", ";", ", while"]
        weights = [50, 25, 25]
        
    # Boost semicolon for strong continuation
    if relationship == "strong_continuation":
        try:
            semicolon_index = connectors.index(";")
            weights[semicolon_index] += 30
        except ValueError:
            pass
            
    return random.choices(connectors, weights=weights)[0]

def get_subject_phrase(main_subject_token):
    """Returns the text of the full phrase connected to the subject token using .subtree."""
    return "".join(t.text_with_ws for t in sorted(main_subject_token.subtree)).strip()

def create_relative_clause(doc1, doc2, first_subject):
    """Create relative clause with improved person detection."""
    if not first_subject:
        return None
    
    PERSON_LIKE_ROLES = {"ceo", "doctor", "scientist", "manager", "developer", "artist", "writer", "chef", "director", "president"}

    if len(doc2) <= 1:
        return None
    
    second_content = doc2[1:].text.strip().rstrip('.')
    if not second_content:
        return None
        
    subject_phrase = get_subject_phrase(first_subject)

    # Enhanced person detection
    is_person = False
    if any(t.ent_type_ == "PERSON" for t in first_subject.subtree):
        is_person = True
    elif first_subject.lemma_.lower() in PERSON_LIKE_ROLES:
        is_person = True
    elif first_subject.has_vector:
        try:
            person_token = nlp("person")
            if person_token.has_vector and first_subject.similarity(person_token) > 0.4:
                is_person = True
        except:
            pass
            
    relative_pronoun = "who" if is_person else "which"
    
    # Get rest of first sentence after subject phrase
    try:
        last_subject_token_index = max(t.i for t in first_subject.subtree)
        first_rest = doc1[last_subject_token_index + 1:].text.strip()
    except:
        first_rest = ""
    
    # Format the relative clause
    if first_rest:
        return f"{subject_phrase}, {relative_pronoun} {second_content}, {first_rest.rstrip('.?!')}."
    else:
        return f"{subject_phrase}, {relative_pronoun} {second_content}."

def create_connector_merge(first_sentence, second_sentence, doc1, doc2):
    """Create connector merge with proper capitalization."""
    connector = get_smart_connector_from_docs(doc1, doc2)
    
    if connector == ";":
        # For semicolon, lowercase the next sentence
        return first_sentence.strip('.?!') + connector + " " + second_sentence[0].lower() + second_sentence[1:]
    elif connector.startswith(" "):
        return first_sentence.strip('.?!') + connector + " " + second_sentence[0].lower() + second_sentence[1:]
    else:
        return first_sentence.strip('.?!') + connector + " " + second_sentence[0].lower() + second_sentence[1:]

def get_merge_strategy():
    """Randomly choose merge strategy with weighted probabilities."""
    strategies = [("connector", 60), ("relative_clause", 20), ("no_merge", 20)]
    strategy_names, weights = zip(*strategies)
    return random.choices(strategy_names, weights=weights)[0]

def combine_sentences_recursive(text, short_threshold=6):
    """Recursively combine sentences with improved logic."""
    blob = TextBlob(text)
    sentences = list(blob.sentences)
    
    new_sentences = []
    changes_made = False
    
    i = 0
    while i < len(sentences):
        if i >= len(sentences) - 1:
            new_sentences.append(str(sentences[i]))
            i += 1
            continue

        # Parse sentences once
        doc1 = nlp(str(sentences[i]))
        doc2 = nlp(str(sentences[i+1]))
        
        # Use TextBlob word count for consistency
        length1 = len(sentences[i].words)
        length2 = len(sentences[i+1].words)
        
        # Check conditions with consistent length counting
        if (length1 <= short_threshold and length2 <= short_threshold and 
            doc2[0].pos_ == "PRON" and 
            enhanced_pronoun_check_from_docs(doc1, doc2)):
            
            strategy = get_merge_strategy()
            combined = None
            
            if strategy == "relative_clause":
                subject = next((t for t in doc1 if t.dep_ in ("nsubj", "nsubjpass")), None)
                combined = create_relative_clause(doc1, doc2, subject)
                if not combined:
                    strategy = "connector" 
                
            if strategy == "connector":
                combined = create_connector_merge(str(sentences[i]), str(sentences[i+1]), doc1, doc2)
            
            if strategy != "no_merge" and combined:
                new_sentences.append(combined)
                changes_made = True
                i += 2
            else:
                new_sentences.append(str(sentences[i]))
                i += 1
        else:
            new_sentences.append(str(sentences[i]))
            i += 1

    result_text = ' '.join(new_sentences)
    
    if changes_made:
        return combine_sentences_recursive(result_text, short_threshold)
    else:
        return result_text

# ================== NEW SENTENCE SPLITTING FUNCTIONS ==================

def pre_filter_split_candidates(doc, min_part_length=4):
    """
    Quick pre-filtering to find only plausible split points.
    Returns list of token indices that could be good split points.
    """
    candidates = []
    
    for i in range(min_part_length, len(doc) - min_part_length + 1):
        token = doc[i] if i < len(doc) else None
        prev_token = doc[i - 1] if i > 0 else None
        
        # Only analyze tokens that could realistically be split points
        is_candidate = False
        
        if prev_token and token:
            # After punctuation followed by conjunctions/connectors
            if (prev_token.text in [',', ';', ':'] and 
                token.pos_ in ['CCONJ', 'SCONJ', 'ADV']):
                is_candidate = True
            
            # After punctuation followed by relative pronouns
            elif (prev_token.text == ',' and 
                token.lemma_ in ['which', 'who', 'that']):
                is_candidate = True
            
            # Direct after semicolons (natural sentence breaks)
            elif prev_token.text == ';':
                is_candidate = True
            
            # At the start of new clauses (identified by subjects)
            elif token.dep_ in ['nsubj', 'nsubjpass'] and prev_token.text == ',':
                is_candidate = True
            
            # After certain prepositions in long prepositional phrases
            elif (token.pos_ == 'ADP' and 
                prev_token.text in [',', ';']):
                is_candidate = True
        
        if is_candidate:
            candidates.append(i)
    
    return candidates

def analyze_split_quality(doc, split_index, target_middle):
    """
    Analyzes the quality of a potential split point using linguistic features.
    Returns a score (higher = better split).
    """
    if split_index <= 0 or split_index >= len(doc):
        return 0
    
    first_part_len = split_index
    second_part_len = len(doc) - split_index
    
    # Base score starts at 0
    score = 0
    
    # 1. BALANCE SCORE: Prefer splits closer to the middle
    distance_from_middle = abs(split_index - target_middle)
    max_distance = max(target_middle, len(doc) - target_middle)
    balance_score = (max_distance - distance_from_middle) / max_distance * 100
    score += balance_score
    
    # 2. MINIMUM LENGTH PENALTY: Heavily penalize if either part is too short
    min_length = 4
    if first_part_len < min_length or second_part_len < min_length:
        score -= 200  # Heavy penalty for imbalanced splits
    
    # 3. LINGUISTIC BOUNDARY SCORE: Analyze the split point linguistically
    split_token = doc[split_index] if split_index < len(doc) else None
    prev_token = doc[split_index - 1] if split_index > 0 else None
    
    if prev_token and split_token:
        # Coordinating conjunctions are excellent split points
        if (prev_token.text == ',' and 
            split_token.pos_ == 'CCONJ'):  # and, but, or, etc.
            score += 50
            
        # Subordinating conjunctions are good split points
        elif (prev_token.text == ',' and 
            split_token.pos_ == 'SCONJ'):  # because, although, etc.
            score += 40
            
        # Semicolons are natural breaks
        elif prev_token.text == ';':
            score += 45
            
        # Relative pronouns after commas
        elif (prev_token.text == ',' and 
            split_token.lemma_ in ['which', 'who', 'that']):
            score += 35
            
        # Adverbial connectors
        elif (prev_token.text == ',' and 
            split_token.pos_ == 'ADV' and 
            split_token.lemma_ in ['however', 'moreover', 'furthermore', 'therefore']):
            score += 30
    
    # 4. SYNTACTIC DEPENDENCY SCORE: Prefer splits that don't break tight dependencies
    if split_token and prev_token:
        # Avoid splitting between tokens with strong dependencies
        strong_deps = ['det', 'amod', 'compound', 'aux', 'auxpass']
        
        if (split_token.dep_ in strong_deps and split_token.head == prev_token) or \
        (prev_token.dep_ in strong_deps and prev_token.head == split_token):
            score -= 30  # Penalty for breaking strong dependencies
    
    # 5. CLAUSE BOUNDARY SCORE: Prefer splits at clause boundaries
    if split_index > 0 and split_index < len(doc):
        # Check if we're at the start of a new clause
        for token in doc[split_index:split_index+3]:  # Look ahead a few tokens
            if token.dep_ in ['nsubj', 'nsubjpass']:  # New subject = new clause
                score += 25
                break
    
    # 6. PUNCTUATION CONTEXT SCORE
    if prev_token:
        if prev_token.text in [',', ';', ':']:
            score += 20  # Good to split after punctuation
        elif prev_token.text in ['.', '!', '?']:
            score -= 50  # Don't split after sentence endings

    #7. Penalize splits before conjunctions/prepositions if there is no comma
    if split_token.pos_ in ['SCONJ', 'ADP'] and prev_token.text != ',':
        score -= 1000
        
    return score

def find_best_split_points(doc, min_part_length=4):
    """
    Optimized version: Pre-filter candidates then analyze only the promising ones.
    Returns list of (index, score) tuples sorted by score (best first).
    """
    target_middle = len(doc) // 2
    
    # OPTIMIZATION 1: Pre-filter to get only plausible candidates
    candidate_indices = pre_filter_split_candidates(doc, min_part_length)
    
    # If no candidates found, fallback to analyzing all positions (rare case)
    if not candidate_indices:
        candidate_indices = list(range(min_part_length, len(doc) - min_part_length + 1))
    
    scored_candidates = []
    
    # OPTIMIZATION 2: Only analyze the pre-filtered candidates
    for i in candidate_indices:
        score = analyze_split_quality(doc, i, target_middle)
        if score > 0:  # Only consider positive scores
            scored_candidates.append((i, score))
    
    # Sort by score (highest first)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates

def clean_sentence_ending(text):
    """Ensures proper sentence ending punctuation."""
    text = text.strip()
    
    # Remove trailing commas and semicolons
    while text.endswith((',', ';')):
        text = text[:-1].strip()
    
    # Add period if no ending punctuation
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def clean_sentence_beginning(text):
    """Ensures proper sentence beginning capitalization."""
    text = text.strip()
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    
    return text

def split_long_sentence(sentence_text, max_length=15, min_part_length=4):
    """
    Intelligently splits long sentences using linguistic analysis.
    Returns a list of sentences.
    """
    doc = nlp(sentence_text)
    
    # Check if splitting is needed
    if len(doc) <= max_length:
        return [sentence_text]
    
    # Find the best split points (now optimized)
    split_candidates = find_best_split_points(doc, min_part_length)
    
    if not split_candidates:
        return [sentence_text]  # No good split found
    
    # Use the best split point
    best_split_index, best_score = split_candidates[0]
    
    # Extract the two parts
    first_part = doc[:best_split_index]
    second_part = doc[best_split_index:]
    
    # Clean up the text
    first_text = first_part.text.strip()
    second_text = second_part.text.strip()
    
    # Handle punctuation intelligently
    first_text = clean_sentence_ending(first_text)
    second_text = clean_sentence_beginning(second_text)
    
    return [first_text, second_text]

def split_text_recursively(text, max_sentence_length=15, min_part_length=4):
    """
    Optimized version using spaCy's sentence segmentation for consistency.
    Recursively splits all long sentences in a text.
    """
    # Use spaCy for sentence segmentation instead of TextBlob
    doc = nlp(text)
    sentences = list(doc.sents)
    
    new_sentences = []
    changes_made = False

    if random.random() < 0.8:
        for sentence in sentences:
            sentence_str = sentence.text.strip()
            sentence_doc = nlp(sentence_str)  # Re-parse individual sentence for accuracy
        
            if len(sentence_doc) > max_sentence_length:
                split_results = split_long_sentence(sentence_str, max_sentence_length, min_part_length)
                new_sentences.extend(split_results)
                if len(split_results) > 1:
                    changes_made = True
            else:
                new_sentences.append(sentence_str)
    
    # If we made changes, recursively check again (some splits might still be too long)
        if changes_made:
            result_text = ' '.join(new_sentences)
        # Check if any sentence is still too long using spaCy
            result_doc = nlp(result_text)
            needs_more_splitting = any(len(nlp(sent.text)) > max_sentence_length for sent in result_doc.sents)
        
            if needs_more_splitting:
                return split_text_recursively(result_text, max_sentence_length, min_part_length)
    
        return ' '.join(new_sentences)

# --- REVISED, HIGH-QUALITY NORMALIZER ---
def normalize_sentence_lengths(text, target_min=15, target_max=25, ideal_range=(18, 22)):
    """
    Unified sentence length normalizer with multi-pass convergence.
    Runs splitting and merging passes until no more beneficial changes can be made.
    """
    last_text = ""
    current_text = text
    
    # Run for a maximum of 3 passes to prevent infinite loops
   # for pass_num in range(3):
    if current_text == last_text:
        return current_text  # No changes in this pass, we've converged
    last_text = current_text
        
        # Pass 1: Split any sentences that are too long
    split_text = split_text_recursively(
        current_text, 
        max_sentence_length=target_max
        )
        
        # Pass 2: Merge any sentences that are too short
    current_text = combine_sentences_recursive(
            split_text, 
            short_threshold=target_min
        )
    
    return current_text

def combine_two_sentences(sent1, sent2):
    """
    Intelligently combine two sentences using the existing combination logic.
    This is a helper function for the unified approach.
    """
    # Create documents for analysis
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)
    
    # Use the existing relationship detection and combination logic
    relationship = detect_sentence_relationship(doc1, doc2)
    
    # Get appropriate connector and combine
    connector = get_smart_connector_from_docs(doc1, doc2)
    if not connector:
        return None
        
    # Perform pronoun validation
    if not enhanced_pronoun_check_from_docs(doc1, doc2):
        return None
        
    return sent1 + connector + " " + sent2

# ================== MAIN HUMANIZE FUNCTION ==================

def clean_text_output(text):
    """Clean up formatting issues in the final output."""
    # Fix stray punctuation patterns
    text = re.sub(r'\s*\.\s*;\s*', '. ', text)  # Fix ". ;" → ". "
    text = re.sub(r'\s*,\s*and\s+', ', and ', text)  # Fix ", and " spacing
    text = re.sub(r'\s+', ' ', text)  # Fix multiple spaces
    text = re.sub(r'\s+([.!?])', r'\1', text)  # Fix space before punctuation
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after punctuation
    
    # Fix sentence fragments that start with conjunctions
    text = re.sub(r'\.\s*(and|but|or|so)\s+([A-Z])', r', \1 \2', text)
    
    return text.strip()

def humanize_text(text, mode="unified", synonym_fraction=0.3, target_min=15, target_max=25, 
                 filler_style="neutral", filler_ratio=0.0, apply_fillers=False):
    """
    Main function to humanize AI-generated text using a unified approach with optional fillers.
    """
    try:
        if mode == "synonyms":
            result = add_synonums(text, fraction=synonym_fraction)
        elif mode == "legacy":
            # Old behavior: apply synonyms, then merge, then split
            result = add_synonums(text, fraction=synonym_fraction)
            result = combine_sentences_recursive(result, short_threshold=6)
            result = split_text_recursively(result, max_sentence_length=15)
        elif mode == "unified":
            # New unified approach: apply synonyms, then normalize lengths intelligently
            result = add_synonums(text, fraction=synonym_fraction)
            result = normalize_sentence_lengths(result, target_min=target_min, target_max=target_max)
        else:
            raise ValueError("Mode must be 'synonyms', 'legacy', or 'unified'")
        
        # Apply fillers if requested
        if apply_fillers and filler_ratio > 0:
            filler_config = FillerConfig(
                ratio=filler_ratio,
                style=filler_style,
                structured_plan=False,
                enforce_semantics=True
            )
            result = add_fillers(result, config=filler_config)
        
        # Clean up the final output
        result = clean_text_output(result)
        
        return result
    
    except Exception as e:
        print(f"Error in humanize_text: {e}")
        return text  # Return original text if processing fails

# Example usagecas
if __name__ == "__main__":
    text = '''
The library stood at the edge of the city, older than the streets that surrounded it, older than the city itself. Its gates were wrought of black iron, twisted into shapes that seemed to shift when stared at too long. Inside, the shelves stretched endlessly upward, vanishing into a darkness no candle could pierce. The air was rich with dust, parchment, and ink, yet also with something heavier—like the weight of thoughts too vast for human minds. Books whispered when unopened, their voices weaving together in a thousand conversations. Some volumes glowed faintly, others wept ink that pooled on the floor. Scholars entered seeking wisdom, but not all left with sanity intact. It was said the library remembered every reader, and that the more one learned, the more the library learned in return.
'''
     # Apply unified humanization approach with fillers
    humanized_text = humanize_text(
        text, 
        mode="unified", 
        synonym_fraction=0.2, 
        target_min=8, 
        target_max=22,
        filler_style="casual",
        filler_ratio=0.1,
        apply_fillers=True
    )
    print(humanized_text)
