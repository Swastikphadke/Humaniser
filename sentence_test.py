from textblob import TextBlob
import spacy
import random

# Use larger model with word vectors, fallback to small model
try:
    nlp = spacy.load("en_core_web_md")  # Has word vectors
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback to small model
    print("Warning: Using small model without word vectors. Consider installing en_core_web_md")

# ...existing code...

FILLER_PHRASES = [
    "to be honest", "frankly", "actually", "basically", "in fact",
    "as it turns out", "interestingly", "after all", "for the most part",
    "generally speaking", "quite honestly", "you know", "truthfully",
    "if you think about it", "in a way"
]

def _choose_filler(rng):
    return rng.choice(FILLER_PHRASES)

def _insert_filler_into_sentence(sent_text, filler, rng):
    sent_text = sent_text.strip()
    if not sent_text:
        return sent_text
    # Avoid altering very short sentences
    if len(sent_text.split()) < 4:
        return sent_text
    # Do not prepend to questions or exclamations (optional heuristic)
    lower = sent_text.lower()
    # Decide insertion position: 0 = prefix, 1 = mid, 2 = suffix
    mode_weights = [0.55, 0.25, 0.20]
    mode = rng.choices([0,1,2], weights=mode_weights, k=1)[0]

    # Ensure filler has proper form
    filler_form = filler
    # Capitalize if going at start
    if mode == 0:
        filler_form = filler_form[0].upper() + filler_form[1:]
        # If sentence already starts with a capital and filler ends w/o comma, add comma after
        return f"{filler_form}, {sent_text[0].lower() + sent_text[1:] if sent_text[:1].isalpha() else sent_text}"

    words = sent_text.split()
    if mode == 1 and len(words) >= 8:
        # Mid insertion near a clause boundary (after first comma if exists, else near 1/3)
        if ',' in sent_text:
            first_comma_idx = sent_text.find(',')
            before = sent_text[:first_comma_idx+1].rstrip()
            after = sent_text[first_comma_idx+1:].lstrip()
            return f"{before} {filler}, {after}"
        insert_index = max(2, len(words)//3)
        return " ".join(words[:insert_index] + [filler + ","] + words[insert_index:])
    # Suffix insertion – before final punctuation
    if sent_text[-1] in ".!?":
        core = sent_text[:-1].rstrip()
        punct = sent_text[-1]
    else:
        core = sent_text
        punct = ""
    return f"{core}, {filler}{punct}"

def add_fillers(text, ratio=0.2, seed=None):
    """
    Insert filler discourse markers into approximately ratio fraction of sentences.
    ratio: 0–1
    """
    if ratio <= 0:
        return text
    rng = random.Random(seed)
    doc = nlp(text)
    sentences = [s.text for s in doc.sents]
    if not sentences:
        return text
    target_count = max(1, int(len(sentences) * ratio))
    # Select indices (avoid already very short sentences)
    candidate_indices = [i for i, s in enumerate(sentences) if len(s.split()) >= 4]
    rng.shuffle(candidate_indices)
    chosen = set(candidate_indices[:target_count])
    out = []
    for i, s in enumerate(sentences):
        if i in chosen:
            filler = _choose_filler(rng)
            out.append(_insert_filler_into_sentence(s, filler, rng))
        else:
            out.append(s.strip())
    return " ".join(out)

def humanize_text(
    text,
    mode="unified",
    synonym_fraction=0.3,
    sentence_threshold=6,
    max_sentence_length=22,
    target_min=15,
    target_max=25,
    filler_ratio=0.2,
    apply_fillers=True
):
    # ...existing logic before final return...
    # Assume unified normalization + synonyms already applied earlier in this function.
    result = text  # replace with existing pipeline variable
    # ...existing code...
    if apply_fillers and filler_ratio > 0:
        result = add_fillers(result, ratio=filler_ratio)
    return result

# ...existing code...

# Test cases
test_sentences = [
    "The algorithm ran through millions of data points, but it failed to find a conclusive pattern.",
    "This is a short sentence.",
    "The company, which was founded in 1995, has grown significantly over the years, and it now employs over 500 people worldwide, making it one of the largest employers in the region.",
    "She studied hard for the exam; however, she was still nervous about the results because the subject was particularly challenging.",
    "Although the weather was terrible, we decided to go hiking because we had planned this trip for months, and we didn't want to waste the opportunity.",
    "The research team discovered that the new drug was effective in treating the disease, but they also found several side effects that needed further investigation."
]

print("=== OPTIMIZED ADAPTIVE SENTENCE SPLITTING TESTS ===\n")

# Run performance test first
test_performance()
print("\n" + "="*80 + "\n")

for i, sentence in enumerate(test_sentences, 1):
    print(f"Test {i}: {sentence}")
    print(f"Original length: {len(nlp(sentence))} words")
    
    # Show split candidates with scores
    doc = nlp(sentence)
    if len(doc) > 15:
        candidates = find_best_split_points(doc)
        print("Top 3 split candidates:")
        for j, (idx, score) in enumerate(candidates[:3]):
            first_part = doc[:idx].text
            second_part = doc[idx:].text
            print(f"  {j+1}. Score: {score:.1f} | Split at token {idx}")
            print(f"     '{first_part}' + '{second_part}'")
    
    result = split_long_sentence(sentence)
    print(f"Final result:")
    for j, part in enumerate(result):
        part_doc = nlp(part)
        print(f"  Part {j+1}: {len(part_doc)} words - '{part}'")
    print("-" * 80)

# Place in a separate test block or script
sample = "The model performed well on benchmarks, but it struggled on nuanced reasoning tasks. Results were promising. Further evaluation will be necessary to confirm robustness across domains."
print(add_fillers(sample, ratio=0.4, seed=42))