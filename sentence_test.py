from textblob import TextBlob
import spacy
import random

# Use larger model with word vectors, fallback to small model
try:
    nlp = spacy.load("en_core_web_md")  # Has word vectors
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback to small model
    print("Warning: Using small model without word vectors. Consider installing en_core_web_md")

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
            elif token.dep_ in ['nsubj', 'nsubjpass']:
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
    
    # Add period if no ending punctuation
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def split_text_recursively(text, max_sentence_length=15, min_part_length=4):
    """
    Optimized version using spaCy's sentence segmentation for consistency.
    Recursively splits all long sentences in a text.
    """
    # OPTIMIZATION: Use spaCy for sentence segmentation instead of TextBlob
    doc = nlp(text)
    sentences = list(doc.sents)
    
    new_sentences = []
    changes_made = False
    
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

# Performance testing function
def test_performance():
    """Test the performance improvement of pre-filtering."""
    import time
    
    # Create a very long sentence for testing
    long_sentence = ("The algorithm ran through millions of data points, analyzing each one carefully, " +
                    "but it failed to find a conclusive pattern, which was disappointing, " +
                    "because the researchers had spent months preparing the data, " +
                    "and they had high hopes for breakthrough results, " +
                    "however the computational complexity proved to be much higher than expected, " +
                    "so they decided to try a different approach, " +
                    "which involved machine learning techniques that were more suitable for this type of problem.")
    
    doc = nlp(long_sentence)
    print(f"Testing sentence with {len(doc)} tokens")
    
    # Test optimized version
    start_time = time.time()
    candidates = find_best_split_points(doc)
    optimized_time = time.time() - start_time
    
    print(f"Optimized version: {optimized_time:.4f} seconds")
    print(f"Found {len(candidates)} candidates")
    if candidates:
        print(f"Best split score: {candidates[0][1]:.1f}")

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