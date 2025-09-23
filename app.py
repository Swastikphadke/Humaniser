import spacy
from textblob import TextBlob, Word
import random
import time
from collections import Counter

# Use larger model with word vectors, fallback to small model
try:
    nlp = spacy.load("en_core_web_md")  # Has word vectors
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback to small model

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
        return first_sentence.strip('.?!') + connector + " " + second_sentence.lower()
    elif connector.startswith(" "):
        return first_sentence.strip('.?!') + connector + " " + second_sentence.lower()
    else:
        return first_sentence.strip('.?!') + connector + " " + second_sentence.lower()

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

def normalize_sentence_lengths(text, target_min=15, target_max=25, ideal_range=(18, 22)):
    """
    Unified sentence length normalizer that intelligently adjusts sentence lengths
    toward an ideal range, avoiding the merge-then-split conflict.
    
    Args:
        text: Input text to normalize
        target_min: Minimum acceptable sentence length (words)
        target_max: Maximum acceptable sentence length (words) 
        ideal_range: Tuple of (min_ideal, max_ideal) word counts
    
    Returns:
        Text with normalized sentence lengths
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return text
        
    i = 0
    while i < len(sentences):
        current_sent = sentences[i]
        word_count = len(current_sent.split())
        
        # If sentence is too long, split it
        if word_count > target_max:
            split_result = split_long_sentence(current_sent)
            if len(split_result) > 1:
                # Replace current sentence with split results
                sentences[i:i+1] = split_result
                # Don't increment i, recheck the first new sentence
                continue
                
        # If sentence is too short, try to merge with next
        elif word_count < target_min and i + 1 < len(sentences):
            next_sent = sentences[i + 1]
            next_word_count = len(next_sent.split())
            combined_length = word_count + next_word_count
            
            # Only merge if result won't be too long
            if combined_length <= target_max:
                combined = combine_two_sentences(current_sent, next_sent)
                if combined and combined != current_sent + " " + next_sent:
                    # Replace both sentences with combined result
                    sentences[i:i+2] = [combined]
                    # Don't increment i, recheck the new combined sentence
                    continue
        
        # If in ideal range or no changes possible, move to next
        i += 1
    
    return " ".join(sentences)

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
    if relationship == "none":
        return None
        
    # Get appropriate connector and combine
    connector = get_smart_connector_from_docs(doc1, doc2)
    if not connector:
        return None
        
    # Perform pronoun validation
    if not enhanced_pronoun_check_from_docs(doc1, doc2, connector):
        return None
        
    return sent1 + " " + connector + " " + sent2

def split_long_sentence(sentence):
    """
    Split a single long sentence using the existing splitting logic.
    This is a helper function for the unified approach.
    """
    doc = nlp(sentence)
    tokens = [token for token in doc if not token.is_space]
    
    if len(tokens) <= 15:  # Not long enough to split
        return [sentence]
        
    # Use existing split logic with pre-filtering
    split_candidates = pre_filter_split_candidates(doc, min_part_length=4)
    if not split_candidates:
        return [sentence]
        
    target_middle = len(tokens) // 2
    best_split = min(split_candidates, 
                    key=lambda x: abs(x - target_middle))
    
    # Create the splits
    first_part = " ".join([t.text for t in tokens[:best_split]])
    second_part = " ".join([t.text for t in tokens[best_split:]])
    
    # Clean up and return
    first_part = first_part.strip()
    second_part = second_part.strip()
    
    if not second_part:
        return [sentence]
        
    # Capitalize second part if needed
    if second_part and second_part[0].islower():
        second_part = second_part[0].upper() + second_part[1:]
        
    return [first_part + ".", second_part]

# ================== MAIN HUMANIZE FUNCTION ==================

def humanize_text(text, mode="unified", synonym_fraction=0.3, target_min=15, target_max=25):
    """
    Main function to humanize AI-generated text using a unified approach.
    
    Args:
        text: Input text to humanize
        mode: "unified" (recommended), "synonyms", "legacy" for old behavior
        synonym_fraction: Fraction of frequent words to replace (default 0.3)
        target_min: Minimum acceptable sentence length in words (default 15)
        target_max: Maximum acceptable sentence length in words (default 25)
    """
    if mode == "synonyms":
        return add_synonums(text, fraction=synonym_fraction)
    elif mode == "legacy":
        # Old behavior: apply synonyms, then merge, then split
        text = add_synonums(text, fraction=synonym_fraction)
        text = combine_sentences_recursive(text, short_threshold=6)
        text = split_text_recursively(text, max_sentence_length=15)
        return text
    elif mode == "unified":
        # New unified approach: apply synonyms, then normalize lengths intelligently
        text = add_synonums(text, fraction=synonym_fraction)
        text = normalize_sentence_lengths(text, target_min=target_min, target_max=target_max)
        return text
    else:
        raise ValueError("Mode must be 'synonyms', 'legacy', or 'unified'")

# Example usage
if __name__ == "__main__":
    # Sample text for demonstration
    text = '''
    The village of Eldenwood sat quietly at the edge of a vast forest, where the trees grew so tall their tops seemed to brush the clouds. Most villagers avoided the forest after dusk, for it was said to be alive with whispers and shadows that didn't belong to any living thing.

    One evening, a young girl named Aria lingered too long by the riverbank. The sun slipped below the horizon, and the familiar path home dissolved into darkness. Instead of fear, curiosity bloomed in her chest. She stepped deeper into the woods, drawn by a soft golden glow flickering between the trunks.

    The light led her to a clearing where an ancient oak stood, hollow at its base. Inside, tiny orbs of light swirled like stars in miniature. They pulsed gently, as if breathing. Aria reached out, and one orb settled into her palm, warm as sunlight.

    Suddenly, the whispers grew louder—but they weren't frightening. They carried words, old and kind: "Guardian chosen."

    The next morning, when Aria returned to the village, her eyes shimmered faintly with golden light. The people of Eldenwood noticed. The forest, long feared, had chosen her as its keeper.
    '''
    
    # Apply unified humanization approach (recommended)
    humanized_text = humanize_text(text, mode="unified", synonym_fraction=0.3, target_min=15, target_max=25)
    print(humanized_text)
