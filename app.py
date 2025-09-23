import spacy
from textblob import TextBlob, Word
import random
from collections import Counter

# Use larger model with word vectors, fallback to small model
try:
    nlp = spacy.load("en_core_web_md")  # Has word vectors
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback to small model
    print("Warning: Using small model without word vectors. Consider installing en_core_web_md")

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
    """Improved relationship detection with better causal logic."""
    # Check for basic "Action -> State" pattern
    has_action_verb = any(t.pos_ == "VERB" and t.dep_ == "ROOT" and t.lemma_ not in ["be", "have", "seem"] for t in doc1)
    
    # Check if second sentence describes a state with "be" + adjective
    is_state_description = False
    state_adjective = None
    if len(doc2) > 1 and doc2[1].lemma_ == "be":
        adj_token = next((t for t in doc2 if t.pos_ == "ADJ"), None)
        if adj_token:
            is_state_description = True
            state_adjective = adj_token

    # If basic pattern doesn't match, check for strong continuation
    if not (has_action_verb and is_state_description):
        try:
            if doc1.has_vector and doc2.has_vector and doc1.similarity(doc2) > 0.65:
                return "strong_continuation"
        except:
            pass
        return "continuation"

    # Enhanced semantic check for causality
    subject_noun = next((t for t in doc1 if t.dep_ in ("nsubj", "nsubjpass")), None)
    action_verb = next((t for t in doc1 if t.dep_ == "ROOT"), None)

    if not (subject_noun and action_verb and state_adjective):
        return "continuation"

    # Check for emotional/psychological states that can be caused by actions
    emotion_adjectives = {"happy", "sad", "tired", "excited", "angry", "confident", "worried", "proud", "embarrassed", "surprised"}
    physical_adjectives = {"fast", "slow", "strong", "weak", "big", "small", "heavy", "light"}
    
    adj_lemma = state_adjective.lemma_.lower()
    
    # Emotions can be caused by actions
    if adj_lemma in emotion_adjectives:
        # Special check: avoid illogical causality
        if action_verb.lemma_ in ["run", "walk", "move"] and adj_lemma in ["fast", "slow"]:
            return "continuation"  # "ran fast" is descriptive, not causal
        return "causal"
    
    # Physical descriptions are usually not causal
    if adj_lemma in physical_adjectives:
        return "continuation"
    
    # Use word vectors for semantic similarity if available
    try:
        if (subject_noun.has_vector and action_verb.has_vector and state_adjective.has_vector):
            sim_adj_to_verb = state_adjective.similarity(action_verb)
            sim_adj_to_subj = state_adjective.similarity(subject_noun)
            
            # If adjective is more related to the action than subject, it's descriptive
            if sim_adj_to_verb > sim_adj_to_subj + 0.15:
                return "continuation"
            else:
                return "causal"
    except:
        pass
    
    return "continuation"

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

def humanize_text(text, mode="both", synonym_fraction=0.3, sentence_threshold=6):
    """
    Main function to humanize AI-generated text.
    
    Args:
        text: Input text to humanize
        mode: "synonyms", "sentences", or "both"
        synonym_fraction: Fraction of frequent words to replace (default 0.3)
        sentence_threshold: Max words per sentence to consider for merging (default 6)
    """
    if mode == "synonyms":
        return add_synonums(text, fraction=synonym_fraction)
    elif mode == "sentences":
        return combine_sentences_recursive(text, short_threshold=sentence_threshold)
    elif mode == "both":
        # Apply synonyms first, then sentence combination
        text = add_synonums(text, fraction=synonym_fraction)
        text = combine_sentences_recursive(text, short_threshold=sentence_threshold)
        return text
    else:
        raise ValueError("Mode must be 'synonyms', 'sentences', or 'both'")

# Test the functions
text = '''
The village of Eldenwood sat quietly at the edge of a vast forest, where the trees grew so tall their tops seemed to brush the clouds. Most villagers avoided the forest after dusk, for it was said to be alive with whispers and shadows that didn't belong to any living thing.

One evening, a young girl named Aria lingered too long by the riverbank. The sun slipped below the horizon, and the familiar path home dissolved into darkness. Instead of fear, curiosity bloomed in her chest. She stepped deeper into the woods, drawn by a soft golden glow flickering between the trunks.

The light led her to a clearing where an ancient oak stood, hollow at its base. Inside, tiny orbs of light swirled like stars in miniature. They pulsed gently, as if breathing. Aria reached out, and one orb settled into her palm, warm as sunlight.

Suddenly, the whispers grew louder—but they weren't frightening. They carried words, old and kind: "Guardian chosen."

The next morning, when Aria returned to the village, her eyes shimmered faintly with golden light. The people of Eldenwood noticed. The forest, long feared, had chosen her as its keeper.
'''

print("Choose an option:")
print("1. Add synonyms only")
print("2. Change sentence length only") 
print("3. Do both")

choice = input("Enter your choice (1/2/3): ")

if choice == "1":
    updated_text = humanize_text(text, mode="synonyms", synonym_fraction=0.5)
    print("\n" + "="*50)
    print("SYNONYMS ONLY:")
    print("="*50)
elif choice == "2":
    updated_text = humanize_text(text, mode="sentences", sentence_threshold=6)
    print("\n" + "="*50)
    print("SENTENCE COMBINATION ONLY:")
    print("="*50)
elif choice == "3":
    updated_text = humanize_text(text, mode="both", synonym_fraction=0.3, sentence_threshold=6)
    print("\n" + "="*50)
    print("BOTH TECHNIQUES APPLIED:")
    print("="*50)
else:
    print("Invalid choice!")
    updated_text = text

print(updated_text)
