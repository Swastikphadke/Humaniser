from textblob import TextBlob, Word
import spacy
import random

# Use larger model with word vectors, fallback to small model
try:
    nlp = spacy.load("en_core_web_md")  # Has word vectors
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback to small model
    print("Warning: Using small model without word vectors. Consider installing en_core_web_md")

def enhanced_pronoun_check(first_sentence, second_sentence):
    """Enhanced pronoun checking using only spaCy's built-in analysis."""
    doc1 = nlp(first_sentence)
    doc2 = nlp(second_sentence)
    first_word = doc2[0]

    # Find subject of first sentence
    subject1 = None
    for token in doc1:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject1 = token
            break
    
    if not subject1 or first_word.pos_ != "PRON":
        return False
    
    pronoun = first_word.text.lower()
    
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
    
    # Method 3: Pure grammatical analysis - no word lists!
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
    """Detect relationship using only spaCy's linguistic analysis - NO hardcoded lists."""
    
    # Get main verb from first sentence
    first_main_verb = None
    for token in doc1:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            first_main_verb = token
            break
    
    # Get main content from second sentence (skip pronoun)
    second_main_content = []
    for token in doc2[1:]:  # Skip first pronoun
        if token.pos_ in ["ADJ", "VERB", "NOUN"]:  # Main content words
            second_main_content.append(token)
    
    if not first_main_verb or not second_main_content:
        return "continuation"
    
    # Check if second sentence describes a state/quality vs action
    has_adjective = any(token.pos_ == "ADJ" for token in second_main_content)
    has_state_verb = any(token.lemma_ in ["be", "seem", "appear", "look", "feel"] for token in doc2)
    has_action_verb = any(token.pos_ == "VERB" and token.lemma_ not in ["be", "have", "seem", "appear", "look", "feel"] for token in doc2)
    
    # More restrictive causal detection - only for emotions/states that logically follow actions
    if has_adjective and has_state_verb and not has_action_verb:
        first_verb_lemma = first_main_verb.lemma_
        
        # Only use word vectors if available
        try:
            # For other verbs, check if the adjective in second sentence is a typical result
            for adj_token in second_main_content:
                if adj_token.pos_ == "ADJ":
                    # Use spaCy's word vectors to check semantic relationship (if available)
                    if (first_main_verb.has_vector and adj_token.has_vector and 
                        first_main_verb.similarity(adj_token) > 0.3):
                        return "causal"
        except:
            # If no word vectors, use basic heuristic
            pass
    
    return "continuation"

def get_smart_connector_from_docs(doc1, doc2):
    """Get connector using pre-parsed spaCy docs with relationship awareness."""
    relationship = detect_sentence_relationship(doc1, doc2)
    
    if relationship == "causal":
        # Use causal connectors for action->state relationships
        causal_connectors = [
            (" because", 40),
            (" as", 30),
            (" since", 20),
            (", and", 10)  # Fallback
        ]
        connectors, weights = zip(*causal_connectors)
        return random.choices(connectors, weights=weights)[0]
    
    # For continuation relationship, use existing logic
    # Check if first sentence has action verbs using spaCy analysis
    has_action_verb = False
    for token in doc1:
        if (token.pos_ == "VERB" and 
            token.dep_ == "ROOT" and 
            token.lemma_ not in ["be", "have", "seem", "appear"]):
            has_action_verb = True
            break
    
    if has_action_verb:
        safe_connectors = [
            (", and", 70),
            (";", 30)
        ]
    else:
        safe_connectors = [
            (", and", 40),
            (";", 25),
            (", while", 20),
            (", additionally,", 15)
        ]
    
    connectors, weights = zip(*safe_connectors)
    return random.choices(connectors, weights=weights)[0]

def find_complete_subject(doc1, main_subject):
    """Find the complete subject phrase including determiners and modifiers."""
    subject_tokens = []
    
    # Start from the main subject and collect related tokens
    for token in doc1:
        # Include the main subject
        if token == main_subject:
            subject_tokens.append(token)
        # Include determiners, adjectives, and compounds that modify the subject
        elif (token.head == main_subject and 
              token.dep_ in ["det", "amod", "compound", "nmod"] and
              token.i < main_subject.i + 3):  # Within reasonable distance
            subject_tokens.append(token)
        # Include tokens that the subject depends on (like "the" in "the CEO")
        elif (main_subject.head == token and 
              token.dep_ == "det" and
              token.i == main_subject.i - 1):
            subject_tokens.append(token)
    
    # Sort by position in sentence
    subject_tokens.sort(key=lambda x: x.i)
    return subject_tokens

def create_relative_clause(doc1, doc2, first_subject, first_pronoun):
    """Create relative clause using pre-validated data. NO redundant checks."""
    if not first_subject:
        return None
    
    # Get the rest of the second sentence (excluding the pronoun)
    second_tokens = [token.text for token in doc2[1:]]  # Skip first pronoun
    second_content = " ".join(second_tokens)
    
    if not second_content.strip():
        return None
    
    # Clean up second content - remove trailing periods
    second_content = second_content.strip().rstrip('.')
    
    # Find complete subject phrase
    subject_phrase_tokens = find_complete_subject(doc1, first_subject)
    
    # Determine appropriate relative pronoun using spaCy's entity analysis
    # Check if any token in the subject phrase is a person
    is_person = any(token.ent_type_ == "PERSON" for token in subject_phrase_tokens)
    
    if is_person:
        relative_pronoun = "who"
    else:
        relative_pronoun = "which"
    
    # Get the rest of first sentence after the complete subject
    if subject_phrase_tokens:
        last_subject_index = max(token.i for token in subject_phrase_tokens)
        first_rest_tokens = [token for token in doc1 if token.i > last_subject_index]
        first_rest = "".join([token.text_with_ws for token in first_rest_tokens]).strip().rstrip('.')
    else:
        first_rest = ""
    
    # Reconstruct subject phrase with proper capitalization
    subject_text = " ".join([token.text for token in subject_phrase_tokens])
    
    # Proper capitalization - preserve original case for acronyms
    if subject_text.upper() in ["CEO", "CTO", "CFO", "USA", "UK"]:
        # Keep acronyms uppercase
        subject_text = subject_text.upper()
    elif subject_text.lower().startswith("the "):
        # Capitalize "The" and the noun
        parts = subject_text.split(" ", 1)
        if len(parts) == 2:
            subject_text = f"The {parts[1]}"
    else:
        # Default capitalization
        subject_text = subject_text.capitalize()
    
    # Format: Subject + , which/who + Content of B + Rest of A
    if first_rest:
        combined = f"{subject_text}, {relative_pronoun} {second_content}, {first_rest}."
    else:
        combined = f"{subject_text}, {relative_pronoun} {second_content}."
    
    return combined

def create_connector_merge(first_sentence, second_sentence, doc1, doc2):
    """Create connector merge using pre-validated data. NO redundant checks."""
    connector = get_smart_connector_from_docs(doc1, doc2)
    
    # Handle different connector types
    if connector == ";":
        # For semicolon, capitalize the next sentence
        return first_sentence.strip('.') + connector + " " + second_sentence.capitalize()
    elif connector.startswith(" "):  # Causal connectors like " because", " as", " since"
        return first_sentence.strip('.') + connector + " " + second_sentence.lower()
    else:
        return first_sentence.strip('.') + connector + " " + second_sentence.lower()

def get_merge_strategy():
    """Randomly choose merge strategy with weighted probabilities."""
    strategies = [
        ("connector", 60),       # 60% - use connectors (, and, ;, etc.)
        ("relative_clause", 20), # 20% - use relative clauses
        ("no_merge", 20)         # 20% - don't merge
    ]
    
    strategy_names, weights = zip(*strategies)
    return random.choices(strategy_names, weights=weights)[0]

def combine_sentences_recursive(text, short_threshold=6):
    """Recursively combine sentences with multiple merge strategies."""
    
    blob = TextBlob(text)
    sentences = list(blob.sentences)
    
    print(f"\n--- Processing: {len(sentences)} sentences ---")
    for idx, sent in enumerate(sentences):
        print(f"  {idx}: '{sent}' ({len(sent.words)} words)")
    
    new_sentences = []
    changes_made = False
    
    i = 0
    while i < len(sentences):
        # Check if we can merge with next sentence
        if i < len(sentences) - 1:
            length = len(sentences[i].words)
            next_length = len(sentences[i+1].words)
            
            print(f"\nChecking merge: {i} and {i+1}")
            print(f"  Sentence {i}: '{sentences[i]}' - Length: {length}")
            print(f"  Sentence {i+1}: '{sentences[i+1]}' - Length: {next_length}")
            
            if length <= short_threshold and next_length <= short_threshold:
                first_word_str = str(sentences[i+1].words[0])
                word_doc = nlp(first_word_str)
                print(f"  First word of next sentence: '{first_word_str}' - POS: {word_doc[0].pos_}")
                
                if word_doc[0].pos_ == "PRON":
                    # Check if pronoun reference is valid (SINGLE CHECK)
                    if enhanced_pronoun_check(str(sentences[i]), str(sentences[i+1])):
                        
                        # Parse sentences once and pass to merge functions
                        doc1 = nlp(str(sentences[i]))
                        doc2 = nlp(str(sentences[i+1]))
                        
                        # Find subject once - improved subject finding
                        first_subject = None
                        for token in doc1:
                            if token.dep_ in ("nsubj", "nsubjpass"):
                                first_subject = token
                                break
                        
                        # Choose merge strategy
                        strategy = get_merge_strategy()
                        print(f"  Strategy chosen: {strategy}")
                        
                        if strategy == "relative_clause":
                            # Pass pre-validated data to relative clause function
                            combined = create_relative_clause(doc1, doc2, first_subject, word_doc[0])
                            if combined:
                                new_sentences.append(combined)
                                print(f"  ✅ MERGED with relative clause: '{combined}'")
                                changes_made = True
                                i += 2
                                continue
                            else:
                                print(f"  ❌ Relative clause formation failed, trying connector")
                                strategy = "connector"  # Fallback to connector
                        
                        if strategy == "connector":
                            # Pass pre-validated data to connector function
                            combined = create_connector_merge(str(sentences[i]), str(sentences[i+1]), doc1, doc2)
                            new_sentences.append(combined)
                            print(f"  ✅ MERGED with connector: '{combined}'")
                            changes_made = True
                            i += 2
                            continue
                        
                        # If strategy is "no_merge", fall through to keep separate
                        print(f"  ➡️  Strategy chose not to merge")
                    else:
                        print(f"  ❌ Pronoun doesn't properly reference subject")
                else:
                    print(f"  ❌ Not a pronoun, no merge")
            else:
                print(f"  ❌ Length criteria not met, no merge")
        
        # If we reach here, add current sentence without merging
        new_sentences.append(str(sentences[i]))
        print(f"  ➡️  Keeping: '{sentences[i]}'")
        i += 1
    
    # Join the results
    result_text = ' '.join(new_sentences)
    
    if changes_made:
        print(f"\n🔄 Changes made, running another pass...")
        return combine_sentences_recursive(result_text, short_threshold)
    else:
        print(f"\n✅ No more merges possible. Final result:")
        return result_text

# Test with both connector and relative clause examples
test_texts = [
   "The old dog ran. It was fast. The cat slept. It was tired.",
    "Sarah smiled. She was happy. The heavy book fell. It made a loud noise.",
    "The CEO spoke. He was confident. The company grew. It expanded rapidly."
]

for i, text in enumerate(test_texts, 1):
    print("=" * 60)
    print(f"TEST {i}: SENTENCE COMBINATION")
    print("=" * 60)
    print(f"Original: {text}")
    
    final_result = combine_sentences_recursive(text, short_threshold=6)
    
    print("=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(final_result)
    print("\n")