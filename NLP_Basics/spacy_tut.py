import spacy

# Import the Matcher
from spacy.matcher import Matcher

# Load a pipeline and create the nlp object
nlp = spacy.load("en_core_web_sm")

# Initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# Add the pattern to the matcher
pattern = [{"POS":"DET"}, {"OP":"?"}
           ,{"POS":"NOUN"}]
matcher.add("IPHONE_PATTERN", [pattern])

# Process some text
doc = nlp("I brought a Smartphone. Now I'm buying apps.")

# Call the matcher on the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)