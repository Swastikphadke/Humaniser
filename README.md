# Text Humaniser 🤖➡️👤

An advanced NLP-powered tool that transforms AI-generated text into natural, human-like writing using sophisticated linguistic techniques including synonym replacement, intelligent sentence restructuring, discourse filler insertion, and ML-based clause reordering.

## 🌟 Features

### Core Capabilities

- **Intelligent Synonym Replacement**: Replaces frequent adjectives and adverbs with contextually appropriate synonyms while preserving sentiment and part-of-speech accuracy
- **Adaptive Sentence Length Normalization**: Intelligently splits long sentences and merges short ones to achieve natural reading flow
- **Discourse Filler Integration**: Adds natural discourse markers ("to be honest", "in fact", etc.) with linguistic awareness and style control
- **ML-Based Clause Reordering**: Uses machine learning to reorder clauses for more natural sentence structure (e.g., moving "because" clauses to the front)
- **Multiple Processing Modes**: Choose between synonym-only, legacy, or unified processing approaches

### Advanced Linguistic Features

- **Context-Aware Processing**: Uses spaCy's NLP pipeline for deep linguistic analysis
- **Semantic Relevance Gating**: Ensures fillers match sentence semantics (e.g., argumentative fillers only in argumentative contexts)
- **Pronoun Reference Validation**: Verifies pronoun coherence when merging sentences
- **Style Presets**: Multiple filler styles (neutral, casual, formal, formal_strict)
- **Repetition Avoidance**: Sophisticated weighting system to prevent filler overuse
- **Dependency-Aware Splitting**: Splits sentences at natural linguistic boundaries without breaking grammatical dependencies

## 📋 Requirements

```bash
pip install spacy textblob numpy
python -m spacy download en_core_web_md
python -m textblob.download_corpora
```

**Required Models:**
- `en_core_web_md` (preferred) or `en_core_web_sm` (fallback)

## 🚀 Quick Start

### Basic Usage

```python
from app import humanize_text

# Original AI-generated text
text = """
Artificial intelligence is transforming industries. It enables automation. 
Companies use AI for decision-making. The technology continues to evolve rapidly.
"""

# Humanize with default settings
result = humanize_text(text)
print(result)
```

### Advanced Usage

```python
# Full customization
result = humanize_text(
    text,
    mode="unified",                    # Processing mode
    synonym_fraction=0.25,              # Replace 25% of words
    target_min=12,                      # Min sentence length
    target_max=25,                      # Max sentence length
    filler_style="formal",              # Filler style
    filler_ratio=0.15,                  # Add fillers to 15% of sentences
    apply_fillers=True,                 # Enable fillers
    apply_clause_reordering=True,       # Enable ML clause reordering
    clause_threshold=0.5                # Reordering probability threshold
)
```

## ⚙️ Configuration Options

### Processing Modes

| Mode | Description |
|------|-------------|
| `synonyms` | Only applies synonym replacement |
| `legacy` | Synonyms → merge short → split long (original approach) |
| `unified` | Synonyms → intelligent length normalization (recommended) |

### Filler Styles

| Style | Description | Example Fillers |
|-------|-------------|-----------------|
| `neutral` | Balanced, all-purpose | All available fillers |
| `casual` | Conversational tone | "you know", "basically", "to be honest" |
| `formal` | Professional writing | "in fact", "truthfully", "generally speaking" |
| `formal_strict` | Academic/technical | "in fact", "truthfully" (minimal) |

### Parameters

```python
humanize_text(
    text: str,                          # Input text
    mode: str = "unified",              # "synonyms" | "legacy" | "unified"
    synonym_fraction: float = 0.3,      # 0.0-1.0
    target_min: int = 15,               # Minimum sentence length
    target_max: int = 25,               # Maximum sentence length
    filler_style: str = "neutral",      # "neutral" | "casual" | "formal" | "formal_strict"
    filler_ratio: float = 0.0,          # 0.0-1.0
    apply_fillers: bool = False,        # Enable/disable fillers
    apply_clause_reordering: bool = True, # Enable/disable ML reordering
    clause_threshold: float = 0.6       # 0.0-1.0 (lower = more reordering)
)
```

## 🎯 Use Cases

### Academic Writing
```python
result = humanize_text(
    text,
    mode="unified",
    filler_style="formal_strict",
    filler_ratio=0.10,
    apply_fillers=True,
    synonym_fraction=0.20
)
```

### Blog Posts / Articles
```python
result = humanize_text(
    text,
    mode="unified",
    filler_style="casual",
    filler_ratio=0.20,
    apply_fillers=True,
    synonym_fraction=0.30
)
```

### Professional Reports
```python
result = humanize_text(
    text,
    mode="unified",
    filler_style="formal",
    filler_ratio=0.12,
    apply_fillers=True,
    apply_clause_reordering=True
)
```

## 🔧 Advanced Features

### Filler Configuration

```python
from app import add_fillers, FillerConfig

config = FillerConfig(
    ratio=0.15,                         # Percentage of sentences to modify
    style="formal",                     # Filler style
    min_sentence_len=4,                 # Minimum sentence length to consider
    skip_discourse_lead=True,           # Skip sentences starting with connectors
    skip_all_caps=True,                 # Skip all-caps sentences
    skip_dialogue=True,                 # Skip quoted dialogue
    enforce_semantics=True,             # Match fillers to sentence meaning
    use_exponential=False,              # Use exponential decay for repetition penalty
    structured_plan=True                # Return (text, plan) tuple
)

result, plan = add_fillers(text, config=config)
```

### Remove Fillers

```python
from app import remove_fillers

# Remove all added fillers from text
clean_text = remove_fillers(humanized_text)
```

### ML Clause Reordering

```python
from app import reorder_clauses_ml

# Standalone clause reordering
reordered = reorder_clauses_ml(
    text,
    threshold=0.4,                      # Lower = more aggressive reordering
    model_path="clause_patterns.pkl"    # Path to ML model
)
```

## 📊 How It Works

### 1. Synonym Replacement
- Identifies frequently used adjectives and adverbs
- Uses TextBlob and WordNet for synonym discovery
- Preserves part-of-speech and sentiment polarity
- Applies quality filters (length, commonality, context)

### 2. Sentence Normalization
- **Splitting**: Uses linguistic analysis to find natural split points (conjunctions, clauses, punctuation)
- **Merging**: Combines short sentences with validated pronoun references and smart connectors
- **Scoring System**: Evaluates split quality based on balance, syntax, and dependencies

### 3. Filler Insertion
- **Position Modes**: Prefix, mid-sentence, or suffix insertion
- **Repetition Control**: Weighted selection to avoid overuse
- **Semantic Gating**: Matches fillers to sentence meaning
- **Linguistic Validation**: Respects dialogue, quotes, and discourse structure

### 4. Clause Reordering
- **ML Patterns**: Learns from human writing samples
- **Feature Extraction**: Analyzes sentence complexity, length, and structure
- **Confidence Scoring**: Generates multiple candidates with probability scores
- **Common Transformations**:
  - "X because Y" → "Because Y, X"
  - "X if Y" → "If Y, X"
  - Main clause + prep phrase → Prep phrase + main clause

## 🧪 Example Transformations

### Before
```
Virtual Reality is an immersive technology. It simulates environments. 
Users wear head-mounted displays. The displays track movements.
```

### After (Unified Mode + Fillers + Reordering)
```
Virtual Reality is essentially an immersive technology that simulates 
environments, and users wear head-mounted displays which track movements. 
To be honest, this creates realistic experiences that transform how we 
interact with digital content.
```

## 🏗️ Architecture

```
Text Input
    ↓
Synonym Replacement (TextBlob + spaCy)
    ↓
Sentence Normalization (spaCy NLP)
    ├── Split long sentences (linguistic boundary detection)
    └── Merge short sentences (pronoun validation + connectors)
    ↓
ML Clause Reordering (pattern matching + feature extraction)
    ↓
Filler Insertion (weighted selection + semantic gating)
    ↓
Text Cleanup (formatting + punctuation)
    ↓
Humanized Output
```

## 📝 Model Files

- `clause_patterns.pkl`: Trained ML patterns for clause reordering (auto-generated)
- Bootstrap patterns included for first-time use

## ⚠️ Limitations

- Requires good-quality input text (grammatically correct)
- Best results with en_core_web_md model (larger model)
- Processing time increases with text length
- Synonym replacement may occasionally alter subtle nuances
- Clause reordering works best on structured sentences

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional language support
- More sophisticated ML models for clause reordering
- Expanded filler phrase databases
- Better handling of technical/domain-specific text

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Swastik Phadke**
- GitHub: [@Swastikphadke](https://github.com/Swastikphadke)

## 🙏 Acknowledgments

- Built with [spaCy](https://spacy.io/) for NLP
- Uses [TextBlob](https://textblob.readthedocs.io/) for sentiment and synonym analysis
- Inspired by research in natural language generation and style transfer

---

**Note**: This tool is designed to make AI-generated text more natural and human-like. Always review output for accuracy and appropriateness before use.