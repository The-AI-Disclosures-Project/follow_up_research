# Follow-Up Suggestion Research

Analyzing follow-up suggestions in human-AI conversations using the WildChat dataset.

## Overview

This project studies how AI assistants offer follow-up suggestions (explicit offers, invitations, or next-step proposals) and how users respond to them. We process the [allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) dataset to extract and classify follow-up patterns.

## Pipeline

```
WildChat-4.8M → Filter (15k sample) → {Classify Opening Intent, Extract Follow-ups} → Analyze
```

### 1. Download Dataset
```bash
python download_wildchat.py --mode download --output ./data/wildchat
```

Downloads the WildChat dataset from Hugging Face. Supports streaming for memory efficiency.

### 2. Filter Conversations
```bash
python filter_wildchat.py
```

Produces a 15,000 conversation sample (5,000 per year: 2023-2025) with:
- English language only
- 1-30 turns per conversation
- Template filtering (removes bot-generated content with repeated prompts)
- Deduplication by conversation hash

### 3. Classify Opening Intent
```bash
python classify_conversation_intent.py --input data/wildchat/filtered_15k.jsonl
```

Uses DeepSeek v3.2 via OpenRouter to classify each conversation's opening intent:
- `EXPLORATORY`
- `BUILDING`
- `INFORMATIONAL`
- `UNKNOWN`

Classification uses the opening `user + assistant + user` slice for conversations with at least 2 turns, and the first user message only for shorter conversations.

Requires `OPENROUTER_API_KEY` environment variable.

### 4. Extract Follow-ups
```bash
python extract_follow_up_suggestions.py --input data/wildchat/filtered_15k.jsonl
```

Uses DeepSeek v3.2 via OpenRouter to analyze each conversation and extract:
- Follow-up suggestions from each AI message
- Whether each user message acted on any suggestion (current or previous turns)

Requires `OPENROUTER_API_KEY` environment variable.

### 5. Analyze Results
```bash
python analyze_results.py
```

Generates minified datasets and statistics in `data/minified/`.

## Output Data

### analysis_data.jsonl
One record per conversation with turn-level arrays:
- `suggestions_given_per_turn` - Follow-up suggestions offered per turn
- `suggestion_used_flags_per_turn` - Whether user acted on suggestions
- `total_suggestions_given_count`, `total_suggestions_used` - Aggregates

### analysis_data_flattened.jsonl
One record per turn for ML/analysis:
- `suggestion_given_count`, `suggestion_used_flag` - Per-turn metrics
- `conversation_continued` - Whether conversation continued after this turn
- `strict_*` variants - Only count same-turn suggestion usage

See [FIELD_DOCUMENTATION.md](FIELD_DOCUMENTATION.md) for complete field descriptions.

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `datasets` - Hugging Face dataset loading
- `huggingface_hub` - API access
- `tqdm` - Progress bars
- `pandas`, `openai`, `pydantic`, `python-dotenv` - For processing

## Project Structure

```
├── download_wildchat.py      # Dataset download/streaming
├── filter_wildchat.py        # Filtering and sampling
├── classify_conversation_intent.py  # Opening-intent classification
├── extract_follow_up_suggestions.py  # LLM-based extraction
├── analyze_results.py        # Analysis and minification
├── catagory_classification_prompt.md  # Intent classification prompt
├── wildchat_utils.py         # JSONL utilities, reservoir sampling
├── system_prompt.md          # Extraction prompt for LLM
├── FIELD_DOCUMENTATION.md    # Output field documentation
├── analyze_results.ipynb     # Interactive analysis notebook
└── data/
    ├── wildchat/             # Filtered conversations
    ├── processed_data/       # LLM extraction results
    └── minified/             # Final analysis datasets
```

## What Counts as a Follow-up Suggestion?

A follow-up suggestion is an explicit offer or invitation directed at the user that:
- Is action-oriented and specific ("I can draft an email to your team")
- Invites continued interaction with the assistant
- Is future-facing, not a statement of fact

**Not** follow-up suggestions:
- Marketing CTAs in generated content
- Outline bullets that are content, not invitations
- Generic pleasantries ("How can I help?")
- Instructions to act outside the chat ("check your firewall settings")
- Offers already satisfied within the same message

See [system_prompt.md](system_prompt.md) for the complete extraction criteria.

## License

Research use only. WildChat dataset is subject to its own [license terms](https://huggingface.co/datasets/allenai/WildChat-4.8M).
