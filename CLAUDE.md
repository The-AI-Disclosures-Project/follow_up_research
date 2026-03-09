# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project analyzing follow-up suggestions in human-AI conversations using the WildChat dataset. The pipeline extracts and classifies follow-up patterns from ~4.8M conversations.

## Pipeline Commands

### 1. Download Dataset
```bash
python download_wildchat.py --mode download --output ./data/wildchat
```

### 2. Filter Conversations
```bash
python filter_wildchat.py                    # Full filtering (produces 45k sample)
python filter_wildchat.py --analyze          # Analyze template distribution only
python filter_wildchat.py --analyze --refresh  # Force re-scan ignoring cache
```

### 3. Classify Conversation Intent (requires OPENROUTER_API_KEY)
```bash
python classify_conversation_intent.py --input data/wildchat/filtered_45k.jsonl
python classify_conversation_intent.py --input data/wildchat/filtered_45k.jsonl --limit 100  # Test run
python classify_conversation_intent.py --reuse-from data/processed_data/old_intents.jsonl     # Reuse existing
```

### 4. Extract Follow-ups (requires OPENROUTER_API_KEY)
```bash
python extract_follow_up_suggestions.py --input data/wildchat/filtered_45k.jsonl
python extract_follow_up_suggestions.py --input data/wildchat/filtered_45k.jsonl --limit 100  # Test run
python extract_follow_up_suggestions.py --reuse-from data/processed_data/old_results.jsonl    # Reuse existing
```

### 5. Analyze Results
```bash
python analyze_results.py
```

## Architecture

### Data Flow
```
WildChat-4.8M (HuggingFace)
  → download_wildchat.py → data/wildchat/wildchat_train.jsonl
  → filter_wildchat.py   → data/wildchat/filtered_45k.jsonl
  → classify_conversation_intent.py  → data/processed_data/conversation_intents_45k.jsonl
  → extract_follow_up_suggestions.py  → data/processed_data/extracted_follow_ups_45k.jsonl
  → analyze_results.py   → data/minified/{analysis_data.jsonl, analysis_data_flattened.jsonl, summary_stats.json}
```

### Key Design Patterns

**Memory Efficiency**: The codebase handles large datasets through streaming:
- `wildchat_utils.py`: `ReservoirSampler` for bounded-memory random sampling, `WildChatReader` for line-by-line JSONL iteration
- `filter_wildchat.py`: Two-pass approach (build indexes, then filter) with JSONL cache for template analysis
- `classify_conversation_intent.py`: Streams input, classifies the opening turns, appends results incrementally, and reuses results by `conversation_hash`
- `extract_follow_up_suggestions.py`: Streams input, appends results incrementally, uses regex for hash extraction to avoid full JSON parsing

**Parallel Processing**: `classify_conversation_intent.py` and `extract_follow_up_suggestions.py` use `ThreadPoolExecutor` with configurable workers (default 40) for concurrent API calls to OpenRouter.

**Incremental Processing**: Scripts track progress via `conversation_hash` and skip already-processed records, allowing safe resume after interruption.

### Output Formats

- `analysis_data.jsonl`: One record per conversation with turn-level arrays (`suggestions_given_per_turn`, `suggestion_used_flags_per_turn`, etc.)
- `analysis_data_flattened.jsonl`: One record per turn for ML/analysis with scalar fields
- See `FIELD_DOCUMENTATION.md` for complete field descriptions

### Follow-up Classification

The extraction model (DeepSeek v3.2) distinguishes:
- **Follow-up suggestions**: Explicit offers/invitations that invite continued interaction ("I can draft an email")
- **Not suggestions**: Marketing CTAs, outline content, generic pleasantries, troubleshooting steps, instructions for other platforms

The prompt definition is in `system_prompt.md`.

### Conversation Intent Classification

The intent classifier uses the opening turns only:
- If the conversation has at least 2 turns, classify from the opening `user + assistant + user` slice
- If the conversation has fewer than 2 turns, classify from the first user message only

The prompt definition is in `catagory_classification_prompt.md`.

## Configuration Constants

Key values in `filter_wildchat.py`:
- `SAMPLES_PER_YEAR = 15000` (total 45k across 2023-2025)
- `MIN_TEMPLATE_OCCURRENCES = 500` (threshold for banning repeated prompts)
- `MIN_TEMPLATE_LENGTH = 500` chars, `TEMPLATE_SIGNATURE_LENGTH = 500` chars

Key values in `extract_follow_up_suggestions.py`:
- Default model: `deepseek/deepseek-v3.2`
- Default workers: 40
- Default timeout: 180s per request

Key values in `classify_conversation_intent.py`:
- Default model: `deepseek/deepseek-v3.2`
- Default workers: 40
- Default timeout: 180s per request
