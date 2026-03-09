# Field Documentation

This document describes the data files, their fields, and the data pipeline used to generate them.

---

## Data Source

### WildChat Dataset

The source data comes from **[allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M)** hosted on Hugging Face.

- **Size**: ~4.8 million human-AI conversations
- **Content**: Real user conversations with various GPT models (gpt-3.5-turbo, gpt-4, gpt-4o-mini, etc.)
- **Metadata**: Includes timestamps, model versions, language detection, and conversation structure
- **Time Range**: Conversations from 2023, 2024, and 2025

### Follow-up Extraction Model

Follow-up suggestions were extracted using **DeepSeek v3.2** via the OpenRouter API. The model analyzes each conversation to identify:
- Follow-up suggestions made by the AI (explicit offers, invitations, or next-step proposals)
- Whether users acted on those suggestions in subsequent messages

The extraction prompt (`system_prompt.md`) defines what constitutes a follow-up suggestion vs. content/instructions.

---

## Data Pipeline

### Step 1: Download
**Script**: `download_wildchat.py`
**Input**: Hugging Face API
**Output**: `data/wildchat/wildchat_train.jsonl`

Downloads the WildChat dataset via streaming to handle the large size.

### Step 2: Filter
**Script**: `filter_wildchat.py`
**Input**: `data/wildchat/wildchat_train.jsonl`
**Output**: `data/wildchat/filtered_15k.jsonl`

Produces a 15,000 conversation sample (5,000 per year) with:
- English language only
- 1-30 turns per conversation
- Template/system prompt filtering (removes conversations starting with repeated prompts appearing 500+ times)
- Deduplication by conversation hash

### Step 3: Classify Opening Intent
**Script**: `classify_conversation_intent.py`
**Input**: `data/wildchat/filtered_15k.jsonl`
**Output**: `data/processed_data/conversation_intents_45k.jsonl`

Uses DeepSeek v3.2 to classify each conversation's opening intent as one of:
- `EXPLORATORY`
- `BUILDING`
- `INFORMATIONAL`
- `UNKNOWN`

The classifier uses the opening `user + assistant + user` slice when a conversation has at least 2 turns, and falls back to the first user message otherwise.

Logs are saved to `data/processed_data/conversation_intent_logs.txt`.

### Step 4: Extract Follow-ups
**Script**: `extract_follow_up_suggestions.py`
**Input**: `data/wildchat/filtered_15k.jsonl`
**Output**: `data/processed_data/extracted_follow_ups.jsonl`

Uses DeepSeek v3.2 to analyze each conversation and extract:
- Follow-up suggestions from each AI message
- Whether each user message acted on any suggestions (current or previous turns)

Logs are saved to `data/processed_data/logs.txt`.

### Step 5: Analyze & Minify
**Script**: `analyze_results.py`
**Input**: `data/processed_data/extracted_follow_ups.jsonl`
**Output**: `data/minified/` directory containing:
- `analysis_data.jsonl` - One record per conversation
- `analysis_data_flattened.jsonl` - One record per turn
- `summary_stats.json` - Aggregate statistics

---

## Output Files

### data/minified/analysis_data.jsonl

This file contains **one record per conversation** with turn-level data stored as arrays.

#### Conversation-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `conversation_hash` | string | Unique hash identifier for the conversation |
| `timestamp` | string | ISO 8601 timestamp of the conversation (e.g., `2023-12-30 20:03:47`) |
| `year` | integer | Year extracted from timestamp (2023, 2024, or 2025) |
| `language` | string | Language of the conversation. In this dataset, always `"English"` |
| `model` | string | GPT model version used (e.g., `gpt-4-1106-preview`, `gpt-3.5-turbo-0301`) |
| `state` | string | US state (e.g., "California"), null if not available |
| `country` | string | Country name (e.g., "United States") |
| `hashed_ip` | string | Anonymized SHA-256 hash of user's IP address |
| `toxic` | boolean | Whether conversation was flagged as toxic |
| `redacted` | boolean | Whether content was redacted for privacy |
| `header` | object | Request headers with `user-agent` and `accept-language` |
| `turns` | integer | Total number of conversation turns. Calculated as `ceil((len(conversation) - 1) / 2)` |
| `total_word_count` | integer | Total words across all messages in conversation |
| `user_word_count` | integer | Total words in user messages |
| `assistant_word_count` | integer | Total words in assistant messages |

#### Turn-Level Arrays

Each array has one element per turn (0-indexed). If you need per-turn analysis, use the flattened dataset instead.

| Field | Type | Description |
|-------|------|-------------|
| `suggestions_given_per_turn` | array[int] | Number of follow-up suggestions offered by the AI in each turn |
| `suggestion_used_flags_per_turn` | array[bool] | Whether the user used any suggestions in each turn |
| `suggestion_used_counts_per_turn` | array[int] | Number of suggestions used by the user in each turn (includes suggestions from any previous turn) |
| `strict_suggestion_used_counts_per_turn` | array[int] | Number of suggestions used that were offered in that same turn only |
| `strict_suggestion_used_flags_per_turn` | array[bool] | Whether the user used any suggestions offered in that same turn |
| `user_word_counts_per_turn` | array[int] | Word count of user message in each turn |
| `assistant_word_counts_per_turn` | array[int] | Word count of assistant message in each turn |

#### Aggregated Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_suggestions_given_count` | integer | Sum of all suggestions offered across all turns |
| `total_suggestions_used` | integer | Total number of suggestions used across all turns |
| `num_turns_with_suggestions_used` | float | Number of turns where the user used at least one suggestion |

---

### data/minified/analysis_data_flattened.jsonl

This file contains **one record per turn**, with conversation metadata repeated for each turn. This structure simplifies turn-level analysis.

#### Conversation-Level Fields (repeated for each turn)

| Field | Type | Description |
|-------|------|-------------|
| `conversation_hash` | string | Unique hash identifier for the conversation |
| `year` | integer | Year extracted from timestamp |
| `timestamp` | string | ISO 8601 timestamp of the conversation |
| `language` | string | Language of the conversation (always `"English"`) |
| `model` | string | GPT model version used |
| `state` | string | US state (e.g., "California"), null if not available |
| `country` | string | Country name (e.g., "United States") |
| `hashed_ip` | string | Anonymized SHA-256 hash of user's IP address |
| `toxic` | boolean | Whether conversation was flagged as toxic |
| `redacted` | boolean | Whether content was redacted for privacy |
| `header` | object | Request headers with `user-agent` and `accept-language` |

#### Turn-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `turn` | integer | Turn number within the conversation (1-indexed). First turn is `1` |
| `turns` | integer | Total number of turns in the conversation |
| `suggestion_given_count` | integer | Number of follow-up suggestions offered by the AI in this turn |
| `suggestion_used_count` | integer | Number of suggestions used by the user in this turn (from any turn) |
| `suggestion_used_flag` | boolean | Whether the user used any suggestions in this turn |
| `strict_suggestion_used_count` | integer | Number of suggestions used that were offered in this same turn |
| `strict_suggestion_used_flag` | boolean | Whether the user used any suggestions offered in this same turn |
| `conversation_continued` | boolean | `true` if this is not the last turn (`turn < turns`), `false` otherwise |
| `user_word_count` | integer | Word count of user message in this turn |
| `assistant_word_count` | integer | Word count of assistant message in this turn |

---

### data/minified/summary_stats.json

Contains aggregate statistics:
- `total_conversations`: Total number of analyzed conversations
- `conversations_with_suggestions`: Conversations where AI offered at least one suggestion
- `conversations_with_suggestions_used`: Conversations where user acted on at least one suggestion
- `suggestion_usage_rate`: Percentage of suggestion-containing conversations where suggestions were used
- `year_distribution`: Count of conversations per year
- `language_distribution`: Count by language (all English in this dataset)
- `avg_suggestions_by_year`: Average suggestions per conversation by year
- `avg_suggestions_by_year_nonzero`: Average suggestions per conversation (excluding zero-suggestion conversations)

---

## Key Differences Between Files

| Aspect | analysis_data.jsonl | analysis_data_flattened.jsonl |
|--------|---------------------|-------------------------------|
| **Structure** | One record per conversation | One record per turn |
| **Turn data** | Stored as arrays | Scalar values per turn |
| **Field naming** | Plural with `_per_turn` suffix | Singular forms |
| **Extra fields** | - | `conversation_continued` |
| **Best for** | Conversation-level analysis | Turn-level analysis, ML features |

---

## Understanding "Strict" vs Regular Suggestion Usage

- **Regular** (`suggestion_used_count`, `suggestion_used_flag`): Tracks whether the user acted on ANY suggestion offered in the current OR any previous turn. This captures cross-turn references where users revisit earlier offers.

- **Strict** (`strict_suggestion_used_count`, `strict_suggestion_used_flag`): Only counts suggestions that were offered AND used within the same turn. This is a more conservative measure of immediate suggestion uptake.

---

## Notes
- **Turn numbering**: 1-indexed in the flattened file. The arrays in the non-flattened file are 0-indexed.
- **Suggestion IDs**: The extraction model assigns globally unique, sequential IDs to suggestions across the entire conversation. These IDs are used internally to track cross-turn references but are not included in the minified output.
