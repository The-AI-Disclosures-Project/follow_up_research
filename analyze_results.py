"""
Analyze follow-up suggestion results from the jan-15 dataset.
Produces JSON outputs in data/processed_data/json/
"""

import argparse
import pandas as pd
import json
import math
import os
from datetime import datetime

# Ensure output directory exists
OUTPUT_DIR = 'data/minified'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def stream_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)


def load_intent_classifications(file_path):
    """Load conversation intent classifications keyed by conversation_hash."""
    if not file_path or not os.path.exists(file_path):
        return {}

    intent_lookup = {}
    for record in stream_jsonl(file_path):
        conv_hash = record.get('conversation_hash')
        if not conv_hash:
            continue
        intent_lookup[conv_hash] = {
            'conversation_intent_classification': record.get('conversation_intent_classification'),
            'opening_messages_used': record.get('opening_messages_used'),
            'opening_message_count_used': record.get('opening_message_count_used'),
        }
    return intent_lookup


def calculate_turns(conversation):
    """Calculate number of turns (skip the first message)."""
    return math.ceil((len(conversation[1:])) / 2)


def count_words(text):
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def calculate_word_counts(conversation):
    """Calculate word counts per turn and totals.

    Follows existing turn logic: skip first message, then pair (user, assistant).
    """
    user_word_counts = []
    assistant_word_counts = []

    # Match existing turn calculation: skip first message, process pairs
    messages = conversation[1:]
    for i in range(0, len(messages), 2):
        user_msg = messages[i] if i < len(messages) else None
        assistant_msg = messages[i+1] if i+1 < len(messages) else None

        user_words = count_words(user_msg.get('content', '')) if user_msg else 0
        assistant_words = count_words(assistant_msg.get('content', '')) if assistant_msg else 0

        user_word_counts.append(user_words)
        assistant_word_counts.append(assistant_words)

    return {
        'user_word_counts_per_turn': user_word_counts,
        'assistant_word_counts_per_turn': assistant_word_counts,
        'user_word_count': sum(user_word_counts),
        'assistant_word_count': sum(assistant_word_counts),
        'total_word_count': sum(user_word_counts) + sum(assistant_word_counts)
    }


def extract_follow_up_data(follow_up_data_row):
    """Extract follow-up suggestion data from a conversation row."""
    suggestions_given_per_turn = []
    suggestion_used_flags_per_turn = []
    suggestion_used_counts_per_turn = []
    strict_suggestion_used_flags_per_turn = []
    strict_suggestion_used_counts_per_turn = []

    for turn in follow_up_data_row.get('conversation_turns', []):
        ai_message_info = turn.get('AIMessageInformation', {})
        user_message_info = turn.get('UserMessageInformation', {})

        # In the jan-15 dataset, follow_up_suggestions is a direct array
        # (not nested under another follow_up_suggestions key)
        suggestions_offered = ai_message_info.get('follow_up_suggestions', [])

        # Handle both list of dicts with suggestion_id and empty list formats
        turn_suggestion_ids = []
        if suggestions_offered and isinstance(suggestions_offered, list):
            if len(suggestions_offered) > 0 and isinstance(suggestions_offered[0], dict):
                turn_suggestion_ids = [s.get('suggestion_id') for s in suggestions_offered if s.get('suggestion_id')]

        used_suggestion_ids = user_message_info.get('used_follow_up_suggestions_ids', [])
        strict_used_suggestion_ids = list(set(turn_suggestion_ids) & set(used_suggestion_ids))

        num_suggestions_offered = len(suggestions_offered) if suggestions_offered else 0

        suggestions_given_per_turn.append(num_suggestions_offered)
        strict_suggestion_used_counts_per_turn.append(len(strict_used_suggestion_ids))

        suggestion_was_used = len(used_suggestion_ids) > 0
        strict_suggestion_was_used = len(strict_used_suggestion_ids) > 0

        strict_suggestion_used_flags_per_turn.append(strict_suggestion_was_used)
        suggestion_used_flags_per_turn.append(suggestion_was_used)
        suggestion_used_counts_per_turn.append(len(used_suggestion_ids))

    total_suggestions_given_count = sum(suggestions_given_per_turn)

    return {
        'suggestions_given_per_turn': suggestions_given_per_turn,
        'suggestion_used_flags_per_turn': suggestion_used_flags_per_turn,
        'suggestion_used_counts_per_turn': suggestion_used_counts_per_turn,
        'num_turns_with_suggestions_used': sum(suggestion_used_flags_per_turn),
        'total_suggestions_used': sum(suggestion_used_counts_per_turn),
        'strict_suggestion_used_counts_per_turn': strict_suggestion_used_counts_per_turn,
        'strict_suggestion_used_flags_per_turn': strict_suggestion_used_flags_per_turn,
        'total_suggestions_given_count': total_suggestions_given_count
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze extracted follow-up suggestions and optionally merge conversation intent labels.')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed_data/extracted_follow_ups_45k.jsonl',
        help='Path to extracted follow-up suggestions JSONL (default: data/processed_data/extracted_follow_ups_45k.jsonl)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='data/wildchat/filtered_45k.jsonl',
        help='Path to filtered WildChat source JSONL (default: data/wildchat/filtered_45k.jsonl)'
    )
    parser.add_argument(
        '--intent-input',
        type=str,
        default='data/processed_data/conversation_intents_45k.jsonl',
        help='Optional path to conversation intent classifications JSONL (default: data/processed_data/conversation_intents_45k.jsonl)'
    )
    args = parser.parse_args()

    # First pass: stream source metadata to build lookup (memory-efficient)
    print(f"Streaming source metadata from {args.source}...")
    source_metadata = {}
    for record in stream_jsonl(args.source):
        conv_hash = record.get('conversation_hash')
        if conv_hash:
            # Calculate word counts immediately instead of storing full conversation
            conversation = record.get('conversation', [])
            word_counts = calculate_word_counts(conversation) if conversation else None

            source_metadata[conv_hash] = {
                'model': record.get('model'),
                'timestamp': record.get('timestamp'),
                'language': record.get('language'),
                'state': record.get('state'),
                'country': record.get('country'),
                'hashed_ip': record.get('hashed_ip'),
                'toxic': record.get('toxic'),
                'redacted': record.get('redacted'),
                'header': record.get('header'),
                'word_counts': word_counts
            }
    print(f"Loaded metadata for {len(source_metadata)} conversations")

    intent_lookup = load_intent_classifications(args.intent_input)
    if intent_lookup:
        print(f"Loaded intent classifications for {len(intent_lookup)} conversations from {args.intent_input}")
    else:
        print(f"No intent classifications loaded from {args.intent_input}; continuing without intent merge")

    print(f"Loading extracted follow-ups data from {args.input}...")
    df = pd.read_json(args.input, orient='records', lines=True)
    df.index = df['index']
    df.drop(columns=['index'], inplace=True)
    print(f"Loaded {len(df)} conversations")

    # Extract metadata by joining with source data
    print("Extracting metadata...")

    # Build metadata lists for bulk assignment (more efficient and avoids pandas assignment issues)
    timestamps = []
    models = []
    years = []
    languages = []
    states = []
    countries = []
    hashed_ips = []
    toxics = []
    redacteds = []
    headers = []
    conversation_intent_classifications = []
    opening_messages_used_list = []
    opening_message_count_used_list = []
    user_word_counts_per_turn = []
    assistant_word_counts_per_turn = []
    user_word_counts = []
    assistant_word_counts = []
    total_word_counts = []

    for _, row in df.iterrows():
        conv_hash = row['conversation_hash']
        metadata = source_metadata.get(conv_hash, {})

        timestamps.append(metadata.get('timestamp'))
        models.append(metadata.get('model'))
        languages.append(metadata.get('language'))
        states.append(metadata.get('state'))
        countries.append(metadata.get('country'))
        hashed_ips.append(metadata.get('hashed_ip'))
        toxics.append(metadata.get('toxic'))
        redacteds.append(metadata.get('redacted'))
        headers.append(metadata.get('header'))
        intent_data = intent_lookup.get(conv_hash, {})
        conversation_intent_classifications.append(intent_data.get('conversation_intent_classification'))
        opening_messages_used_list.append(intent_data.get('opening_messages_used'))
        opening_message_count_used_list.append(intent_data.get('opening_message_count_used'))

        # Extract year from timestamp
        timestamp = metadata.get('timestamp')
        year = None
        if timestamp:
            try:
                timestamp_dt = pd.to_datetime(timestamp)
                year = timestamp_dt.year
            except:
                pass
        years.append(year)

        # Use pre-computed word counts from streaming step
        word_counts = metadata.get('word_counts')
        if word_counts:
            user_word_counts_per_turn.append(word_counts['user_word_counts_per_turn'])
            assistant_word_counts_per_turn.append(word_counts['assistant_word_counts_per_turn'])
            user_word_counts.append(word_counts['user_word_count'])
            assistant_word_counts.append(word_counts['assistant_word_count'])
            total_word_counts.append(word_counts['total_word_count'])
        else:
            user_word_counts_per_turn.append(None)
            assistant_word_counts_per_turn.append(None)
            user_word_counts.append(None)
            assistant_word_counts.append(None)
            total_word_counts.append(None)

    # Bulk assign all metadata columns
    df['timestamp'] = timestamps
    df['model'] = models
    df['year'] = years
    df['language'] = languages
    df['state'] = states
    df['country'] = countries
    df['hashed_ip'] = hashed_ips
    df['toxic'] = toxics
    df['redacted'] = redacteds
    df['header'] = headers
    df['conversation_intent_classification'] = conversation_intent_classifications
    df['opening_messages_used'] = opening_messages_used_list
    df['opening_message_count_used'] = opening_message_count_used_list
    df['user_word_counts_per_turn'] = user_word_counts_per_turn
    df['assistant_word_counts_per_turn'] = assistant_word_counts_per_turn
    df['user_word_count'] = user_word_counts
    df['assistant_word_count'] = assistant_word_counts
    df['total_word_count'] = total_word_counts

    # Process follow-up data using bulk assignment
    print("Processing follow-up data...")
    suggestions_given_per_turn_list = []
    suggestion_used_flags_per_turn_list = []
    suggestion_used_counts_per_turn_list = []
    strict_suggestion_used_counts_per_turn_list = []
    strict_suggestion_used_flags_per_turn_list = []
    num_turns_with_suggestions_used_list = []
    total_suggestions_used_list = []
    total_suggestions_given_count_list = []

    for _, row in df.iterrows():
        extracted_data = extract_follow_up_data(row['follow_up_data'])
        suggestions_given_per_turn_list.append(extracted_data['suggestions_given_per_turn'])
        suggestion_used_flags_per_turn_list.append(extracted_data['suggestion_used_flags_per_turn'])
        suggestion_used_counts_per_turn_list.append(extracted_data['suggestion_used_counts_per_turn'])
        strict_suggestion_used_counts_per_turn_list.append(extracted_data['strict_suggestion_used_counts_per_turn'])
        strict_suggestion_used_flags_per_turn_list.append(extracted_data['strict_suggestion_used_flags_per_turn'])
        num_turns_with_suggestions_used_list.append(extracted_data['num_turns_with_suggestions_used'])
        total_suggestions_used_list.append(extracted_data['total_suggestions_used'])
        total_suggestions_given_count_list.append(extracted_data['total_suggestions_given_count'])

    df['suggestions_given_per_turn'] = suggestions_given_per_turn_list
    df['suggestion_used_flags_per_turn'] = suggestion_used_flags_per_turn_list
    df['suggestion_used_counts_per_turn'] = suggestion_used_counts_per_turn_list
    df['strict_suggestion_used_counts_per_turn'] = strict_suggestion_used_counts_per_turn_list
    df['strict_suggestion_used_flags_per_turn'] = strict_suggestion_used_flags_per_turn_list
    df['num_turns_with_suggestions_used'] = num_turns_with_suggestions_used_list
    df['total_suggestions_used'] = total_suggestions_used_list
    df['total_suggestions_given_count'] = total_suggestions_given_count_list

    df['turns'] = df['conversation'].apply(calculate_turns)
    df['turns_match'] = df.apply(lambda row: len(row['suggestions_given_per_turn']) == row['turns'], axis=1)

    # Filter to only matching turns
    df_filtered = df[df['turns_match']].copy()
    print(f"Filtered to {len(df_filtered)} conversations with matching turns")

    # Filter out o1 models (they don't have multiple turns at all)
    excluded_models = ['o1-mini-2024-09-12', 'o1-preview-2024-09-12']
    excluded_count = df_filtered['model'].isin(excluded_models).sum()
    df_filtered = df_filtered[~df_filtered['model'].isin(excluded_models)].copy()
    print(f"Excluded {excluded_count} conversations with o1 models, {len(df_filtered)} remaining")

    # Output 1: Flattened data (one row per turn)
    print("Creating flattened output...")
    flattened_data = []
    for i, row in df_filtered.iterrows():
        turns = row.turns
        conversation_hash = row.conversation_hash
        year = row.year
        timestamp = row.timestamp
        language = row.language
        model = row.model
        state = row.state
        country = row.country
        hashed_ip = row.hashed_ip
        toxic = row.toxic
        redacted = row.redacted
        header = row.header
        user_word_counts = row['user_word_counts_per_turn'] or []
        assistant_word_counts = row['assistant_word_counts_per_turn'] or []

        for turn in range(turns):
            suggestion_given_count = row['suggestions_given_per_turn'][turn]
            suggestion_used_count = row['suggestion_used_counts_per_turn'][turn]
            suggestion_used_flag = row['suggestion_used_flags_per_turn'][turn]
            strict_suggestion_used_count = row['strict_suggestion_used_counts_per_turn'][turn]
            strict_suggestion_used_flag = row['strict_suggestion_used_flags_per_turn'][turn]
            conversation_continued = turn < turns - 1
            user_word_count = user_word_counts[turn] if turn < len(user_word_counts) else 0
            assistant_word_count = assistant_word_counts[turn] if turn < len(assistant_word_counts) else 0

            flattened_data.append({
                'conversation_hash': conversation_hash,
                'year': int(year) if pd.notna(year) else None,
                'timestamp': str(timestamp) if pd.notna(timestamp) else None,
                'language': language,
                'model': model,
                'state': state,
                'country': country,
                'hashed_ip': hashed_ip,
                'toxic': bool(toxic) if pd.notna(toxic) else None,
                'redacted': bool(redacted) if pd.notna(redacted) else None,
                'header': header,
                'conversation_intent_classification': row.get('conversation_intent_classification'),
                'opening_messages_used': row.get('opening_messages_used'),
                'opening_message_count_used': int(row['opening_message_count_used']) if pd.notna(row['opening_message_count_used']) else None,
                'turn': turn + 1,
                'turns': turns,
                'suggestion_given_count': suggestion_given_count,
                'suggestion_used_count': suggestion_used_count,
                'suggestion_used_flag': suggestion_used_flag,
                'strict_suggestion_used_count': strict_suggestion_used_count,
                'strict_suggestion_used_flag': strict_suggestion_used_flag,
                'conversation_continued': conversation_continued,
                'user_word_count': user_word_count,
                'assistant_word_count': assistant_word_count
            })

    flattened_path = os.path.join(OUTPUT_DIR, 'analysis_data_flattened.jsonl')
    with open(flattened_path, 'w') as f:
        for record in flattened_data:
            f.write(json.dumps(record) + '\n')
    print(f"Saved flattened data to {flattened_path}")

    # Output 2: Minified results (one row per conversation)
    print("Creating minified output...")
    minified_data = []
    for i, row in df_filtered.iterrows():
        minified_data.append({
            'conversation_hash': row['conversation_hash'],
            'timestamp': str(row['timestamp']) if pd.notna(row['timestamp']) else None,
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'language': row['language'],
            'model': row['model'],
            'state': row['state'],
            'country': row['country'],
            'hashed_ip': row['hashed_ip'],
            'toxic': bool(row['toxic']) if pd.notna(row['toxic']) else None,
            'redacted': bool(row['redacted']) if pd.notna(row['redacted']) else None,
            'header': row['header'],
            'conversation_intent_classification': row.get('conversation_intent_classification'),
            'opening_messages_used': row.get('opening_messages_used'),
            'opening_message_count_used': int(row['opening_message_count_used']) if pd.notna(row['opening_message_count_used']) else None,
            'suggestions_given_per_turn': row['suggestions_given_per_turn'],
            'suggestion_used_flags_per_turn': row['suggestion_used_flags_per_turn'],
            'suggestion_used_counts_per_turn': row['suggestion_used_counts_per_turn'],
            'strict_suggestion_used_counts_per_turn': row['strict_suggestion_used_counts_per_turn'],
            'strict_suggestion_used_flags_per_turn': row['strict_suggestion_used_flags_per_turn'],
            'total_suggestions_given_count': row['total_suggestions_given_count'],
            'total_suggestions_used': row['total_suggestions_used'],
            'num_turns_with_suggestions_used': row['num_turns_with_suggestions_used'],
            'turns': row['turns'],
            'user_word_counts_per_turn': row['user_word_counts_per_turn'],
            'assistant_word_counts_per_turn': row['assistant_word_counts_per_turn'],
            'user_word_count': int(row['user_word_count']) if pd.notna(row['user_word_count']) else None,
            'assistant_word_count': int(row['assistant_word_count']) if pd.notna(row['assistant_word_count']) else None,
            'total_word_count': int(row['total_word_count']) if pd.notna(row['total_word_count']) else None
        })

    minified_path = os.path.join(OUTPUT_DIR, 'analysis_data.jsonl')
    with open(minified_path, 'w') as f:
        for record in minified_data:
            f.write(json.dumps(record) + '\n')
    print(f"Saved minified data to {minified_path}")

    # Output 3: Summary statistics
    print("Calculating summary statistics...")
    total_conversations = len(df_filtered)
    convs_with_suggestions = (df_filtered['total_suggestions_given_count'] > 0).sum()
    convs_with_suggestions_used = (df_filtered['total_suggestions_used'] > 0).sum()

    # Year distribution
    year_counts = df_filtered['year'].value_counts().to_dict()
    year_counts = {int(k) if pd.notna(k) else 'unknown': int(v) for k, v in year_counts.items()}

    # Language distribution
    language_counts = df_filtered['language'].value_counts().head(10).to_dict()

    # Average suggestions per year
    avg_suggestions_by_year = df_filtered.groupby('year')['total_suggestions_given_count'].mean().to_dict()
    avg_suggestions_by_year = {int(k) if pd.notna(k) else 'unknown': float(v) for k, v in avg_suggestions_by_year.items()}

    # Average suggestions per year (excluding zero)
    df_with_suggestions = df_filtered[df_filtered['total_suggestions_given_count'] > 0]
    if len(df_with_suggestions) > 0:
        avg_suggestions_by_year_nonzero = df_with_suggestions.groupby('year')['total_suggestions_given_count'].mean().to_dict()
        avg_suggestions_by_year_nonzero = {int(k) if pd.notna(k) else 'unknown': float(v) for k, v in avg_suggestions_by_year_nonzero.items()}
    else:
        avg_suggestions_by_year_nonzero = {}

    summary = {
        'total_conversations': int(total_conversations),
        'conversations_with_suggestions': int(convs_with_suggestions),
        'conversations_with_suggestions_used': int(convs_with_suggestions_used),
        'suggestion_usage_rate': float(convs_with_suggestions_used / convs_with_suggestions * 100) if convs_with_suggestions > 0 else 0,
        'year_distribution': year_counts,
        'language_distribution': language_counts,
        'avg_suggestions_by_year': avg_suggestions_by_year,
        'avg_suggestions_by_year_nonzero': avg_suggestions_by_year_nonzero
    }

    summary_path = os.path.join(OUTPUT_DIR, 'summary_stats.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to {summary_path}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total conversations: {total_conversations}")
    print(f"Conversations with suggestions: {convs_with_suggestions}")
    print(f"Conversations with suggestions used: {convs_with_suggestions_used}")
    if convs_with_suggestions > 0:
        print(f"Suggestion usage rate: {convs_with_suggestions_used / convs_with_suggestions * 100:.2f}%")
    print(f"\nYear distribution:")
    for year, count in sorted(year_counts.items(), key=lambda x: str(x[0])):
        print(f"  {year}: {count}")
    print(f"\nTop languages:")
    for lang, count in list(language_counts.items())[:5]:
        print(f"  {lang}: {count}")


if __name__ == '__main__':
    main()
