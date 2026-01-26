#!/usr/bin/env python3
"""
Process Wildchat conversations to extract follow-up suggestions.
Saves results incrementally to JSONL and skips already processed conversations.
"""

import argparse
import ast
import datetime as _dt
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime
from typing import Any, Set
import dotenv
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

# Global logger for file logging (thread-safe)
file_logger = None

# Lock for output file writes
output_file_lock = threading.Lock()

dotenv.load_dotenv()

# Regex pattern to extract conversation_hash without full JSON parsing
HASH_PATTERN = re.compile(r'"conversation_hash"\s*:\s*"([^"]+)"')

# ==================== Safe Evaluator for Parsing ====================

class _SafeEvaluator(ast.NodeVisitor):
    """
    Safely evaluate a restricted subset of Python literals plus datetime.datetime(...).
    Disallows arbitrary names, attributes, calls, and comprehensions.
    """
    # Top-level
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    # Literals (3.8+ collapse into Constant)
    def visit_Constant(self, node: ast.Constant) -> Any:
        val = node.value
        if isinstance(val, (str, bytes, int, float, bool)) or val is None:
            return val
        raise ValueError(f"Unsupported constant type: {type(val).__name__}")

    # Containers
    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    # Names (allow only True/False/None; 'datetime' is allowed only as part of datetime.datetime)
    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[node.id]
        if node.id == "datetime":
            # Mark this to be combined with .datetime below
            return ("__NAME__", "datetime")
        raise ValueError(f"Unsupported name: {node.id!r}")

    # Attributes (allow only datetime.datetime)
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        base = self.visit(node.value)
        if base == ("__NAME__", "datetime") and node.attr == "datetime":
            return ("__ATTR__", "datetime.datetime")
        raise ValueError("Unsupported attribute access.")

    # Calls (allow only datetime.datetime(...) with positional integer args)
    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if func == ("__ATTR__", "datetime.datetime"):
            if node.keywords:
                raise ValueError("datetime.datetime with keyword args is not allowed.")
            args = [self.visit(a) for a in node.args]
            if not all(isinstance(a, int) for a in args):
                raise ValueError("datetime.datetime args must be integers.")
            return _dt.datetime(*args)
        raise ValueError("Unsupported function call.")

    # Anything else is rejected
    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def parse_pythonish_record(text: str) -> Any:
    """
    Parse a string that looks like a Python literal (lists/dicts) and may contain
    datetime.datetime(...) into native Python objects—safely.
    """
    # Stage 1: If it's a one-piece quoted string, decode escapes with literal_eval
    if text and text[0] in "\"'" and text[-1] == text[0]:
        try:
            text = ast.literal_eval(text)   # safe: only decodes the outer string literal
        except Exception:
            # If that fails (rare), fall back to removing a single matching pair of quotes
            text = text[1:-1]

    # Stage 2: Parse the inner expression and evaluate safely
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Input is not a valid Python literal expression: {e}") from e

    return _SafeEvaluator().visit(tree)


# ==================== Pydantic Models ====================

class FollowUpSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suggestion_id: int
    suggestion: str  # contextualized suggestion text
    span: str  # minimal quoted span used for matching


class FollowUpSuggestions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    follow_up_suggestions: list[FollowUpSuggestion]


class AIMessageInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    follow_up_suggestions: FollowUpSuggestions


class UserMessageInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_message_uses_follow_up_suggestions: bool
    used_follow_up_suggestions_ids: list[int]
    user_message_exists: bool
    user_message_exists: bool


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_number: int
    AIMessageInformation: AIMessageInformation
    UserMessageInformation: UserMessageInformation | None


class ConversationClassified(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_turns: list[Turn]


# ==================== Utility Functions ====================

def pydantic_to_json_schema(model_class):
    """Convert a Pydantic model to JSON Schema format for structured outputs."""
    schema = model_class.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "strict": True,
            "schema": schema
        }
    }


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def load_system_prompt(prompt_path: str) -> str:
    """Load system prompt from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def build_input_string(row) -> str:
    """Build input string from conversation row."""
    full_messages = row.conversation_compiled

    roles = []
    messages = []
    for message in full_messages:
        roles.append(message['role'])
        messages.append(message['content'])

    if roles[0] != 'user':
        raise ValueError("First message must be a user message")
    if roles[-1] != 'assistant':
        raise ValueError("Last message must be an assistant message")
    for i in range(len(roles) - 1):
        if roles[i] == roles[i+1]:
            raise ValueError("Two consecutive messages cannot have the same role")

    # Skip first user message (we analyze from the first AI response onward)
    roles = roles[1:]
    messages = messages[1:]
    turns = [i // 2 + 1 for i in range(0, len(messages))]

    return "\n".join([f"{role} (turn {turn}): {message}" for role, message, turn in zip(roles, messages, turns)])


def load_processed_hashes(output_file: str) -> Set[str]:
    """Load conversation hashes without parsing full JSON records.

    Uses regex extraction for memory efficiency - avoids allocating
    full dict objects for each record when we only need the hash.
    """
    processed_hashes = set()

    if not os.path.exists(output_file):
        return processed_hashes

    try:
        with open(output_file, 'r') as f:
            for line in f:
                match = HASH_PATTERN.search(line)
                if match:
                    processed_hashes.add(match.group(1))
    except Exception as e:
        print(f"Warning: Error loading processed hashes from {output_file}: {e}")

    return processed_hashes


def load_existing_results(processed_file: str) -> dict[str, dict]:
    """Load all results from a processed file, keyed by conversation_hash."""
    existing_results = {}
    if not os.path.exists(processed_file):
        return existing_results
    try:
        with open(processed_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'conversation_hash' in data:
                        existing_results[data['conversation_hash']] = data
    except Exception as e:
        print(f"Warning: Error loading existing results from {processed_file}: {e}")
    return existing_results


def append_result_to_file(result: dict, output_file: str, output_lock: 'threading.Lock'):
    """Append a single result to the output JSONL file (thread-safe)."""
    line = json.dumps(result, cls=DateTimeEncoder, ensure_ascii=False) + '\n'
    with output_lock:
        with open(output_file, 'a') as f:
            f.write(line)


def setup_file_logger(log_file: str):
    """Set up a thread-safe file logger."""
    global file_logger
    file_logger = logging.getLogger('process_wildchat')
    file_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    file_logger.handlers.clear()

    # Create file handler
    handler = logging.FileHandler(log_file, mode='a')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(message)s'))
    file_logger.addHandler(handler)

    # Prevent propagation to root logger
    file_logger.propagate = False


def process_row(index_row_tuple, system_prompt: str, api_key: str, model: str, log_file: str = None, max_retries: int = 2):
    """Process a single row and return the result. Never raises exceptions."""
    index, row = index_row_tuple
    start_time = time.time()

    def log(msg):
        """Log message using thread-safe logger."""
        if file_logger:
            file_logger.info(msg)

    try:
        # Build input
        try:
            input_text = build_input_string(row)
        except Exception as e:
            log(f"✗ Error building input for index {index}: {e}")
            return None

        # Validate API key
        if not api_key or api_key.strip() == "":
            log(f"✗ Error at index {index}: OPENROUTER_API_KEY not set")
            return None

        # Create client once
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Retry loop for API calls
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text}
                    ],
                    extra_body={
                        "response_format": pydantic_to_json_schema(ConversationClassified),
                        "temperature": 0.0,
                        "reasoning": {
                            "effort": "medium"
                        },
                        "usage": {
                            "include": True
                        },
                        "allow_fallbacks": False,
                        "provider":{
                            "only": [
                                'atlas-cloud/fp8'
                            ]
                        }
                      
                    }
                )

                # Check finish_reason for content filtering or other issues
                if response.choices and len(response.choices) > 0:
                    finish_reason = getattr(response.choices[0], 'finish_reason', None)
                    if finish_reason and finish_reason != 'stop':
                        log(f"⚠ Index {index} attempt {attempt + 1}: finish_reason={finish_reason}")
                        if finish_reason == 'content_filter':
                            log(f"✗ Content filtered at index {index}")
                            return None
                        elif finish_reason == 'length' and attempt < max_retries:
                            # Retry for length issues
                            time.sleep(1)
                            continue

                # Break retry loop if successful
                break

            except Exception as e:
                if attempt < max_retries:
                    log(f"⚠ API error at index {index} attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    elapsed = time.time() - start_time
                    log(f"✗ API error at index {index} after {max_retries + 1} attempts ({elapsed:.1f}s): {e}")
                    return None

        # Parse response
        try:
            # Validate response structure
            if not response.choices or len(response.choices) == 0:
                log(f"✗ No choices in response at index {index}")
                return None

            if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
                log(f"✗ No message in response at index {index}")
                return None

            # Try to get content from message
            message = response.choices[0].message
            content = getattr(message, 'content', None)

            # Log full response for debugging
            log(f"📋 Full response for index {index}:")
            if hasattr(response, 'model_dump'):
                try:
                    log(json.dumps(response.model_dump(), indent=2, default=str))
                except Exception as e:
                    log(f"  Could not serialize response: {e}")

            # Get reasoning trace if available
            reasoning_trace = getattr(message, 'reasoning', None)
            if reasoning_trace:
                log(f"📝 Reasoning trace for index {index}:")
                log(f"  {reasoning_trace}")

            # Some APIs might use 'text' instead of 'content'
            if not content:
                content = getattr(message, 'text', None)

            # Log empty or problematic responses
            if not content or not content.strip():
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                # Check if there's a refusal message
                refusal = getattr(response.choices[0].message, 'refusal', None)

                # Debug: log full response details
                log(f"✗ Empty response at index {index}:")
                log(f"  finish_reason: {finish_reason}")
                log(f"  content type: {type(content)}")
                log(f"  content repr: {repr(content)}")
                log(f"  content is None: {content is None}")
                log(f"  refusal: {refusal}")

                # Log all message attributes to see what's available
                message_attrs = [attr for attr in dir(message) if not attr.startswith('_')]
                log(f"  message attributes: {message_attrs}")

                # Try to serialize the message to see its full structure
                try:
                    if hasattr(message, 'model_dump'):
                        log(f"  message dump: {message.model_dump()}")
                    elif hasattr(message, 'dict'):
                        log(f"  message dict: {message.dict()}")
                except Exception as e:
                    log(f"  Could not dump message: {e}")

                # Also check usage to see if model generated any tokens
                if hasattr(response, 'usage') and response.usage:
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                    log(f"  completion_tokens: {completion_tokens}")

                return None

            # Log the raw content for debugging if it fails to parse
            try:
                data = json.loads(content)
            except json.JSONDecodeError as json_err:
                log(f"✗ JSON decode error at index {index}: {json_err}")
                log(f"  Raw content: {content}")
                return None

            # Extract usage information
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                    'cost': getattr(response.usage, 'cost', 0.0)
                }
                # Try to get reasoning tokens if available
                if hasattr(response.usage, 'completion_tokens_details'):
                    details = response.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        usage_info['reasoning_tokens'] = details.reasoning_tokens

            elapsed = time.time() - start_time
            tokens_str = f"{usage_info.get('total_tokens', 0)} tokens" if usage_info else "no usage info"
            cost_str = f"${usage_info.get('cost', 0):.6f}" if usage_info and usage_info.get('cost') else "no cost"
            reasoning_str = f"reasoning={usage_info.get('reasoning_tokens', 0)}" if usage_info else ""
            log(f"✓ Index {index}: {len(data['conversation_turns'])} turns, {len(row.conversation_compiled)} messages, {tokens_str}, {reasoning_str}, {cost_str} ({elapsed:.1f}s)")

            # Include reasoning trace in the result if available
            result = {
                'follow_up_data': data,
                'conversation_hash': row.conversation_hash,
                'conversation': row.conversation_compiled,
                'index': index,
                'usage': usage_info
            }

            # Add reasoning trace if it exists
            if reasoning_trace:
                result['reasoning_trace'] = reasoning_trace

            return result
        except Exception as e:
            elapsed = time.time() - start_time
            log(f"✗ Parse error at index {index} ({elapsed:.1f}s): {e}")
            return None

    except Exception as e:
        # Catch-all for any unexpected errors
        elapsed = time.time() - start_time
        log(f"✗ Unexpected error at index {index} ({elapsed:.1f}s): {e}")
        return None


def print_statistics(output_data: list):
    """Print token usage and cost statistics."""
    print("\n" + "="*80)
    print("TOKEN USAGE AND COST STATISTICS")
    print("="*80)

    if not output_data:
        print("\n⚠️  No successful requests to calculate statistics")
        return

    # Collect usage stats from successful requests
    usage_stats = [item.get('usage', {}) for item in output_data if item.get('usage')]

    if not usage_stats:
        print("\n⚠️  No usage data available in the results")
        return

    total_prompt_tokens = sum(u.get('prompt_tokens', 0) for u in usage_stats)
    total_completion_tokens = sum(u.get('completion_tokens', 0) for u in usage_stats)
    total_tokens = sum(u.get('total_tokens', 0) for u in usage_stats)
    total_cost = sum(u.get('cost', 0) for u in usage_stats)
    total_reasoning_tokens = sum(u.get('reasoning_tokens', 0) for u in usage_stats)

    num_successful = len(usage_stats)

    print(f"\nTotal Requests with Usage Data: {num_successful}")
    print(f"\nToken Counts:")
    print(f"  Prompt Tokens:      {total_prompt_tokens:,}")
    print(f"  Completion Tokens:  {total_completion_tokens:,}")
    print(f"  Reasoning Tokens:   {total_reasoning_tokens:,}")
    print(f"  Total Tokens:       {total_tokens:,}")

    print(f"\nCost:")
    print(f"  Total Cost:         ${total_cost:.6f}")
    print(f"  Mean Cost per Item: ${total_cost / num_successful:.6f}")

    print(f"\nAverages per Item:")
    print(f"  Avg Prompt Tokens:      {total_prompt_tokens / num_successful:,.1f}")
    print(f"  Avg Completion Tokens:  {total_completion_tokens / num_successful:,.1f}")
    print(f"  Avg Reasoning Tokens:   {total_reasoning_tokens / num_successful:,.1f}")
    print(f"  Avg Total Tokens:       {total_tokens / num_successful:,.1f}")

    print("="*80)


# ==================== Main Processing Function ====================

def main():
    parser = argparse.ArgumentParser(
        description="Process Wildchat conversations to extract follow-up suggestions"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/wildchat/filtered_45k.jsonl',
        help='Input JSONL file containing wildchat conversations (default: data/wildchat/filtered_45k.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_data/extracted_follow_ups_45k.jsonl',
        help='Output JSONL file to save results (default: data/processed_data/extracted_follow_ups_45k.jsonl)'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default='system_prompt.md',
        help='Path to system prompt file (default: system_prompt.md)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek/deepseek-v3.2',
        help='Model to use for processing (default: deepseek/deepseek-v3.2)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=40,
        help='Number of parallel workers (default: 40)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=180,
        help='Timeout per request in seconds (default: 180)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only first N conversations (default: None, process all)'
    )
    parser.add_argument(
        '--skip-confirmation',
        action='store_true',
        help='Skip confirmation prompt for large datasets'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='data/processed_data/logs.txt',
        help='Log file for detailed processing logs (default: data/processed_data/logs.txt)'
    )
    parser.add_argument(
        '--reuse-from',
        type=str,
        default=None,
        help='Path to existing processed JSONL file to reuse results from (for conversations with matching hashes)'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Ensure log directory exists
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # Load system prompt
    print(f"Loading system prompt from {args.system_prompt}...")
    system_prompt = load_system_prompt(args.system_prompt)
    print(system_prompt[:100])

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return 1

    # Load already processed hashes
    print(f"Loading processed conversation hashes from {args.output}...")
    processed_hashes = load_processed_hashes(args.output)
    print(f"Found {len(processed_hashes)} already processed conversations")

    # Stream input data - only load unprocessed rows to save memory
    print(f"\nStreaming input data from {args.input} (filtering as we read)...")
    rows_to_process = []
    seen_hashes = set()
    total_lines = 0
    skipped_processed = 0
    skipped_language = 0
    skipped_short = 0
    skipped_duplicate = 0

    with open(args.input, 'r') as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            conv_hash = record.get('conversation_hash')

            # Skip already processed
            if conv_hash in processed_hashes:
                skipped_processed += 1
                continue

            # Skip duplicates within this file
            if conv_hash in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(conv_hash)

            # Filter for English only
            if record.get('language') != 'English':
                skipped_language += 1
                continue

            # Parse conversation
            conv = record.get('conversation')
            if isinstance(conv, str):
                try:
                    conv = parse_pythonish_record(conv)
                except Exception:
                    continue
            record['conversation_compiled'] = conv

            # Filter for min length
            if len(conv) < 2:
                skipped_short += 1
                continue

            # Create a simple namespace object to match expected interface
            class Row:
                pass
            row = Row()
            row.conversation_hash = conv_hash
            row.conversation_compiled = conv
            row.turn = record.get('turn', len(conv) // 2)

            rows_to_process.append(row)

    print(f"  Total lines read: {total_lines}")
    print(f"  Skipped (already processed): {skipped_processed}")
    print(f"  Skipped (non-English): {skipped_language}")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (duplicate): {skipped_duplicate}")
    print(f"\nTotal to process: {len(rows_to_process)} conversations")

    # Handle reuse from existing results file
    reused_count = 0
    if args.reuse_from and len(rows_to_process) > 0:
        print(f"\nLoading existing results to reuse from {args.reuse_from}...")
        reuse_results = load_existing_results(args.reuse_from)
        print(f"Loaded {len(reuse_results)} existing results for potential reuse")

        # Find conversations that can be reused
        current_hashes = {row.conversation_hash for row in rows_to_process}
        reusable_hashes = current_hashes & set(reuse_results.keys())

        if reusable_hashes:
            print(f"Found {len(reusable_hashes)} conversations to reuse (skipping API calls)")

            # Write reused results to output file
            for conv_hash in reusable_hashes:
                existing_result = reuse_results[conv_hash].copy()
                existing_result['reused_from'] = args.reuse_from  # Mark as reused for tracking
                append_result_to_file(existing_result, args.output, output_file_lock)
                reused_count += 1

            # Filter rows_to_process to only keep rows that need processing
            rows_to_process = [row for row in rows_to_process if row.conversation_hash not in reusable_hashes]
            print(f"Remaining conversations to process via API: {len(rows_to_process)}")
        else:
            print("No overlapping conversations found to reuse")

    # Show turn distribution
    if len(rows_to_process) > 0:
        from collections import Counter
        turn_counter = Counter(row.turn for row in rows_to_process)
        print(f"\nTurn distribution:")
        cumulative = 0
        for turns in sorted(turn_counter.keys()):
            count = turn_counter[turns]
            cumulative += count
            pct = count / len(rows_to_process) * 100
            cum_pct = cumulative / len(rows_to_process) * 100
            bar = '#' * int(pct / 2)
            print(f"  {turns:2d} turns: {count:6,} ({pct:5.1f}%) cum:{cum_pct:5.1f}% {bar}")

    # Limit to first N if requested
    if args.limit:
        print(f"Limiting to first {args.limit} conversations...")
        rows_to_process = rows_to_process[:args.limit]

    if len(rows_to_process) == 0:
        print("\nNo new conversations to process!")
        return 0

    # Confirmation
    if not args.skip_confirmation and len(rows_to_process) > 50:
        response = input(f"\nProcess {len(rows_to_process)} conversations? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled by user")
            return 0

    # Prepare rows as (index, row) tuples for process_row
    rows_to_process = [(idx, row) for idx, row in enumerate(rows_to_process)]

    print(f"\n{'='*80}")
    print(f"PROCESSING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model:              {args.model}")
    print(f"Workers:            {args.workers}")
    print(f"Timeout:            {args.timeout}s")
    print(f"Conversations:      {len(rows_to_process)}")
    print(f"Output file:        {args.output}")
    if args.log_file:
        print(f"Log file:           {args.log_file}")
        setup_file_logger(args.log_file)
    print(f"{'='*80}\n")

    # Process in parallel
    output_data = []
    start_time = time.time()

    # Tracking variables
    completed = 0
    failed = 0
    timed_out = 0
    total_cost = 0.0
    total_tokens = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_row, row_data, system_prompt, api_key, args.model, args.log_file): row_data[0]
                for row_data in rows_to_process
            }

            total = len(future_to_index)

            # Track submission time for each future
            future_start_times = {future: time.time() for future in future_to_index}

            # Progress bar
            with tqdm(total=total, desc="Processing", unit="conv") as pbar:
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    submit_time = future_start_times[future]
                    wait_time = time.time() - submit_time

                    try:
                        # Try to get result with timeout
                        result = future.result(timeout=args.timeout)
                        if result is not None:
                            # Save immediately to file
                            append_result_to_file(result, args.output, output_file_lock)
                            output_data.append(result)
                            completed += 1

                            # Update cost and token tracking
                            if 'usage' in result and result['usage']:
                                total_cost += result['usage'].get('cost', 0.0)
                                total_tokens += result['usage'].get('total_tokens', 0)
                        else:
                            failed += 1
                    except TimeoutError:
                        tqdm.write(f"⏱ TIMEOUT Index {idx} after {wait_time:.1f}s")
                        future.cancel()
                        timed_out += 1
                        failed += 1
                    except Exception as e:
                        tqdm.write(f"✗ Index {idx}: Error after {wait_time:.1f}s: {e}")
                        failed += 1

                    # Update progress bar
                    processed = completed + failed
                    fail_rate = (failed / processed * 100) if processed > 0 else 0
                    pbar.set_postfix({
                        'Success': completed,
                        'Failed': f"{failed} ({fail_rate:.1f}%)",
                        'Cost': f"${total_cost:.4f}",
                        'Tokens': f"{total_tokens:,}"
                    })
                    pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
    except Exception as e:
        print(f"\n\n⚠️  Fatal error in main loop: {e}")

    # Final statistics
    elapsed_total = time.time() - start_time
    print(f"\n✓ Completed: {len(output_data)}/{len(rows_to_process)} in {elapsed_total/60:.1f}m")
    print(f"Success: {completed}, Failed: {failed - timed_out}, Timed out: {timed_out}")

    # Show which indices failed (using actual dataframe indices, not sequential 0-N)
    processed_indices = {item['index'] for item in output_data}
    all_indices = {idx for idx, _ in rows_to_process}
    failed_indices = all_indices - processed_indices
    if failed_indices:
        print(f"\nFailed/timed-out indices: {sorted(failed_indices)}")

    # Print statistics
    print_statistics(output_data)

    print(f"\n✓ Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
