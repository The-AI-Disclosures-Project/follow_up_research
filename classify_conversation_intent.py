#!/usr/bin/env python3
"""
Classify conversation intent from the WildChat opening turns.
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from typing import Any, Literal, Set

import dotenv
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


class _SafeEvaluator(ast.NodeVisitor):
    """
    Safely evaluate a restricted subset of Python literals plus datetime.datetime(...).
    Disallows arbitrary names, attributes, calls, and comprehensions.
    """

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        val = node.value
        if isinstance(val, (str, bytes, int, float, bool)) or val is None:
            return val
        raise ValueError(f"Unsupported constant type: {type(val).__name__}")

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[node.id]
        if node.id == "datetime":
            return ("__NAME__", "datetime")
        raise ValueError(f"Unsupported name: {node.id!r}")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        base = self.visit(node.value)
        if base == ("__NAME__", "datetime") and node.attr == "datetime":
            return ("__ATTR__", "datetime.datetime")
        raise ValueError("Unsupported attribute access.")

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

    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def parse_pythonish_record(text: str) -> Any:
    """Parse a Python-like literal string safely."""
    if text and text[0] in "\"'" and text[-1] == text[0]:
        try:
            text = ast.literal_eval(text)
        except Exception:
            text = text[1:-1]

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Input is not a valid Python literal expression: {e}") from e

    return _SafeEvaluator().visit(tree)


class ConversationIntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["EXPLORATORY", "BUILDING", "INFORMATIONAL", "UNKNOWN"]


def pydantic_to_json_schema(model_class):
    """Convert a Pydantic model to JSON Schema format for structured outputs."""
    schema = model_class.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "strict": True,
            "schema": schema,
        },
    }


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def load_system_prompt(prompt_path: str) -> str:
    """Load system prompt from file."""
    with open(prompt_path, "r") as f:
        return f.read()


def infer_turn_count(conversation: list[dict]) -> int:
    """Infer turn count from alternating conversation messages."""
    assistant_messages = sum(1 for msg in conversation if msg.get("role") == "assistant")
    return max(assistant_messages, len(conversation) // 2)


def build_opening_input_string(row) -> tuple[str, list[dict]]:
    """Build the opening slice used for conversation intent classification."""
    full_messages = row.conversation_compiled
    if not full_messages:
        raise ValueError("Conversation is empty")

    first_role = full_messages[0].get("role")
    if first_role != "user":
        raise ValueError(f"First message must be from user, got {first_role!r}")

    turn_count = row.turn if row.turn is not None else infer_turn_count(full_messages)

    if turn_count >= 2:
        if len(full_messages) < 3:
            raise ValueError("Conversation has fewer than 3 messages for a 2-turn opening slice")

        opening_messages = full_messages[:3]
        opening_roles = [msg.get("role") for msg in opening_messages]
        if opening_roles != ["user", "assistant", "user"]:
            raise ValueError(f"Opening slice must be user/assistant/user, got {opening_roles!r}")
    else:
        opening_messages = [full_messages[0]]

    formatted_lines = []
    for idx, message in enumerate(opening_messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content") or ""
        formatted_lines.append(f"{role} (message {idx}): {content}")

    return "\n".join(formatted_lines), opening_messages


def load_processed_hashes(output_file: str) -> Set[str]:
    """Load conversation hashes without parsing full JSON records."""
    processed_hashes = set()

    if not os.path.exists(output_file):
        return processed_hashes

    try:
        with open(output_file, "r") as f:
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
        with open(processed_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "conversation_hash" in data:
                        existing_results[data["conversation_hash"]] = data
    except Exception as e:
        print(f"Warning: Error loading existing results from {processed_file}: {e}")

    return existing_results


def append_result_to_file(result: dict, output_file: str, output_lock: "threading.Lock"):
    """Append a single result to the output JSONL file (thread-safe)."""
    line = json.dumps(result, cls=DateTimeEncoder, ensure_ascii=False) + "\n"
    with output_lock:
        with open(output_file, "a") as f:
            f.write(line)


def setup_file_logger(log_file: str):
    """Set up a thread-safe file logger."""
    global file_logger
    file_logger = logging.getLogger("classify_conversation_intent")
    file_logger.setLevel(logging.DEBUG)
    file_logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode="a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    file_logger.addHandler(handler)
    file_logger.propagate = False


def process_row(index_row_tuple, system_prompt: str, api_key: str, model: str, log_file: str = None, max_retries: int = 2):
    """Process a single row and return the result. Never raises exceptions."""
    index, row = index_row_tuple
    start_time = time.time()

    def log(msg):
        if file_logger:
            file_logger.info(msg)

    try:
        try:
            input_text, opening_messages = build_opening_input_string(row)
        except Exception as e:
            log(f"✗ Error building input for index {index}: {e}")
            return None

        if not api_key or api_key.strip() == "":
            log(f"✗ Error at index {index}: OPENROUTER_API_KEY not set")
            return None

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    extra_body={
                        "response_format": pydantic_to_json_schema(ConversationIntentClassification),
                        "temperature": 0.0,
                        "reasoning": {
                            "effort": "medium",
                        },
                        "usage": {
                            "include": True,
                        },
                        "allow_fallbacks": False,
                        "provider": {
                            "only": [
                                "atlas-cloud/fp8",
                            ]
                        },
                    },
                )

                if response.choices and len(response.choices) > 0:
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    if finish_reason and finish_reason != "stop":
                        log(f"⚠ Index {index} attempt {attempt + 1}: finish_reason={finish_reason}")
                        if finish_reason == "content_filter":
                            log(f"✗ Content filtered at index {index}")
                            return None
                        if finish_reason == "length" and attempt < max_retries:
                            time.sleep(1)
                            continue

                break
            except Exception as e:
                if attempt < max_retries:
                    log(f"⚠ API error at index {index} attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    elapsed = time.time() - start_time
                    log(f"✗ API error at index {index} after {max_retries + 1} attempts ({elapsed:.1f}s): {e}")
                    return None

        try:
            if not response.choices or len(response.choices) == 0:
                log(f"✗ No choices in response at index {index}")
                return None

            if not hasattr(response.choices[0], "message") or not response.choices[0].message:
                log(f"✗ No message in response at index {index}")
                return None

            message = response.choices[0].message
            content = getattr(message, "content", None)

            log(f"📋 Full response for index {index}:")
            if hasattr(response, "model_dump"):
                try:
                    log(json.dumps(response.model_dump(), indent=2, default=str))
                except Exception as e:
                    log(f"  Could not serialize response: {e}")

            reasoning_trace = getattr(message, "reasoning", None)
            if reasoning_trace:
                log(f"📝 Reasoning trace for index {index}:")
                log(f"  {reasoning_trace}")

            if not content:
                content = getattr(message, "text", None)

            if not content or not content.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                refusal = getattr(response.choices[0].message, "refusal", None)
                log(f"✗ Empty response at index {index}:")
                log(f"  finish_reason: {finish_reason}")
                log(f"  content repr: {repr(content)}")
                log(f"  refusal: {refusal}")
                return None

            try:
                data = json.loads(content)
            except json.JSONDecodeError as json_err:
                log(f"✗ JSON decode error at index {index}: {json_err}")
                log(f"  Raw content: {content}")
                return None

            usage_info = {}
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                    "cost": getattr(response.usage, "cost", 0.0),
                }
                if hasattr(response.usage, "completion_tokens_details"):
                    details = response.usage.completion_tokens_details
                    if hasattr(details, "reasoning_tokens"):
                        usage_info["reasoning_tokens"] = details.reasoning_tokens

            elapsed = time.time() - start_time
            tokens_str = f"{usage_info.get('total_tokens', 0)} tokens" if usage_info else "no usage info"
            cost_str = f"${usage_info.get('cost', 0):.6f}" if usage_info and usage_info.get("cost") else "no cost"
            log(
                f"✓ Index {index}: intent={data.get('intent')}, "
                f"{len(opening_messages)} opening messages, {tokens_str}, {cost_str} ({elapsed:.1f}s)"
            )

            result = dict(row.source_record)
            result["conversation_compiled"] = row.conversation_compiled
            result["conversation_intent_classification"] = data
            result["opening_messages_used"] = opening_messages
            result["opening_message_count_used"] = len(opening_messages)
            result["index"] = index
            result["usage"] = usage_info

            if reasoning_trace:
                result["reasoning_trace"] = reasoning_trace

            return result
        except Exception as e:
            elapsed = time.time() - start_time
            log(f"✗ Parse error at index {index} ({elapsed:.1f}s): {e}")
            return None

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"✗ Unexpected error at index {index} ({elapsed:.1f}s): {e}")
        return None


def print_statistics(output_data: list):
    """Print token usage and cost statistics."""
    print("\n" + "=" * 80)
    print("TOKEN USAGE AND COST STATISTICS")
    print("=" * 80)

    if not output_data:
        print("\n⚠️  No successful requests to calculate statistics")
        return

    usage_stats = [item.get("usage", {}) for item in output_data if item.get("usage")]

    if not usage_stats:
        print("\n⚠️  No usage data available in the results")
        return

    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage_stats)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in usage_stats)
    total_tokens = sum(u.get("total_tokens", 0) for u in usage_stats)
    total_cost = sum(u.get("cost", 0) for u in usage_stats)
    total_reasoning_tokens = sum(u.get("reasoning_tokens", 0) for u in usage_stats)

    num_successful = len(usage_stats)

    print(f"\nTotal Requests with Usage Data: {num_successful}")
    print("\nToken Counts:")
    print(f"  Prompt Tokens:      {total_prompt_tokens:,}")
    print(f"  Completion Tokens:  {total_completion_tokens:,}")
    print(f"  Reasoning Tokens:   {total_reasoning_tokens:,}")
    print(f"  Total Tokens:       {total_tokens:,}")
    print("\nCost:")
    print(f"  Total Cost:         ${total_cost:.6f}")
    print(f"  Mean Cost per Item: ${total_cost / num_successful:.6f}")
    print("\nAverages per Item:")
    print(f"  Avg Prompt Tokens:      {total_prompt_tokens / num_successful:,.1f}")
    print(f"  Avg Completion Tokens:  {total_completion_tokens / num_successful:,.1f}")
    print(f"  Avg Reasoning Tokens:   {total_reasoning_tokens / num_successful:,.1f}")
    print(f"  Avg Total Tokens:       {total_tokens / num_successful:,.1f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Classify WildChat conversations from their opening turns"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/wildchat/filtered_45k.jsonl",
        help="Input JSONL file containing WildChat conversations (default: data/wildchat/filtered_45k.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_data/conversation_intents_45k.jsonl",
        help="Output JSONL file to save intent classifications (default: data/processed_data/conversation_intents_45k.jsonl)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="catagory_classification_prompt.md",
        help="Path to intent classification prompt file (default: catagory_classification_prompt.md)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="Model to use for processing (default: deepseek/deepseek-v3.2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="Number of parallel workers (default: 40)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout per request in seconds (default: 180)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N conversations (default: None, process all)",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip confirmation prompt for large datasets",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="data/processed_data/conversation_intent_logs.txt",
        help="Log file for detailed processing logs (default: data/processed_data/conversation_intent_logs.txt)",
    )
    parser.add_argument(
        "--reuse-from",
        type=str,
        default=None,
        help="Path to existing processed JSONL file to reuse results from (for conversations with matching hashes)",
    )

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    print(f"Loading system prompt from {args.system_prompt}...")
    system_prompt = load_system_prompt(args.system_prompt)
    print(system_prompt[:100])

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return 1

    print(f"Loading processed conversation hashes from {args.output}...")
    processed_hashes = load_processed_hashes(args.output)
    print(f"Found {len(processed_hashes)} already processed conversations")

    print(f"\nStreaming input data from {args.input} (filtering as we read)...")
    rows_to_process = []
    seen_hashes = set()
    total_lines = 0
    skipped_processed = 0
    skipped_language = 0
    skipped_malformed = 0
    skipped_duplicate = 0

    with open(args.input, "r") as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_malformed += 1
                continue

            conv_hash = record.get("conversation_hash")
            if conv_hash in processed_hashes:
                skipped_processed += 1
                continue

            if conv_hash in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(conv_hash)

            if record.get("language") != "English":
                skipped_language += 1
                continue

            conv = record.get("conversation")
            if isinstance(conv, str):
                try:
                    conv = parse_pythonish_record(conv)
                except Exception:
                    skipped_malformed += 1
                    continue

            if not isinstance(conv, list) or len(conv) == 0:
                skipped_malformed += 1
                continue

            first_role = conv[0].get("role")
            if first_role != "user":
                skipped_malformed += 1
                continue

            record["conversation_compiled"] = conv

            class Row:
                pass

            row = Row()
            row.conversation_hash = conv_hash
            row.conversation_compiled = conv
            row.turn = record.get("turn", infer_turn_count(conv))
            row.source_record = record
            rows_to_process.append(row)

    print(f"  Total lines read: {total_lines}")
    print(f"  Skipped (already processed): {skipped_processed}")
    print(f"  Skipped (non-English): {skipped_language}")
    print(f"  Skipped (malformed/empty): {skipped_malformed}")
    print(f"  Skipped (duplicate): {skipped_duplicate}")
    print(f"\nTotal to process: {len(rows_to_process)} conversations")

    reused_count = 0
    if args.reuse_from and len(rows_to_process) > 0:
        print(f"\nLoading existing results to reuse from {args.reuse_from}...")
        reuse_results = load_existing_results(args.reuse_from)
        print(f"Loaded {len(reuse_results)} existing results for potential reuse")

        current_hashes = {row.conversation_hash for row in rows_to_process}
        reusable_hashes = current_hashes & set(reuse_results.keys())

        if reusable_hashes:
            print(f"Found {len(reusable_hashes)} conversations to reuse (skipping API calls)")
            for conv_hash in reusable_hashes:
                existing_result = reuse_results[conv_hash].copy()
                existing_result["reused_from"] = args.reuse_from
                append_result_to_file(existing_result, args.output, output_file_lock)
                reused_count += 1

            rows_to_process = [row for row in rows_to_process if row.conversation_hash not in reusable_hashes]
            print(f"Remaining conversations to process via API: {len(rows_to_process)}")
        else:
            print("No overlapping conversations found to reuse")

    if args.limit:
        print(f"Limiting to first {args.limit} conversations...")
        rows_to_process = rows_to_process[:args.limit]

    if len(rows_to_process) == 0:
        print("\nNo new conversations to process!")
        if reused_count > 0:
            print(f"Reused {reused_count} existing results")
        return 0

    if not args.skip_confirmation and len(rows_to_process) > 50:
        response = input(f"\nProcess {len(rows_to_process)} conversations? (y/n): ")
        if response.lower() != "y":
            print("Cancelled by user")
            return 0

    rows_to_process = [(idx, row) for idx, row in enumerate(rows_to_process)]

    print(f"\n{'=' * 80}")
    print("PROCESSING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Model:              {args.model}")
    print(f"Workers:            {args.workers}")
    print(f"Timeout:            {args.timeout}s")
    print(f"Conversations:      {len(rows_to_process)}")
    print(f"Output file:        {args.output}")
    if args.log_file:
        print(f"Log file:           {args.log_file}")
        setup_file_logger(args.log_file)
    print(f"{'=' * 80}\n")

    output_data = []
    start_time = time.time()
    completed = 0
    failed = 0
    timed_out = 0
    total_cost = 0.0
    total_tokens = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {
                executor.submit(process_row, row_data, system_prompt, api_key, args.model, args.log_file): row_data[0]
                for row_data in rows_to_process
            }

            total = len(future_to_index)
            future_start_times = {future: time.time() for future in future_to_index}

            with tqdm(total=total, desc="Processing", unit="conv") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    submit_time = future_start_times[future]
                    wait_time = time.time() - submit_time

                    try:
                        result = future.result(timeout=args.timeout)
                        if result is not None:
                            append_result_to_file(result, args.output, output_file_lock)
                            output_data.append(result)
                            completed += 1
                            if "usage" in result and result["usage"]:
                                total_cost += result["usage"].get("cost", 0.0)
                                total_tokens += result["usage"].get("total_tokens", 0)
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

                    processed = completed + failed
                    fail_rate = (failed / processed * 100) if processed > 0 else 0
                    pbar.set_postfix(
                        {
                            "Success": completed,
                            "Failed": f"{failed} ({fail_rate:.1f}%)",
                            "Cost": f"${total_cost:.4f}",
                            "Tokens": f"{total_tokens:,}",
                        }
                    )
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
    except Exception as e:
        print(f"\n\n⚠️  Fatal error in main loop: {e}")

    elapsed_total = time.time() - start_time
    print(f"\n✓ Completed: {len(output_data)}/{len(rows_to_process)} in {elapsed_total / 60:.1f}m")
    print(f"Success: {completed}, Failed: {failed - timed_out}, Timed out: {timed_out}, Reused: {reused_count}")

    processed_indices = {item["index"] for item in output_data}
    all_indices = {idx for idx, _ in rows_to_process}
    failed_indices = all_indices - processed_indices
    if failed_indices:
        print(f"\nFailed/timed-out indices: {sorted(failed_indices)}")

    print_statistics(output_data)
    print(f"\n✓ Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
