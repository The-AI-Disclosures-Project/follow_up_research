#!/usr/bin/env python3
"""
Download the allenai/WildChat-4.8M dataset from Hugging Face.

This dataset is very large (~4.8M conversations), so this script provides
options for:
1. Streaming (no local storage, process on-the-fly)
2. Partial download (first N examples)
3. Full download (requires significant disk space)

Usage:
    python download_wildchat.py --mode stream --limit 100
    python download_wildchat.py --mode download --output ./data/wildchat
    python download_wildchat.py --mode download --limit 10000 --output ./data/wildchat_sample
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def download_dataset(output_dir: str, limit: int | None = None, split: str = "train"):
    """
    Download the WildChat dataset to disk as JSONL files.

    Args:
        output_dir: Directory to save the dataset
        limit: Maximum number of examples to download (None for all)
        split: Dataset split to download
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading WildChat dataset (split: {split})...")
    print("Note: This dataset is ~4.8M conversations and may take a while.")

    # Use streaming to avoid loading everything into memory
    dataset = load_dataset(
        "allenai/WildChat-4.8M",
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    output_file = output_path / f"wildchat_{split}.jsonl"

    print(f"Saving to {output_file}...")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False, cls=DateTimeEncoder) + "\n")
            count += 1

            if count % 10000 == 0:
                print(f"  Saved {count:,} examples...")

            if limit and count >= limit:
                print(f"  Reached limit of {limit:,} examples.")
                break

    print(f"Done! Saved {count:,} examples to {output_file}")
    return output_file


def stream_dataset(limit: int | None = None, split: str = "train"):
    """
    Stream the WildChat dataset without downloading.
    Useful for quick inspection or processing without storage.

    Args:
        limit: Maximum number of examples to process
        split: Dataset split to stream

    Yields:
        dict: Each conversation example
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return

    print(f"Streaming WildChat dataset (split: {split})...")

    dataset = load_dataset(
        "allenai/WildChat-4.8M",
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    count = 0
    for example in dataset:
        yield example
        count += 1

        if limit and count >= limit:
            break


def inspect_dataset(num_examples: int = 5):
    """
    Inspect the first few examples of the dataset to understand the structure.

    Args:
        num_examples: Number of examples to inspect
    """
    print(f"Inspecting first {num_examples} examples from WildChat dataset...")
    print("=" * 80)

    for i, example in enumerate(stream_dataset(limit=num_examples)):
        print(f"\n--- Example {i + 1} ---")
        print(f"Keys: {list(example.keys())}")

        # Show conversation structure
        if "conversation" in example:
            conv = example["conversation"]
            print(f"Conversation length: {len(conv)} turns")
            if conv:
                print(f"First turn keys: {list(conv[0].keys()) if isinstance(conv[0], dict) else type(conv[0])}")

        # Show other metadata
        for key in example:
            if key != "conversation":
                value = example[key]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"{key}: {value}")

        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download or stream the allenai/WildChat-4.8M dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["download", "stream", "inspect"],
        default="inspect",
        help="Mode: 'download' saves to disk, 'stream' processes on-the-fly, 'inspect' shows structure"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "data" / "wildchat"),
        help="Output directory for downloaded data (defaults to ./data/wildchat in script directory)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (None for all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
    )

    args = parser.parse_args()

    if args.mode == "download":
        download_dataset(args.output, limit=args.limit, split=args.split)
    elif args.mode == "stream":
        print("Streaming mode - processing examples...")
        for i, example in enumerate(stream_dataset(limit=args.limit, split=args.split)):
            print(f"Example {i + 1}: {len(example.get('conversation', []))} turns")
    elif args.mode == "inspect":
        inspect_dataset(num_examples=args.limit or 5)


if __name__ == "__main__":
    main()
