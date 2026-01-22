#!/usr/bin/env python3
"""
Utilities for efficiently reading the WildChat dataset line by line.

These utilities are designed for memory-efficient processing of the large
WildChat-4.8M dataset, supporting:
- Line-by-line iteration (JSONL files)
- Random access by line number
- Batched reading
- Filtering and sampling

Usage:
    from wildchat_utils import WildChatReader

    # Iterate line by line
    reader = WildChatReader("data/wildchat/wildchat_train.jsonl")
    for conversation in reader:
        process(conversation)

    # Get specific line
    conv = reader.get_line(1000)

    # Get batch of lines
    batch = reader.get_batch(start=0, size=100)

    # Random sample
    samples = reader.random_sample(n=50)
"""

import json
import mmap
import os
import random
from pathlib import Path
from typing import Iterator, Callable


class ReservoirSampler:
    """
    Reservoir sampling (Algorithm R) for memory-efficient random sampling.

    Maintains a fixed-size sample in O(k) memory regardless of input size.
    Each item in the stream has equal probability of being in the final sample.

    Usage:
        sampler = ReservoirSampler(k=1000)
        for item in huge_stream:
            sampler.add(item)
        sample = sampler.get_sample()  # Exactly 1000 items (or fewer if stream was smaller)
    """

    def __init__(self, k: int, seed: int | None = None):
        """
        Initialize the reservoir sampler.

        Args:
            k: Maximum number of items to keep in the sample
            seed: Random seed for reproducibility
        """
        self.k = k
        self.reservoir: list = []
        self.n = 0  # Total items seen
        if seed is not None:
            random.seed(seed)

    def add(self, item):
        """
        Add an item to the reservoir.

        Args:
            item: Item to potentially include in the sample
        """
        self.n += 1
        if len(self.reservoir) < self.k:
            # Reservoir not full yet, add directly
            self.reservoir.append(item)
        else:
            # Reservoir full, randomly decide whether to replace
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.reservoir[j] = item

    def get_sample(self) -> list:
        """
        Get the current sample.

        Returns:
            List of sampled items (up to k items)
        """
        return self.reservoir

    def __len__(self) -> int:
        """Return number of items currently in reservoir."""
        return len(self.reservoir)

    def total_seen(self) -> int:
        """Return total number of items that were added."""
        return self.n


class WildChatReader:
    """
    Memory-efficient reader for WildChat JSONL files.

    Supports streaming iteration without loading the entire file into memory.
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize the reader.

        Args:
            filepath: Path to the JSONL file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self._line_offsets: list[int] | None = None
        self._total_lines: int | None = None

    def __iter__(self) -> Iterator[dict]:
        """Iterate through all conversations line by line."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def __len__(self) -> int:
        """Return total number of lines (builds index if needed)."""
        if self._total_lines is None:
            self._build_line_index()
        return self._total_lines

    def _build_line_index(self):
        """Build an index of line offsets for random access."""
        print(f"Building line index for {self.filepath}...")
        self._line_offsets = [0]

        with open(self.filepath, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self._line_offsets.append(f.tell())

        # Remove the last offset (EOF position)
        self._line_offsets.pop()
        self._total_lines = len(self._line_offsets)
        print(f"Indexed {self._total_lines:,} lines.")

    def get_line(self, line_number: int) -> dict:
        """
        Get a specific line by number (0-indexed).

        Args:
            line_number: The line number to retrieve

        Returns:
            dict: The parsed JSON object
        """
        if self._line_offsets is None:
            self._build_line_index()

        if line_number < 0 or line_number >= len(self._line_offsets):
            raise IndexError(f"Line {line_number} out of range (0-{len(self._line_offsets) - 1})")

        with open(self.filepath, "r", encoding="utf-8") as f:
            f.seek(self._line_offsets[line_number])
            line = f.readline().strip()
            return json.loads(line)

    def get_batch(self, start: int = 0, size: int = 100) -> list[dict]:
        """
        Get a batch of consecutive lines.

        Args:
            start: Starting line number (0-indexed)
            size: Number of lines to retrieve

        Returns:
            list[dict]: List of parsed JSON objects
        """
        if self._line_offsets is None:
            self._build_line_index()

        end = min(start + size, len(self._line_offsets))
        results = []

        with open(self.filepath, "r", encoding="utf-8") as f:
            f.seek(self._line_offsets[start])
            for _ in range(end - start):
                line = f.readline().strip()
                if line:
                    results.append(json.loads(line))

        return results

    def random_sample(self, n: int, seed: int | None = None) -> list[dict]:
        """
        Get a random sample of lines.

        Args:
            n: Number of samples to retrieve
            seed: Random seed for reproducibility

        Returns:
            list[dict]: List of randomly sampled conversations
        """
        if self._line_offsets is None:
            self._build_line_index()

        if seed is not None:
            random.seed(seed)

        n = min(n, len(self._line_offsets))
        indices = random.sample(range(len(self._line_offsets)), n)
        indices.sort()  # Sort for sequential file access (more efficient)

        results = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for idx in indices:
                f.seek(self._line_offsets[idx])
                line = f.readline().strip()
                if line:
                    results.append(json.loads(line))

        return results

    def iterate_with_filter(
        self,
        filter_fn: Callable[[dict], bool],
        limit: int | None = None
    ) -> Iterator[dict]:
        """
        Iterate with a filter function.

        Args:
            filter_fn: Function that returns True to include the item
            limit: Maximum number of items to yield

        Yields:
            dict: Conversations that pass the filter
        """
        count = 0
        for item in self:
            if filter_fn(item):
                yield item
                count += 1
                if limit and count >= limit:
                    break

    def iterate_batched(self, batch_size: int = 100) -> Iterator[list[dict]]:
        """
        Iterate in batches for batch processing.

        Args:
            batch_size: Number of items per batch

        Yields:
            list[dict]: Batches of conversations
        """
        batch = []
        for item in self:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class WildChatStreamer:
    """
    Stream WildChat directly from Hugging Face without local storage.

    Use this when you don't want to download the dataset.
    """

    def __init__(self, split: str = "train"):
        """
        Initialize the streamer.

        Args:
            split: Dataset split to stream
        """
        self.split = split
        self._dataset = None

    def _load_dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "datasets library required. Install with: pip install datasets"
                )

            self._dataset = load_dataset(
                "allenai/WildChat",
                split=self.split,
                streaming=True,
                trust_remote_code=True
            )

    def __iter__(self) -> Iterator[dict]:
        """Iterate through all conversations."""
        self._load_dataset()
        for item in self._dataset:
            yield item

    def take(self, n: int) -> list[dict]:
        """
        Take the first n items.

        Args:
            n: Number of items to take

        Returns:
            list[dict]: First n conversations
        """
        self._load_dataset()
        results = []
        for i, item in enumerate(self._dataset):
            if i >= n:
                break
            results.append(item)
        return results

    def skip_take(self, skip: int, take: int) -> list[dict]:
        """
        Skip some items then take some.

        Args:
            skip: Number of items to skip
            take: Number of items to take after skipping

        Returns:
            list[dict]: Conversations after skipping
        """
        self._load_dataset()
        results = []
        for i, item in enumerate(self._dataset):
            if i < skip:
                continue
            if i >= skip + take:
                break
            results.append(item)
        return results


def count_lines(filepath: str | Path) -> int:
    """
    Quickly count lines in a file using memory mapping.

    Args:
        filepath: Path to the file

    Returns:
        int: Number of lines
    """
    filepath = Path(filepath)
    with open(filepath, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        count = 0
        while mm.readline():
            count += 1
        mm.close()
    return count


def iter_jsonl(filepath: str | Path) -> Iterator[dict]:
    """
    Simple generator to iterate over a JSONL file.

    Args:
        filepath: Path to the JSONL file

    Yields:
        dict: Each parsed JSON line
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_line(filepath: str | Path, line_number: int) -> dict:
    """
    Read a specific line from a JSONL file.

    Note: This is O(n) for each call. For repeated access,
    use WildChatReader which builds an index.

    Args:
        filepath: Path to the JSONL file
        line_number: Line number to read (0-indexed)

    Returns:
        dict: The parsed JSON object
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line.strip())
    raise IndexError(f"Line {line_number} not found in file")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wildchat_utils.py <path_to_jsonl>")
        print("\nOr use from Python:")
        print("  from wildchat_utils import WildChatReader, WildChatStreamer")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Reading {filepath}...")

    reader = WildChatReader(filepath)
    print(f"Total lines: {len(reader):,}")

    print("\nFirst 3 conversations:")
    for i, conv in enumerate(reader):
        if i >= 3:
            break
        print(f"  {i + 1}. {len(conv.get('conversation', []))} turns")

    print("\nRandom sample of 5:")
    for conv in reader.random_sample(5, seed=42):
        print(f"  - {len(conv.get('conversation', []))} turns")
