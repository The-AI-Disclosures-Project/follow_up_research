#!/usr/bin/env python3
"""
Filter WildChat dataset for follow-up detection research.

Produces a 15k sample (5k per year) with:
- English only
- 2-30 turns
- No repeated template prompts (automated usage detection)
- Capped IPs to prevent single-user domination

Two-pass approach for memory efficiency:
- Pass 1: Build IP counts and template signature counts
- Pass 2: Filter and collect candidates, then sample

Usage:
    python filter_wildchat.py                    # Run full filtering
    python filter_wildchat.py --analyze          # Analyze template distribution only
    python filter_wildchat.py --analyze --top 50 # Show top 50 templates
    python filter_wildchat.py --analyze --refresh # Force re-scan (ignore cache)
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Iterator

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            print(f"{desc}...")
        return iterable

from wildchat_utils import iter_jsonl, ReservoirSampler

# =============================================================================
# Configuration
# =============================================================================

LANGUAGE_FILTER = "English"
YEARS = [2023, 2024, 2025]
SAMPLES_PER_YEAR = 15000
MIN_TURNS = 1
MAX_TURNS = 50

# IP capping disabled
MAX_PER_IP = None

# Template detection
MIN_TEMPLATE_LENGTH = 500  # chars - only check messages this long or longer
TEMPLATE_SIGNATURE_LENGTH = 500  # first N chars to compare
MIN_TEMPLATE_OCCURRENCES = 500  # threshold for "suspicious" templates

# Paths
INPUT_FILE = Path(__file__).parent / "data" / "wildchat" / "wildchat_train.jsonl"
OUTPUT_FILE = Path(__file__).parent / "data" / "wildchat" / "filtered_45k.jsonl"
CACHE_FILE = Path(__file__).parent / "data" / "wildchat" / "template_analysis_cache.jsonl"

RANDOM_SEED = 42


# =============================================================================
# Helper Functions
# =============================================================================

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def get_template_signature(content: str) -> str | None:
    """
    Extract a template signature from content.
    Returns None if content is too short to be a template.

    Args:
        content: Message content to analyze

    Returns:
        Normalized first N chars if content is long enough, else None
    """
    if not content or len(content) < MIN_TEMPLATE_LENGTH:
        return None

    # Normalize: strip whitespace, lowercase, take first N chars
    normalized = content[:TEMPLATE_SIGNATURE_LENGTH].strip().lower()
    return normalized


def extract_year(timestamp) -> int | None:
    """
    Extract year from timestamp (handles various formats).

    Args:
        timestamp: Timestamp string or datetime object

    Returns:
        Year as int, or None if parsing fails
    """
    if timestamp is None:
        return None

    if isinstance(timestamp, datetime):
        return timestamp.year

    if isinstance(timestamp, str):
        try:
            # Try ISO format first
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.year
        except (ValueError, AttributeError):
            pass

        try:
            # Try common format
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return dt.year
        except (ValueError, AttributeError):
            pass

    return None


def get_first_message_content(conversation: list) -> str:
    """
    Get the content of the first message in a conversation.

    Args:
        conversation: List of message dicts

    Returns:
        First message content, or empty string if not found
    """
    if not conversation or len(conversation) == 0:
        return ""

    first_msg = conversation[0]
    if isinstance(first_msg, dict):
        return first_msg.get('content', '') or ''

    return ""


def save_template_cache(
    template_counts: Counter,
    template_examples: dict,
    template_lengths: dict,
    ip_counts: Counter,
    cache_path: Path,
    template_ips: dict = None,
    ip_daily_counts: dict = None,
    total_conversations: int = 0,
):
    """
    Save template analysis results to disk in JSONL format for memory-efficient loading.

    Format (one JSON object per line):
        Line 1: {"type": "metadata", "version": 5, "total_conversations": N}
        Template lines: {"type": "template", "sig": "...", "count": N, "example": "...", "lengths": [...], "ips": {...}}
        IP count lines: {"type": "ip_counts_batch", "counts": {"ip1": N, "ip2": M, ...}}  (batched for efficiency)
        IP daily lines: {"type": "ip_daily", "ip": "...", "daily": {...}}

    Args:
        template_counts: Counter of template signatures
        template_examples: Dict of signature -> full example
        template_lengths: Dict of signature -> list of lengths
        ip_counts: Counter of IP addresses
        cache_path: Path to save cache
        template_ips: Dict of signature -> dict of IP -> count
        ip_daily_counts: Dict of IP -> dict of date -> count
        total_conversations: Total number of conversations in dataset
    """
    template_ips = template_ips or {}
    ip_daily_counts = ip_daily_counts or {}

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'w', encoding='utf-8') as f:
        # Line 1: Metadata
        metadata = {
            "type": "metadata",
            "version": 5,
            "total_conversations": total_conversations,
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

        # Template lines (one per template signature)
        for sig, count in template_counts.items():
            template_line = {
                "type": "template",
                "sig": sig,
                "count": count,
                "example": template_examples.get(sig, ""),
                "lengths": list(template_lengths.get(sig, [])),
                "ips": template_ips.get(sig, {}),
            }
            f.write(json.dumps(template_line, ensure_ascii=False) + '\n')

        # IP counts in batches (1000 per line to reduce line count)
        BATCH_SIZE = 1000
        ip_items = list(ip_counts.items())
        for i in range(0, len(ip_items), BATCH_SIZE):
            batch = dict(ip_items[i:i + BATCH_SIZE])
            ip_batch_line = {
                "type": "ip_counts_batch",
                "counts": batch,
            }
            f.write(json.dumps(ip_batch_line, ensure_ascii=False) + '\n')

        # IP daily counts (one per IP that has daily data)
        for ip, daily in ip_daily_counts.items():
            ip_daily_line = {
                "type": "ip_daily",
                "ip": ip,
                "daily": daily,
            }
            f.write(json.dumps(ip_daily_line, ensure_ascii=False) + '\n')

    print(f"\nCache saved to: {cache_path}")


def load_template_cache(cache_path: Path) -> tuple[Counter, dict, dict, Counter, dict, dict, int] | None:
    """
    Load template analysis results from JSONL cache, streaming line by line.

    Args:
        cache_path: Path to cache file

    Returns:
        Tuple of (template_counts, template_examples, template_lengths, ip_counts, template_ips, ip_daily_counts, total_conversations)
        or None if not found/outdated
    """
    if not cache_path.exists():
        return None

    try:
        template_counts = Counter()
        template_examples = {}
        template_lengths = defaultdict(list)
        ip_counts = Counter()
        template_ips = {}
        ip_daily_counts = {}
        total_conversations = 0

        with open(cache_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                record_type = record.get("type")

                if record_type == "metadata":
                    if record.get("version") != 5:
                        print(f"Cache version mismatch (got {record.get('version')}, need 5), will re-scan...")
                        return None
                    total_conversations = record.get("total_conversations", 0)

                elif record_type == "template":
                    sig = record["sig"]
                    template_counts[sig] = record["count"]
                    template_examples[sig] = record.get("example", "")
                    template_lengths[sig] = record.get("lengths", [])
                    template_ips[sig] = record.get("ips", {})

                elif record_type == "ip_counts_batch":
                    ip_counts.update(record["counts"])

                elif record_type == "ip_daily":
                    ip = record["ip"]
                    ip_daily_counts[ip] = record["daily"]

        return template_counts, template_examples, template_lengths, ip_counts, template_ips, ip_daily_counts, total_conversations

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Cache corrupted ({e}), will re-scan...")
        return None


def estimate_total_lines(filepath: Path) -> int:
    """
    Estimate total lines in file based on file size and sample.

    Args:
        filepath: Path to the file

    Returns:
        Estimated number of lines
    """
    file_size = filepath.stat().st_size

    # Sample first 1000 lines to get average line size
    sample_size = 0
    sample_lines = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            sample_size += len(line.encode('utf-8'))
            sample_lines += 1

    if sample_lines == 0:
        return 0

    avg_line_size = sample_size / sample_lines
    return int(file_size / avg_line_size)


# =============================================================================
# Pass 1: Build Indexes
# =============================================================================

def build_indexes(filepath: Path) -> tuple[Counter, Counter]:
    """
    First pass: build IP counts and template signature counts.
    Only considers English conversations.

    Args:
        filepath: Path to the JSONL file

    Returns:
        Tuple of (ip_counts, template_counts)
    """
    ip_counts = Counter()
    template_counts = Counter()

    # Estimate total for progress bar
    estimated_total = estimate_total_lines(filepath)

    print(f"Pass 1: Building indexes from {filepath}")
    print(f"Estimated lines: {estimated_total:,}")
    print(f"Filtering to: {LANGUAGE_FILTER} only")

    for conv in tqdm(iter_jsonl(filepath), total=estimated_total, desc="Pass 1"):
        # Skip non-English conversations
        if conv.get('language') != LANGUAGE_FILTER:
            continue

        # Count by IP
        ip = conv.get('hashed_ip')
        if ip:
            ip_counts[ip] += 1

        # Detect templates in first message
        conversation = conv.get('conversation', [])
        first_content = get_first_message_content(conversation)
        sig = get_template_signature(first_content)
        if sig:
            template_counts[sig] += 1

    return ip_counts, template_counts


# =============================================================================
# Pass 2: Filter and Collect
# =============================================================================

def filter_conversations(
    filepath: Path,
    banned_templates: set[str],
    max_per_ip: int = MAX_PER_IP,
    samples_per_year: int = SAMPLES_PER_YEAR,
    seed: int = RANDOM_SEED
) -> dict[int, list[dict]]:
    """
    Second pass: filter and sample candidates by year using reservoir sampling.

    Uses reservoir sampling to maintain bounded memory - only keeps samples_per_year
    conversations per year in memory, regardless of how many pass filtering.

    Filters applied:
    1. Language == "English"
    2. MIN_TURNS <= turn <= MAX_TURNS
    3. Year in YEARS
    4. Not using a banned template
    5. IP not exceeding max quota (tracked during collection)

    Args:
        filepath: Path to the JSONL file
        banned_templates: Set of template signatures to filter out
        max_per_ip: Maximum conversations to collect per IP
        samples_per_year: Number of samples to keep per year (reservoir size)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping year -> list of sampled conversations
    """
    # Use reservoir samplers instead of lists to bound memory usage
    random.seed(seed)
    samplers_by_year = {year: ReservoirSampler(samples_per_year) for year in YEARS}
    ip_usage = Counter()  # Track how many we've taken per IP
    seen_hashes = set()  # Track conversation hashes to avoid duplicates

    # Stats tracking
    stats = {
        'total': 0,
        'filtered_language': 0,
        'filtered_turns': 0,
        'filtered_year': 0,
        'filtered_template': 0,
        'filtered_duplicate': 0,
        'filtered_ip_cap': 0,
        'passed': 0,
    }

    # Estimate total for progress bar
    estimated_total = estimate_total_lines(filepath)

    print(f"\nPass 2: Filtering conversations")
    print(f"Filters: language={LANGUAGE_FILTER}, turns={MIN_TURNS}-{MAX_TURNS}, years={YEARS}")
    print(f"Banned templates: {len(banned_templates)}, max_per_ip: {max_per_ip}")

    for conv in tqdm(iter_jsonl(filepath), total=estimated_total, desc="Pass 2"):
        stats['total'] += 1

        # Filter 1: Top-level language
        if conv.get('language') != LANGUAGE_FILTER:
            stats['filtered_language'] += 1
            continue

        # Filter 1b: At least 75% of messages must be English
        conversation = conv.get('conversation', [])
        if conversation:
            english_msgs = sum(1 for msg in conversation if msg.get('language') == LANGUAGE_FILTER)
            if english_msgs < len(conversation) * 0.75:
                stats['filtered_language'] += 1
                continue

        # Filter 2: Turn range
        turn = conv.get('turn', 0)
        if not (MIN_TURNS <= turn <= MAX_TURNS):
            stats['filtered_turns'] += 1
            continue

        # Filter 3: Year
        year = extract_year(conv.get('timestamp'))
        if year not in YEARS:
            stats['filtered_year'] += 1
            continue

        # Filter 4: Template detection
        first_content = get_first_message_content(conversation)
        sig = get_template_signature(first_content)
        if sig and sig in banned_templates:
            stats['filtered_template'] += 1
            continue

        # Filter 5: Deduplicate by conversation hash
        conv_hash = conv.get('conversation_hash')
        if conv_hash in seen_hashes:
            stats['filtered_duplicate'] += 1
            continue
        seen_hashes.add(conv_hash)

        # Filter 6: IP cap (disabled when max_per_ip is None)
        ip = conv.get('hashed_ip')
        if max_per_ip is not None and ip and ip_usage[ip] >= max_per_ip:
            stats['filtered_ip_cap'] += 1
            continue

        # Passed all filters - add to reservoir sampler
        stats['passed'] += 1
        samplers_by_year[year].add(conv)
        if max_per_ip is not None and ip:
            ip_usage[ip] += 1

    # Print filter stats
    print(f"\nFilter Statistics:")
    print(f"  Total processed: {stats['total']:,}")
    print(f"  Filtered by language: {stats['filtered_language']:,}")
    print(f"  Filtered by turns: {stats['filtered_turns']:,}")
    print(f"  Filtered by year: {stats['filtered_year']:,}")
    print(f"  Filtered by template: {stats['filtered_template']:,}")
    print(f"  Filtered by duplicate: {stats['filtered_duplicate']:,}")
    print(f"  Filtered by IP cap: {stats['filtered_ip_cap']:,}")
    print(f"  Passed all filters: {stats['passed']:,}")

    for year in YEARS:
        sampler = samplers_by_year[year]
        print(f"  {year}: {len(sampler)} sampled from {sampler.total_seen():,} candidates")

    # Return samples directly from reservoirs
    return {year: sampler.get_sample() for year, sampler in samplers_by_year.items()}


# =============================================================================
# Sampling
# =============================================================================

def sample_balanced(
    candidates_by_year: dict[int, list[dict]],
    samples_per_year: int,
    seed: int = RANDOM_SEED
) -> list[dict]:
    """
    Sample equally from each year.

    Args:
        candidates_by_year: Dict mapping year -> list of conversations
        samples_per_year: Number of samples per year
        seed: Random seed for reproducibility

    Returns:
        Combined list of sampled conversations
    """
    random.seed(seed)
    final_samples = []

    print(f"\nSampling {samples_per_year} per year (seed={seed})...")

    for year in YEARS:
        pool = candidates_by_year[year]
        if len(pool) < samples_per_year:
            print(f"  WARNING: {year} only has {len(pool)} candidates, taking all")
            final_samples.extend(pool)
        else:
            sampled = random.sample(pool, samples_per_year)
            final_samples.extend(sampled)
            print(f"  {year}: sampled {len(sampled)} from {len(pool)} candidates")

    return final_samples


# =============================================================================
# Output
# =============================================================================

def write_output(samples: list[dict], output_path: Path):
    """
    Write samples to JSONL file.

    Args:
        samples: List of conversations to write
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in samples:
            f.write(json.dumps(conv, cls=DateTimeEncoder, ensure_ascii=False) + '\n')


def analyze_templates(filepath: Path, top_n: int = 100, threshold: int = None, refresh: bool = False):
    """
    Analyze and display the distribution of long first messages (templates/system prompts).

    Args:
        filepath: Path to the JSONL file
        top_n: Number of top templates to display
        threshold: Occurrence threshold for filtering (default: MIN_TEMPLATE_OCCURRENCES)
        refresh: If True, ignore cache and re-scan
    """
    if threshold is None:
        threshold = MIN_TEMPLATE_OCCURRENCES

    print("=" * 80)
    print("TEMPLATE / SYSTEM PROMPT ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing first messages longer than {MIN_TEMPLATE_LENGTH} chars...")
    print(f"Using first {TEMPLATE_SIGNATURE_LENGTH} chars as signature")
    print(f"Filter threshold: {threshold}+ occurrences\n")

    # Try to load from cache
    cached_data = None
    total_conversations = 0
    if not refresh:
        cached_data = load_template_cache(CACHE_FILE)
        if cached_data:
            print(f"Loaded from cache: {CACHE_FILE}")
            template_counts, template_examples, template_lengths, ip_counts, template_ips, ip_daily_counts, total_conversations = cached_data

    # If no cache or refresh requested, scan the file
    if cached_data is None:
        print("Scanning dataset (this will be cached for future runs)...")
        print(f"Filtering to: {LANGUAGE_FILTER} only")

        # Collect template data
        template_counts = Counter()
        template_examples = {}  # Store one full example per signature
        template_lengths = defaultdict(list)  # Store lengths per signature
        template_ip_counts = defaultdict(Counter)  # Store IP counts per template (not full lists)
        ip_counts = Counter()
        ip_daily_counts = defaultdict(Counter)  # IP -> date -> count

        estimated_total = estimate_total_lines(filepath)

        # First pass: collect template IPs (English only)
        template_related_ips = set()
        total_conversations = 0

        for conv in tqdm(iter_jsonl(filepath), total=estimated_total, desc="Analyzing"):
            # Skip non-English conversations
            if conv.get('language') != LANGUAGE_FILTER:
                continue

            total_conversations += 1
            ip = conv.get('hashed_ip')
            if ip:
                ip_counts[ip] += 1

            # Analyze templates
            conversation = conv.get('conversation', [])
            first_content = get_first_message_content(conversation)
            sig = get_template_signature(first_content)

            if sig:
                template_counts[sig] += 1
                # Only store a few lengths to save memory
                if len(template_lengths[sig]) < 100:
                    template_lengths[sig].append(len(first_content))
                # Track IP counts per template (memory efficient)
                if ip:
                    template_ip_counts[sig][ip] += 1
                    template_related_ips.add(ip)
                # Store first example we see
                if sig not in template_examples:
                    template_examples[sig] = first_content

        # Second pass: only collect daily counts for template-related IPs (English convos only)
        print(f"Collecting daily patterns for {len(template_related_ips):,} template-related IPs...")
        for conv in tqdm(iter_jsonl(filepath), total=estimated_total, desc="Daily counts"):
            # Skip non-English conversations
            if conv.get('language') != LANGUAGE_FILTER:
                continue

            ip = conv.get('hashed_ip')
            if ip and ip in template_related_ips:
                timestamp = conv.get('timestamp')
                if timestamp:
                    date_str = str(timestamp)[:10]  # YYYY-MM-DD
                    ip_daily_counts[ip][date_str] += 1

        # Convert defaultdicts for JSON serialization
        template_ips = {k: dict(v) for k, v in template_ip_counts.items()}  # Store as {sig: {ip: count}}
        ip_daily_counts = {k: dict(v) for k, v in ip_daily_counts.items()}

        # Save cache for next time
        save_template_cache(template_counts, template_examples, template_lengths, ip_counts, CACHE_FILE, template_ips, ip_daily_counts, total_conversations)

    # Summary statistics
    total_with_templates = sum(template_counts.values())
    unique_templates = len(template_counts)

    print(f"\n{'=' * 80}")
    print("SUMMARY (English only)")
    print(f"{'=' * 80}")
    print(f"Total English conversations: {total_conversations:,}")
    print(f"Conversations with long first messages: {total_with_templates:,} ({total_with_templates/total_conversations*100:.1f}%)")
    print(f"Unique template signatures: {unique_templates:,}")

    # Distribution by occurrence count
    occurrence_dist = Counter()  # unique templates per bucket
    usage_dist = Counter()  # total conversations per bucket
    for count in template_counts.values():
        if count >= 5000:
            occurrence_dist['5000+'] += 1
            usage_dist['5000+'] += count
        elif count >= 2500:
            occurrence_dist['2500-4999'] += 1
            usage_dist['2500-4999'] += count
        elif count >= 1000:
            occurrence_dist['1000-2499'] += 1
            usage_dist['1000-2499'] += count
        elif count >= 500:
            occurrence_dist['500-999'] += 1
            usage_dist['500-999'] += count
        elif count >= 100:
            occurrence_dist['100-499'] += 1
            usage_dist['100-499'] += count
        elif count >= 50:
            occurrence_dist['50-99'] += 1
            usage_dist['50-99'] += count
        elif count >= 20:
            occurrence_dist['20-49'] += 1
            usage_dist['20-49'] += count
        elif count >= 10:
            occurrence_dist['10-19'] += 1
            usage_dist['10-19'] += count
        elif count >= 5:
            occurrence_dist['5-9'] += 1
            usage_dist['5-9'] += count
        else:
            occurrence_dist['1-4'] += 1
            usage_dist['1-4'] += count

    print(f"\nTemplates by occurrence frequency:")
    print(f"  {'Bucket':<12} {'Templates':>10} {'Convos':>12} {'% of dataset':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    for bucket in ['5000+', '2500-4999', '1000-2499', '500-999', '100-499', '50-99', '20-49', '10-19', '5-9', '1-4']:
        if bucket in occurrence_dist:
            pct = (usage_dist[bucket] / total_conversations * 100) if total_conversations > 0 else 0
            print(f"  {bucket:<12} {occurrence_dist[bucket]:>10,} {usage_dist[bucket]:>12,} {pct:>11.2f}%")

    # Templates that would be filtered
    would_filter = sum(1 for c in template_counts.values() if c >= threshold)
    convs_filtered = sum(c for c in template_counts.values() if c >= threshold)
    pct_filtered = (convs_filtered / total_conversations * 100) if total_conversations > 0 else 0
    print(f"\nWith threshold >= {threshold}:")
    print(f"  Would filter {would_filter:,} unique templates")
    print(f"  Affecting {convs_filtered:,} conversations ({pct_filtered:.2f}% of dataset)")

    # Analyze IP daily patterns for high-frequency templates
    print(f"\n{'=' * 80}")
    print("IP DAILY LIMIT ANALYSIS")
    print(f"{'=' * 80}")
    print("Would limiting IPs to N conversations/day eliminate templates?\n")

    # For templates that would be filtered, check how many unique IPs use them
    # and what their daily patterns look like
    filtered_templates = [(sig, c) for sig, c in template_counts.most_common() if c >= threshold]

    if filtered_templates and template_ips:
        total_template_convos = sum(c for _, c in filtered_templates)

        # Check what % of template conversations come from IPs with >N/day
        for daily_limit in [5, 10, 20, 50]:
            convos_from_high_volume_ips = 0
            for sig, count in filtered_templates:
                # template_ips is now {sig: {ip: count}} format
                ip_counts_for_template = template_ips.get(sig, {})
                for ip, ip_conv_count in ip_counts_for_template.items():
                    daily_counts = ip_daily_counts.get(ip, {})
                    max_daily = max(daily_counts.values()) if daily_counts else 0
                    if max_daily > daily_limit:
                        convos_from_high_volume_ips += ip_conv_count

            pct = (convos_from_high_volume_ips / total_template_convos * 100) if total_template_convos > 0 else 0
            print(f"  Limit {daily_limit}/day: {pct:.1f}% of template convos would be filtered by IP limit alone")

    # Show top templates with IP info
    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} MOST COMMON TEMPLATES (with IP analysis)")
    print(f"{'=' * 80}")

    for i, (sig, count) in enumerate(template_counts.most_common(top_n), 1):
        avg_len = sum(template_lengths[sig]) / len(template_lengths[sig])
        example = template_examples[sig]

        # Create preview (first 200 chars, cleaned)
        preview = example[:200].replace('\n', '\\n').replace('\r', '')
        if len(example) > 200:
            preview += "..."

        filter_status = "FILTERED" if count >= threshold else "kept"

        # IP analysis for this template (template_ips is now {sig: {ip: count}})
        ip_counts_for_template = template_ips.get(sig, {})
        unique_ips = len(ip_counts_for_template)

        # Check max daily usage for IPs using this template
        max_daily_for_template = 0
        for ip in ip_counts_for_template.keys():
            daily_counts = ip_daily_counts.get(ip, {})
            if daily_counts:
                max_daily_for_template = max(max_daily_for_template, max(daily_counts.values()))

        print(f"\n{i}. [{filter_status}] Count: {count:,} | Avg len: {avg_len:.0f} | Unique IPs: {unique_ips} | Max daily/IP: {max_daily_for_template}")
        print(f"   Preview: {preview}")

    # Option to save full analysis
    print(f"\n{'=' * 80}")
    print("To see full template content, templates are stored in template_examples dict")
    print(f"Threshold for filtering: {threshold}+ occurrences")
    print(f"{'=' * 80}")

    return template_counts, template_examples


def print_final_statistics(samples: list[dict]):
    """Print comprehensive statistics about the final sample."""
    print("\n" + "=" * 60)
    print("FINAL SAMPLE STATISTICS")
    print("=" * 60)

    # Year distribution
    year_dist = Counter(extract_year(s.get('timestamp')) for s in samples)

    # Turn distribution
    turn_dist = Counter(s.get('turn', 0) for s in samples)

    # IP diversity
    ips = [s.get('hashed_ip') for s in samples if s.get('hashed_ip')]
    unique_ips = len(set(ips))

    # Model distribution
    model_dist = Counter(s.get('model', 'unknown') for s in samples)

    print(f"\nTotal samples: {len(samples)}")
    print(f"\nBy year:")
    for year in sorted(year_dist.keys()):
        print(f"  {year}: {year_dist[year]:,}")

    print(f"\nUnique IPs: {unique_ips:,}")

    print(f"\nTurn distribution:")
    print(f"  Min: {min(turn_dist.keys())}")
    print(f"  Max: {max(turn_dist.keys())}")
    print(f"  Mean: {sum(t * c for t, c in turn_dist.items()) / len(samples):.1f}")

    print(f"\nTop 5 models:")
    for model, count in model_dist.most_common(5):
        print(f"  {model}: {count:,}")

    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter WildChat dataset for follow-up detection research"
    )
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze template/system prompt distribution only (no filtering)'
    )
    parser.add_argument(
        '--top', '-t',
        type=int,
        default=100,
        help='Number of top templates to show in analysis (default: 100)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help=f'Override template occurrence threshold (default: {MIN_TEMPLATE_OCCURRENCES})'
    )
    parser.add_argument(
        '--refresh', '-r',
        action='store_true',
        help='Force re-scan, ignoring cached analysis'
    )

    args = parser.parse_args()

    # Use threshold from args or default
    template_threshold = args.threshold if args.threshold is not None else MIN_TEMPLATE_OCCURRENCES

    # Check input file exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return 1

    # Analysis mode
    if args.analyze:
        analyze_templates(INPUT_FILE, top_n=args.top, threshold=template_threshold, refresh=args.refresh)
        return 0

    # Full filtering mode
    print("=" * 60)
    print("WildChat Filtering Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Language: {LANGUAGE_FILTER}")
    print(f"  Years: {YEARS}")
    print(f"  Samples per year: {SAMPLES_PER_YEAR}")
    print(f"  Turn range: {MIN_TURNS}-{MAX_TURNS}")
    print(f"  Max per IP: {MAX_PER_IP}")
    print(f"  Template length threshold: {MIN_TEMPLATE_LENGTH}")
    print(f"  Template occurrence threshold: {template_threshold}")
    print()

    # Pass 1: Build indexes (try cache first)
    cached_data = load_template_cache(CACHE_FILE)
    if cached_data and not args.refresh:
        print("Using cached index data...")
        template_counts, _, _, ip_counts, _, _, _ = cached_data
    else:
        print("Building indexes (will be cached)...")
        ip_counts, template_counts = build_indexes(INPUT_FILE)
        # Save cache for future runs
        save_template_cache(template_counts, {}, defaultdict(list), ip_counts, CACHE_FILE, {}, {})

    print(f"\nIndex Statistics:")
    print(f"  Unique IPs: {len(ip_counts):,}")
    print(f"  Unique template signatures: {len(template_counts):,}")

    # Identify banned templates (appear too frequently)
    banned_templates = {
        sig for sig, count in template_counts.items()
        if count >= template_threshold
    }
    print(f"  Banned templates (>={template_threshold} occurrences): {len(banned_templates)}")

    # Show top banned templates
    if banned_templates:
        print(f"\n  Top 5 most common templates filtered:")
        top_templates = sorted(
            [(sig, template_counts[sig]) for sig in banned_templates],
            key=lambda x: -x[1]
        )[:5]
        for sig, count in top_templates:
            preview = sig[:80].replace('\n', ' ')
            print(f"    ({count:,}x) {preview}...")

    # Pass 2: Filter and sample using reservoir sampling (memory-efficient)
    samples_by_year = filter_conversations(
        INPUT_FILE,
        banned_templates,
        samples_per_year=SAMPLES_PER_YEAR,
        seed=RANDOM_SEED
    )

    # Combine samples from all years
    samples = []
    for year in YEARS:
        year_samples = samples_by_year[year]
        samples.extend(year_samples)
        print(f"  {year}: {len(year_samples)} samples")

    # Write output
    print(f"\nWriting {len(samples)} conversations to {OUTPUT_FILE}...")
    write_output(samples, OUTPUT_FILE)

    # Final statistics
    print_final_statistics(samples)

    print(f"\nDone! Output saved to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    exit(main())
