#!/bin/bash
# Run WildChat filtering to produce filtered_15k.jsonl
# Seed is fixed at 42 for reproducibility
# exact command used for filtering
cd "$(dirname "$0")"
python3 filter_wildchat.py