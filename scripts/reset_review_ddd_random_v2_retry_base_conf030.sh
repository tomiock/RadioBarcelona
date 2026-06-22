#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG=outputs/review_logs/review_log_ddd_random_v2_retry_base_conf030.jsonl

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

echo "Reset review log:"
echo "$LOG"
wc -l "$LOG"
