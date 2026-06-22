#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

lsof -ti:5000 | xargs -r kill -9 || true

env \
REVIEW_METADATA=outputs/object_crops_ddd_random_reviewed_baseclasses_v1_conf030/metadata.jsonl \
REVIEW_LOG=outputs/review_logs/review_log_ddd_random_reviewed_baseclasses_v1_conf030.jsonl \
python review_app/app.py
