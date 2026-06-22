#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec ./scripts/run_review_ddd_random_v2_retry_base_conf030.sh
