#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${REVIEW_CONFIG:-configs/review_current.json}

REVIEW_LOG=$(python - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(cfg["review_log"])
PY
)

mkdir -p "$(dirname "$REVIEW_LOG")"
: > "$REVIEW_LOG"

echo "Reset review log:"
echo "$REVIEW_LOG"
wc -l "$REVIEW_LOG"
