#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${REVIEW_CONFIG:-configs/review_current.json}

METADATA=$(python - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(cfg["metadata"])
PY
)

REVIEW_LOG=$(python - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(cfg["review_log"])
PY
)

lsof -ti:5000 | xargs -r kill -9 || true

env \
REVIEW_METADATA="$METADATA" \
REVIEW_LOG="$REVIEW_LOG" \
python review_app/app.py
