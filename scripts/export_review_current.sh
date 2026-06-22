#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${REVIEW_CONFIG:-configs/review_current.json}

read_cfg () {
  python - "$CONFIG" "$1" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(cfg[sys.argv[2]])
PY
}

METADATA=$(read_cfg metadata)
REVIEW_LOG=$(read_cfg review_log)
INDEX_OUTPUT_DIR=$(read_cfg index_output_dir)

python tools/review_tools/build_review_indexes.py \
  --project-root . \
  --metadata "$METADATA" \
  --review-log "$REVIEW_LOG" \
  --output-dir "$INDEX_OUTPUT_DIR" \
  --export-package
