#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${REVIEW_CONFIG:-configs/review_current.json}

read_cfg () {
  python - "$CONFIG" "$1" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
value = cfg[sys.argv[2]]
if isinstance(value, list):
    print(" ".join(value))
else:
    print(value)
PY
}

CROP_DIR=$(read_cfg crop_dir)
METADATA=$(read_cfg metadata)
REVIEW_LOG=$(read_cfg review_log)
FAISS_DIR=$(read_cfg faiss_dir)
FAISS_TMP_DIR=$(read_cfg faiss_tmp_dir)
VAE_DIR=$(read_cfg vae_dir)
VAE_MODEL=$(read_cfg vae_model)
CLASSES=$(read_cfg classes)

rm -rf "$FAISS_DIR" "$FAISS_TMP_DIR"

python visual_search/build_embeddings.py \
  --input-crops "$CROP_DIR" \
  --metadata "$METADATA" \
  --output-dir "$FAISS_DIR" \
  --classes $CLASSES \
  --use-metadata-crop-paths

mkdir -p "$FAISS_TMP_DIR"

python visual_search/build_faiss_index.py \
  --embeddings "$FAISS_DIR/embeddings.npy" \
  --metadata "$FAISS_DIR/metadata.jsonl" \
  --output-dir "$FAISS_TMP_DIR" \
  --index-name visual_index.faiss \
  --metric cosine

cp "$FAISS_TMP_DIR/visual_index.faiss" "$FAISS_DIR/visual_index.faiss"
cp "$FAISS_TMP_DIR/faiss_config.json" "$FAISS_DIR/faiss_config.json" 2>/dev/null || true

python visual_search/build_vae_faiss.py \
  --project-root . \
  --metadata "$METADATA" \
  --review-log "$REVIEW_LOG" \
  --model "$VAE_MODEL" \
  --output-dir "$VAE_DIR" \
  --by-type

echo "Visual search rebuilt from config: $CONFIG"
