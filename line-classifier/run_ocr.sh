#!/usr/bin/env bash
# OCR pipeline launcher for Radio Barcelona PageXML documents.
# Crops lines, classifies H/T, routes to Tesseract or Loghi HTR,
# and writes <TextEquiv> back into PageXML.
#
# Usage:
#   ./run_ocr.sh <year_month_dir>
#   ./run_ocr.sh guiradbcn_a1937m10
#
# Or override any variable at call time:
#   BATCH=32 ./run_ocr.sh guiradbcn_a1937m10

set -euo pipefail

# ── configurable paths ────────────────────────────────────────────────────────
BATCH=${BATCH:-64}
TESS_LANG=${TESS_LANG:-"spa+cat"}   # avoid $LANG — that's the system locale
LOGHI_DOCKER=${LOGHI_DOCKER:-"loghi/docker.htr:2.3.0"}

PDF_IMAGES_BASE="/data/storage/datasets/RadioBarcelona/pdf_images"
PAGEXML_BASE="/data/storage/users/tockier/laypa_vis"
CHECKPOINT="/data/storage/users/tockier/laypa_train/model/best.pth"
LOGHI_MODEL=""   # set to absolute path when H/T model is trained
WORK_BASE="/tmp/ocr_work"
OUTPUT_BASE="/data/storage/users/tockier/ocr_output"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_ocr_pipeline.py"

# ── argument ──────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <year_month_dir>  (e.g. guiradbcn_a1937m10)"
    exit 1
fi

DIR_NAME="$1"
XML_DIR="$PAGEXML_BASE/$DIR_NAME/page"
IMG_DIR="$PDF_IMAGES_BASE/$DIR_NAME"
WORK_DIR="$WORK_BASE/$DIR_NAME"
OUTPUT_XML_DIR="$OUTPUT_BASE/$DIR_NAME/page"

# ── checks ────────────────────────────────────────────────────────────────────
if [ ! -d "$XML_DIR" ]; then
    echo "ERROR: PageXML dir not found: $XML_DIR"
    exit 1
fi
if [ ! -d "$IMG_DIR" ]; then
    echo "ERROR: Image dir not found: $IMG_DIR"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$WORK_DIR" "$OUTPUT_XML_DIR"

# ── build optional loghi flag ─────────────────────────────────────────────────
LOGHI_ARGS=""
if [ -n "$LOGHI_MODEL" ]; then
    LOGHI_ARGS="--loghi-model $LOGHI_MODEL --loghi-docker $LOGHI_DOCKER"
fi

# ── run ───────────────────────────────────────────────────────────────────────
echo "=== OCR pipeline: $DIR_NAME ==="
echo "  XML dir:    $XML_DIR"
echo "  Image dir:  $IMG_DIR"
echo "  Output:     $OUTPUT_XML_DIR"
echo "  Work dir:   $WORK_DIR"
echo "  Checkpoint: $CHECKPOINT"
[ -n "$LOGHI_MODEL" ] && echo "  Loghi model: $LOGHI_MODEL" || echo "  Loghi model: (not set — handwritten lines will be skipped)"
echo ""

conda run -n docs python "$PIPELINE" \
    --xml-dir        "$XML_DIR" \
    --img-dir        "$IMG_DIR" \
    --checkpoint     "$CHECKPOINT" \
    --work-dir       "$WORK_DIR" \
    --output-xml-dir "$OUTPUT_XML_DIR" \
    --batch-size     "$BATCH" \
    --tesseract-lang "$TESS_LANG" \
    $LOGHI_ARGS \
    2>&1 | grep -v "WARN\|grfmt"

echo ""
echo "Done. Output PageXML: $OUTPUT_XML_DIR"
