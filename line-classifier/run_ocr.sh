#!/usr/bin/env bash
# OCR pipeline launcher for Radio Barcelona PageXML documents.
#
# Crops lines, runs Tesseract on all of them, then routes lines with no
# Tesseract output to ODAOCR (handwritten-focused ViTSTR model).
#
# Usage:
#   ./run_ocr.sh <year_month_dir>
#   ./run_ocr.sh guiradbcn_a1937m10
#
# Override any variable at call time:
#   TESS_LANG=spa ./run_ocr.sh guiradbcn_a1937m10

set -euo pipefail

# ── configurable ─────────────────────────────────────────────────────────────
TESS_LANG=${TESS_LANG:-"spa+cat"}   # avoid $LANG — that's the system locale

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ODAOCR_DIR="$(cd "$REPO_DIR/../ODAOCR" && pwd)"

ODAOCR_CHECKPOINT="${ODAOCR_CHECKPOINT:-"$ODAOCR_DIR/MODELS/model_metalearnt.pt"}"
ODAOCR_TOKENIZER_DIR="${ODAOCR_TOKENIZER_DIR:-"$ODAOCR_DIR/MODELS"}"

PDF_IMAGES_BASE="/data/storage/datasets/RadioBarcelona/pdf_images"
PAGEXML_BASE="/data/storage/users/tockier/laypa_vis"
WORK_BASE="/tmp/ocr_work"
OUTPUT_BASE="/data/storage/users/tockier/ocr_output"

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
    echo "ERROR: PageXML dir not found: $XML_DIR"; exit 1
fi
if [ ! -d "$IMG_DIR" ]; then
    echo "ERROR: Image dir not found: $IMG_DIR"; exit 1
fi
if [ ! -f "$ODAOCR_CHECKPOINT" ]; then
    echo "ERROR: ODAOCR checkpoint not found: $ODAOCR_CHECKPOINT"; exit 1
fi
if [ ! -f "$ODAOCR_TOKENIZER_DIR/tokenizer.json" ]; then
    echo "ERROR: tokenizer.json not found in: $ODAOCR_TOKENIZER_DIR"; exit 1
fi

mkdir -p "$WORK_DIR" "$OUTPUT_XML_DIR"

# ── run ───────────────────────────────────────────────────────────────────────
echo "=== OCR pipeline: $DIR_NAME ==="
echo "  XML dir:          $XML_DIR"
echo "  Image dir:        $IMG_DIR"
echo "  Output:           $OUTPUT_XML_DIR"
echo "  Tesseract lang:   $TESS_LANG"
echo "  ODAOCR checkpoint: $ODAOCR_CHECKPOINT"
echo ""

conda run -n docs python "$PIPELINE" \
    --xml-dir               "$XML_DIR" \
    --img-dir               "$IMG_DIR" \
    --output-xml-dir        "$OUTPUT_XML_DIR" \
    --work-dir              "$WORK_DIR" \
    --tesseract-lang        "$TESS_LANG" \
    --odaocr-checkpoint     "$ODAOCR_CHECKPOINT" \
    --odaocr-tokenizer-dir  "$ODAOCR_TOKENIZER_DIR" \
    2>&1 | grep -v "WARN\|grfmt"

echo ""
echo "Done. Output PageXML: $OUTPUT_XML_DIR"
