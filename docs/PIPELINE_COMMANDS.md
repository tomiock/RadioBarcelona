# RadioBarcelona Review + YOLO + VAE Pipeline Commands

This document summarizes the full local pipeline used for the current `stamp-detection-alex` branch.
It is written so the pipeline can be reproduced from a clean checkout **without losing reviewed crops, review logs, trained models, or generated indexes**.

---

## 0. Safety first: do not delete these folders/files

The most important reviewed/trained data lives under `outputs/` and `runs/`. Before cleaning or regenerating anything, keep these paths safe:

```text
outputs/review_logs/review_log.jsonl
outputs/object_crops_raw/metadata.jsonl
outputs/object_crops_raw/
outputs/object_crops_reviewed/
outputs/object_crops_rejected/
outputs/object_crops_skipped/
outputs/index/
outputs/review_exports/
outputs/review_yolo_dataset_v2/
outputs/vae/
outputs/faiss/
runs/detect/review_detector_test/
runs/detect/review_detector_v2/
```

Avoid destructive commands such as:

```bash
rm -rf outputs
rm -rf runs
git clean -fdx
```

`git clean -fdx` is especially dangerous because it removes ignored and untracked generated data.

---

## 1. Create a backup snapshot before major changes

Run this from the project root:

```bash
cd ~/Desktop/UNI/3º/Synthesis/RadioBarcelona-main

BACKUP_DIR="backups/review_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp -a outputs/review_logs "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/object_crops_raw "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/object_crops_reviewed "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/object_crops_rejected "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/object_crops_skipped "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/index "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/review_exports "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/review_yolo_dataset_v2 "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/vae "$BACKUP_DIR/" 2>/dev/null || true
cp -a outputs/faiss "$BACKUP_DIR/" 2>/dev/null || true
cp -a runs/detect "$BACKUP_DIR/" 2>/dev/null || true

echo "Backup written to: $BACKUP_DIR"
```

To inspect backup size:

```bash
du -sh "$BACKUP_DIR"
```

---

## 2. Environment setup from zero

```bash
cd ~/Desktop/UNI/3º/Synthesis/RadioBarcelona-main

git checkout stamp-detection-alex
git pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies. If the repository has `requirements.txt`, use it first:

```bash
pip install -r requirements.txt
```

Then install the main tools used by the review/YOLO/VAE pipeline:

```bash
pip install ultralytics faiss-cpu torch torchvision pillow numpy flask opencv-python
```

Quick checks:

```bash
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
try:
    import faiss
    print("faiss OK")
except Exception as e:
    print("faiss error", e)
PY
```

---

## 3. Start the manual review app

```bash
cd ~/Desktop/UNI/3º/Synthesis/RadioBarcelona-main
source .venv/bin/activate
python review_app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The review app reads:

```text
outputs/object_crops_raw/metadata.jsonl
outputs/review_logs/review_log.jsonl
review_app/review_schema.json
```

Useful review filters:

```text
All / Pending / Reviewed / Accepted / Rejected / Skipped / Exportable / Accepted but not exportable
Predicted type filters: stamp, handwritten_text, crossout, etc.
Effective type filters: stamp, false_positive, handwritten_text, crossout, etc.
```

---

## 4. Build review indexes and export package

Build JSONL indexes only:

```bash
python review_tools/build_review_indexes.py --project-root .
```

Build indexes and create an export package:

```bash
python review_tools/build_review_indexes.py --project-root . --export-package
```

Store the latest export path:

```bash
EXPORT_DIR=$(ls -td outputs/review_exports/export_* | head -1)
echo "$EXPORT_DIR"
```

Inspect stats:

```bash
python -m json.tool "$EXPORT_DIR/review_stats.json" | less
```

Count exportable labels by class:

```bash
grep '"reviewed_type": "stamp"' "$EXPORT_DIR/exportable_metadata.jsonl" | wc -l
grep '"reviewed_type": "handwritten_text"' "$EXPORT_DIR/exportable_metadata.jsonl" | wc -l
grep '"reviewed_type": "crossout"' "$EXPORT_DIR/exportable_metadata.jsonl" | wc -l
grep '"reviewed_type": "typewritten_line"' "$EXPORT_DIR/exportable_metadata.jsonl" | wc -l
```

Expected current example after reviewing more stamps:

```text
43 exportable labels
23 stamp
11 handwritten_text
9 crossout
```

---

## 5. Export reviewed crops to YOLO format

```bash
python review_tools/export_review_to_yolo.py \
  --project-root . \
  --export-dir "$EXPORT_DIR" \
  --output-dir outputs/review_yolo_dataset_v2 \
  --val-ratio 0.2
```

Inspect conversion report:

```bash
cat outputs/review_yolo_dataset_v2/conversion_report.json
python -m json.tool outputs/review_yolo_dataset_v2/conversion_report.json | less
```

Expected structure:

```text
outputs/review_yolo_dataset_v2/
├── data.yaml
├── conversion_report.json
├── page_map.json
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 6. Train YOLO from reviewed crops

```bash
python stamp_detection_module/stamp_detection/train_yolo_stamps.py \
  --data outputs/review_yolo_dataset_v2/data.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 4 \
  --name review_detector_v2
```

Results are saved to:

```text
runs/detect/review_detector_v2/
runs/detect/review_detector_v2/weights/best.pt
runs/detect/review_detector_v2/weights/last.pt
```

Inspect result plots:

```bash
ls runs/detect/review_detector_v2
ls runs/detect/review_detector_v2/weights
xdg-open runs/detect/review_detector_v2/results.png
xdg-open runs/detect/review_detector_v2/labels.jpg
xdg-open runs/detect/review_detector_v2/confusion_matrix.png
```

If KDE/Gwenview prints MIME warnings but the images open, they are just desktop viewer warnings and do not affect training.

---

## 7. Train the VAE visual model

The VAE does not require perfect labels. It learns visual similarity from crop images.

```bash
python visual_search/train_vae.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --output-dir outputs/vae \
  --image-size 128 \
  --latent-dim 64 \
  --epochs 20 \
  --batch-size 32
```

Expected outputs:

```text
outputs/vae/vae_final.pt
outputs/vae/vae_best.pt
```

---

## 8. Build VAE + FAISS indexes

```bash
python visual_search/build_vae_faiss.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --model outputs/vae/vae_best.pt \
  --output-dir outputs/faiss/vae/global \
  --by-type
```

Expected outputs:

```text
outputs/faiss/vae/global/
├── visual_index.faiss
├── metadata.jsonl
├── embeddings.npy
└── config.json

outputs/faiss/vae/by_type/
├── stamp/
├── handwritten_text/
├── crossout/
├── false_positive/
└── typewritten_line/
```

Example current counts:

```text
stamp: 71 vectors
false_positive: 29 vectors
handwritten_text: 524 vectors
crossout: 114 vectors
typewritten_line: 1 vector
```

---

## 9. Search similar crops with VAE + FAISS

Pick a known crop id:

```bash
CROP_ID=stamp_000676
```

Search in the global VAE index:

```bash
python visual_search/search_similar.py \
  --project-root . \
  --crop-id "$CROP_ID" \
  --index outputs/faiss/vae/global/visual_index.faiss \
  --metadata outputs/faiss/vae/global/metadata.jsonl \
  --model outputs/vae/vae_best.pt \
  --top-k 10
```

Search only among stamps:

```bash
python visual_search/search_similar.py \
  --project-root . \
  --crop-id "$CROP_ID" \
  --index outputs/faiss/vae/by_type/stamp/visual_index.faiss \
  --metadata outputs/faiss/vae/by_type/stamp/metadata.jsonl \
  --model outputs/vae/vae_best.pt \
  --top-k 10
```

Open crops manually:

```bash
xdg-open outputs/object_crops_raw/stamp/stamp_000676.jpg
xdg-open outputs/object_crops_raw/stamp/stamp_000682.jpg
```

---

## 10. Review app with normal FAISS + VAE FAISS

The updated review app shows two similarity columns:

```text
Similar crops      → previous/simple FAISS embeddings
VAE similar crops  → VAE latent embeddings + FAISS, by type when available
```

Run:

```bash
python review_app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

If the VAE column is empty, rebuild the VAE FAISS index:

```bash
python visual_search/build_vae_faiss.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --model outputs/vae/vae_best.pt \
  --output-dir outputs/faiss/vae/global \
  --by-type
```

---

## 11. Optional: run VAE API for final UI integration

```bash
python visual_search/api.py
```

Example endpoint, if implemented in the current API version:

```text
http://127.0.0.1:5050/api/similar-crops/stamp_000676?k=10
```

This is useful for handing results to a future UI without embedding the logic inside Flask review app.

---

## 12. Git workflow

Check changes:

```bash
git status
```

Commit source code and docs only:

```bash
git add review_app/app.py docs/PIPELINE_COMMANDS.md
git commit -m "Add VAE similarities to review app and pipeline guide"
git push
```

Do not commit generated data unless explicitly needed:

```text
outputs/
runs/
real_test_current_pages_50/
real_test_test_random_pages_10/
review_app/_app.py
```

If you want to preserve generated outputs for the team, prefer a backup/export zip outside git.

---

## 13. Minimal full cycle summary

```bash
cd ~/Desktop/UNI/3º/Synthesis/RadioBarcelona-main
source .venv/bin/activate

# Manual review
python review_app/app.py

# Export reviewed labels
python review_tools/build_review_indexes.py --project-root . --export-package
EXPORT_DIR=$(ls -td outputs/review_exports/export_* | head -1)

# Convert to YOLO
python review_tools/export_review_to_yolo.py \
  --project-root . \
  --export-dir "$EXPORT_DIR" \
  --output-dir outputs/review_yolo_dataset_v2 \
  --val-ratio 0.2

# Train YOLO
python stamp_detection_module/stamp_detection/train_yolo_stamps.py \
  --data outputs/review_yolo_dataset_v2/data.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 4 \
  --name review_detector_v2

# Train VAE
python visual_search/train_vae.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --output-dir outputs/vae \
  --image-size 128 \
  --latent-dim 64 \
  --epochs 20 \
  --batch-size 32

# Build VAE FAISS
python visual_search/build_vae_faiss.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --model outputs/vae/vae_best.pt \
  --output-dir outputs/faiss/vae/global \
  --by-type
```
