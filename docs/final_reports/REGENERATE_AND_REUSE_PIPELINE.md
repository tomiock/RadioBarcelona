# E.C.H.O. Internal Pipeline — Reuse, Regeneration and Training Guide

## Purpose

This document explains how to reuse and regenerate the E.C.H.O. detection, review and visual similarity pipeline.

It is intended for developers or project members who need to reuse the delivered UI package, regenerate detections, run the Review App, rebuild FAISS/VAE indexes, or fine-tune the detector again with new reviewed data.

The Review App is an internal developer/review tool. It is not the final archivist-facing UI.

## Current final state

The project has two complementary branches.

### General multi-class reviewed branch

```text
configs/review_current_general_multiclass_final_20260624.json
outputs/review_exports/export_20260624_172950
```

This branch preserves the manual review work.

Review summary:

```text
Reviewed: 545
Accepted: 350
Rejected: 184
Skipped: 11
```

It contains reviewed examples of stamps, handwriting, typewritten text, crossouts, censorship blocks and table fragments. It is useful for future multi-class training, generator assets and detector error analysis.

### Stamp-focused UI branch

```text
configs/review_current.json
configs/radio_barcelona_real_full_stamp_finetuned_conf060_i1024.json
outputs/models/stamp_review_finetune_20260624/best.pt
outputs/batches/radio_barcelona_real_full_stamp_finetuned_conf060_i1024
outputs/faiss/current
outputs/faiss/vae/global
outputs/faiss/vae/by_type/stamp
```

This branch is prepared for UI integration. It contains the stamp-focused model, 700 stamp candidates, metadata and visual similarity indexes.

## Quick reuse path

If the delivery ZIP is available, the UI team does not need to regenerate the pipeline.

Use:

```text
outputs/delivery/ECHO_UI_DELIVERY_PACKAGE_20260624.zip
```

Main UI resources:

```text
models/stamp_review_finetune_20260624/best.pt
stamp_finetuned_batch/radio_barcelona_real_full_stamp_finetuned_conf060_i1024/object_crops/metadata.jsonl
stamp_finetuned_batch/radio_barcelona_real_full_stamp_finetuned_conf060_i1024/object_crops/stamp/
visual_search_indexes/faiss_current/
visual_search_indexes/vae_global/
visual_search_indexes/vae_by_type/stamp/
```

## Regeneration path from Git

Clone the repository, create the environment and install dependencies:

```bash
git clone <repo_url>
cd RadioBarcelona-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full UAB page dataset is not included in Git. It must be available locally as rendered page images:

```text
data/sources/radio_barcelona_real_full/pages
```

If the team only has PDFs, they must first render them into page images.

## Rebuilding the stamp-focused batch

```bash
python tools/pipeline/build_review_batch.py   --batch-id radio_barcelona_real_full_stamp_finetuned_conf060_i1024   --description "Full real Radio Barcelona dataset using stamp-focused fine-tuned detector for UI integration."   --source-pages data/sources/radio_barcelona_real_full/pages   --weights outputs/models/stamp_review_finetune_20260624/best.pt   --conf 0.60   --imgsz 1024   --min-width 100   --min-height 60   --classes stamp   --set-current
```

Current result:

```text
Processed documents: 2771
Saved crops: 700
Class: stamp
```

## Rebuilding FAISS/VAE

After changing the active batch, metadata or crops:

```bash
./scripts/rebuild_visual_search_current.sh
```

This creates outputs/faiss/current, outputs/faiss/vae/global and outputs/faiss/vae/by_type/stamp.

Current result: 700 vectors in the simple FAISS index and 700 vectors in the VAE stamp index.

## Recreating the YOLO stamp training dataset

The stamp-only YOLO dataset was generated from the reviewed export:

```bash
python tools/review_tools/export_review_to_yolo.py   --project-root .   --export-dir outputs/review_exports/export_20260624_172950   --output-dir outputs/training/yolo_stamp_review_20260624   --classes stamp   --val-ratio 0.20   --seed 42   --copy-mode copy   --min-box-size 1
```

Result: 90 images, 121 labels, 72 train images, 18 validation images, class stamp.

## Fine-tuning the stamp detector

The stamp model was fine-tuned from the general detector:

```text
runs/detect/layout_detector_typewritten_v1/weights/best.pt
```

Final validation:

```text
Precision: 0.848
Recall: 0.826
mAP50: 0.911
mAP50-95: 0.536
```

Clean model copy:

```text
outputs/models/stamp_review_finetune_20260624/best.pt
```

## What should be preserved

Always preserve:

```text
outputs/review_exports/export_20260624_172950
outputs/models/stamp_review_finetune_20260624/best.pt
configs/review_current_general_multiclass_final_20260624.json
configs/radio_barcelona_real_full_stamp_finetuned_conf060_i1024.json
docs/
```

These preserve the human decisions, final model and reproducible settings.

## What can be regenerated

Regenerable: predicted_layouts/, object_crops/, metadata.jsonl, FAISS indexes, VAE/FAISS indexes and stamp-focused batch.

Not easily regenerable without human work: manual review decisions, review_log human annotations and accepted/rejected labels.

## Git policy

Commit to Git: tools/, scripts/, review_app/, configs/, docs/ and requirements.txt.

Do not commit to Git: data/sources/, outputs/batches/, outputs/faiss/, outputs/review_exports/, outputs/models/*.pt, raw PDFs or large image datasets.

Large artifacts should be shared through the delivery ZIP or external storage.
