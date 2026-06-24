# E.C.H.O. Review App — Detailed Report

## 1. Purpose

The Review App is an internal developer/review tool for the E.C.H.O. project.

It is designed to help validate automatic detections over historical Radio Barcelona documents. It is not the final archivist-facing UI.

Its main goal is to accelerate manual review and produce useful reviewed data for future detector training, model error analysis, synthetic generator assets, VAE/FAISS visual search and final UI integration.

The detector proposes candidates, but the human reviewer validates them.

## 2. General workflow

```text
page images
→ YOLO detector
→ predicted layouts
→ object crops
→ Review App
→ human decisions
→ export package
→ future training / UI / generator / analysis
```

The app turns a large page-level problem into a crop-level review workflow.

## 3. YOLO detector

The YOLO detector scans page images and predicts bounding boxes for visual elements.

In the general pipeline, the detector can propose classes such as stamp, handwritten_text, typewritten_text, crossout, censorship_block and table_fragment.

The current UI-oriented fine-tuned model focuses on stamp candidates.

The detector output is not treated as ground truth. It is a candidate generator.

## 4. Predicted layouts

For each page, the detector writes a JSON prediction file under predicted_layouts/.

These JSON files store bbox, predicted class, confidence and page/document id. They are useful for crop generation, detector debugging, page-level bbox overlays and UI context.

## 5. Crop extraction

The cropper extracts regions from page images using predicted bounding boxes.

Outputs:

```text
object_crops/
object_crops/metadata.jsonl
```

The metadata file connects crop_id, crop_path, page/document, bbox, predicted type and confidence.

## 6. Main Review App interface

The app shows the main crop, detector confidence, predicted/effective class, page preview link, crop link, review actions, attributes, FAISS similar crops, VAE similar crops, filters and statistics.

The UI was simplified so that long crop IDs and document IDs are hidden behind short links.

## 7. Review actions

Accept is used when the crop is useful. The best training value is accepted + bbox_quality=good.

Reject is used when the crop is wrong, noisy or misleading. Rejected examples are valuable because they document false positives.

Skip is used when the reviewer cannot decide quickly.

## 8. Predicted vs effective filters

Predicted means the class proposed by the detector. Effective means the reviewed/human-corrected class if available.

The filter panel can switch between predicted and effective classes.

## 9. Statistics

The app shows total crops, pending, reviewed, accepted, rejected, skipped, exportable assets, accepted not exportable, predicted type counts and effective type counts.

Final reviewed export summary:

```text
Reviewed: 545
Accepted: 350
Rejected: 184
Skipped: 11
```

## 10. Attributes

The app allows attributes such as faded, low_contrast, rotated, background_noise, paper, stain, ink_bleed, margin note, long note, numeric, word, overlaps, crossed_out, line, underline, typewritten, handwritten, stamp, table, photo, signature and correction/censorship.

Attributes help describe visual properties that may matter for training, UI filtering or future research.

## 11. FAISS similarity

Simple FAISS similarity uses visual embeddings to find near-duplicate or visually close crops.

It is useful for repeated patterns, quick visual lookup, debugging and fallback when VAE is not available.

## 12. VAE + FAISS similarity

The VAE compresses crops into a learned latent representation. FAISS searches this latent space.

This is especially useful for stamps because historical stamps may be faded, partial, rotated, degraded or low contrast.

VAE/FAISS helps group visually similar stamps even when they are not pixel-identical.

## 13. Buttons and actions

Rebuild FAISS/VAE rebuilds visual similarity indexes. It does not delete reviews.

Rebuild indexes recalculates internal review/export summaries from metadata and review log.

Export package creates a reviewed export package under outputs/review_exports/.

## 14. Main JSON/JSONL files

- metadata.jsonl: crop index produced by the cropper
- review_log.jsonl: human decisions recorded by the Review App
- predicted_layouts/*.json: detector predictions per page
- review_stats.json: summary generated in export packages
- manifest.json: export package description

## 15. Final export

Final reviewed export:

```text
outputs/review_exports/export_20260624_172950
```

It includes exportable_metadata.jsonl, accepted_not_exportable_metadata.jsonl, rejected_metadata.jsonl, review_log_snapshot.jsonl, review_stats.json, manifest.json and EXPORT_NOTES.txt.

## 16. Current limitations

The detector is still a candidate generator, not a final archival authority.

Known limitations include fragmented text lines, remaining false positives, lack of class-specific thresholds, and the need for more data for table/censorship classes.

## 17. Future improvements

Recommended future work:

- Batch Manager inside the app
- class-specific thresholds
- text line grouping into larger OCR/HTR blocks
- multi-class retraining with more reviewed data
- stable FAISS/VAE API for final UI
- large-scale indexing
- clearer reviewed/confirmed distinction in final UI

## 18. Conclusion

The Review App provides a practical human-in-the-loop layer for E.C.H.O.

It accelerates validation, stores human decisions, supports training data creation, helps visual search and prepares useful assets for the final UI.

It is not the final archivist interface, but it produces the reviewed evidence and technical base that the final UI can use.
