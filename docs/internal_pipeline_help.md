# Pipeline Help & Files Guide — Internal Review Pipeline

## 1. What this pipeline is for

This tool is an internal development and review pipeline for the E.C.H.O. project.

It is not the final interface for archivists. It is the technical layer that helps the team prepare better data, inspect detector errors, validate crops, reuse useful visual regions, and build visual search indexes that can later support the final UI.

The main idea is simple:

```text
We start with scanned document pages.
The detector proposes interesting regions.
The cropper extracts them.
The review app lets us validate them.
FAISS/VAE helps us find visually similar crops.
The export package saves reviewed material for future use.
```

So the pipeline is useful because it connects several parts of the project:

```text
generation
detection
manual review
visual similarity
future retraining
final UI integration
```

---

## 2. Mental model: source → batch → review → export

The pipeline is easier to understand if we separate four concepts.

### Source

A source is a local collection of page images.

Example:

```text
data/sources/radio_barcelona_real_full/pages
```

This is the raw material: the pages we want to process.

### Batch

A batch is one concrete processing run over a source.

It stores:

```text
the selected pages
the detector predictions
the generated crops
the metadata
the review log
the config used for this run
```

Example:

```text
outputs/batches/radio_barcelona_real_full_conf060_i1024_min100x60/
```

### Review

Review is the human validation step.

The detector proposes. The reviewer decides.

The review log records:

```text
accepted
rejected
skipped
corrected classes
attributes
bbox quality
manual corrections
```

### Export

Export packages are how reviewed decisions leave the app.

They can later be used for:

```text
future detector training
generator assets
UI integration
backup
delivery
analysis of model errors
```

---

## 3. Current recommended setup

Current active config:

```text
configs/review_current.json
```

Current recommended batch:

```text
radio_barcelona_real_full_conf060_i1024_min100x60
```

Current recommended parameters:

```text
confidence threshold: 0.60
YOLO imgsz: 1024
minimum crop width: 100 px
minimum crop height: 60 px
main target: stamp candidates
secondary classes: handwritten_text, typewritten_text, crossout
prepared future classes: censorship_block, table_fragment
```

This setup tries to avoid excessive noise. It is stricter than exploratory detection, but still keeps enough stamp candidates for review.

Important limitation:

```text
min_width and min_height are global filters.
They work reasonably well for stamps,
but they are not ideal for text blocks.
```

For text, the detector may still produce line-level or fragment-level crops. A future postprocess should group nearby text lines into larger handwritten or typewritten blocks.

---

## 4. Main folders and what they mean

### `data/sources/`

This contains local datasets.

Example:

```text
data/sources/radio_barcelona_real_full/
  pages/
  raw_pdfs/
  manifest.jsonl
  source_config.json
```

Use:

```text
Input source data.
Ignored by Git.
Can be large.
```

Meaning:

```text
pages/          page images used by the detector
raw_pdfs/       original PDFs, useful for provenance or re-rendering
manifest.jsonl  page list and basic metadata
source_config   description of the dataset
```

The pipeline works from page images, not directly from PDFs. PDFs can be kept for traceability, but they are not needed for review once pages are rendered.

---

### `outputs/batches/<batch_id>/`

This contains everything produced by one batch run.

Example:

```text
outputs/batches/radio_barcelona_real_full_conf060_i1024_min100x60/
```

Inside:

```text
pages/
predicted_layouts/
object_crops/
review_log.jsonl
batch_config.json
```

These files are not just temporary outputs. They are the bridge between detection, review, similarity search and future reuse.

---

## 5. What each batch folder/file is used for

### `pages/`

These are the page images used in the batch.

They may be real copies or symlinks to the source dataset.

Used by:

```text
detector
page preview in the review app
manual bbox correction
future traceability
```

For UI integration:

```text
The final UI can use these pages to show context around reviewed stamps or annotations.
```

---

### `predicted_layouts/`

This folder contains one JSON prediction file per page.

Each file stores the automatic detector output:

```text
bbox coordinates
predicted class
confidence
page/document id
```

Used by:

```text
crop generation
detector debugging
future comparison between raw prediction and human review
```

For development:

```text
If the detector fails, predicted_layouts tells us what it saw and what it missed.
```

For OCR/HTR:

```text
These layout predictions can help identify candidate text regions,
but the current detector is not yet optimized for full text blocks.
```

---

### `object_crops/`

This contains cropped regions generated from detector predictions.

Example:

```text
object_crops/
  stamp/
  handwritten_text/
  typewritten_text/
  crossout/
  metadata.jsonl
```

Used by:

```text
review app
FAISS/VAE indexing
manual validation
future dataset construction
```

For generator/GAN work:

```text
Accepted good crops can become real visual assets.
For example, accepted stamps can be reused as stamp assets.
Accepted crossouts or handwriting can also help create more realistic synthetic documents.
```

For detector training:

```text
Accepted good crops help define positive examples.
Rejected crops help identify false positives and model weaknesses.
```

Important:

```text
Raw automatic crops are not ground truth.
They only become reliable after review.
```

---

### `metadata.jsonl`

This is the main index of all crops in the batch.

It stores information such as:

```text
crop_id
crop_path
source page
bbox
predicted type
confidence
document/page id
```

Used by:

```text
review app
FAISS/VAE rebuild
export tools
review statistics
```

If metadata is wrong or points to missing files, the app cannot review correctly.

---

### `review_log.jsonl`

This is the human review record.

It stores:

```text
accepted / rejected / skipped
reviewed class
bbox quality
attributes
notes
manual corrections
manual crops
```

This is the most important human-generated file.

Used by:

```text
export package
review indexes
future training preparation
model error analysis
```

For future training:

```text
accepted + good bbox = potential positive data
rejected = false positive evidence / hard negative analysis
```

For the final UI:

```text
review_log separates validated evidence from raw detector guesses.
```

---

### `batch_config.json`

This stores the exact parameters used to create the batch.

It records:

```text
source pages
weights
confidence
imgsz
classes
crop thresholds
metadata path
review log path
FAISS/VAE settings
```

Used for:

```text
reproducibility
debugging
sharing settings with teammates
rerunning the same workflow
```

If someone asks “how was this batch created?”, this file answers it.

---

## 6. How the outputs are reused later

### For the generator / synthetic data

Reviewed real crops can become better assets for the generator.

Examples:

```text
accepted stamps → real stamp assets
accepted handwriting → handwriting style examples
accepted crossouts → correction/degradation assets
accepted table fragments → layout assets
```

This improves synthetic documents because future generated pages can use **real reviewed** visual material instead of only artificial assets.

---

### For GAN / restoration / augmentation work

If a team module works on document degradation, cleaning, restoration or style transfer, reviewed crops can provide examples of real artifacts:

```text
stains
ink bleed
paper texture
faded marks
crossouts
stamps
handwriting overlays
```

The review app helps separate useful real artifacts from detector noise.

---

### For OCR / HTR

The detector can help identify candidate regions that may contain handwriting or typewritten text.

However, current text detections may be small or fragmented. For OCR/HTR, future grouping is recommended:

```text
many nearby typewritten lines → larger typewritten block
many nearby handwritten lines → larger handwritten block
```

So the current pipeline can support OCR/HTR, but it should not yet be treated as a final text segmentation system.

---

### For the final UI

The final UI can use reviewed crops and similarity search to help users explore repeated visual elements.

For example:

```text
click a stamp
→ retrieve visually similar stamps
→ show pages where similar stamps appear
```

The UI should not expose all raw detector predictions as final truth. It should prioritize reviewed or high-confidence material.

The review pipeline prepares that material.

---

### For future detector training

The export package is the bridge to future training.

Good reviewed examples can be used to improve the detector.

Recommended policy:

```text
Do not train directly from unreviewed detector predictions.
Train from reviewed exports.
Use accepted + good as positives.
Use rejected examples for error analysis and hard-negative mining.
```

---

## 7. Why there are two similarity systems

The app has two similarity columns because they help in different ways.

### Simple FAISS similarity

This uses direct visual embeddings from crops.

It is useful as a baseline and for finding near-duplicate crops.

Good for:

```text
quick visual lookup
debugging
simple repeated patterns
fallback when VAE is not available
```

### VAE + FAISS similarity

This uses a VAE to compress crops into a learned visual space, then uses FAISS to search that space.

Good for:

```text
noisy historical crops
faded stamps
repeated but imperfect visual patterns
visual grouping by style
stamp review
```

In practice, VAE/FAISS is often more useful for stamps because stamps can be faded, partial, rotated or degraded.

### Why keep both?

Because they are complementary.

```text
Main detector → finds candidate crops
Simple FAISS → finds direct visual similarity
VAE/FAISS → finds learned visual similarity
Human review → decides what is actually useful
```

For a developer, having both helps compare whether similarity is driven by raw appearance or learned visual structure.

For the UI, VAE/FAISS is the more promising option for scalable visual exploration.

---

## 8. How to actually review

Start with the main crop.

Ask:

```text
Is this crop useful?
Is the predicted class correct?
Is the bbox good enough?
Does it belong to a repeated visual pattern?
```

### Accept

Use Accept when the crop is useful.

For training/export quality:

```text
Accept + bbox_quality=good
```

is the most valuable case.

### Reject

Use Reject when the crop is wrong, noisy or misleading.

Rejected crops are not useless. They help show what the detector gets wrong.

### Skip

Use Skip when you cannot decide quickly.

Use it instead of forcing a bad decision.

---

## 9. Should I review from Main detector, Similar crops or VAE similarity?

Use them in this order:

### 1. Main detector crop

This is the primary candidate.

Inspect crop and page context.

### 2. VAE similar crops

Use this when the crop is visually meaningful, especially for stamps.

Good workflow:

```text
find a good stamp
look at VAE similar crops
accept clear repeated stamps
reject repeated false positives
```

### 3. Simple FAISS similar crops

Use this as a second opinion or fallback.

It may find near duplicates or visually close crops that VAE does not surface.

### Practical rule

```text
Use the main detector to enter the review.
Use VAE/FAISS to accelerate repeated decisions.
Do not accept similar crops blindly.
```

---

## 10. Troubleshooting

### VAE similar crops do not appear

Try:

```bash
./scripts/rebuild_visual_search_current.sh
```

Then restart the app:

```bash
lsof -ti:5000 | xargs -r kill -9
./scripts/run_review_current.sh
```

Also check:

```bash
find outputs/faiss/vae -maxdepth 4 -type f | sort
```

You should see:

```text
outputs/faiss/vae/global/visual_index.faiss
outputs/faiss/vae/global/metadata.jsonl
```

---

### App is slow

Possible causes:

```text
too many crops
very large FAISS index
low confidence threshold
large imgsz
too many small detections
```

Solutions:

```text
increase confidence
increase min_width/min_height
use a smaller sample batch
restart the app after rebuilding indexes
```

---

### Too many small crops

Use stricter crop filters:

```text
--min-width 100
--min-height 60
```

For text blocks, this is still not perfect. Future grouping is needed.

---

### No detections appear

Check:

```text
source path
weights path
confidence threshold
class list
metadata path in configs/review_current.json
```

If confidence is too high, the detector may return very few objects.

---

### FAISS works but VAE does not

Rebuild VAE/FAISS and restart the app.

If it still fails, check whether by-type indexes exist:

```bash
find outputs/faiss/vae -maxdepth 3 -type d | sort
```

---

## 11. Necessary vs unnecessary files

### Necessary to run current review

```text
configs/review_current.json
outputs/batches/<batch_id>/object_crops/metadata.jsonl
outputs/batches/<batch_id>/review_log.jsonl
outputs/batches/<batch_id>/pages/
outputs/faiss/current/
outputs/faiss/vae/global/
review_app/app.py
review_app/review_schema.json
```

### Useful but reproducible

```text
predicted_layouts/
batch_config.json
build logs
FAISS logs
```

### Useful for provenance, not always needed for review

```text
raw_pdfs/
source_config.json
manifest.jsonl
```

### Should not be committed to Git

```text
data/sources/
outputs/batches/
outputs/faiss/
outputs/review_exports/
raw PDFs
large images
model weights
```

---

## 12. Current position

The current version is good enough as an internal development/review tool.

It is prepared to:

```text
review stamp candidates
use VAE/FAISS similarity
export reviewed examples
support future retraining
support generator asset improvement
inform final UI development
```

It is not yet:

```text
a final archivist-facing interface
a fully automatic detector
a complete text block segmenter
a fully app-configured batch manager
```

These are future development directions.
