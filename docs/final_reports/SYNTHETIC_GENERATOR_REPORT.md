# E.C.H.O. Synthetic Dataset Generator — Partial Report

## 1. Purpose

The synthetic document generator was created to bootstrap the detection pipeline for historical Radio Barcelona scripts.

Manual annotations for archival documents are scarce and expensive. Synthetic generation helps create controlled training examples containing typewritten text, handwriting, stamps, crossouts, stains, paper degradation and correction/censorship marks.

The generator is not the final goal. It is a support module for training and experimentation.

## 2. Role in the full project

The generator supports the E.C.H.O. pipeline by providing early data for detector development.

Its role connects with YOLO detection, the Review App, GAN/restoration work, OCR/HTR preprocessing and final UI visual exploration.

The general idea is:

```text
synthetic documents
→ detector training
→ real document detection
→ manual review
→ reviewed real assets
→ improved generation and retraining
```

This creates a feedback loop between generated data and reviewed real data.

## 3. How synthetic generation works

The generator creates document-like images by combining controlled layers such as background paper, typewritten text, handwritten annotations, stamps, crossouts, noise/stains and layout annotations.

Example output structure:

```text
final_merged.jpg
layer0_clean.png
layer1_clean.png
text_original.txt
layout_annotations.json
```

The annotations can be converted into YOLO-style labels for detector training.

## 4. Assets

The generator can use asset folders such as assets/stamps, assets/censorship, assets/patches and iam_samples/.

Common asset types include stamp images, crossout marks, handwriting samples, degradation patches and paper effects.

These assets make synthetic documents more similar to real archival material.

## 5. Why synthetic data was useful

Synthetic data was useful because it allowed early experiments before enough real reviewed data existed.

It helped test class definitions, bounding box extraction, YOLO training, layout annotations, crop generation and the Review App workflow.

Without synthetic data, the project would have needed a larger manual annotation effort from the beginning.

## 6. Connection with reviewed real crops

The Review App now produces real reviewed crops.

These can improve the generator because accepted crops can become real assets:

```text
accepted stamps → real stamp assets
accepted crossouts → realistic correction marks
accepted handwriting → handwriting references
accepted stains/noise → degradation examples
```

Real reviewed assets are more faithful to the archive than purely artificial elements.

## 7. Connection with GAN/restoration work

If a module works on restoration, degradation or image-to-image translation, reviewed crops and synthetic layers can provide examples of paper noise, ink bleed, stains, faded stamps, handwriting overlays, crossouts and damaged document regions.

Synthetic data can create paired or controlled examples, while reviewed real crops provide realistic targets or references.

## 8. Connection with OCR/HTR

Synthetic and reviewed layout information can help OCR/HTR by identifying regions likely to contain typewritten text, handwritten annotations, non-textual marks, stamps and crossouts.

However, the current detector is not yet a final text segmentation system. Text detections may be fragmented. Future work should group nearby text-line detections into larger text blocks before OCR/HTR.

## 9. Connection with the final UI

The final UI benefits indirectly from synthetic and reviewed data.

The generator helps improve the detector. The detector creates crops and metadata. The Review App validates them. FAISS/VAE indexes support visual search.

For the final UI, the most relevant current result is the stamp-focused pipeline: fine-tuned stamp detector, 700 stamp crops, metadata and VAE/FAISS indexes.

## 10. Limitations

Current limitations:

- synthetic documents may not fully match real archival variability
- real stains/stamps/handwriting are more diverse
- some generated layouts may be too clean or artificial
- large text blocks still need better handling
- classes such as table_fragment or censorship_block need more reviewed real data

Synthetic data should support human-reviewed real data, not replace it.

## 11. Future improvements

Recommended improvements:

- use reviewed real stamps as generator assets
- add more realistic paper aging and stains
- generate larger typewritten/handwritten text blocks
- include real crossout and censorship assets
- create class-specific generation profiles
- combine synthetic and reviewed real examples for retraining

## 12. Conclusion

The synthetic generator provides an important bootstrap mechanism for E.C.H.O.

Its strongest future use is as part of a loop:

```text
generate synthetic data
train detector
detect on real pages
review real crops
reuse reviewed assets
generate better synthetic data
retrain
```

This makes it a useful technical component for long-term improvement of the detection and UI pipeline.
