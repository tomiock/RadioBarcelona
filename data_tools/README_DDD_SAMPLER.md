# DDD random page sampler

Downloads a small random sample of PDFs/pages from the UAB DDD search results for **Guions de Ràdio Barcelona**.

## Install

```bash
pip install requests beautifulsoup4
sudo apt install poppler-utils
```

## Dry run

```bash
python data_tools/sample_ddd_random_pages.py --num-records 3 --pages-per-record 5 --dry-run
```

## Real run

```bash
python data_tools/sample_ddd_random_pages.py \
  --num-records 3 \
  --pages-per-record 5 \
  --seed 42 \
  --dpi 200 \
  --output-dir data/ddd_random
```

## Output

```text
data/ddd_random/
├── raw_pdfs/
├── pages/
└── manifest.jsonl
```

Each manifest row stores record URL, PDF URL, local PDF path, local page path, page number, page count, seed and DPI.

## Recommended use

1. Download random pages.
2. Run the detector/layout pipeline on `data/ddd_random/pages/`.
3. Open detections in the review app.
4. Use manual bbox selector where detections are missing or wrong.
5. Export reviewed labels to YOLO.
6. Retrain the detector.
7. Send selected good crops to `assets_real_reviewed/` for generator reuse.
