# visual_search — VAE + FAISS module

Independent visual retrieval module for the final UI. It does not replace the detector, OCR, GAN or review app.

## 1. Train VAE

```bash
python visual_search/train_vae.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --output-dir outputs/vae \
  --image-size 128 \
  --latent-dim 64 \
  --epochs 30 \
  --batch-size 32
```

## 2. Build FAISS index

```bash
python visual_search/build_vae_faiss.py \
  --project-root . \
  --metadata outputs/object_crops_raw/metadata.jsonl \
  --review-log outputs/review_logs/review_log.jsonl \
  --model outputs/vae/vae_best.pt \
  --output-dir outputs/faiss/vae/global \
  --by-type
```

## 3. Test search from terminal

```bash
python visual_search/search_similar.py \
  --project-root . \
  --crop-id stamp_000123 \
  --top-k 10
```

## 4. Optional API for final UI

```bash
python visual_search/api.py
```

Then query:

```text
http://127.0.0.1:5050/api/similar-crops/<crop_id>?k=10
http://127.0.0.1:5050/api/health
```

The API returns JSON only; Mateo's final UI can consume it and decide how to display the results.
