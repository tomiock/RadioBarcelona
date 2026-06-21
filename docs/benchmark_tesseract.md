# Tesseract OCR Benchmark — Radio Barcelona `page_examples`

Ground truth: 1 393 lines manually reviewed via the annotation tool.
All scores use **Tesseract-only lines** (`--filter-engine tesseract`).
CER = Character Error Rate · WER = Word Error Rate (lower is better).

---

## Overall results

| Image set | Lines | CER | WER | Δ CER vs baseline |
|---|---:|---:|---:|---:|
| Original scans (colour RGB) | 1163 | **40.21%** | 77.57% | — |
| `results/*_text.png` (colour RGB, pipeline-processed) | 1195 | **62.23%** | 114.14% | +22.02pp |
| `results_text/*_text.png` (grayscale, pre-processed) | 1220 | **63.55%** | 115.49% | +23.34pp |

---

## Per-page CER

| Page | Original (baseline) | `results` (colour) | `results_text` (grayscale) |
|---|---:|---:|---:|
| 1925_1 | 69.7% (n=51) | 96.3% (n=72) | 97.3% (n=71) |
| 1931_1 | 81.2% (n=25) | 84.2% (n=26) | 86.0% (n=27) |
| 1931_2 | 57.0% (n=65) | 75.5% (n=69) | 80.9% (n=67) |
| 1936_1 | 31.2% (n=133) | 55.4% (n=137) | 53.8% (n=138) |
| 1938_1 | 68.8% (n=82) | 68.8% (n=75) | 75.4% (n=85) |
| 1940_1 | 14.3% (n=15) | 59.1% (n=15) | 61.2% (n=13) |
| 1940_2 | 71.4% (n=17) | 86.8% (n=16) | 90.9% (n=18) |
| 1943_1 | 6.0% (n=44) | 48.9% (n=44) | 47.6% (n=45) |
| 1944_1 | 49.9% (n=41) | 78.5% (n=41) | 78.8% (n=40) |
| 1944_2 | 40.2% (n=125) | 91.0% (n=123) | 90.2% (n=129) |
| 1944_3 | 79.6% (n=18) | 88.9% (n=11) | 88.6% (n=12) |
| 1945_1 | 33.8% (n=15) | 50.2% (n=16) | 50.0% (n=17) |
| 1945_2 | 65.2% (n=25) | 75.9% (n=29) | 74.8% (n=30) |
| 1949_1 | 17.7% (n=44) | 43.7% (n=45) | 43.8% (n=45) |
| 1950_1 | 40.5% (n=125) | 60.9% (n=122) | 60.8% (n=121) |
| 1950_2 | 20.1% (n=21) | 31.7% (n=32) | 32.6% (n=30) |
| 1953_1 | 19.3% (n=140) | 45.6% (n=134) | 45.2% (n=139) |
| 1953_2 | 18.6% (n=140) | 45.0% (n=134) | 45.2% (n=139) |
| 1953_3 | 29.4% (n=37) | 43.4% (n=54) | 44.3% (n=54) |

---

## Key observations

- **Baseline dominates**: the original scans achieve the lowest CER on every page
  and every era. Both processed variants are significantly worse.
- **`results` ≈ `results_text`**: the two processed variants score almost
  identically (62.2% vs 63.6% CER), despite one being colour and the other
  grayscale. The pre-processing step — not the colour depth — is responsible
  for the quality drop.
- **Biggest regressions** (original → processed) occur on the early
  handwritten/mixed pages (1925, 1940, 1943–1944), where the processing
  appears to degrade ink contrast that Tesseract relies on.
- **1950 / 1953** (dense typewritten): the processed variants close the gap,
  suggesting the processing is more neutral on clean typewritten content.

---

## Reproduction

```bash
# baseline
python line-classifier/benchmark_ocr.py \
    --gt  /data/storage/users/tockier/ocr_output/page_examples_vllm/page \
    --hyp /data/storage/users/tockier/ocr_output/page_examples/page \
    --filter-engine tesseract

# results/_text.png
python line-classifier/benchmark_ocr.py \
    --gt  /data/storage/users/tockier/ocr_output/page_examples_vllm/page \
    --hyp /data/storage/users/tockier/ocr_output/page_examples_results/page \
    --filter-engine tesseract

# results_text/_text.png
python line-classifier/benchmark_ocr.py \
    --gt  /data/storage/users/tockier/ocr_output/page_examples_vllm/page \
    --hyp /data/storage/users/tockier/ocr_output/page_examples_results_text/page \
    --filter-engine tesseract
```