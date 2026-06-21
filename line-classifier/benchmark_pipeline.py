"""
Benchmark run_ocr_pipeline.py stage by stage.

Times three stages for each page:
  1. crop   — collect lines from PageXML + extract JPEG crops
  2. tess   — Tesseract PSM-7 on every line
  3. odaocr — word-split ODAOCR on Tesseract-empty lines

Prints per-page rows and a summary table at the end.

Usage:
    conda activate docs
    python line-classifier/benchmark_pipeline.py \\
        --batch guiradbcn_a1937m10 \\
        --n-pages 5

    # to skip ODAOCR (fast Tesseract-only timing):
    python line-classifier/benchmark_pipeline.py --batch guiradbcn_a1937m10 --no-odaocr
"""

import argparse
import sys
import time
import tempfile
import shutil
from pathlib import Path

# make sure run_ocr_pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ocr_pipeline as pipe

# ── paths ─────────────────────────────────────────────────────────────────────

PDF_IMAGES_BASE = Path("/data/storage/datasets/RadioBarcelona/pdf_images")
PAGEXML_BASE    = Path("/data/storage/users/tockier/laypa_vis")
ODAOCR_REPO     = Path("/home/tockier/uni/radio/ODAOCR")
ODAOCR_CKPT     = ODAOCR_REPO / "MODELS/model_metalearnt.pt"
ODAOCR_TOK_DIR  = ODAOCR_REPO / "MODELS"


# ── helpers ───────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._t = None
        self.elapsed = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t


def fmt(seconds: float, width: int = 7) -> str:
    return f"{seconds:{width}.2f}s"


def fmt_ms(seconds: float, width: int = 7) -> str:
    return f"{seconds * 1000:{width}.1f}ms"


# ── per-page benchmark ────────────────────────────────────────────────────────

def benchmark_page(xml_path: Path, img_dir: Path, work_dir: Path,
                   tess_lang: str, workers: int,
                   model, tokenizer, decoder,
                   run_odaocr: bool) -> dict:

    crops_dir = work_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # ── stage 1: collect + crop ───────────────────────────────────────────────
    with Timer() as t_crop:
        lines   = pipe.collect_lines(xml_path.parent, img_dir)
        # filter to this page only
        lines   = [l for l in lines if l["xml_path"] == xml_path]
        records = pipe.extract_crops(lines, crops_dir, 28, 10, 8)

    n_lines = len(records)

    # ── stage 2: tesseract ────────────────────────────────────────────────────
    with Timer() as t_tess:
        pipe.run_tesseract(records, tess_lang, workers=workers)

    tess_hit  = [r for r in records if r["ocr_text"]]
    tess_miss = [r for r in records if not r["ocr_text"]]

    # ── stage 3: odaocr ───────────────────────────────────────────────────────
    t_oda = Timer()
    oda_hit = 0
    if run_odaocr and tess_miss and model is not None:
        with t_oda:
            pipe.run_odaocr(tess_miss, model, tokenizer, decoder, "cuda")
        oda_hit = sum(1 for r in tess_miss if r["ocr_text"])
    else:
        t_oda.elapsed = 0.0

    return {
        "stem":      xml_path.stem,
        "n_lines":   n_lines,
        "n_tess":    len(tess_hit),
        "n_hw":      len(tess_miss),
        "n_oda_hit": oda_hit,
        "t_crop":    t_crop.elapsed,
        "t_tess":    t_tess.elapsed,
        "t_oda":     t_oda.elapsed,
        "t_total":   t_crop.elapsed + t_tess.elapsed + t_oda.elapsed,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Benchmark OCR pipeline per page")
    p.add_argument("--batch",   default="guiradbcn_a1937m10",
                   help="Batch directory name")
    p.add_argument("--n-pages", type=int, default=5,
                   help="Number of pages to benchmark (default 5, 0 = all)")
    p.add_argument("--tess-lang", default="spa+cat")
    p.add_argument("--workers",   type=int, default=0,
                   help="Tesseract threads (0 = all CPU cores)")
    p.add_argument("--no-odaocr", action="store_true",
                   help="Skip ODAOCR stage (faster Tesseract-only timing)")
    p.add_argument("--odaocr-repo",        default=str(ODAOCR_REPO))
    p.add_argument("--odaocr-checkpoint",  default=str(ODAOCR_CKPT))
    p.add_argument("--odaocr-tokenizer-dir", default=str(ODAOCR_TOK_DIR))
    return p.parse_args()


def main():
    args   = get_args()
    batch  = args.batch
    xml_dir = PAGEXML_BASE / batch / "page"
    img_dir = PDF_IMAGES_BASE / batch

    if not xml_dir.exists():
        sys.exit(f"ERROR: PageXML dir not found: {xml_dir}")
    if not img_dir.exists():
        sys.exit(f"ERROR: Image dir not found: {img_dir}")

    xmls = sorted(xml_dir.glob("*.xml"))
    if args.n_pages > 0:
        xmls = xmls[: args.n_pages]

    import os as _os
    n_workers = args.workers if args.workers > 0 else (_os.cpu_count() or 1)
    print(f"Batch   : {batch}")
    print(f"Pages   : {len(xmls)}")
    print(f"Workers : {n_workers} threads (Tesseract)")
    print(f"Stages  : crop  +  tesseract{''  if args.no_odaocr else '  +  odaocr'}\n")

    # ── load ODAOCR once (heavy, ~4s) ─────────────────────────────────────────
    model = tokenizer = decoder = None
    if not args.no_odaocr:
        print("Loading ODAOCR model …", end=" ", flush=True)
        with Timer() as t_load:
            model, tokenizer, decoder = pipe.load_odaocr(
                Path(args.odaocr_repo),
                Path(args.odaocr_checkpoint),
                Path(args.odaocr_tokenizer_dir),
                "cuda",
            )
        print(f"done ({t_load.elapsed:.1f}s)\n")

    # ── header ────────────────────────────────────────────────────────────────
    col = "{:<14}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7}"
    hdr = col.format(
        "page", "lines", "tess_hit", "hw_lines",
        "crop", "tesseract", "odaocr", "total", "ms/line"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for xml_path in xmls:
        work_dir = Path(tempfile.mkdtemp(prefix="ocr_bench_"))
        try:
            r = benchmark_page(
                xml_path, img_dir, work_dir,
                args.tess_lang, args.workers,
                model, tokenizer, decoder,
                run_odaocr=not args.no_odaocr,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        ms_per_line = (r["t_total"] / r["n_lines"] * 1000) if r["n_lines"] else 0
        print(col.format(
            r["stem"],
            r["n_lines"],
            r["n_tess"],
            r["n_hw"],
            fmt(r["t_crop"], 7),
            fmt(r["t_tess"], 7),
            fmt(r["t_oda"],  7),
            fmt(r["t_total"], 6),
            f"{ms_per_line:>6.0f}ms",
        ))
        results.append(r)

    if not results:
        return

    # ── summary ───────────────────────────────────────────────────────────────
    print("-" * len(hdr))

    def avg(key): return sum(r[key] for r in results) / len(results)
    def tot(key): return sum(r[key] for r in results)

    avg_lines = avg("n_lines")
    avg_total = avg("t_total")
    avg_ms    = (avg_total / avg_lines * 1000) if avg_lines else 0

    print(col.format(
        f"AVG ({len(results)} pages)",
        f"{avg_lines:.0f}",
        f"{avg('n_tess'):.0f}",
        f"{avg('n_hw'):.0f}",
        fmt(avg("t_crop"), 7),
        fmt(avg("t_tess"), 7),
        fmt(avg("t_oda"),  7),
        fmt(avg_total, 6),
        f"{avg_ms:>6.0f}ms",
    ))

    print()
    print("Stage breakdown (avg per page):")
    total_t = avg("t_crop") + avg("t_tess") + avg("t_oda")
    for label, key in [("  crop     ", "t_crop"), ("  tesseract", "t_tess"), ("  odaocr   ", "t_oda")]:
        t = avg(key)
        pct = (t / total_t * 100) if total_t else 0
        bar = "█" * int(pct / 2)
        print(f"{label}  {fmt(t)}  {pct:5.1f}%  {bar}")

    print()
    n_tess_total = tot("n_tess")
    n_hw_total   = tot("n_hw")
    n_total      = tot("n_lines")
    if n_tess_total > 0:
        ms_per_tess = tot("t_tess") / n_tess_total * 1000
        print(f"  Tesseract  {ms_per_tess:.1f} ms/line  ({n_tess_total} lines)")
    if n_hw_total > 0 and not args.no_odaocr:
        ms_per_oda = tot("t_oda") / n_hw_total * 1000 if n_hw_total else 0
        print(f"  ODAOCR     {ms_per_oda:.1f} ms/line  ({n_hw_total} lines, word-split)")
    print(f"  Total      {avg_total:.2f}s/page  ({avg_ms:.0f} ms/line avg)")


if __name__ == "__main__":
    main()
