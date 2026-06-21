"""
Benchmark Laypa baseline detection on Radio Barcelona page images.

Copies N pages into a temp directory and times a single inference.py
subprocess call (which is how it runs in production). Reports total time,
per-page throughput, and seconds-per-page.

Usage:
    conda activate docs   # or base — this script launches laypa env itself
    python line-classifier/benchmark_laypa.py --n-pages 5

    # test different dataloader worker counts:
    python line-classifier/benchmark_laypa.py --n-pages 10 --num-workers 8
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

PDF_IMAGES_BASE = Path("/data/storage/datasets/RadioBarcelona/pdf_images")
LAYPA_DIR       = Path("/home/tockier/loghi/laypa")
LAYPA_CONFIG    = LAYPA_DIR / "configs/segmentation/baseline/baseline_general.yaml"
LAYPA_WEIGHTS   = Path("/home/tockier/loghi/models/public-models/laypa/general/baseline2/model_best_mIoU.pth")
LAYPA_CONDA_ENV = "laypa"


# ── helpers ───────────────────────────────────────────────────────────────────

def find_images(batch_dir: Path, n: int) -> list[Path]:
    imgs = sorted(
        p for p in batch_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    return imgs[:n] if n > 0 else imgs


def fmt_s(t: float) -> str:
    if t >= 60:
        m, s = divmod(t, 60)
        return f"{int(m)}m{s:.1f}s"
    return f"{t:.2f}s"


# ── benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(images: list[Path], num_workers: int,
                  warmup: bool) -> dict:
    """
    Copy images to a temp dir, run laypa inference, return timing dict.
    warmup=True means this run's time is labelled as warm-up.
    """
    tmp_in  = Path(tempfile.mkdtemp(prefix="laypa_bench_in_"))
    tmp_out = Path(tempfile.mkdtemp(prefix="laypa_bench_out_"))

    try:
        for p in images:
            shutil.copy2(p, tmp_in / p.name)

        cmd = [
            "conda", "run", "--no-capture-output", "-n", LAYPA_CONDA_ENV,
            "python", str(LAYPA_DIR / "inference.py"),
            "-c", str(LAYPA_CONFIG),
            "-i", str(tmp_in),
            "-o", str(tmp_out),
            "--num_workers", str(num_workers),
            "--opts",
            "TEST.WEIGHTS", str(LAYPA_WEIGHTS),
        ]

        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=str(LAYPA_DIR),
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print("\nLaypa stderr (last 20 lines):")
            for line in result.stderr.strip().splitlines()[-20:]:
                print(" ", line)
            sys.exit(f"ERROR: Laypa exited with code {result.returncode}")

        # count produced XMLs
        n_xml = len(list((tmp_out / "page").glob("*.xml"))) if (tmp_out / "page").exists() else 0

        return {
            "n_images":    len(images),
            "n_xml":       n_xml,
            "elapsed":     elapsed,
            "s_per_page":  elapsed / len(images),
            "pages_per_s": len(images) / elapsed,
            "warmup":      warmup,
            "num_workers": num_workers,
        }
    finally:
        shutil.rmtree(tmp_in,  ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Benchmark Laypa baseline detection")
    p.add_argument("--batch",       default="guiradbcn_a1937m10")
    p.add_argument("--n-pages",     type=int, default=5,
                   help="Pages per run (0 = full batch)")
    p.add_argument("--n-runs",      type=int, default=3,
                   help="Number of timed runs (default 3)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Laypa dataloader workers (default 4)")
    p.add_argument("--no-warmup",   action="store_true",
                   help="Skip the warm-up run")
    return p.parse_args()


def main():
    args = get_args()

    batch_dir = PDF_IMAGES_BASE / args.batch
    if not batch_dir.exists():
        sys.exit(f"ERROR: batch dir not found: {batch_dir}")
    if not LAYPA_CONFIG.exists():
        sys.exit(f"ERROR: config not found: {LAYPA_CONFIG}")
    if not LAYPA_WEIGHTS.exists():
        sys.exit(f"ERROR: weights not found: {LAYPA_WEIGHTS}")

    images = find_images(batch_dir, args.n_pages)
    if not images:
        sys.exit(f"ERROR: no images found in {batch_dir}")

    print(f"Batch      : {args.batch}")
    print(f"Pages/run  : {len(images)}")
    print(f"Runs       : {args.n_runs}{' + 1 warm-up' if not args.no_warmup else ''}")
    print(f"DL workers : {args.num_workers}")
    print(f"Weights    : {LAYPA_WEIGHTS.name}")
    print()

    results = []

    # warm-up run (model load, CUDA init — not counted in stats)
    if not args.no_warmup:
        print("Warm-up run (model/GPU init) …", end=" ", flush=True)
        r = run_benchmark(images[:1], args.num_workers, warmup=True)
        print(f"done ({fmt_s(r['elapsed'])})")
        print()

    # timed runs
    for i in range(1, args.n_runs + 1):
        print(f"Run {i}/{args.n_runs} ({len(images)} pages) …", end=" ", flush=True)
        r = run_benchmark(images, args.num_workers, warmup=False)
        print(f"{fmt_s(r['elapsed'])}  ({r['s_per_page']:.2f}s/page  "
              f"{r['pages_per_s']:.2f} pages/s)  [{r['n_xml']} XMLs produced]")
        results.append(r)

    if not results:
        return

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed_vals = [r["elapsed"] for r in results]
    spp_vals     = [r["s_per_page"] for r in results]

    avg_elapsed = sum(elapsed_vals) / len(elapsed_vals)
    min_elapsed = min(elapsed_vals)
    avg_spp     = sum(spp_vals) / len(spp_vals)
    min_spp     = min(spp_vals)

    print()
    print("─" * 52)
    print(f"  Pages per run  : {len(images)}")
    print(f"  Avg total time : {fmt_s(avg_elapsed)}")
    print(f"  Best total time: {fmt_s(min_elapsed)}")
    print(f"  Avg s/page     : {avg_spp:.2f}s")
    print(f"  Best s/page    : {min_spp:.2f}s")
    print(f"  Throughput     : {1/avg_spp:.2f} pages/s  "
          f"({60/avg_spp:.0f} pages/min)")
    print()

    # estimate full batch time
    all_imgs = find_images(batch_dir, 0)
    est_full = avg_spp * len(all_imgs)
    print(f"  Full batch ({len(all_imgs)} pages) estimate: {fmt_s(est_full)}")
    print("─" * 52)


if __name__ == "__main__":
    main()
