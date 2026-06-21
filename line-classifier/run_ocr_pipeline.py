"""
OCR routing pipeline for Radio Barcelona PageXML documents.

For each TextLine in the input PageXML:
  1. Crop the line from the page image (using Laypa baseline coordinates)
  2. Run Tesseract (PSM 7, spa+cat) on every line
  3. Lines that got text  → keep Tesseract result (typewritten)
     Lines that got nothing → run through ODAOCR (handwritten focus)
  4. Write recognised text back as <TextEquiv><Unicode> in the PageXML

The routing heuristic relies on the empirical observation that Tesseract
returns an empty string on handwritten content but reliably transcribes
typewritten text — so empty Tesseract output is used as the handwritten
signal rather than a CNN classifier.

Prerequisites (conda env docs):
  - tesseract + tesseract-ocr-spa + tesseract-ocr-cat
  - pip install pytesseract pillow

ODAOCR:
  - The ODAOCR repo must be reachable (--odaocr-repo or ODAOCR_REPO env var)
  - checkpoint: --odaocr-checkpoint  (e.g. ODAOCR/MODELS/model_metalearnt.pt)
  - tokenizer:  --odaocr-tokenizer-dir (dir containing tokenizer.json)

Usage:
    python line-classifier/run_ocr_pipeline.py \\
        --xml-dir        /data/.../laypa_vis/guiradbcn_a1937m10/page \\
        --img-dir        /data/.../pdf_images/guiradbcn_a1937m10 \\
        --output-xml-dir /data/.../ocr_output/guiradbcn_a1937m10/page \\
        --work-dir       /tmp/ocr_work/guiradbcn_a1937m10 \\
        --odaocr-checkpoint /path/to/ODAOCR/MODELS/model_metalearnt.pt \\
        --odaocr-tokenizer-dir /path/to/ODAOCR/MODELS
"""

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Register the PageXML namespace once so ET never emits ns0: prefix
_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", _NS)


# ── line extraction ───────────────────────────────────────────────────────────

def parse_points(points_str: str) -> np.ndarray:
    pts = []
    for tok in points_str.strip().split():
        x, y = tok.split(",")
        pts.append([int(x), int(y)])
    return np.array(pts)


def collect_lines(xml_dir: Path, img_dir: Path) -> list[dict]:
    lines = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        stem = xml_path.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            c = img_dir / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for tl in root.findall(f".//{{{_NS}}}TextLine"):
            bl = tl.find(f".//{{{_NS}}}Baseline")
            if bl is None:
                continue
            lines.append({
                "img_path": img_path,
                "baseline":  parse_points(bl.attrib["points"]),
                "stem":      stem,
                "xml_path":  xml_path,
                "line_id":   tl.attrib.get("id", ""),
            })
    return lines


def crop_line(img: np.ndarray, baseline: np.ndarray,
              pad_top: int, pad_bot: int, pad_side: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x_min = max(0, int(baseline[:, 0].min()) - pad_side)
    x_max = min(w, int(baseline[:, 0].max()) + pad_side)
    y_min = max(0, int(baseline[:, 1].min()) - pad_top)
    y_max = min(h, int(baseline[:, 1].max()) + pad_bot)
    if x_max - x_min < 10 or y_max - y_min < 4:
        return None
    return img[y_min:y_max, x_min:x_max]


def extract_crops(lines: list[dict], crops_dir: Path,
                  pad_top: int, pad_bot: int, pad_side: int) -> list[dict]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    records = []
    img_cache: dict[str, np.ndarray] = {}
    skipped = 0

    for entry in lines:
        key = str(entry["img_path"])
        if key not in img_cache:
            img = cv2.imread(key)
            if img is None:
                skipped += 1
                continue
            img_cache = {key: img}   # one image at a time to limit memory

        crop = crop_line(img_cache[key], entry["baseline"], pad_top, pad_bot, pad_side)
        if crop is None:
            skipped += 1
            continue

        safe_id  = entry["line_id"].replace("/", "_").replace(":", "_")
        crop_path = crops_dir / f"{entry['stem']}_{safe_id}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        records.append({
            "crop_path": crop_path,
            "xml_path":  entry["xml_path"],
            "line_id":   entry["line_id"],
            "stem":      entry["stem"],
            "ocr_text":  None,
            "engine":    None,
        })

    if skipped:
        print(f"  Skipped {skipped} lines (unreadable image or too small)")
    return records


# ── Tesseract ─────────────────────────────────────────────────────────────────

def _tessdata_dir() -> str:
    result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    m = re.search(r'in "([^"]+)"', result.stdout + result.stderr)
    return m.group(1).rstrip("/") if m else ""


def run_tesseract(records: list[dict], lang: str, workers: int = 0) -> None:
    """Run Tesseract PSM 7 on every record; fills ocr_text in-place.

    workers: number of parallel threads. 0 = os.cpu_count().
    Each call launches a Tesseract subprocess, so threads don't fight the GIL
    and scale linearly up to the number of physical cores.
    """
    try:
        import pytesseract
    except ImportError:
        print("pytesseract not installed — cannot run Tesseract.")
        return

    tessdata = _tessdata_dir()
    tessdata_flag = f"--tessdata-dir {tessdata}" if tessdata else ""
    config = f"--psm 7 --oem 3 -l {lang} {tessdata_flag}".strip()
    n_workers = workers if workers > 0 else (os.cpu_count() or 1)

    def _run_one(rec: dict) -> None:
        img = cv2.imread(str(rec["crop_path"]))
        if img is None:
            rec["ocr_text"] = ""
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            text = pytesseract.image_to_string(rgb, config=config).strip()
        except Exception as e:
            print(f"  Tesseract error on {rec['crop_path'].name}: {e}")
            text = ""
        rec["ocr_text"] = text
        rec["engine"]   = "tesseract" if text else None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, rec): rec for rec in records}
        for fut in as_completed(futures):
            if fut.exception():
                print(f"  Tesseract thread error: {fut.exception()}")


# ── ODAOCR (handwritten lines) ────────────────────────────────────────────────

def load_odaocr(odaocr_repo: Path, checkpoint: Path,
                tokenizer_dir: Path, device: str):
    """Load ODAOCR model + tokenizer. Returns (model, tokenizer, decoder)."""
    sys.path.insert(0, str(odaocr_repo))
    from constructors import prepare_model, GreedyTextDecoder, CharTokenizer  # noqa: E402

    tokenizer = CharTokenizer(False, str(tokenizer_dir), "tokenizer")
    model = prepare_model(
        len(tokenizer), device=device,
        load_checkpoint=True, checkpoint_name=str(checkpoint),
    )
    model.eval()
    decoder = GreedyTextDecoder()
    return model, tokenizer, decoder


def _split_words(crop_bgr: np.ndarray, min_gap: int = 8, min_w: int = 10) -> list[np.ndarray]:
    """Split a line crop into word-sized crops via vertical ink projection."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = ((255 - binary) // 255).sum(axis=0).astype(int)
    w = crop_bgr.shape[1]
    in_word, segs, x0 = False, [], 0
    for x in range(w):
        if proj[x] > 0:
            if not in_word:
                x0, in_word = x, True
        else:
            if in_word:
                in_word = False
                if x - x0 >= min_w:
                    segs.append([x0, x])
    if in_word and w - x0 >= min_w:
        segs.append([x0, w])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [crop_bgr[:, max(0, a - 4):min(w, b + 4)] for a, b in merged]


def run_odaocr(records: list[dict], model, tokenizer, decoder, device: str) -> None:
    """Run ODAOCR on records using word-split inference; fills ocr_text in-place."""
    from constructors import make_inference  # already on sys.path
    from PIL import Image

    def _infer(crop_bgr: np.ndarray) -> str:
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        r = make_inference(model, tokenizer, decoder, pil, device)
        return r[0].strip() if r else ""

    def _infer_words(crop_bgr: np.ndarray) -> str:
        words = _split_words(crop_bgr)
        if not words:
            return _infer(crop_bgr)
        return " ".join(p for p in (_infer(wc) for wc in words) if p)

    for rec in records:
        try:
            crop = cv2.imread(str(rec["crop_path"]))
            if crop is None:
                raise ValueError("unreadable crop")
            text = _infer_words(crop)
        except Exception as e:
            print(f"  ODAOCR error on {rec['crop_path'].name}: {e}")
            text = ""
        rec["ocr_text"] = text
        rec["engine"]   = "odaocr" if text else None


# ── PageXML write-back ────────────────────────────────────────────────────────

def write_back_xml(xml_path: Path, records: list[dict], output_xml_dir: Path) -> None:
    id_to_rec = {r["line_id"]: r for r in records if r["ocr_text"] is not None}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for tl in root.findall(f".//{{{_NS}}}TextLine"):
        rec = id_to_rec.get(tl.attrib.get("id", ""))
        if rec is None:
            continue
        existing = tl.find(f"{{{_NS}}}TextEquiv")
        if existing is not None:
            tl.remove(existing)
        te = ET.SubElement(tl, f"{{{_NS}}}TextEquiv")
        if rec.get("engine"):
            te.set("engine", rec["engine"])
        ET.SubElement(te, f"{{{_NS}}}Unicode").text = rec["ocr_text"]

    out_path = output_xml_dir / xml_path.name
    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)


# ── argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="OCR pipeline for Radio Barcelona PageXML")
    p.add_argument("--xml-dir",            required=True)
    p.add_argument("--img-dir",            required=True)
    p.add_argument("--output-xml-dir",     required=True)
    p.add_argument("--work-dir",           required=True)
    p.add_argument("--tesseract-lang",     default="spa+cat")
    p.add_argument("--workers",            type=int, default=0,
                   help="Tesseract threads (0 = all CPU cores)")
    p.add_argument("--odaocr-repo",        default="",
                   help="Path to ODAOCR repo (default: sibling of this script's parent)")
    p.add_argument("--odaocr-checkpoint",  default="",
                   help="Path to ODAOCR .pt checkpoint")
    p.add_argument("--odaocr-tokenizer-dir", default="",
                   help="Dir containing tokenizer.json")
    p.add_argument("--batch-size",         type=int, default=64, help="(unused, kept for compat)")
    p.add_argument("--pad-top",            type=int, default=28)
    p.add_argument("--pad-bot",            type=int, default=10)
    p.add_argument("--pad-side",           type=int, default=8)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    xml_dir        = Path(args.xml_dir)
    img_dir        = Path(args.img_dir)
    output_xml_dir = Path(args.output_xml_dir)
    work_dir       = Path(args.work_dir)
    crops_dir      = work_dir / "crops"

    output_xml_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── collect + crop ────────────────────────────────────────────────────────
    print("\nCollecting lines from PageXML …")
    lines = collect_lines(xml_dir, img_dir)
    print(f"  {len(lines)} TextLines across {len(list(xml_dir.glob('*.xml')))} XML files")

    print("Cropping lines …")
    records = extract_crops(lines, crops_dir, args.pad_top, args.pad_bot, args.pad_side)
    print(f"  {len(records)} crops saved")

    # ── Tesseract: all lines ──────────────────────────────────────────────────
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    print(f"\nRunning Tesseract ({args.tesseract_lang}) on all {len(records)} lines "
          f"[{n_workers} threads] …")
    run_tesseract(records, args.tesseract_lang, workers=args.workers)

    tess_hit  = [r for r in records if r["ocr_text"]]
    tess_miss = [r for r in records if not r["ocr_text"]]
    print(f"  Got text:    {len(tess_hit)}")
    print(f"  Empty (→ handwritten OCR): {len(tess_miss)}")

    # ── ODAOCR: lines Tesseract left empty ───────────────────────────────────
    if tess_miss:
        # resolve ODAOCR repo path
        script_dir  = Path(__file__).resolve().parent
        default_repo = script_dir.parent.parent / "ODAOCR"
        odaocr_repo = Path(args.odaocr_repo) if args.odaocr_repo else default_repo

        if not args.odaocr_checkpoint or not args.odaocr_tokenizer_dir:
            print("\n--odaocr-checkpoint / --odaocr-tokenizer-dir not set; "
                  "skipping handwritten OCR.")
            for rec in tess_miss:
                rec["ocr_text"] = ""
        else:
            ckpt_path = Path(args.odaocr_checkpoint)
            tok_dir   = Path(args.odaocr_tokenizer_dir)
            device    = "cuda"

            print(f"\nLoading ODAOCR from {odaocr_repo} …")
            model, tokenizer, decoder = load_odaocr(odaocr_repo, ckpt_path, tok_dir, device)

            print(f"Running ODAOCR on {len(tess_miss)} handwritten lines …")
            run_odaocr(tess_miss, model, tokenizer, decoder, device)

            oda_hit = sum(1 for r in tess_miss if r["ocr_text"])
            print(f"  ODAOCR got text: {oda_hit}/{len(tess_miss)}")

    # ── write back ────────────────────────────────────────────────────────────
    print("\nWriting results into PageXML …")
    by_xml: dict[Path, list[dict]] = defaultdict(list)
    for rec in records:
        by_xml[rec["xml_path"]].append(rec)

    for xml_path, xml_records in by_xml.items():
        write_back_xml(xml_path, xml_records, output_xml_dir)

    total_text = sum(1 for r in records if r["ocr_text"])
    print(f"Done. {total_text}/{len(records)} lines with text.")
    print(f"Output PageXML: {output_xml_dir}  ({len(by_xml)} files)")


if __name__ == "__main__":
    main()
