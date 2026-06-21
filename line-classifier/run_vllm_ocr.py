"""
VLM-based OCR pipeline for Radio Barcelona PageXML documents.

For each TextLine in the input PageXML:
  1. Crop the line from the page image (using Laypa Baseline coordinates)
  2. Send the crop to a vLLM-hosted vision model via the OpenAI-compatible API
  3. Write the response back as <TextEquiv engine="vllm"><Unicode> in the PageXML

Intended use: generate a high-quality reference transcription that is then
corrected by a human annotator in annotate_ocr.py to produce a benchmark.

Usage:
    conda activate docs
    python line-classifier/run_vllm_ocr.py \\
        --xml-dir        /data/storage/users/tockier/laypa_vis/page_examples/page \\
        --img-dir        /data/storage/datasets/RadioBarcelona/pdf_images/page_examples \\
        --output-xml-dir /data/storage/users/tockier/ocr_output/page_examples_vllm/page \\
        --work-dir       /tmp/ocr_work/page_examples_vllm
"""

import argparse
import base64
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", _NS)

VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_ID    = "Qwen/Qwen3.5-27B-FP8"

SYSTEM_PROMPT = (
    "You are an OCR engine for historical Spanish and Catalan documents "
    "from the Radio Barcelona archive (1924–1953). "
    "Documents are a mix of typewritten and handwritten text. "
    "Transcribe exactly what is written in the image — do not correct spelling, "
    "do not add punctuation that is not there, do not translate. "
    "Output only the transcribed text, nothing else. "
    "If the image contains no readable text, output an empty string."
)

OCR_PROMPT = "Transcribe the text in this image."


# ── PageXML helpers (reused from run_ocr_pipeline) ───────────────────────────

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
                "baseline": parse_points(bl.attrib["points"]),
                "stem":     stem,
                "xml_path": xml_path,
                "line_id":  tl.attrib.get("id", ""),
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
            img_cache = {key: img}
        crop = crop_line(img_cache[key], entry["baseline"], pad_top, pad_bot, pad_side)
        if crop is None:
            skipped += 1
            continue
        safe_id   = entry["line_id"].replace("/", "_").replace(":", "_")
        crop_path = crops_dir / f"{entry['stem']}_{safe_id}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "crop_path": crop_path,
            "xml_path":  entry["xml_path"],
            "line_id":   entry["line_id"],
            "stem":      entry["stem"],
            "ocr_text":  None,
            "engine":    "vllm",
        })
    if skipped:
        print(f"  Skipped {skipped} lines (unreadable image or too small)")
    return records


# ── vLLM inference ────────────────────────────────────────────────────────────

def _crop_to_b64(crop_path: Path) -> str:
    with open(crop_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _call_vllm(b64_img: str, session) -> str:
    """Send one crop to vLLM; returns stripped transcription text."""
    payload = {
        "model": MODEL_ID,
        "max_tokens": 16384,   # enough budget for thinking + short answer
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                {"type": "text", "text": OCR_PROMPT},
            ]},
        ],
    }
    resp = session.post(VLLM_URL, json=payload, timeout=600)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Qwen3 thinking lands in content, terminated by </think>.
    # Take everything after the last </think> tag.
    # If </think> is absent the model hit the token limit mid-thought — no answer.
    if "</think>" in content:
        content = content[content.rfind("</think>") + len("</think>"):].strip()
    else:
        content = ""
    return content


def run_vllm(records: list[dict], workers: int = 8) -> None:
    """Run vLLM OCR on all records in parallel; fills ocr_text in-place."""
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        sys.exit("ERROR: pip install requests")

    def make_session():
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5,
                      status_forcelist=[500, 502, 503, 504])
        s.mount("http://", HTTPAdapter(max_retries=retry))
        return s

    errors = 0

    def _process(rec: dict) -> tuple[dict, str, str]:
        # skip crops too small to contain readable text (noise, stamps, rules)
        img = cv2.imread(str(rec["crop_path"]))
        if img is not None:
            h, w = img.shape[:2]
            if w < 60 or h < 15 or w * h < 3000:
                return rec, "", None
        session = make_session()
        b64 = _crop_to_b64(rec["crop_path"])
        text = _call_vllm(b64, session)
        return rec, text, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, rec): rec for rec in records}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                rec, text, _ = fut.result()
                rec["ocr_text"] = text
                rec["engine"]   = "vllm" if text else None
            except Exception as e:
                rec = futures[fut]
                rec["ocr_text"] = ""
                rec["engine"]   = None
                errors += 1
                print(f"  vLLM error on {rec['crop_path'].name}: {e}")
            if done % 50 == 0 or done == len(records):
                hit = sum(1 for r in records[:done] if r["ocr_text"] is not None and r["ocr_text"])
                print(f"  {done}/{len(records)} done  ({hit} with text)", flush=True)

    if errors:
        print(f"  {errors} errors total")


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


# ── main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="vLLM OCR pipeline for Radio Barcelona PageXML")
    p.add_argument("--xml-dir",          required=True)
    p.add_argument("--img-dir",          required=True)
    p.add_argument("--output-xml-dir",   required=True)
    p.add_argument("--work-dir",         required=True)
    p.add_argument("--workers",  type=int, default=2,
                   help="Parallel vLLM requests (default 2)")
    p.add_argument("--page",     default="",
                   help="Process only this page stem (e.g. 1940_1). Omit to process all pages.")
    p.add_argument("--pad-top",  type=int, default=28)
    p.add_argument("--pad-bot",  type=int, default=10)
    p.add_argument("--pad-side", type=int, default=8)
    return p.parse_args()


def main():
    args = get_args()
    xml_dir        = Path(args.xml_dir)
    img_dir        = Path(args.img_dir)
    output_xml_dir = Path(args.output_xml_dir)
    work_dir       = Path(args.work_dir)
    crops_dir      = work_dir / "crops"

    output_xml_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\nCollecting lines from PageXML …")
    lines = collect_lines(xml_dir, img_dir)
    if args.page:
        lines = [l for l in lines if l["stem"] == args.page]
        if not lines:
            sys.exit(f"ERROR: no lines found for page '{args.page}'")
    n_pages = len(set(l["stem"] for l in lines))
    print(f"  {len(lines)} TextLines across {n_pages} page(s)")

    print("Cropping lines …")
    records = extract_crops(lines, crops_dir, args.pad_top, args.pad_bot, args.pad_side)
    print(f"  {len(records)} crops saved")

    t0 = time.perf_counter()
    print(f"\nRunning vLLM OCR ({args.workers} workers) on {len(records)} lines …")
    run_vllm(records, workers=args.workers)
    elapsed = time.perf_counter() - t0

    hit = sum(1 for r in records if r["ocr_text"])
    print(f"\n  {hit}/{len(records)} lines with text  ({elapsed:.1f}s total, "
          f"{elapsed/len(records)*1000:.0f}ms/line avg)")

    print("\nWriting results into PageXML …")
    by_xml: dict[Path, list[dict]] = defaultdict(list)
    for rec in records:
        by_xml[rec["xml_path"]].append(rec)
    for xml_path, xml_records in by_xml.items():
        write_back_xml(xml_path, xml_records, output_xml_dir)

    print(f"Done. Output: {output_xml_dir}  ({len(by_xml)} files)")


if __name__ == "__main__":
    main()
