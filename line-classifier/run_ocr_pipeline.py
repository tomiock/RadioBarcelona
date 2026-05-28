"""
OCR routing pipeline for Radio Barcelona PageXML documents.

For each TextLine in the input PageXML:
  1. Crop the line from the page image (using Laypa baseline coordinates)
  2. Classify it as H (handwritten) or T (typewritten) with a LineCNN
  3. Route T lines → Tesseract (pytesseract, PSM 7, spa+cat)
     Route H lines → Loghi HTR (Docker)
  4. Write the recognised text back as <TextEquiv><Unicode> in the PageXML

Prerequisites:
  - tesseract + tesseract-ocr-spa + tesseract-ocr-cat installed
    (sudo apt-get install tesseract-ocr tesseract-ocr-spa tesseract-ocr-cat)
  - pip install pytesseract
  - Docker available with image loghi/docker.htr:2.3.0 pulled
  - A trained H/T checkpoint from train_ht_classifier.py

Usage (Step A — classify only, no OCR):
    python scripts/run_ocr_pipeline.py \
        --xml-dir    /data/.../page \
        --img-dir    /data/.../pdf_images/guiradbcn_a1931m10 \
        --checkpoint /data/.../laypa_ht_classify/model/best.pth \
        --work-dir   /tmp/ocr_work \
        --output-xml-dir /data/.../page_ocr \
        --classify-only

Full run:
    python scripts/run_ocr_pipeline.py \
        --xml-dir    /data/.../page \
        --img-dir    /data/.../pdf_images/guiradbcn_a1931m10 \
        --checkpoint /data/.../laypa_ht_classify/model/best.pth \
        --loghi-model /data/.../loghi_model \
        --work-dir   /tmp/ocr_work \
        --output-xml-dir /data/.../page_ocr
"""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Register the PageXML namespace once at import time so ET never emits ns0: prefix
_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", _NS)


# ── model (matches train_ht_classifier.py) ────────────────────────────────────

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, pool=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        super().__init__(*layers)


class LineCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ── line extraction ───────────────────────────────────────────────────────────

def parse_points(points_str: str) -> np.ndarray:
    pts = []
    for tok in points_str.strip().split():
        x, y = tok.split(",")
        pts.append([int(x), int(y)])
    return np.array(pts)


def collect_lines_with_id(xml_dir: Path, img_dir: Path) -> list[dict]:
    """Collect all TextLine baselines, including xml_path and line @id for write-back."""
    lines = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        stem = xml_path.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for tl in root.findall(f".//{{{_NS}}}TextLine"):
            bl = tl.find(f".//{{{_NS}}}Baseline")
            if bl is None:
                continue
            line_id = tl.attrib.get("id", "")
            pts = parse_points(bl.attrib["points"])
            lines.append({
                "img_path": img_path,
                "baseline": pts,
                "stem": stem,
                "xml_path": xml_path,
                "line_id": line_id,
            })
    return lines


def crop_line(img: np.ndarray, baseline: np.ndarray,
              pad_top: int, pad_bot: int, pad_side: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x_min = int(baseline[:, 0].min()) - pad_side
    x_max = int(baseline[:, 0].max()) + pad_side
    y_min = int(baseline[:, 1].min()) - pad_top
    y_max = int(baseline[:, 1].max()) + pad_bot
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    if x_max - x_min < 10 or y_max - y_min < 4:
        return None
    return img[y_min:y_max, x_min:x_max]


def extract_crops(lines: list[dict], crops_dir: Path,
                  pad_top: int, pad_bot: int, pad_side: int) -> list[dict]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    records = []
    img_cache: dict[str, np.ndarray] = {}
    skipped = 0

    for i, entry in enumerate(lines):
        img_key = str(entry["img_path"])
        if img_key not in img_cache:
            img = cv2.imread(img_key)
            if img is None:
                skipped += 1
                continue
            img_cache = {img_key: img}

        crop = crop_line(img_cache[img_key], entry["baseline"], pad_top, pad_bot, pad_side)
        if crop is None:
            skipped += 1
            continue

        safe_id = entry["line_id"].replace("/", "_").replace(":", "_")
        crop_name = f"{entry['stem']}_{safe_id}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        records.append({
            "idx":       i,
            "crop_path": crop_path,
            "xml_path":  entry["xml_path"],
            "line_id":   entry["line_id"],
            "stem":      entry["stem"],
            "label":     None,
            "ocr_text":  None,
        })

    if skipped:
        print(f"  Skipped {skipped} lines (unreadable image or too small)")
    return records


# ── H/T classification ────────────────────────────────────────────────────────

def collate_pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of (1, H, W) tensors to the same spatial size."""
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = torch.zeros(len(tensors), 1, max_h, max_w)
    for i, t in enumerate(tensors):
        padded[i, :, :t.shape[1], :t.shape[2]] = t
    return padded


_LABEL_MAP = {0: "H", 1: "T"}


@torch.no_grad()
def classify_crops(model: nn.Module, records: list[dict],
                   device: torch.device, batch_size: int) -> None:
    """Fills record['label'] in-place with 'H' or 'T'."""
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        tensors = []
        for rec in batch:
            img = cv2.imread(str(rec["crop_path"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((38, 64), dtype=np.uint8)
            tensors.append(
                torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            )
        padded = collate_pad_tensors(tensors).to(device)
        preds = model(padded).argmax(dim=1).cpu().tolist()
        for rec, pred in zip(batch, preds):
            rec["label"] = _LABEL_MAP[pred]


# ── Tesseract (typewritten lines) ─────────────────────────────────────────────

def _tessdata_dir() -> str:
    """Detect the tessdata directory from the tesseract binary to avoid
    TESSDATA_PREFIX not being propagated into the subprocess."""
    import re
    result = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True
    )
    # output: 'List of available languages in "/path/to/tessdata/" (N):'
    m = re.search(r'in "([^"]+)"', result.stdout + result.stderr)
    return m.group(1).rstrip("/") if m else ""


def run_tesseract_batch(records: list[dict], lang: str) -> None:
    """OCR typewritten line crops with Tesseract PSM 7 (single text line)."""
    try:
        import pytesseract
    except ImportError:
        print("pytesseract not installed — skipping Tesseract OCR.")
        for rec in records:
            rec["ocr_text"] = ""
        return

    tessdata = _tessdata_dir()
    tessdata_flag = f"--tessdata-dir {tessdata}" if tessdata else ""
    config = f"--psm 7 --oem 3 -l {lang} {tessdata_flag}".strip()
    for rec in records:
        img = cv2.imread(str(rec["crop_path"]))
        if img is None:
            rec["ocr_text"] = ""
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            text = pytesseract.image_to_string(rgb, config=config).strip()
        except Exception as e:
            print(f"  Tesseract error on {rec['crop_path'].name}: {e}")
            text = ""
        rec["ocr_text"] = text


# ── Loghi HTR (handwritten lines) ─────────────────────────────────────────────

def run_loghi_htr(records: list[dict], work_dir: Path,
                  loghi_model: str, docker_image: str, batch_size: int) -> None:
    """
    Write an inference list, call Loghi HTR via Docker, parse the results TSV.
    work_dir and loghi_model must be absolute paths — they are mounted into the
    container at the same path so crop file references stay valid.
    """
    lines_file   = work_dir / "loghi_input.txt"
    results_file = work_dir / "loghi_results.txt"

    with open(lines_file, "w") as f:
        for rec in records:
            f.write(str(rec["crop_path"].resolve()) + "\n")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_dir.resolve()}:{work_dir.resolve()}",
        "-v", f"{loghi_model}:{loghi_model}",
        docker_image,
        "bash", "-c",
        (
            f"python3 /src/loghi-htr/src/main.py"
            f" --model {loghi_model}"
            f" --batch_size {batch_size}"
            f" --inference_list {lines_file.resolve()}"
            f" --results_file {results_file.resolve()}"
        ),
    ]

    print(f"  Running Loghi HTR Docker on {len(records)} lines …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [Loghi] stderr:\n{result.stderr[:1000]}")
        raise RuntimeError("Loghi HTR Docker invocation failed")

    path_to_text: dict[str, str] = {}
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    path_to_text[parts[0]] = parts[1]

    for rec in records:
        rec["ocr_text"] = path_to_text.get(str(rec["crop_path"].resolve()), "")


# ── PageXML write-back ────────────────────────────────────────────────────────

def write_back_xml(xml_path: Path, records: list[dict], output_xml_dir: Path) -> None:
    """Insert <TextEquiv><Unicode> for each processed line into the PageXML."""
    id_to_text = {rec["line_id"]: rec["ocr_text"] for rec in records
                  if rec["ocr_text"] is not None}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for tl in root.findall(f".//{{{_NS}}}TextLine"):
        line_id = tl.attrib.get("id", "")
        text = id_to_text.get(line_id)
        if text is None:
            continue

        existing = tl.find(f"{{{_NS}}}TextEquiv")
        if existing is not None:
            tl.remove(existing)

        te = ET.SubElement(tl, f"{{{_NS}}}TextEquiv")
        unicode_el = ET.SubElement(te, f"{{{_NS}}}Unicode")
        unicode_el.text = text

    out_path = output_xml_dir / xml_path.name
    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)


# ── argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="OCR routing pipeline for Radio Barcelona PageXML")
    p.add_argument("--xml-dir",        required=True,  help="PageXML input directory")
    p.add_argument("--img-dir",        required=True,  help="Page image directory")
    p.add_argument("--output-xml-dir", required=True,  help="Where to write updated PageXML")
    p.add_argument("--checkpoint",     required=True,  help="H/T LineCNN checkpoint (best.pth)")
    p.add_argument("--loghi-model",    default="",     help="Absolute path to Loghi HTR model")
    p.add_argument("--work-dir",       required=True,  help="Temp dir for crops and Loghi I/O")
    p.add_argument("--batch-size",     type=int, default=64)
    p.add_argument("--tesseract-lang", default="spa+cat")
    p.add_argument("--loghi-docker",   default="loghi/docker.htr:2.3.0")
    p.add_argument("--pad-top",        type=int, default=28)
    p.add_argument("--pad-bot",        type=int, default=10)
    p.add_argument("--pad-side",       type=int, default=8)
    p.add_argument("--classify-only",  action="store_true",
                   help="Stop after classification; print H/T stats, skip OCR and XML write-back")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    xml_dir        = Path(args.xml_dir)
    img_dir        = Path(args.img_dir)
    output_xml_dir = Path(args.output_xml_dir)
    work_dir       = Path(args.work_dir)

    output_xml_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = work_dir / "crops"

    # ── load H/T model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = LineCNN(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded H/T checkpoint: val_acc={ckpt.get('val_acc', '?'):.3f}")

    # ── collect and crop ──────────────────────────────────────────────────────
    print("\nCollecting lines from PageXML …")
    lines = collect_lines_with_id(xml_dir, img_dir)
    print(f"  Found {len(lines)} TextLines across {len(list(xml_dir.glob('*.xml')))} XML files")

    print("Cropping lines …")
    records = extract_crops(lines, crops_dir, args.pad_top, args.pad_bot, args.pad_side)
    print(f"  Cropped {len(records)} lines")

    # ── classify ──────────────────────────────────────────────────────────────
    print("Classifying H/T …")
    classify_crops(model, records, device, args.batch_size)

    h_records = [r for r in records if r["label"] == "H"]
    t_records = [r for r in records if r["label"] == "T"]
    print(f"  H (handwritten): {len(h_records)}  |  T (typewritten): {len(t_records)}")

    if args.classify_only:
        print("\n--classify-only set; stopping before OCR. Stats printed above.")
        # Save a classification summary
        summary = [{"crop": str(r["crop_path"]), "label": r["label"],
                    "line_id": r["line_id"], "stem": r["stem"]} for r in records]
        summary_path = work_dir / "classification_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Classification summary saved to: {summary_path}")
        return

    # ── Tesseract (typewritten) ───────────────────────────────────────────────
    if t_records:
        print(f"\nRunning Tesseract on {len(t_records)} typewritten lines …")
        run_tesseract_batch(t_records, args.tesseract_lang)
        done = sum(1 for r in t_records if r["ocr_text"])
        print(f"  Tesseract: {done}/{len(t_records)} lines produced text")

    # ── Loghi HTR (handwritten) ───────────────────────────────────────────────
    if h_records:
        if not args.loghi_model:
            print("\n--loghi-model not provided; skipping Loghi HTR for handwritten lines.")
            for rec in h_records:
                rec["ocr_text"] = ""
        else:
            print(f"\nRunning Loghi HTR on {len(h_records)} handwritten lines …")
            run_loghi_htr(h_records, work_dir, args.loghi_model,
                          args.loghi_docker, args.batch_size)
            done = sum(1 for r in h_records if r["ocr_text"])
            print(f"  Loghi HTR: {done}/{len(h_records)} lines produced text")

    # ── write back to PageXML ─────────────────────────────────────────────────
    print("\nWriting OCR results into PageXML …")
    by_xml: dict[Path, list[dict]] = defaultdict(list)
    for rec in records:
        by_xml[rec["xml_path"]].append(rec)

    for xml_path, xml_records in by_xml.items():
        write_back_xml(xml_path, xml_records, output_xml_dir)

    print(f"Updated PageXML written to: {output_xml_dir}")
    print(f"  Files: {len(by_xml)}")
    total_text = sum(1 for r in records if r["ocr_text"])
    print(f"  Lines with text: {total_text}/{len(records)}")


if __name__ == "__main__":
    main()
