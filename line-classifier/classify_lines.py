"""
Crop 500 text lines from Laypa PageXML output and serve a browser UI
for binary classification (H = handwritten, L = typewritten/letterpress).

Usage:
    conda activate laypa
    python scripts/classify_lines.py \
        --xml-dir  /data/storage/users/tockier/laypa_vis/guiradbcn_a1937m10/page \
        --img-dir  /data/storage/datasets/RadioBarcelona/pdf_images/guiradbcn_a1937m10 \
        --output   /data/storage/users/tockier/laypa_classify \
        --n        500 \
        --port     5050

Then open http://localhost:5050 in your browser.
Keys:  H → handwritten   L → typewritten   ← Backspace → undo last
Results are saved to <output>/labels.json after every classification.
"""

import argparse
import base64
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ── argument parsing ──────────────────────────────────────────────────────────

def get_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-line binary classifier UI")
    p.add_argument("--xml-dir",  required=True, help="Dir with PageXML files (page/)")
    p.add_argument("--img-dir",  required=True, help="Dir with original PNG images")
    p.add_argument("--output",   required=True, help="Output dir (crops + labels.json)")
    p.add_argument("--n",        type=int, default=500, help="Number of lines to sample")
    p.add_argument("--port",     type=int, default=5050, help="Flask port")
    p.add_argument("--pad-top",  type=int, default=28,  help="Padding above baseline (px)")
    p.add_argument("--pad-bot",  type=int, default=10,  help="Padding below baseline (px)")
    p.add_argument("--pad-side", type=int, default=8,   help="Horizontal padding (px)")
    p.add_argument("--seed",     type=int, default=42,  help="Random seed")
    return p.parse_args()


# ── line cropping ─────────────────────────────────────────────────────────────

def parse_points(points_str: str) -> np.ndarray:
    pts = []
    for tok in points_str.strip().split():
        x, y = tok.split(",")
        pts.append([int(x), int(y)])
    return np.array(pts)


def collect_lines(xml_dir: Path, img_dir: Path):
    import xml.etree.ElementTree as ET
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
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
        for tl in root.findall(f".//{{{ns}}}TextLine"):
            bl = tl.find(f".//{{{ns}}}Baseline")
            if bl is None:
                continue
            pts = parse_points(bl.attrib["points"])
            lines.append({"img_path": img_path, "baseline": pts, "stem": stem})
    return lines


def crop_line(img: np.ndarray, baseline: np.ndarray,
              pad_top: int, pad_bot: int, pad_side: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x_min = int(baseline[:, 0].min()) - pad_side
    x_max = int(baseline[:, 0].max()) + pad_side
    y_min = int(baseline[:, 1].min()) - pad_top
    y_max = int(baseline[:, 1].max()) + pad_bot
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    if x_max - x_min < 10 or y_max - y_min < 4:
        return None
    return img[y_min:y_max, x_min:x_max]


def prepare_crops(args, output_dir: Path) -> list[dict]:
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        print("Found existing manifest, reusing crops.")
        with open(manifest_path) as f:
            return json.load(f)

    print("Collecting text lines from PageXML …")
    all_lines = collect_lines(Path(args.xml_dir), Path(args.img_dir))
    print(f"  Found {len(all_lines)} lines total.")

    random.seed(args.seed)
    sample = random.sample(all_lines, min(args.n, len(all_lines)))

    manifest = []
    img_cache: dict[str, np.ndarray] = {}
    skipped = 0

    for i, entry in enumerate(sample):
        img_key = str(entry["img_path"])
        if img_key not in img_cache:
            img = cv2.imread(img_key)
            if img is None:
                skipped += 1
                continue
            img_cache = {img_key: img}  # keep only last to save RAM

        crop = crop_line(
            img_cache[img_key], entry["baseline"],
            args.pad_top, args.pad_bot, args.pad_side,
        )
        if crop is None:
            skipped += 1
            continue

        crop_name = f"line_{i:04d}_{entry['stem']}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({
            "id": i,
            "crop": str(crop_path),
            "source_image": str(entry["img_path"]),
            "stem": entry["stem"],
        })

    print(f"  Cropped {len(manifest)} lines ({skipped} skipped).")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ── Flask app ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Line Classifier</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a; color: #eee;
    font-family: 'Segoe UI', sans-serif;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100vh; overflow: hidden;
  }
  #progress-bar-wrap {
    width: 90vw; max-width: 900px;
    background: #333; border-radius: 4px;
    height: 6px; margin-bottom: 16px;
  }
  #progress-bar {
    height: 6px; border-radius: 4px;
    background: #4caf50;
    transition: width 0.2s;
  }
  #counter {
    font-size: 14px; color: #888; margin-bottom: 20px; letter-spacing: 1px;
  }
  #line-wrap {
    background: #fff; border-radius: 6px;
    padding: 20px 30px;
    display: flex; align-items: center; justify-content: center;
    min-height: 80px; max-width: 90vw;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }
  #line-img {
    max-width: 85vw; max-height: 200px;
    image-rendering: pixelated;
  }
  #label-badge {
    margin-top: 18px; font-size: 20px; font-weight: bold;
    height: 32px; letter-spacing: 2px;
  }
  .badge-h { color: #81c784; }
  .badge-l { color: #64b5f6; }
  #hints {
    margin-top: 24px; font-size: 13px; color: #555; text-align: center;
    line-height: 2;
  }
  kbd {
    background: #2e2e2e; border: 1px solid #555;
    border-radius: 4px; padding: 2px 8px;
    font-size: 13px; font-family: monospace;
  }
  #done-screen {
    display: none; text-align: center;
  }
  #done-screen h1 { font-size: 2rem; margin-bottom: 12px; color: #81c784; }
  #done-screen p  { color: #888; }
</style>
</head>
<body>

<div id="main">
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  <div id="counter"></div>
  <div id="line-wrap">
    <img id="line-img" src="" alt="text line">
  </div>
  <div id="label-badge"></div>
  <div id="hints">
    <kbd>H</kbd> Handwritten &nbsp;&nbsp;
    <kbd>L</kbd> Typewritten &nbsp;&nbsp;
    <kbd>Backspace</kbd> Undo last
  </div>
</div>

<div id="done-screen">
  <h1>All done!</h1>
  <p id="done-msg"></p>
</div>

<script>
const total   = {{ total }};
let index     = {{ start_index }};
let labels    = {{ labels_json }};

function imgSrc(idx) { return '/crop/' + idx; }

function updateUI() {
  if (index >= total) { showDone(); return; }
  const pct = (index / total * 100).toFixed(1);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('counter').textContent =
    index + ' / ' + total + '  (' + pct + '%)';
  document.getElementById('line-img').src = imgSrc(index);
  const lbl = labels[index];
  const badge = document.getElementById('label-badge');
  if (lbl === 'H') { badge.textContent = 'HANDWRITTEN'; badge.className = 'badge-h'; }
  else if (lbl === 'L') { badge.textContent = 'TYPEWRITTEN'; badge.className = 'badge-l'; }
  else { badge.textContent = ''; badge.className = ''; }
}

function classify(key) {
  fetch('/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: index, label: key})
  }).then(r => r.json()).then(d => {
    labels[index] = key;
    index++;
    updateUI();
  });
}

function undo() {
  if (index === 0) return;
  index--;
  fetch('/undo', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: index})
  }).then(() => { labels[index] = null; updateUI(); });
}

function showDone() {
  document.getElementById('main').style.display = 'none';
  const ds = document.getElementById('done-screen');
  ds.style.display = 'block';
  const labeled = Object.keys(labels).length;
  document.getElementById('done-msg').textContent =
    labeled + ' lines classified. Results saved to labels.json.';
}

document.addEventListener('keydown', e => {
  if (e.key === 'h' || e.key === 'H') classify('H');
  else if (e.key === 'l' || e.key === 'L') classify('L');
  else if (e.key === 'Backspace') { e.preventDefault(); undo(); }
});

updateUI();
</script>
</body>
</html>
"""


def build_app(manifest: list[dict], output_dir: Path):
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    labels_path = output_dir / "labels.json"

    # Load existing labels if resuming
    if labels_path.exists():
        with open(labels_path) as f:
            saved = json.load(f)
        labels: dict[int, str] = {int(k): v for k, v in saved.items()}
    else:
        labels = {}

    def save_labels():
        with open(labels_path, "w") as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=2)

    def img_to_data_url(path: str) -> str:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"

    @app.route("/")
    def index():
        start_index = max((k for k in labels), default=-1) + 1 if labels else 0
        labels_json = json.dumps({str(k): v for k, v in labels.items()})
        html = (HTML
                .replace("{{ total }}", str(len(manifest)))
                .replace("{{ start_index }}", str(start_index))
                .replace("{{ labels_json }}", labels_json))
        return Response(html, mimetype="text/html")

    @app.route("/crop/<int:idx>")
    def serve_crop(idx: int):
        if idx < 0 or idx >= len(manifest):
            return Response("not found", status=404)
        path = manifest[idx]["crop"]
        with open(path, "rb") as f:
            data = f.read()
        return Response(data, mimetype="image/jpeg")

    @app.route("/label", methods=["POST"])
    def set_label():
        body = request.get_json()
        idx = int(body["index"])
        lbl = body["label"]
        if lbl not in ("H", "L"):
            return jsonify({"error": "invalid label"}), 400
        labels[idx] = lbl
        save_labels()
        return jsonify({"ok": True, "labeled": len(labels)})

    @app.route("/undo", methods=["POST"])
    def undo():
        body = request.get_json()
        idx = int(body["index"])
        labels.pop(idx, None)
        save_labels()
        return jsonify({"ok": True})

    return app


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_arguments()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = prepare_crops(args, output_dir)
    if not manifest:
        print("No lines could be cropped. Check --xml-dir and --img-dir.")
        sys.exit(1)

    app = build_app(manifest, output_dir)

    print(f"\nClassifier ready — open http://localhost:{args.port}")
    print("Keys: H = handwritten, L = typewritten, Backspace = undo\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
