"""
Sample text-line crops and serve a browser UI to label each as
Strikethrough (S) or Normal (N).

Low-quality crops are silently skipped using the derived heuristic
(width >= 99, height >= 41, aspect_ratio >= 1.9) so you only see
clean lines worth annotating.

Usage:
    conda activate laypa
    python scripts/label_strikethrough.py \
        --xml-dir  /data/storage/users/tockier/laypa_vis/guiradbcn_a1937m10/page \
        --img-dir  /data/storage/datasets/RadioBarcelona/pdf_images/guiradbcn_a1937m10 \
        --output   /data/storage/users/tockier/laypa_strikethrough \
        --n 500 --port 5053

Keys:  S → strikethrough   N → normal   Backspace → undo
Results saved to <output>/strikethrough_labels.json after every label.
"""

import argparse
import base64
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def get_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--xml-dir",          required=True)
    p.add_argument("--img-dir",          required=True)
    p.add_argument("--output",           required=True)
    p.add_argument("--n",                type=int,   default=500)
    p.add_argument("--port",             type=int,   default=5053)
    p.add_argument("--pad-top",          type=int,   default=28)
    p.add_argument("--pad-bot",          type=int,   default=10)
    p.add_argument("--pad-side",         type=int,   default=8)
    p.add_argument("--seed",             type=int,   default=13)
    p.add_argument("--min-width",        type=float, default=99)
    p.add_argument("--min-height",       type=float, default=41)
    p.add_argument("--min-aspect-ratio", type=float, default=1.9)
    return p.parse_args()


def parse_points(s):
    pts = []
    for tok in s.strip().split():
        x, y = tok.split(",")
        pts.append([int(x), int(y)])
    return np.array(pts)


def collect_lines(xml_dir, img_dir):
    lines = []
    for xml_path in sorted(Path(xml_dir).glob("*.xml")):
        stem = xml_path.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            c = Path(img_dir) / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            continue
        root = ET.parse(xml_path).getroot()
        for tl in root.findall(f".//{{{NS}}}TextLine"):
            bl = tl.find(f".//{{{NS}}}Baseline")
            if bl is None:
                continue
            pts = parse_points(bl.attrib["points"])
            lines.append({"img_path": img_path, "baseline": pts, "stem": stem})
    return lines


def crop_line(img, baseline, pad_top, pad_bot, pad_side):
    h, w = img.shape[:2]
    x_min = max(0, int(baseline[:, 0].min()) - pad_side)
    x_max = min(w, int(baseline[:, 0].max()) + pad_side)
    y_min = max(0, int(baseline[:, 1].min()) - pad_top)
    y_max = min(h, int(baseline[:, 1].max()) + pad_bot)
    if x_max - x_min < 4 or y_max - y_min < 4:
        return None
    return img[y_min:y_max, x_min:x_max]


def passes_quality(w, h, min_w, min_h, min_ratio):
    ratio = w / h if h else 0
    return w >= min_w and h >= min_h and ratio >= min_ratio


def prepare_crops(args, output_dir):
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "strikethrough_manifest.json"

    if manifest_path.exists():
        print("Found existing manifest, reusing crops.")
        return json.load(open(manifest_path))

    print("Collecting lines from PageXML …")
    all_lines = collect_lines(args.xml_dir, args.img_dir)
    print(f"  {len(all_lines)} lines found.")

    random.seed(args.seed)
    random.shuffle(all_lines)

    manifest = []
    img_cache = {}
    skipped_bad_img = 0
    skipped_quality = 0
    crop_id = 0

    for entry in all_lines:
        if len(manifest) >= args.n:
            break

        key = str(entry["img_path"])
        if key not in img_cache:
            img = cv2.imread(key)
            if img is None:
                skipped_bad_img += 1
                continue
            img_cache = {key: img}

        crop = crop_line(img_cache[key], entry["baseline"],
                         args.pad_top, args.pad_bot, args.pad_side)
        if crop is None:
            skipped_quality += 1
            continue

        ch, cw = crop.shape[:2]
        if not passes_quality(cw, ch,
                              args.min_width, args.min_height,
                              args.min_aspect_ratio):
            skipped_quality += 1
            continue

        crop_name = f"st_{crop_id:04d}_{entry['stem']}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({
            "id":           crop_id,
            "crop":         str(crop_path),
            "stem":         entry["stem"],
            "width":        cw,
            "height":       ch,
            "aspect_ratio": round(cw / ch, 3),
        })
        crop_id += 1

    print(f"  Saved {len(manifest)} crops "
          f"({skipped_bad_img} bad images, {skipped_quality} failed quality filter).")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return manifest


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Strikethrough Labeler</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a; color: #eee;
    font-family: 'Segoe UI', sans-serif;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100vh;
  }
  #progress-wrap { width: 80vw; max-width: 800px; background: #333; border-radius: 4px; height: 6px; margin-bottom: 14px; }
  #progress-bar  { height: 6px; border-radius: 4px; background: #e91e63; transition: width 0.2s; }
  #counter { font-size: 13px; color: #777; margin-bottom: 18px; letter-spacing: 1px; }
  #line-wrap {
    background: #fff; border-radius: 6px; padding: 16px 24px;
    display: flex; align-items: center; justify-content: center;
    max-width: 90vw; min-height: 60px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6);
  }
  #line-img { max-width: 85vw; max-height: 180px; image-rendering: pixelated; }
  #meta { margin-top: 12px; font-size: 13px; color: #666; letter-spacing: 1px; }
  #badge { margin-top: 14px; font-size: 18px; font-weight: bold; height: 28px; letter-spacing: 2px; }
  .badge-S { color: #e57373; }
  .badge-N { color: #81c784; }
  #hints { margin-top: 20px; font-size: 13px; color: #555; }
  kbd { background: #2e2e2e; border: 1px solid #555; border-radius: 4px; padding: 2px 8px; font-family: monospace; }
  #legend {
    margin-top: 16px; font-size: 12px; color: #555;
    display: flex; gap: 20px;
  }
  #done { display:none; text-align:center; }
  #done h1 { font-size: 2rem; color: #81c784; margin-bottom: 10px; }
  #done p  { color: #888; }
</style>
</head>
<body>
<div id="main">
  <div id="progress-wrap"><div id="progress-bar"></div></div>
  <div id="counter"></div>
  <div id="line-wrap"><img id="line-img" src="" alt="crop"></div>
  <div id="meta"></div>
  <div id="badge"></div>
  <div id="hints">
    <kbd>S</kbd> Strikethrough &nbsp;&nbsp; <kbd>N</kbd> Normal &nbsp;&nbsp; <kbd>Backspace</kbd> Undo
  </div>
  <div id="legend">
    <span>S = line has a visible strikethrough mark</span>
    <span>N = normal text line, no strikethrough</span>
  </div>
</div>
<div id="done">
  <h1>Done!</h1>
  <p id="done-msg"></p>
</div>

<script>
const total   = {{ total }};
const meta    = {{ meta_json }};
let index     = {{ start_index }};
let labels    = {{ labels_json }};

function updateUI() {
  if (index >= total) { showDone(); return; }
  const pct = (index / total * 100).toFixed(1);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('counter').textContent = index + ' / ' + total + '  (' + pct + '%)';
  document.getElementById('line-img').src = '/crop/' + index;
  const m = meta[index];
  document.getElementById('meta').textContent =
    m.width + ' × ' + m.height + ' px  |  ratio ' + m.aspect_ratio.toFixed(1) + '  |  ' + m.stem;
  const lbl = labels[index];
  const badge = document.getElementById('badge');
  if (lbl === 'S') { badge.textContent = 'STRIKETHROUGH'; badge.className = 'badge-S'; }
  else if (lbl === 'N') { badge.textContent = 'NORMAL';        badge.className = 'badge-N'; }
  else { badge.textContent = ''; badge.className = ''; }
}

function classify(key) {
  fetch('/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index, label: key})
  }).then(r => r.json()).then(() => { labels[index] = key; index++; updateUI(); });
}

function undo() {
  if (index === 0) return;
  index--;
  fetch('/undo', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index})
  }).then(() => { labels[index] = null; updateUI(); });
}

function showDone() {
  document.getElementById('main').style.display = 'none';
  const ds = document.getElementById('done');
  ds.style.display = 'block';
  const n = Object.keys(labels).length;
  const nS = Object.values(labels).filter(v => v === 'S').length;
  const nN = Object.values(labels).filter(v => v === 'N').length;
  document.getElementById('done-msg').textContent =
    n + ' lines labeled  —  ' + nS + ' strikethrough, ' + nN + ' normal.  Results in strikethrough_labels.json.';
}

document.addEventListener('keydown', e => {
  if (e.key === 's' || e.key === 'S') classify('S');
  else if (e.key === 'n' || e.key === 'N') classify('N');
  else if (e.key === 'Backspace') { e.preventDefault(); undo(); }
});

updateUI();
</script>
</body>
</html>
"""


def build_app(manifest, output_dir):
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    labels_path = output_dir / "strikethrough_labels.json"
    labels = {}
    if labels_path.exists():
        saved = json.load(open(labels_path))
        labels = {int(k): v for k, v in saved.items()}

    def save():
        json.dump({str(k): v for k, v in labels.items()}, open(labels_path, "w"), indent=2)

    meta_json = json.dumps([{
        "width":        e["width"],
        "height":       e["height"],
        "aspect_ratio": e["aspect_ratio"],
        "stem":         e["stem"],
    } for e in manifest])

    @app.route("/")
    def index():
        start = max((k for k in labels), default=-1) + 1 if labels else 0
        html = (HTML
                .replace("{{ total }}", str(len(manifest)))
                .replace("{{ start_index }}", str(start))
                .replace("{{ labels_json }}", json.dumps({str(k): v for k, v in labels.items()}))
                .replace("{{ meta_json }}", meta_json))
        return Response(html, mimetype="text/html")

    @app.route("/crop/<int:idx>")
    def serve_crop(idx):
        if idx < 0 or idx >= len(manifest):
            return Response("not found", status=404)
        with open(manifest[idx]["crop"], "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")

    @app.route("/label", methods=["POST"])
    def set_label():
        body = request.get_json()
        lbl = body["label"]
        if lbl not in ("S", "N"):
            return jsonify({"error": "invalid"}), 400
        labels[int(body["index"])] = lbl
        save()
        return jsonify({"ok": True})

    @app.route("/undo", methods=["POST"])
    def undo():
        labels.pop(int(request.get_json()["index"]), None)
        save()
        return jsonify({"ok": True})

    return app


def main():
    args = get_arguments()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = prepare_crops(args, output_dir)
    if not manifest:
        print("No crops produced — check --xml-dir and --img-dir paths.")
        sys.exit(1)

    labels_path = output_dir / "strikethrough_labels.json"
    labeled_so_far = 0
    if labels_path.exists():
        labeled_so_far = len(json.load(open(labels_path)))

    app = build_app(manifest, output_dir)
    print(f"\nStrikethrough labeler ready — open http://localhost:{args.port}")
    print(f"  {len(manifest)} crops total, {labeled_so_far} already labeled")
    print("Keys: S = strikethrough, N = normal, Backspace = undo\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
