"""
Sample 2000 text-line crops, filter low-quality ones, run the trained CNN to
get initial predictions, then serve a browser UI so you can confirm or correct
each label. Only quality-passing crops are shown.

The model prediction is pre-filled for every sample. If the model is right,
just press H or L to confirm and move on. If it's wrong, press the other key.
Backspace undoes the last label.

Usage:
    conda activate laypa
    python scripts/annotate_with_model.py \
        --xml-dir    /data/storage/users/tockier/laypa_vis/guiradbcn_a1937m10/page \
        --img-dir    /data/storage/datasets/RadioBarcelona/pdf_images/guiradbcn_a1937m10 \
        --checkpoint /data/storage/users/tockier/laypa_classify/model/best.pth \
        --output     /data/storage/users/tockier/laypa_annotate \
        --n 2000 --port 5052

Keys: H = handwritten   L = typewritten   Backspace = undo
Results saved to <output>/labels.json after every label.
"""

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


# ── model ─────────────────────────────────────────────────────────────────────

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
    def __init__(self, num_classes=2):
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


def load_model(checkpoint_path, device):
    model = LineCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, img_gray, device):
    t = torch.from_numpy(img_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    probs = torch.softmax(model(t), dim=1)[0]
    idx = int(probs.argmax())
    return ("H" if idx == 0 else "L"), float(probs[idx])


# ── cropping ──────────────────────────────────────────────────────────────────

def parse_points(s):
    return np.array([[int(x), int(y)] for tok in s.strip().split()
                     for x, y in [tok.split(",")]])


def collect_lines(xml_dir, img_dir):
    lines = []
    for xml_path in sorted(Path(xml_dir).glob("*.xml")):
        stem = xml_path.stem
        img_path = next(
            (Path(img_dir) / (stem + ext)
             for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
             if (Path(img_dir) / (stem + ext)).exists()),
            None,
        )
        if img_path is None:
            continue
        root = ET.parse(xml_path).getroot()
        for tl in root.findall(f".//{{{NS}}}TextLine"):
            bl = tl.find(f".//{{{NS}}}Baseline")
            if bl is None:
                continue
            lines.append({"img_path": img_path, "baseline": parse_points(bl.attrib["points"]), "stem": stem})
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


def passes_filter(h, w, min_width, min_height, min_aspect_ratio):
    return w >= min_width and h >= min_height and (w / h) >= min_aspect_ratio


# ── args ──────────────────────────────────────────────────────────────────────

def get_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--xml-dir",           required=True)
    p.add_argument("--img-dir",           required=True)
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--output",            required=True)
    p.add_argument("--n",                 type=int, default=2000)
    p.add_argument("--port",              type=int, default=5052)
    p.add_argument("--pad-top",           type=int, default=28)
    p.add_argument("--pad-bot",           type=int, default=10)
    p.add_argument("--pad-side",          type=int, default=8)
    p.add_argument("--min-width",         type=float, default=99)
    p.add_argument("--min-height",        type=float, default=41)
    p.add_argument("--min-aspect-ratio",  type=float, default=1.9)
    p.add_argument("--seed",              type=int, default=123)
    return p.parse_args()


# ── prepare crops + inference ─────────────────────────────────────────────────

def prepare(args, output_dir):
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        print("Found existing manifest, reusing crops.")
        return json.load(open(manifest_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} …")
    model = load_model(args.checkpoint, device)

    print("Collecting lines from PageXML …")
    all_lines = collect_lines(args.xml_dir, args.img_dir)
    print(f"  {len(all_lines)} lines found.")

    random.seed(args.seed)
    random.shuffle(all_lines)

    manifest = []
    img_cache = {}
    filtered = skipped = 0
    i = 0

    for entry in all_lines:
        if len(manifest) >= args.n:
            break

        key = str(entry["img_path"])
        if key not in img_cache:
            img = cv2.imread(key)
            if img is None:
                skipped += 1
                continue
            img_cache = {key: img}

        crop_bgr = crop_line(img_cache[key], entry["baseline"],
                             args.pad_top, args.pad_bot, args.pad_side)
        if crop_bgr is None:
            skipped += 1
            continue

        ch, cw = crop_bgr.shape[:2]
        if not passes_filter(ch, cw, args.min_width, args.min_height, args.min_aspect_ratio):
            filtered += 1
            continue

        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        pred, conf = predict(model, crop_gray, device)

        crop_name = f"line_{i:04d}_{entry['stem']}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        manifest.append({
            "id":         i,
            "crop":       str(crop_path),
            "stem":       entry["stem"],
            "width":      cw,
            "height":     ch,
            "model_pred": pred,
            "model_conf": round(conf, 4),
        })
        i += 1

        if len(manifest) % 200 == 0:
            print(f"  {len(manifest)} / {args.n} crops ready …")

    print(f"  Done: {len(manifest)} crops kept  |  {filtered} filtered  |  {skipped} skipped")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return manifest


# ── Flask UI ──────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotate Lines</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a; color: #eee;
    font-family: 'Segoe UI', sans-serif;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100vh;
  }
  #progress-wrap { width: 80vw; max-width: 900px; background: #333; border-radius: 4px; height: 6px; margin-bottom: 14px; }
  #progress-bar  { height: 6px; border-radius: 4px; background: #4caf50; transition: width 0.2s; }
  #counter { font-size: 13px; color: #777; margin-bottom: 18px; letter-spacing: 1px; }
  #line-wrap {
    background: #fff; border-radius: 6px; padding: 18px 28px;
    display: flex; align-items: center; justify-content: center;
    max-width: 92vw; min-height: 70px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6);
  }
  #line-img { max-width: 88vw; max-height: 200px; image-rendering: pixelated; }
  #suggestion {
    margin-top: 14px; font-size: 13px; color: #666; letter-spacing: 1px;
  }
  #suggestion span { font-weight: bold; }
  .sug-H { color: #81c784; }
  .sug-L { color: #64b5f6; }
  #badge { margin-top: 10px; font-size: 20px; font-weight: bold; height: 30px; letter-spacing: 2px; }
  .badge-H { color: #81c784; }
  .badge-L { color: #64b5f6; }
  #hints { margin-top: 22px; font-size: 13px; color: #555; }
  kbd { background: #2e2e2e; border: 1px solid #555; border-radius: 4px; padding: 2px 8px; font-family: monospace; }
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
  <div id="suggestion"></div>
  <div id="badge"></div>
  <div id="hints">
    <kbd>H</kbd> Handwritten &nbsp;&nbsp; <kbd>L</kbd> Typewritten &nbsp;&nbsp; <kbd>Backspace</kbd> Undo
  </div>
</div>
<div id="done">
  <h1>All done!</h1>
  <p id="done-msg"></p>
</div>

<script>
const total       = {{ total }};
const modelPreds  = {{ model_preds_json }};
const modelConfs  = {{ model_confs_json }};
let   index       = {{ start_index }};
let   labels      = {{ labels_json }};

function updateUI() {
  if (index >= total) { showDone(); return; }
  const pct = (index / total * 100).toFixed(1);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('counter').textContent = index + ' / ' + total + '  (' + pct + '%)';
  document.getElementById('line-img').src = '/crop/' + index;

  const pred = modelPreds[index];
  const conf = (modelConfs[index] * 100).toFixed(1);
  const cls  = pred === 'H' ? 'sug-H' : 'sug-L';
  const name = pred === 'H' ? 'Handwritten' : 'Typewritten';
  document.getElementById('suggestion').innerHTML =
    'Model suggests: <span class="' + cls + '">' + name + '</span>  (' + conf + '%)';

  const lbl   = labels[index];
  const badge = document.getElementById('badge');
  if      (lbl === 'H') { badge.textContent = 'HANDWRITTEN'; badge.className = 'badge-H'; }
  else if (lbl === 'L') { badge.textContent = 'TYPEWRITTEN'; badge.className = 'badge-L'; }
  else                  { badge.textContent = '';             badge.className = '';        }
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
  document.getElementById('done-msg').textContent =
    Object.keys(labels).length + ' lines annotated. Results in labels.json.';
}

document.addEventListener('keydown', e => {
  if      (e.key === 'h' || e.key === 'H') classify('H');
  else if (e.key === 'l' || e.key === 'L') classify('L');
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
    labels_path = output_dir / "labels.json"
    labels = {}
    if labels_path.exists():
        labels = {int(k): v for k, v in json.load(open(labels_path)).items()}

    def save():
        json.dump({str(k): v for k, v in labels.items()}, open(labels_path, "w"), indent=2)

    model_preds = json.dumps([e["model_pred"] for e in manifest])
    model_confs = json.dumps([e["model_conf"] for e in manifest])

    @app.route("/")
    def index():
        start = max((k for k in labels), default=-1) + 1 if labels else 0
        html = (HTML
                .replace("{{ total }}", str(len(manifest)))
                .replace("{{ start_index }}", str(start))
                .replace("{{ labels_json }}", json.dumps({str(k): v for k, v in labels.items()}))
                .replace("{{ model_preds_json }}", model_preds)
                .replace("{{ model_confs_json }}", model_confs))
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
        if lbl not in ("H", "L"):
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_arguments()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = prepare(args, output_dir)
    if not manifest:
        print("No crops passed the quality filter. Check thresholds.")
        sys.exit(1)

    app = build_app(manifest, output_dir)
    print(f"\nAnnotation UI ready — open http://localhost:{args.port}")
    print("Keys: H = handwritten, L = typewritten, Backspace = undo\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
