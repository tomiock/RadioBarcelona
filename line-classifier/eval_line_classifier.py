"""
Run the trained LineCNN on all labeled crops and produce a self-contained HTML
report for manual inspection.

The report shows every crop with its ground-truth label, predicted label, and
confidence. Misclassifications are highlighted. A top bar lets you filter by
All / Correct / Wrong and sort by confidence.

Pass --min-width / --min-height / --min-aspect-ratio to skip low-quality crops
(use the thresholds from derive_filter_heuristic.py).

Usage:
    conda activate laypa
    python scripts/eval_line_classifier.py \
        --manifest /data/storage/users/tockier/laypa_classify/manifest.json \
        --labels   /data/storage/users/tockier/laypa_classify/labels.json \
        --checkpoint /data/storage/users/tockier/laypa_classify/model/best.pth \
        --output   /data/storage/users/tockier/laypa_classify/eval_report.html \
        --min-width 99 --min-height 41 --min-aspect-ratio 1.9
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ── model (must match train_line_classifier.py) ───────────────────────────────

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


# ── inference ─────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "H", 1: "L"}
GT_MAP    = {"H": 0, "L": 1}


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = LineCNN(num_classes=2).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def img_to_tensor(path: str) -> torch.Tensor | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@torch.no_grad()
def run_inference(model, entries, device):
    results = []
    for e in entries:
        t = img_to_tensor(e["crop"])
        if t is None:
            continue
        logits = model(t.to(device))
        probs  = torch.softmax(logits, dim=1)[0]
        pred_idx  = int(probs.argmax())
        pred_lbl  = LABEL_MAP[pred_idx]
        confidence = float(probs[pred_idx])
        correct    = pred_lbl == e["gt"]
        results.append({
            "id":         e["id"],
            "crop":       e["crop"],
            "stem":       e["stem"],
            "gt":         e["gt"],
            "pred":       pred_lbl,
            "conf":       confidence,
            "correct":    correct,
        })
    return results


# ── HTML report ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Line Classifier Evaluation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #111; color: #ddd;
    font-family: 'Segoe UI', sans-serif;
    padding: 20px;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 6px; }}
  #summary {{ font-size: 13px; color: #888; margin-bottom: 16px; }}
  #controls {{
    display: flex; gap: 12px; align-items: center;
    margin-bottom: 20px; flex-wrap: wrap;
  }}
  button {{
    background: #2a2a2a; color: #ccc; border: 1px solid #444;
    border-radius: 4px; padding: 6px 14px; cursor: pointer; font-size: 13px;
  }}
  button.active {{ background: #3a6ea5; color: #fff; border-color: #3a6ea5; }}
  select {{
    background: #2a2a2a; color: #ccc; border: 1px solid #444;
    border-radius: 4px; padding: 6px 10px; font-size: 13px;
  }}
  #grid {{
    display: flex; flex-wrap: wrap; gap: 12px;
  }}
  .card {{
    background: #1e1e1e; border-radius: 6px; padding: 10px;
    display: flex; flex-direction: column; align-items: center;
    border: 2px solid transparent; max-width: 420px;
  }}
  .card.wrong {{ border-color: #c0392b; }}
  .card.correct {{ border-color: #27ae60; }}
  .card img {{
    max-width: 400px; max-height: 120px;
    background: #fff; border-radius: 3px;
    image-rendering: pixelated;
  }}
  .card .meta {{
    margin-top: 7px; font-size: 12px; color: #888;
    text-align: center; line-height: 1.6;
  }}
  .pred-H {{ color: #81c784; font-weight: bold; }}
  .pred-L {{ color: #64b5f6; font-weight: bold; }}
  .gt-H {{ color: #a5d6a7; }}
  .gt-L {{ color: #90caf9; }}
  .conf {{ font-size: 11px; color: #666; }}
</style>
</head>
<body>
<h1>Line Classifier — Evaluation Report</h1>
<div id="summary">{summary}</div>
<div id="controls">
  <button class="active" onclick="filter('all',this)">All ({total})</button>
  <button onclick="filter('correct',this)">Correct ({correct})</button>
  <button onclick="filter('wrong',this)">Wrong ({wrong})</button>
  <label style="font-size:13px;color:#888">Sort:
    <select onchange="sortCards(this.value)">
      <option value="id">Original order</option>
      <option value="conf_asc">Confidence ↑</option>
      <option value="conf_desc">Confidence ↓</option>
    </select>
  </label>
</div>
<div id="grid">{cards}</div>

<script>
const cards = Array.from(document.querySelectorAll('.card'));
let current = 'all';

function filter(mode, btn) {{
  current = mode;
  document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  cards.forEach(c => {{
    if (mode === 'all') c.style.display = '';
    else if (mode === 'correct') c.style.display = c.classList.contains('correct') ? '' : 'none';
    else c.style.display = c.classList.contains('wrong') ? '' : 'none';
  }});
}}

function sortCards(mode) {{
  const grid = document.getElementById('grid');
  const visible = [...cards].filter(c => c.style.display !== 'none');
  const sorted = visible.sort((a, b) => {{
    if (mode === 'id')        return +a.dataset.id   - +b.dataset.id;
    if (mode === 'conf_asc')  return +a.dataset.conf - +b.dataset.conf;
    if (mode === 'conf_desc') return +b.dataset.conf - +a.dataset.conf;
  }});
  sorted.forEach(c => grid.appendChild(c));
}}
</script>
</body>
</html>
"""

CARD_TEMPLATE = """<div class="card {cls}" data-id="{id}" data-conf="{conf:.4f}">
  <img src="data:image/jpeg;base64,{b64}" alt="line {id}">
  <div class="meta">
    <span class="pred-{pred}">pred: {pred}</span> &nbsp;|&nbsp;
    <span class="gt-{gt}">gt: {gt}</span>
    <br><span class="conf">{conf_pct:.1f}% confidence &nbsp;·&nbsp; #{id} &nbsp;·&nbsp; {stem}</span>
  </div>
</div>"""


def build_report(results: list[dict]) -> str:
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    wrong   = total - correct
    acc     = correct / total if total else 0

    summary = (f"{total} samples &nbsp;·&nbsp; "
               f"accuracy {acc:.1%} &nbsp;·&nbsp; "
               f"{correct} correct &nbsp;·&nbsp; "
               f"{wrong} wrong")

    cards_html = []
    for r in results:
        b64 = img_to_b64(r["crop"])
        cls = "correct" if r["correct"] else "wrong"
        cards_html.append(CARD_TEMPLATE.format(
            cls=cls, id=r["id"], conf=r["conf"],
            b64=b64, pred=r["pred"], gt=r["gt"],
            conf_pct=r["conf"] * 100, stem=r["stem"],
        ))

    return HTML_TEMPLATE.format(
        summary=summary,
        total=total, correct=correct, wrong=wrong,
        cards="\n".join(cards_html),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",          required=True)
    p.add_argument("--labels",            required=True)
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--output",            required=True)
    p.add_argument("--min-width",         type=float, default=0)
    p.add_argument("--min-height",        type=float, default=0)
    p.add_argument("--min-aspect-ratio",  type=float, default=0)
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = json.load(open(args.manifest))
    labels   = json.load(open(args.labels))

    entries = []
    discarded = 0
    for e in manifest:
        gt = labels.get(str(e["id"]))
        if gt not in ("H", "L"):
            continue
        if args.min_width > 0 or args.min_height > 0 or args.min_aspect_ratio > 0:
            img = cv2.imread(e["crop"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                discarded += 1
                continue
            h, w = img.shape
            ratio = w / h if h else 0
            if w < args.min_width or h < args.min_height or ratio < args.min_aspect_ratio:
                discarded += 1
                continue
        entries.append({**e, "gt": gt})

    if discarded:
        print(f"Filtered out {discarded} low-quality crops "
              f"(min_width={args.min_width}, min_height={args.min_height}, "
              f"min_aspect_ratio={args.min_aspect_ratio})")
    print(f"Running inference on {len(entries)} samples …")
    model   = load_model(args.checkpoint, device)
    results = run_inference(model, entries, device)

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")

    print("Building HTML report …")
    html = build_report(results)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
