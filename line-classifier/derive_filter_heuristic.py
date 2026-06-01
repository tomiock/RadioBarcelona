"""
Analyse the quality labels produced by label_line_quality.py and derive
simple threshold rules (width, height, aspect ratio) that best separate
Keep from Discard samples.

For each feature the script finds the threshold that maximises F1 on the
labeled set, prints a recommended filter function, and saves a self-contained
HTML scatter report so you can visually inspect the decision boundaries.

Usage:
    conda activate laypa
    python scripts/derive_filter_heuristic.py \
        --manifest /data/storage/users/tockier/laypa_quality/quality_manifest.json \
        --labels   /data/storage/users/tockier/laypa_quality/quality_labels.json \
        --output   /data/storage/users/tockier/laypa_quality/heuristic_report.html
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np


# ── feature extraction ────────────────────────────────────────────────────────

FEATURES = ["width", "height", "aspect_ratio"]


def load_data(manifest_path, labels_path):
    manifest = json.load(open(manifest_path))
    labels   = json.load(open(labels_path))
    rows = []
    for e in manifest:
        lbl = labels.get(str(e["id"]))
        if lbl not in ("K", "D"):
            continue
        rows.append({
            "id":           e["id"],
            "crop":         e["crop"],
            "stem":         e["stem"],
            "width":        e["width"],
            "height":       e["height"],
            "aspect_ratio": e["aspect_ratio"],
            "keep":         lbl == "K",
        })
    return rows


# ── threshold search ──────────────────────────────────────────────────────────

def best_threshold(values, keep_flags, feature_name):
    """
    For each candidate threshold t and direction (>=t or <=t) compute F1
    where 'positive' = Keep. Return the best (threshold, direction, f1, accuracy).
    """
    vals   = np.array(values, dtype=float)
    labels = np.array(keep_flags, dtype=bool)
    n      = len(vals)

    candidates = np.unique(vals)
    # add midpoints between consecutive unique values for finer search
    mids = (candidates[:-1] + candidates[1:]) / 2
    candidates = np.sort(np.concatenate([candidates, mids]))

    best = {"f1": -1}
    for t in candidates:
        for direction in (">=", "<="):
            pred = (vals >= t) if direction == ">=" else (vals <= t)
            tp = np.sum(pred & labels)
            fp = np.sum(pred & ~labels)
            fn = np.sum(~pred & labels)
            tn = np.sum(~pred & ~labels)
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec  = tp / (tp + fn) if (tp + fn) else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            acc  = (tp + tn) / n
            if f1 > best["f1"]:
                best = {"threshold": t, "direction": direction,
                        "f1": f1, "accuracy": acc,
                        "tp": int(tp), "fp": int(fp),
                        "fn": int(fn), "tn": int(tn)}
    return best


def combined_accuracy(rows, rules):
    """Apply all rules conjunctively and compute accuracy."""
    correct = 0
    for r in rows:
        keep_pred = all(
            (r[feat] >= t) if direction == ">=" else (r[feat] <= t)
            for feat, t, direction in rules
        )
        if keep_pred == r["keep"]:
            correct += 1
    return correct / len(rows)


# ── HTML report ───────────────────────────────────────────────────────────────

def img_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Filter Heuristic Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #ddd; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 6px; }}
  h2 {{ font-size: 1.1rem; margin: 24px 0 8px; color: #aaa; }}
  pre {{ background: #1e1e1e; border: 1px solid #333; border-radius: 4px; padding: 14px;
        font-size: 13px; line-height: 1.7; overflow-x: auto; color: #b5cea8; }}
  table {{ border-collapse: collapse; font-size: 13px; margin-bottom: 12px; }}
  th, td {{ padding: 6px 14px; border: 1px solid #333; text-align: right; }}
  th {{ background: #1e1e1e; color: #aaa; text-align: center; }}
  td:first-child {{ text-align: left; }}
  #controls {{ display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }}
  button {{ background:#2a2a2a; color:#ccc; border:1px solid #444; border-radius:4px;
            padding:5px 12px; cursor:pointer; font-size:12px; }}
  button.active {{ background:#3a6ea5; color:#fff; border-color:#3a6ea5; }}
  #grid {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .card {{ background:#1e1e1e; border-radius:5px; padding:8px;
           display:flex; flex-direction:column; align-items:center;
           border:2px solid transparent; max-width:400px; }}
  .card.TP {{ border-color:#27ae60; }}
  .card.TN {{ border-color:#c0392b; }}
  .card.FP {{ border-color:#e67e22; }}
  .card.FN {{ border-color:#8e44ad; }}
  .card img {{ max-width:380px; max-height:100px; background:#fff; border-radius:2px; image-rendering:pixelated; }}
  .card .meta {{ font-size:11px; color:#666; margin-top:5px; text-align:center; line-height:1.6; }}
  .legend {{ display:flex; gap:16px; font-size:12px; margin-bottom:12px; flex-wrap:wrap; }}
  .dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:4px; }}
</style>
</head>
<body>
<h1>Filter Heuristic Report</h1>
<p style="font-size:13px;color:#777;margin-bottom:20px">{n_total} samples &nbsp;·&nbsp; {n_keep} keep &nbsp;·&nbsp; {n_discard} discard</p>

<h2>Per-feature best thresholds</h2>
{threshold_table}

<h2>Recommended filter (combined)</h2>
<pre>{filter_code}</pre>
<p style="font-size:13px;color:#777;margin-top:6px">Combined accuracy on labeled set: <strong>{combined_acc:.1%}</strong></p>

<h2>Sample browser</h2>
<div class="legend">
  <span><span class="dot" style="background:#27ae60"></span>TP — correctly kept</span>
  <span><span class="dot" style="background:#c0392b"></span>TN — correctly discarded</span>
  <span><span class="dot" style="background:#e67e22"></span>FP — predicted keep, should discard</span>
  <span><span class="dot" style="background:#8e44ad"></span>FN — predicted discard, should keep</span>
</div>
<div id="controls">
  <button class="active" onclick="show('all',this)">All ({n_total})</button>
  <button onclick="show('TP',this)">TP ({n_tp})</button>
  <button onclick="show('TN',this)">TN ({n_tn})</button>
  <button onclick="show('FP',this)">FP ({n_fp})</button>
  <button onclick="show('FN',this)">FN ({n_fn})</button>
</div>
<div id="grid">{cards}</div>

<script>
const cards = Array.from(document.querySelectorAll('.card'));
function show(cls, btn) {{
  document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  cards.forEach(c => {{
    c.style.display = (cls === 'all' || c.classList.contains(cls)) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

CARD = """<div class="card {outcome}">
  <img src="data:image/jpeg;base64,{b64}" alt="{id}">
  <div class="meta">{w}×{h}px &nbsp; ratio {ratio:.1f} &nbsp; gt:{gt} &nbsp; {stem}</div>
</div>"""

TABLE_ROW = "<tr><td>{feat}</td><td>{direction} {threshold:.1f}</td><td>{f1:.3f}</td><td>{accuracy:.1%}</td><td>{tp}/{fp}/{fn}/{tn}</td></tr>"


def build_report(rows, rules, combined_acc):
    n_total   = len(rows)
    n_keep    = sum(r["keep"] for r in rows)
    n_discard = n_total - n_keep

    # classify each sample with the combined rule
    outcomes = []
    for r in rows:
        pred_keep = all(
            (r[feat] >= t) if direction == ">=" else (r[feat] <= t)
            for feat, t, direction in rules
        )
        if pred_keep and r["keep"]:     outcomes.append("TP")
        elif not pred_keep and not r["keep"]: outcomes.append("TN")
        elif pred_keep and not r["keep"]:     outcomes.append("FP")
        else:                                  outcomes.append("FN")

    counts = {k: outcomes.count(k) for k in ("TP", "TN", "FP", "FN")}

    # threshold table
    feat_rows = []
    for feat, t, direction in rules:
        result = best_threshold([r[feat] for r in rows], [r["keep"] for r in rows], feat)
        feat_rows.append(TABLE_ROW.format(feat=feat, direction=direction, threshold=t,
            f1=result["f1"], accuracy=result["accuracy"],
            tp=result["tp"], fp=result["fp"], fn=result["fn"], tn=result["tn"]))
    table = ("<table><tr><th>Feature</th><th>Rule</th><th>F1</th>"
             "<th>Accuracy</th><th>TP/FP/FN/TN</th></tr>"
             + "".join(feat_rows) + "</table>")

    # filter code
    conditions = " and\n        ".join(
        f"width >= {t:.0f}" if feat == "width" and direction == ">=" else
        f"height >= {t:.0f}" if feat == "height" and direction == ">=" else
        f"height <= {t:.0f}" if feat == "height" and direction == "<=" else
        f"(width / height) >= {t:.1f}" if feat == "aspect_ratio" and direction == ">=" else
        f"(width / height) <= {t:.1f}"
        for feat, t, direction in rules
    )
    filter_code = f"def keep_line(width, height):\n    return (\n        {conditions}\n    )"

    # cards
    cards_html = []
    for r, outcome in zip(rows, outcomes):
        cards_html.append(CARD.format(
            outcome=outcome, id=r["id"],
            b64=img_b64(r["crop"]),
            w=r["width"], h=r["height"],
            ratio=r["aspect_ratio"],
            gt="K" if r["keep"] else "D",
            stem=r["stem"],
        ))

    return REPORT_TEMPLATE.format(
        n_total=n_total, n_keep=n_keep, n_discard=n_discard,
        threshold_table=table,
        filter_code=filter_code,
        combined_acc=combined_acc,
        n_tp=counts["TP"], n_tn=counts["TN"],
        n_fp=counts["FP"], n_fn=counts["FN"],
        cards="".join(cards_html),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--output",   required=True)
    return p.parse_args()


def main():
    args = get_args()
    rows = load_data(args.manifest, args.labels)
    print(f"Loaded {len(rows)} labeled samples  "
          f"({sum(r['keep'] for r in rows)} keep, "
          f"{sum(not r['keep'] for r in rows)} discard)")

    rules = []
    print("\nPer-feature threshold search:")
    for feat in FEATURES:
        values = [r[feat] for r in rows]
        keeps  = [r["keep"] for r in rows]
        result = best_threshold(values, keeps, feat)
        t  = result["threshold"]
        d  = result["direction"]
        f1 = result["f1"]
        print(f"  {feat:14s}  {d} {t:7.1f}   F1={f1:.3f}  acc={result['accuracy']:.1%}"
              f"  (TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']})")
        rules.append((feat, t, d))

    # keep only rules that actually help (F1 > 0.8 per feature)
    # then recompute combined accuracy
    acc = combined_accuracy(rows, rules)
    print(f"\nCombined accuracy (all rules AND-ed): {acc:.1%}")

    print("\nRecommended filter:")
    for feat, t, d in rules:
        print(f"  {feat} {d} {t:.1f}")

    print("\nBuilding HTML report …")
    html = build_report(rows, rules, acc)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
