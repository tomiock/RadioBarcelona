"""
Radio Barcelona — Presentation OCR Viewer

White-background viewer designed for oral presentations.
Click / arrow-key through lines; shows a zoomed crop + large OCR text.

Usage:
    conda activate docs
    python line-classifier/present_ocr.py \\
        --ocr-output /data/storage/users/tockier/ocr_output \\
        --img-base   /data/storage/datasets/RadioBarcelona/pdf_images \\
        --port 5056

Then open http://localhost:5056
Arrow keys → navigate lines on the page. Arrow Left/Right → navigate pages.
"""

import argparse
import base64
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template_string, request

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
DISPLAY_WIDTH = 900   # page image width in the left panel

IMG_BASES: list[Path] = []


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_points(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in tok.split(",")) for tok in s.strip().split()]


def load_page(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    lines = []
    for tl in root.findall(f".//{{{NS}}}TextLine"):
        coords_el  = tl.find(f"{{{NS}}}Coords")
        te_el      = tl.find(f"{{{NS}}}TextEquiv")
        unicode_el = te_el.find(f"{{{NS}}}Unicode") if te_el is not None else None
        if coords_el is None:
            continue
        pts    = parse_points(coords_el.attrib.get("points", ""))
        text   = (unicode_el.text or "").strip() if unicode_el is not None else ""
        engine = te_el.attrib.get("engine", "") if te_el is not None else ""
        lines.append({"id": tl.attrib.get("id", ""), "pts": pts,
                      "text": text, "engine": engine})
    return {"lines": lines}


def find_image(stem: str, batch: str = "") -> Path | None:
    """Search image bases for stem, checking batch subdir first."""
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for base in IMG_BASES:
        search_dirs: list[Path] = []
        if batch:
            search_dirs.append(base / batch)
        search_dirs.append(base)
        search_dirs.extend(d for d in base.iterdir() if d.is_dir() and d.name != batch)
        for d in search_dirs:
            for ext in exts:
                p = d / (stem + ext)
                if p.exists():
                    return p
    return None


def encode_jpg(img: np.ndarray, quality: int = 88) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def crop_line(img: np.ndarray, pts: list[tuple[int,int]],
              pad: int = 6) -> np.ndarray | None:
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    h, w = img.shape[:2]
    x1 = max(0, min(xs) - pad); y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad); y2 = min(h, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


# ── Flask app ─────────────────────────────────────────────────────────────────

def build_app(ocr_output_dir: Path) -> Flask:
    app = Flask(__name__)

    def all_batches():
        return sorted((d.name for d in ocr_output_dir.iterdir()
                        if d.is_dir() and (d / "page").exists()), reverse=True)

    def page_stems(batch: str) -> list[str]:
        return sorted(x.stem for x in (ocr_output_dir / batch / "page").glob("*.xml"))

    # ── index ─────────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        batches = all_batches()
        return render_template_string(INDEX_HTML, batches=batches)

    # ── batch ─────────────────────────────────────────────────────────────────
    @app.route("/<batch>")
    def batch_view(batch):
        stems = page_stems(batch)
        if not stems:
            abort(404)
        from flask import redirect
        return redirect(f"/{batch}/{stems[0]}")

    # ── page view ─────────────────────────────────────────────────────────────
    @app.route("/<batch>/<stem>")
    def page_view(batch, stem):
        xml_path = ocr_output_dir / batch / "page" / (stem + ".xml")
        if not xml_path.exists():
            abort(404)

        img_path = find_image(stem, batch)
        if img_path is None:
            abort(404, f"Image not found for {stem}")

        img = cv2.imread(str(img_path))
        orig_h, orig_w = img.shape[:2]
        scale = DISPLAY_WIDTH / orig_w
        disp_h = int(orig_h * scale)

        page = load_page(xml_path)
        scaled_lines = []
        for ln in page["lines"]:
            pts_s = [(int(x * scale), int(y * scale)) for x, y in ln["pts"]]
            xs = [p[0] for p in pts_s]; ys = [p[1] for p in pts_s]
            scaled_lines.append({
                "id":     ln["id"],
                "text":   ln["text"],
                "engine": ln["engine"] or "tesseract",
                "pts":    pts_s,
                "x": min(xs), "y": min(ys),
                "w": max(xs) - min(xs),
                "h": max(ys) - min(ys),
            })

        stems = page_stems(batch)
        idx = stems.index(stem) if stem in stems else 0
        prev_stem = stems[idx - 1] if idx > 0 else None
        next_stem = stems[idx + 1] if idx < len(stems) - 1 else None

        return render_template_string(
            PAGE_HTML,
            batch=batch, stem=stem,
            img_w=DISPLAY_WIDTH, img_h=disp_h,
            lines_json=json.dumps(scaled_lines),
            prev_stem=prev_stem, next_stem=next_stem,
            page_num=idx + 1, total_pages=len(stems),
        )

    # ── crop endpoint: returns jpeg of a line crop ────────────────────────────
    @app.route("/crop/<batch>/<stem>/<line_id>")
    def serve_crop(batch, stem, line_id):
        xml_path = ocr_output_dir / batch / "page" / (stem + ".xml")
        if not xml_path.exists():
            abort(404)

        img_path = find_image(stem, batch)
        if img_path is None:
            abort(404)

        img = cv2.imread(str(img_path))
        page = load_page(xml_path)
        ln = next((l for l in page["lines"] if l["id"] == line_id), None)
        if ln is None:
            abort(404)

        crop = crop_line(img, ln["pts"])
        if crop is None:
            abort(404)

        # scale crop to max height 160 for display
        ch, cw = crop.shape[:2]
        max_h = 160
        if ch > max_h:
            crop = cv2.resize(crop, (int(cw * max_h / ch), max_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    # ── page image endpoint ────────────────────────────────────────────────────
    @app.route("/img/<batch>/<stem>")
    def serve_img(batch, stem):
        img_path = find_image(stem, batch)
        if img_path is None:
            abort(404)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        if w > DISPLAY_WIDTH:
            img = cv2.resize(img, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)),
                             interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    return app


# ── templates ─────────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Radio Barcelona — Presentation Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #fff; color: #222; font-family: 'Segoe UI', Arial, sans-serif;
  padding: 60px 80px; }
h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; color: #111; }
p.sub { color: #888; margin-bottom: 40px; font-size: 15px; }
.grid { display: flex; flex-wrap: wrap; gap: 16px; }
.card {
  background: #f5f5f5; border-radius: 10px; padding: 22px 28px;
  text-decoration: none; color: #222; border: 2px solid #e0e0e0;
  min-width: 200px; transition: border-color .15s, box-shadow .15s;
}
.card:hover { border-color: #1976d2; box-shadow: 0 4px 16px rgba(25,118,210,.12); }
.card h2 { font-size: 1.1rem; margin-bottom: 4px; }
.card span { font-size: 13px; color: #888; }
</style></head>
<body>
<h1>Radio Barcelona OCR</h1>
<p class="sub">{{ batches|length }} result set{{ 's' if batches|length != 1 else '' }}</p>
<div class="grid">
  {% for b in batches %}
  <a class="card" href="/{{ b }}">
    <h2>{{ b }}</h2><span>open →</span>
  </a>
  {% endfor %}
</div>
</body></html>"""


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{{ stem }}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #ffffff;
  color: #111;
  font-family: 'Segoe UI', Arial, sans-serif;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

/* ── top bar ── */
#topbar {
  display: flex; align-items: center; gap: 14px;
  padding: 10px 20px;
  background: #1a237e;
  color: #fff;
  flex-shrink: 0;
  font-size: 14px;
}
#topbar a { color: #90caf9; text-decoration: none; }
#topbar a:hover { text-decoration: underline; }
.sep { color: #5c6bc0; }
.nav-btn {
  padding: 5px 16px; background: #283593; border: 1px solid #3949ab;
  border-radius: 5px; color: #e8eaf6; text-decoration: none; font-size: 13px;
  cursor: pointer;
}
.nav-btn:hover { background: #3949ab; }
.nav-btn.disabled { opacity: .3; pointer-events: none; }
#page-counter { color: #9fa8da; margin-left: auto; font-size: 13px; }

/* ── main split ── */
#main { display: flex; flex: 1; overflow: hidden; }

/* ── image panel ── */
#img-pane {
  flex: 0 0 {{ img_w }}px;
  overflow-y: auto; overflow-x: hidden;
  background: #fafafa;
  border-right: 2px solid #e0e0e0;
  position: relative;
}
#img-wrap { position: relative; display: inline-block; }
#page-img { display: block; }
#overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
.line-poly {
  cursor: pointer; pointer-events: all;
  stroke-width: 2; fill: transparent;
  transition: fill .1s, stroke .1s;
}
.line-poly.tesseract { stroke: rgba(25,118,210,0.5); }
.line-poly.tesseract:hover,
.line-poly.tesseract.active { fill: rgba(25,118,210,0.18); stroke: #1565c0; stroke-width: 2.5; }
.line-poly.odaocr { stroke: rgba(230,81,0,0.5); }
.line-poly.odaocr:hover,
.line-poly.odaocr.active { fill: rgba(230,81,0,0.18); stroke: #e65100; stroke-width: 2.5; }
.line-poly.empty { stroke: rgba(150,150,150,0.3); }
.line-poly.empty:hover { fill: rgba(150,150,150,0.1); stroke: #aaa; }

/* ── right panel ── */
#right {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  background: #fff;
}

/* ── detail card ── */
#detail {
  flex-shrink: 0;
  padding: 24px 28px 20px;
  border-bottom: 2px solid #e8eaf6;
  background: #f3f4ff;
  min-height: 200px;
}
#detail-hint {
  color: #aaa; font-size: 15px; font-style: italic; margin-top: 40px;
  text-align: center;
}
#crop-wrap {
  background: #fff; border: 1px solid #ddd; border-radius: 6px;
  padding: 10px; display: none; margin-bottom: 14px;
  text-align: center;
}
#crop-img { max-width: 100%; max-height: 140px; border-radius: 3px; }
#ocr-text {
  font-size: 1.55rem; font-weight: 600; color: #0d47a1; line-height: 1.4;
  word-break: break-word; display: none;
}
#engine-badge {
  display: none; font-size: 11px; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; padding: 3px 8px; border-radius: 12px;
  margin-bottom: 10px; width: fit-content;
}
#engine-badge.tesseract { background: #e3f2fd; color: #1565c0; }
#engine-badge.odaocr    { background: #fff3e0; color: #e65100; }

/* ── transcript list ── */
#list-header {
  padding: 10px 20px 6px;
  font-size: 12px; color: #888; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase;
  border-bottom: 1px solid #eee;
  display: flex; justify-content: space-between; align-items: center;
  flex-shrink: 0;
}
#transcript {
  flex: 1; overflow-y: auto; padding: 4px 0;
}
.line-entry {
  padding: 6px 20px; font-size: 13.5px; line-height: 1.45;
  cursor: pointer; border-left: 3px solid transparent;
  color: #333;
  transition: background .08s;
}
.line-entry:hover { background: #f0f4ff; }
.line-entry.active { background: #e8eaf6; border-left-color: #1a237e; }
.line-entry.empty { color: #bbb; font-style: italic; font-size: 12px; }
#show-empty { font-size: 11px; color: #1976d2; cursor: pointer; }
#show-empty:hover { text-decoration: underline; }

/* ── line counter ── */
#line-nav {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 20px; border-top: 1px solid #eee; flex-shrink: 0;
  background: #fafafa; font-size: 13px; color: #555;
}
#line-nav button {
  padding: 4px 14px; border: 1px solid #bdbdbd; border-radius: 4px;
  background: #fff; cursor: pointer; font-size: 13px;
}
#line-nav button:hover { background: #e8eaf6; border-color: #3f51b5; }
#line-nav button:disabled { opacity: .3; cursor: default; }
</style>
</head>
<body>

<div id="topbar">
  <a href="/">Home</a>
  <span class="sep">/</span>
  <a href="/{{ batch }}">{{ batch }}</a>
  <span class="sep">/</span>
  <strong>{{ stem }}</strong>
  <a class="nav-btn {% if not prev_stem %}disabled{% endif %}"
     href="{% if prev_stem %}/{{ batch }}/{{ prev_stem }}{% endif %}">← prev page</a>
  <a class="nav-btn {% if not next_stem %}disabled{% endif %}"
     href="{% if next_stem %}/{{ batch }}/{{ next_stem }}{% endif %}">next page →</a>
  <span id="page-counter">Page {{ page_num }} / {{ total_pages }}</span>
</div>

<div id="main">
  <!-- left: page image -->
  <div id="img-pane">
    <div id="img-wrap">
      <img id="page-img" src="/img/{{ batch }}/{{ stem }}"
           width="{{ img_w }}" height="{{ img_h }}" alt="{{ stem }}">
      <svg id="overlay" width="{{ img_w }}" height="{{ img_h }}"></svg>
    </div>
  </div>

  <!-- right: detail + transcript -->
  <div id="right">
    <div id="detail">
      <div id="engine-badge"></div>
      <div id="crop-wrap"><img id="crop-img" src="" alt="line crop"></div>
      <div id="ocr-text"></div>
      <div id="detail-hint">Click a line on the image or use ↑↓ to navigate</div>
    </div>

    <div id="list-header">
      <span id="line-count"></span>
      <span id="show-empty" onclick="toggleEmpty()">show empty</span>
    </div>
    <div id="transcript"></div>

    <div id="line-nav">
      <button id="btn-prev-line" onclick="stepLine(-1)" disabled>↑ prev line</button>
      <span id="line-counter">—</span>
      <button id="btn-next-line" onclick="stepLine(1)">↓ next line</button>
    </div>
  </div>
</div>

<script>
const LINES    = {{ lines_json | safe }};
const STEM     = "{{ stem }}";
const overlay  = document.getElementById('overlay');
const trans    = document.getElementById('transcript');
let showEmpty  = false;
let activeIdx  = -1;
const visIdx   = [];   // indices of non-empty lines in reading order

// ── build SVG polygons + sidebar entries ──────────────────────────────────────
LINES.forEach((ln, i) => {
  const hasText   = ln.text && ln.text.length > 0;
  const engClass  = hasText ? (ln.engine || 'tesseract') : 'empty';

  // polygon
  const pts = ln.pts.map(p => p[0] + ',' + p[1]).join(' ');
  const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  poly.setAttribute('points', pts);
  poly.setAttribute('class', 'line-poly ' + engClass);
  poly.dataset.idx = i;
  poly.addEventListener('click', () => activate(i));
  overlay.appendChild(poly);

  // sidebar entry
  const div = document.createElement('div');
  div.className = 'line-entry' + (hasText ? '' : ' empty');
  div.dataset.idx = i;
  div.textContent = hasText ? ln.text : '—';
  if (!hasText) div.style.display = showEmpty ? '' : 'none';
  div.addEventListener('click', () => activate(i));
  trans.appendChild(div);

  if (hasText) visIdx.push(i);
});

// stats
const nTess = LINES.filter(l => l.engine === 'tesseract' && l.text).length;
const nOda  = LINES.filter(l => l.engine === 'odaocr'    && l.text).length;
document.getElementById('line-count').textContent =
  `${nTess + nOda} / ${LINES.length} lines with text`;

// ── activate a line ────────────────────────────────────────────────────────────
function activate(idx) {
  // deactivate old
  if (activeIdx >= 0) {
    overlay.querySelector(`[data-idx="${activeIdx}"]`)?.classList.remove('active');
    trans.querySelector(`[data-idx="${activeIdx}"]`)?.classList.remove('active');
  }
  activeIdx = idx;
  const ln = LINES[idx];

  // highlight polygon
  overlay.querySelector(`[data-idx="${idx}"]`)?.classList.add('active');

  // sidebar
  const entry = trans.querySelector(`[data-idx="${idx}"]`);
  if (entry) {
    entry.classList.add('active');
    entry.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // scroll image to line
  const imgPane = document.getElementById('img-pane');
  imgPane.scrollTo({ top: Math.max(0, ln.y - 120), behavior: 'smooth' });

  // detail panel
  const badge  = document.getElementById('engine-badge');
  const cropW  = document.getElementById('crop-wrap');
  const cropI  = document.getElementById('crop-img');
  const ocrT   = document.getElementById('ocr-text');
  const hint   = document.getElementById('detail-hint');
  hint.style.display = 'none';

  badge.className = 'engine-badge ' + (ln.engine || 'tesseract');
  badge.textContent = ln.engine ? ln.engine.toUpperCase() : 'TESSERACT';
  badge.style.display = 'inline-block';
  badge.className = (ln.engine || 'tesseract');
  badge.id = 'engine-badge';

  if (ln.text) {
    ocrT.textContent = ln.text;
    ocrT.style.display = 'block';
    cropW.style.display = 'block';
    cropI.src = `/crop/{{ batch }}/${STEM}/${encodeURIComponent(ln.id)}`;
  } else {
    ocrT.textContent = '(no text)';
    ocrT.style.color = '#bbb';
    ocrT.style.display = 'block';
    cropW.style.display = 'block';
    cropI.src = `/crop/{{ batch }}/${STEM}/${encodeURIComponent(ln.id)}`;
  }

  // update line counter
  const posInVis = visIdx.indexOf(idx);
  const lineN = posInVis >= 0 ? posInVis + 1 : '?';
  document.getElementById('line-counter').textContent =
    `Line ${lineN} / ${visIdx.length}`;
  document.getElementById('btn-prev-line').disabled = posInVis <= 0;
  document.getElementById('btn-next-line').disabled =
    posInVis < 0 || posInVis >= visIdx.length - 1;
}

function stepLine(dir) {
  const pos = visIdx.indexOf(activeIdx);
  const next = pos + dir;
  if (next >= 0 && next < visIdx.length) activate(visIdx[next]);
}

function toggleEmpty() {
  showEmpty = !showEmpty;
  document.getElementById('show-empty').textContent =
    showEmpty ? 'hide empty' : 'show empty';
  trans.querySelectorAll('.line-entry.empty').forEach(el => {
    el.style.display = showEmpty ? '' : 'none';
  });
}

// ── keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    stepLine(e.key === 'ArrowDown' ? 1 : -1);
    return;
  }
  if (e.key === 'ArrowRight') {
    const btn = document.querySelectorAll('.nav-btn')[1];
    if (btn && !btn.classList.contains('disabled')) window.location = btn.href;
  }
  if (e.key === 'ArrowLeft') {
    const btn = document.querySelectorAll('.nav-btn')[0];
    if (btn && !btn.classList.contains('disabled')) window.location = btn.href;
  }
});

// auto-select first line with text
if (visIdx.length > 0) activate(visIdx[0]);
</script>
</body></html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ocr-output",
                   default="/data/storage/users/tockier/ocr_output",
                   help="Root dir with batch subdirectories (default: %(default)s)")
    p.add_argument("--img-base", action="append", dest="img_bases", default=[],
                   help="Image base directory (repeat for multiple)")
    p.add_argument("--port", type=int, default=5056)
    return p.parse_args()


def main():
    args = get_args()
    IMG_BASES.extend(Path(b) for b in args.img_bases)
    if not IMG_BASES:
        IMG_BASES.append(Path("/data/storage/datasets/RadioBarcelona/pdf_images"))

    ocr_output_dir = Path(args.ocr_output)
    if not ocr_output_dir.exists():
        raise SystemExit(f"OCR output dir not found: {ocr_output_dir}")

    app = build_app(ocr_output_dir)
    print(f"Presentation viewer ready — http://localhost:{args.port}")
    print("  ↑↓  navigate lines      ←→  navigate pages")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
