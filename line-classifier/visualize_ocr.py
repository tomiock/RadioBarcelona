"""
Web viewer for Radio Barcelona OCR output.

Shows page images with interactive text-line overlays. Hovering a line
highlights it and shows the recognised text; the right panel lists the
full transcript in reading order.

Usage:
    conda activate docs
    python line-classifier/visualize_ocr.py --port 5055

Then open http://localhost:5055

Paths are configurable via CLI args or env vars.
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

# ── config ────────────────────────────────────────────────────────────────────

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
IMG_BASES = [
    Path("/data/storage/datasets/RadioBarcelona/pdf_images"),
]
DISPLAY_WIDTH = 1100   # px — image is scaled to this width for display


# ── PageXML parsing ───────────────────────────────────────────────────────────

def parse_points(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in tok.split(",")) for tok in s.strip().split()]


def load_page(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

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

    return {"lines": lines, "xml_path": xml_path}


def find_image(batch: str, stem: str) -> Path | None:
    for base in IMG_BASES:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            p = base / batch / (stem + ext)
            if p.exists():
                return p
    return None


# ── image serving ─────────────────────────────────────────────────────────────

def serve_image(img_path: Path, max_width: int = DISPLAY_WIDTH) -> bytes:
    img = cv2.imread(str(img_path))
    if img is None:
        return b""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def make_thumbnail(img_path: Path, height: int = 200) -> bytes:
    img = cv2.imread(str(img_path))
    if img is None:
        return b""
    h, w = img.shape[:2]
    new_w = int(w * height / h)
    img = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


# ── Flask app ─────────────────────────────────────────────────────────────────

def build_app(ocr_output_dir: Path) -> Flask:
    app = Flask(__name__)

    # ── index: list batches ───────────────────────────────────────────────────
    @app.route("/")
    def index():
        batches = sorted(
            [d.name for d in ocr_output_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        return render_template_string(INDEX_HTML, batches=batches)

    # ── batch: list pages ─────────────────────────────────────────────────────
    @app.route("/<batch>")
    def batch_view(batch):
        page_dir = ocr_output_dir / batch / "page"
        if not page_dir.exists():
            abort(404)
        xmls = sorted(page_dir.glob("*.xml"))
        pages = [x.stem for x in xmls]
        n_total = sum(
            len(ET.parse(x).getroot().findall(f".//{{{NS}}}TextLine"))
            for x in xmls
        )
        n_text = 0
        for x in xmls:
            for tl in ET.parse(x).getroot().findall(f".//{{{NS}}}TextLine"):
                te = tl.find(f"{{{NS}}}TextEquiv/{{{NS}}}Unicode")
                if te is not None and te.text and te.text.strip():
                    n_text += 1
        return render_template_string(
            BATCH_HTML, batch=batch, pages=pages,
            n_total=n_total, n_text=n_text,
        )

    # ── page view ─────────────────────────────────────────────────────────────
    @app.route("/<batch>/<stem>")
    def page_view(batch, stem):
        xml_path = ocr_output_dir / batch / "page" / (stem + ".xml")
        if not xml_path.exists():
            abort(404)
        page = load_page(xml_path)
        img_path = find_image(batch, stem)
        if img_path is None:
            abort(404, f"Image not found for {batch}/{stem}")

        img = cv2.imread(str(img_path))
        orig_h, orig_w = img.shape[:2]
        scale = DISPLAY_WIDTH / orig_w

        # scale coordinates for display
        scaled_lines = []
        for ln in page["lines"]:
            pts_s = [(int(x * scale), int(y * scale)) for x, y in ln["pts"]]
            xs = [p[0] for p in pts_s]
            ys = [p[1] for p in pts_s]
            scaled_lines.append({
                "id":     ln["id"],
                "text":   ln["text"],
                "engine": ln["engine"],
                "pts":    pts_s,
                "x":      min(xs), "y": min(ys),
                "w":      max(xs) - min(xs),
                "h":      max(ys) - min(ys),
            })

        # build prev/next navigation
        page_dir = ocr_output_dir / batch / "page"
        all_stems = sorted(x.stem for x in page_dir.glob("*.xml"))
        idx = all_stems.index(stem) if stem in all_stems else 0
        prev_stem = all_stems[idx - 1] if idx > 0 else None
        next_stem = all_stems[idx + 1] if idx < len(all_stems) - 1 else None

        lines_json = json.dumps(scaled_lines)
        img_h = int(orig_h * scale)

        return render_template_string(
            PAGE_HTML,
            batch=batch, stem=stem,
            img_w=DISPLAY_WIDTH, img_h=img_h,
            lines_json=lines_json,
            prev_stem=prev_stem, next_stem=next_stem,
            page_num=idx + 1, total_pages=len(all_stems),
        )

    # ── image endpoints ───────────────────────────────────────────────────────
    @app.route("/img/<batch>/<stem>")
    def serve_img(batch, stem):
        img_path = find_image(batch, stem)
        if img_path is None:
            abort(404)
        data = serve_image(img_path)
        return Response(data, mimetype="image/jpeg")

    @app.route("/thumb/<batch>/<stem>")
    def serve_thumb(batch, stem):
        img_path = find_image(batch, stem)
        if img_path is None:
            abort(404)
        data = make_thumbnail(img_path)
        return Response(data, mimetype="image/jpeg")

    return app


# ── templates ─────────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Radio Barcelona OCR Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: 'Segoe UI', sans-serif; padding: 40px; }
h1 { font-size: 1.6rem; margin-bottom: 6px; }
p.sub { color: #666; margin-bottom: 32px; font-size: 14px; }
.grid { display: flex; flex-wrap: wrap; gap: 16px; }
.card {
  background: #1e1e1e; border-radius: 8px; padding: 20px 24px;
  text-decoration: none; color: #ddd;
  border: 1px solid #2a2a2a; min-width: 220px;
  transition: border-color .15s;
}
.card:hover { border-color: #4caf50; }
.card h2 { font-size: 1rem; margin-bottom: 4px; }
.card span { font-size: 12px; color: #666; }
</style></head>
<body>
<h1>Radio Barcelona — OCR Viewer</h1>
<p class="sub">{{ batches|length }} batch{{ 'es' if batches|length != 1 else '' }} available</p>
<div class="grid">
  {% for b in batches %}
  <a class="card" href="/{{ b }}">
    <h2>{{ b }}</h2>
    <span>view pages →</span>
  </a>
  {% endfor %}
</div>
</body></html>"""


BATCH_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{{ batch }} — OCR Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: 'Segoe UI', sans-serif; padding: 32px; }
a { color: #4caf50; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.4rem; margin-bottom: 4px; }
.meta { font-size: 13px; color: #666; margin-bottom: 28px; }
.grid { display: flex; flex-wrap: wrap; gap: 14px; }
.card {
  background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
  overflow: hidden; text-decoration: none; color: #ddd;
  transition: border-color .15s; width: 160px;
}
.card:hover { border-color: #4caf50; }
.card img { width: 100%; display: block; background: #2a2a2a; min-height: 90px; }
.card .label { padding: 8px 10px; font-size: 12px; color: #888; }
</style></head>
<body>
<p><a href="/">← batches</a></p>
<h1 style="margin-top:12px">{{ batch }}</h1>
<p class="meta">{{ pages|length }} pages &nbsp;·&nbsp; {{ n_text }} / {{ n_total }} lines with text</p>
<div class="grid">
  {% for p in pages %}
  <a class="card" href="/{{ batch }}/{{ p }}">
    <img src="/thumb/{{ batch }}/{{ p }}" alt="{{ p }}" loading="lazy">
    <div class="label">{{ p }}</div>
  </a>
  {% endfor %}
</div>
</body></html>"""


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{{ stem }} — {{ batch }}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: 'Segoe UI', sans-serif;
  display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* top bar */
#topbar {
  display: flex; align-items: center; gap: 16px;
  padding: 10px 18px; background: #1a1a1a;
  border-bottom: 1px solid #2a2a2a; flex-shrink: 0;
  font-size: 13px;
}
#topbar a { color: #4caf50; text-decoration: none; }
#topbar a:hover { text-decoration: underline; }
#topbar .sep { color: #444; }
.nav-btn {
  padding: 4px 12px; background: #2a2a2a; border: 1px solid #444;
  border-radius: 4px; color: #ccc; text-decoration: none; font-size: 12px;
}
.nav-btn:hover { border-color: #4caf50; color: #fff; text-decoration: none; }
.nav-btn.disabled { opacity: .3; pointer-events: none; }
#page-counter { color: #666; font-size: 12px; }

/* main split */
#main { display: flex; flex: 1; overflow: hidden; }

/* image pane */
#img-pane {
  flex: 1; overflow: auto; position: relative;
  background: #0d0d0d;
}
#img-wrap { position: relative; display: inline-block; }
#page-img { display: block; }
#overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
.line-rect {
  stroke-width: 1.5; cursor: pointer; pointer-events: all;
  transition: fill .1s;
}
/* Tesseract — green */
.line-rect.tesseract { fill: rgba(76,175,80,.12); stroke: rgba(76,175,80,.5); }
.line-rect.tesseract:hover { fill: rgba(76,175,80,.35); stroke: rgba(76,175,80,1); }
.line-rect.tesseract.active { fill: rgba(76,175,80,.45); stroke: #4caf50; stroke-width: 2; }
/* ODAOCR — orange */
.line-rect.odaocr { fill: rgba(255,152,0,.12); stroke: rgba(255,152,0,.5); }
.line-rect.odaocr:hover { fill: rgba(255,152,0,.35); stroke: rgba(255,152,0,1); }
.line-rect.odaocr.active { fill: rgba(255,152,0,.45); stroke: #ff9800; stroke-width: 2; }
/* no text — blue */
.line-rect.empty { stroke: rgba(100,181,246,.4); fill: rgba(100,181,246,.06); }
.line-rect.empty:hover { fill: rgba(100,181,246,.2); }
.line-rect.empty.active { stroke: #64b5f6; stroke-width: 2; }

/* tooltip */
#tooltip {
  position: fixed; background: #222; border: 1px solid #444;
  border-radius: 5px; padding: 6px 10px; font-size: 12px;
  max-width: 420px; pointer-events: none; z-index: 100;
  display: none; box-shadow: 0 4px 12px rgba(0,0,0,.5);
  white-space: pre-wrap; word-break: break-word;
}

/* sidebar */
#sidebar {
  width: 340px; flex-shrink: 0; overflow-y: auto;
  background: #161616; border-left: 1px solid #2a2a2a;
  display: flex; flex-direction: column;
}
#sidebar-header {
  padding: 12px 16px; border-bottom: 1px solid #2a2a2a;
  font-size: 13px; color: #888; flex-shrink: 0;
  display: flex; align-items: center; justify-content: space-between;
}
#filter-toggle { font-size: 11px; color: #4caf50; cursor: pointer; }
#transcript { padding: 8px 0; overflow-y: auto; flex: 1; }
.line-entry {
  padding: 4px 16px; font-size: 12px; line-height: 1.5;
  cursor: pointer; border-left: 3px solid transparent;
  transition: background .1s;
}
.line-entry:hover { background: #1e1e1e; }
.line-entry.tesseract.active { background: #1e2a1e; border-left-color: #4caf50; }
.line-entry.odaocr.active    { background: #2a1e00; border-left-color: #ff9800; }
.line-entry.empty { color: #444; font-style: italic; }
.line-entry.empty:hover { background: #181818; }
#show-empty-label { font-size: 11px; color: #555; cursor: pointer; }
#legend { display: flex; gap: 14px; padding: 8px 16px;
  border-bottom: 1px solid #2a2a2a; font-size: 11px; flex-shrink: 0; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%;
  display: inline-block; margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>

<div id="topbar">
  <a href="/">Home</a><span class="sep">/</span>
  <a href="/{{ batch }}">{{ batch }}</a><span class="sep">/</span>
  <span>{{ stem }}</span>
  <span style="flex:1"></span>
  <span id="page-counter">{{ page_num }} / {{ total_pages }}</span>
  <a class="nav-btn {% if not prev_stem %}disabled{% endif %}"
     href="{% if prev_stem %}/{{ batch }}/{{ prev_stem }}{% endif %}">← prev</a>
  <a class="nav-btn {% if not next_stem %}disabled{% endif %}"
     href="{% if next_stem %}/{{ batch }}/{{ next_stem }}{% endif %}">next →</a>
</div>

<div id="main">
  <div id="img-pane">
    <div id="img-wrap">
      <img id="page-img" src="/img/{{ batch }}/{{ stem }}"
           width="{{ img_w }}" height="{{ img_h }}" alt="{{ stem }}">
      <svg id="overlay" width="{{ img_w }}" height="{{ img_h }}"></svg>
    </div>
  </div>

  <div id="sidebar">
    <div id="sidebar-header">
      <span id="line-count"></span>
      <span id="show-empty-label" onclick="toggleEmpty()">show empty</span>
    </div>
    <div id="legend">
      <span><span class="legend-dot" style="background:#4caf50"></span>Tesseract</span>
      <span><span class="legend-dot" style="background:#ff9800"></span>ODAOCR</span>
      <span><span class="legend-dot" style="background:#64b5f6"></span>empty</span>
    </div>
    <div id="transcript"></div>
  </div>
</div>

<div id="tooltip"></div>

<script>
const LINES = {{ lines_json | safe }};
const overlay = document.getElementById('overlay');
const transcript = document.getElementById('transcript');
const tooltip = document.getElementById('tooltip');
let showEmpty = false;
let activeId = null;

// build rects + sidebar entries
LINES.forEach((ln, i) => {
  const hasText = ln.text && ln.text.length > 0;
  const engineClass = hasText ? (ln.engine || 'tesseract') : 'empty';

  // SVG rect
  const pts = ln.pts.map(p => p[0] + ',' + p[1]).join(' ');
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  el.setAttribute('points', pts);
  el.setAttribute('class', 'line-rect ' + engineClass);
  el.dataset.id = ln.id;
  el.dataset.idx = i;

  el.addEventListener('mouseenter', e => {
    tooltip.style.display = 'block';
    tooltip.textContent = hasText ? ln.text : '(no text)';
  });
  el.addEventListener('mousemove', e => {
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 8)  + 'px';
  });
  el.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  el.addEventListener('click', () => activateLine(ln.id, i));
  overlay.appendChild(el);

  // sidebar entry
  const div = document.createElement('div');
  div.className = 'line-entry' + (hasText ? ' ' + engineClass : ' empty');
  div.dataset.id = ln.id;
  div.dataset.idx = i;
  div.textContent = hasText ? ln.text : '—';
  if (!hasText) div.style.display = showEmpty ? '' : 'none';
  div.addEventListener('click', () => activateLine(ln.id, i));
  transcript.appendChild(div);
});

// stats
const nTess = LINES.filter(l => l.engine === 'tesseract').length;
const nOda  = LINES.filter(l => l.engine === 'odaocr').length;
document.getElementById('line-count').textContent =
  `${nTess + nOda} / ${LINES.length} lines`;

function activateLine(id, idx) {
  // deactivate previous
  if (activeId) {
    overlay.querySelector(`[data-id="${activeId}"]`)?.classList.remove('active');
    transcript.querySelector(`[data-id="${activeId}"]`)?.classList.remove('active');
  }
  activeId = id;
  overlay.querySelector(`[data-id="${id}"]`)?.classList.add('active');
  const entry = transcript.querySelector(`[data-id="${id}"]`);
  if (entry) {
    entry.classList.add('active');
    entry.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  // scroll image pane to region
  const ln = LINES[idx];
  const imgPane = document.getElementById('img-pane');
  const targetY = ln.y - 80;
  imgPane.scrollTo({ top: Math.max(0, targetY), behavior: 'smooth' });
}

function toggleEmpty() {
  showEmpty = !showEmpty;
  document.getElementById('show-empty-label').textContent =
    showEmpty ? 'hide empty' : 'show empty';
  transcript.querySelectorAll('.line-entry.empty').forEach(el => {
    el.style.display = showEmpty ? '' : 'none';
  });
}

// keyboard navigation
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    const link = document.querySelector(
      e.key === 'ArrowRight' ? 'a[href*="next"]' : 'a[href*="prev"]'
    );
    // use the actual nav buttons
    const btn = e.key === 'ArrowRight'
      ? document.querySelectorAll('.nav-btn')[1]
      : document.querySelectorAll('.nav-btn')[0];
    if (btn && !btn.classList.contains('disabled')) {
      window.location = btn.href;
    }
  }
});
</script>
</body></html>"""


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ocr-output", default="/data/storage/users/tockier/ocr_output",
                   help="Root dir containing batch subdirectories")
    p.add_argument("--port", type=int, default=5055)
    p.add_argument("--img-base", action="append", dest="img_bases",
                   help="Additional image base dirs (can repeat)")
    return p.parse_args()


def main():
    args = get_args()
    if args.img_bases:
        IMG_BASES.extend(Path(b) for b in args.img_bases)

    ocr_output_dir = Path(args.ocr_output)
    if not ocr_output_dir.exists():
        print(f"ERROR: OCR output dir not found: {ocr_output_dir}")
        return

    app = build_app(ocr_output_dir)
    print(f"Viewer ready — open http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
