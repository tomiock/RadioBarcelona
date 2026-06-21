"""
Manual annotation / correction tool for Radio Barcelona OCR output.

Serves all PageXML files in a directory as an interactive annotation
interface. Features:
  - Multi-page navigation (dropdown + prev/next, keyboard ←/→)
  - Per-line editable text with Tab/Shift-Tab to advance
  - Line-crop zoom strip for the active line
  - Nearly-transparent bbox overlays; active line highlighted
  - Persistent change history written to history.jsonl in the output dir
  - Ctrl+S / Save button writes corrections back to PageXML

Usage:
    conda activate docs
    python line-classifier/annotate_ocr.py \\
        --xml-dir /data/storage/users/tockier/ocr_output/page_examples_vllm/page \\
        --img-dir /data/storage/datasets/RadioBarcelona/pdf_images/page_examples \\
        --port 5056
"""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, redirect, render_template_string, request, url_for

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", NS)
DISPLAY_WIDTH = 1080


# ── PageXML helpers ───────────────────────────────────────────────────────────

def parse_points(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in tok.split(",")) for tok in s.strip().split()]


def load_page(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    lines = []
    for tl in root.findall(f".//{{{NS}}}TextLine"):
        coords_el = tl.find(f"{{{NS}}}Coords")
        if coords_el is None:
            continue
        te_el      = tl.find(f"{{{NS}}}TextEquiv")
        unicode_el = te_el.find(f"{{{NS}}}Unicode") if te_el is not None else None
        lines.append({
            "id":     tl.attrib.get("id", ""),
            "pts":    parse_points(coords_el.attrib.get("points", "")),
            "text":   (unicode_el.text or "").strip() if unicode_el is not None else "",
            "engine": te_el.attrib.get("engine", "") if te_el is not None else "",
        })
    return lines


def save_corrections(xml_path: Path, edits: dict[str, str],
                     history_path: Path, stem: str, lines: list[dict]) -> int:
    id_to_orig = {l["id"]: l["text"] for l in lines}

    tree = ET.parse(xml_path)
    root = tree.getroot()
    saved = 0
    history_entries = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for tl in root.findall(f".//{{{NS}}}TextLine"):
        lid = tl.attrib.get("id", "")
        if lid not in edits:
            continue
        new_text = edits[lid]
        existing = tl.find(f"{{{NS}}}TextEquiv")
        if existing is not None:
            tl.remove(existing)
        te = ET.SubElement(tl, f"{{{NS}}}TextEquiv")
        te.set("engine", "corrected")
        ET.SubElement(te, f"{{{NS}}}Unicode").text = new_text
        history_entries.append({
            "ts":        now,
            "stem":      stem,
            "line_id":   lid,
            "original":  id_to_orig.get(lid, ""),
            "corrected": new_text,
        })
        saved += 1

    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True)

    with open(history_path, "a", encoding="utf-8") as f:
        for entry in history_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return saved


# ── image helpers ─────────────────────────────────────────────────────────────

def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def serve_image(img_path: Path) -> bytes:
    img = cv2.imread(str(img_path))
    if img is None:
        return b""
    h, w = img.shape[:2]
    if w > DISPLAY_WIDTH:
        img = cv2.resize(img, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 87])
    return buf.tobytes()


# ── app ───────────────────────────────────────────────────────────────────────

def build_app(xml_dir: Path, img_dir: Path, history_path: Path) -> Flask:
    app = Flask(__name__)
    stems = sorted(p.stem for p in xml_dir.glob("*.xml"))

    if not stems:
        raise SystemExit(f"No XML files found in {xml_dir}")

    def page_data(stem: str) -> dict:
        xml_path  = xml_dir / (stem + ".xml")
        img_path  = find_image(img_dir, stem)
        lines     = load_page(xml_path)
        img       = cv2.imread(str(img_path)) if img_path else None
        orig_w    = img.shape[1] if img is not None else DISPLAY_WIDTH
        orig_h    = img.shape[0] if img is not None else 1000
        scale     = DISPLAY_WIDTH / orig_w
        scaled    = []
        for ln in lines:
            pts_s = [(int(x * scale), int(y * scale)) for x, y in ln["pts"]]
            xs, ys = [p[0] for p in pts_s], [p[1] for p in pts_s]
            scaled.append({**ln, "pts": pts_s,
                           "x": min(xs), "y": min(ys),
                           "w": max(xs)-min(xs), "h": max(ys)-min(ys)})
        idx = stems.index(stem)
        return {
            "stem":       stem,
            "lines_json": json.dumps(scaled),
            "img_w":      DISPLAY_WIDTH,
            "img_h":      int(orig_h * scale),
            "stems_json": json.dumps(stems),
            "page_idx":   idx,
            "prev":       stems[idx-1] if idx > 0 else None,
            "next":       stems[idx+1] if idx < len(stems)-1 else None,
            "total":      len(stems),
        }

    @app.route("/")
    def index():
        return redirect(f"/page/{stems[0]}")

    @app.route("/page/<stem>")
    def page_view(stem):
        if stem not in stems:
            abort(404)
        return render_template_string(PAGE_HTML, **page_data(stem))

    @app.route("/img/<stem>")
    def serve_img(stem):
        img_path = find_image(img_dir, stem)
        if not img_path:
            abort(404)
        return Response(serve_image(img_path), mimetype="image/jpeg")

    @app.route("/save/<stem>", methods=["POST"])
    def save(stem):
        if stem not in stems:
            abort(404)
        payload  = request.get_json(force=True)
        edits    = payload.get("edits", {})
        xml_path = xml_dir / (stem + ".xml")
        lines    = load_page(xml_path)
        saved    = save_corrections(xml_path, edits, history_path, stem, lines)
        return {"ok": True, "saved": saved}

    @app.route("/history")
    def history():
        entries = []
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return jsonify(entries[-200:])   # last 200 edits

    return app


# ── HTML template ─────────────────────────────────────────────────────────────

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotate · {{ stem }}</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0f0f0f; color: #d4d4d4;
  font-family: 'Segoe UI', system-ui, sans-serif;
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}
kbd {
  font-size: 10px; background: #2a2a2a; border: 1px solid #444;
  border-radius: 3px; padding: 1px 5px; color: #999;
}

/* ── topbar ── */
#topbar {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 14px; background: #161616;
  border-bottom: 1px solid #252525; flex-shrink: 0; font-size: 13px;
  min-height: 44px;
}
#page-select {
  background: #222; border: 1px solid #383838; color: #ccc;
  border-radius: 5px; padding: 4px 8px; font-size: 12px; cursor: pointer;
}
#page-select:focus { outline: none; border-color: #555; }
.nav-btn {
  padding: 4px 10px; background: #222; border: 1px solid #383838;
  border-radius: 4px; color: #aaa; font-size: 12px; cursor: pointer;
  transition: border-color .12s, color .12s;
}
.nav-btn:hover:not(:disabled) { border-color: #64b5f6; color: #fff; }
.nav-btn:disabled { opacity: .3; cursor: default; }
#page-counter { font-size: 11px; color: #555; }
#dirty-badge {
  font-size: 11px; padding: 2px 8px; border-radius: 10px;
  background: #ff9800; color: #111; font-weight: 700; display: none;
}
#save-btn {
  margin-left: auto; padding: 5px 18px; background: #2e7d32;
  border: 1px solid #388e3c; border-radius: 5px; color: #fff;
  font-size: 13px; font-weight: 600; cursor: pointer;
  transition: background .15s;
}
#save-btn:hover:not(:disabled) { background: #388e3c; }
#save-btn:disabled { background: #1a1a1a; border-color: #333; color: #444; cursor: default; }
#save-status { font-size: 11px; color: #666; min-width: 80px; }
#hist-btn {
  padding: 4px 10px; background: #222; border: 1px solid #383838;
  border-radius: 4px; color: #aaa; font-size: 12px; cursor: pointer;
}
#hist-btn.active { border-color: #ab47bc; color: #ab47bc; }

/* ── main ── */
#main { display: flex; flex: 1; overflow: hidden; }

/* ── image pane ── */
#img-pane { flex: 1; overflow: auto; background: #080808; }
#img-wrap  { position: relative; display: inline-block; }
#page-img  { display: block; }
#overlay   { position: absolute; top: 0; left: 0; pointer-events: none; }

.lr {
  stroke-width: 1; cursor: pointer; pointer-events: all;
  transition: fill .15s, stroke .15s, stroke-width .15s;
}
/* default: nearly invisible */
.lr.vllm      { fill: rgba(0,188,212,.04);   stroke: rgba(0,188,212,.25); }
.lr.tesseract { fill: rgba(76,175,80,.04);   stroke: rgba(76,175,80,.25); }
.lr.odaocr    { fill: rgba(255,152,0,.04);   stroke: rgba(255,152,0,.25); }
.lr.corrected { fill: rgba(171,71,188,.04);  stroke: rgba(171,71,188,.25); }
.lr.empty     { fill: rgba(255,255,255,.02); stroke: rgba(255,255,255,.12); }

/* hover: subtle glow */
.lr:hover { fill: rgba(255,255,255,.07) !important; stroke: rgba(255,255,255,.5) !important; stroke-width: 1.5 !important; }

/* active (selected + editing) */
.lr.active.vllm      { fill: rgba(0,188,212,.18);  stroke: #00bcd4; stroke-width: 2; }
.lr.active.tesseract { fill: rgba(76,175,80,.18);  stroke: #4caf50; stroke-width: 2; }
.lr.active.odaocr    { fill: rgba(255,152,0,.18);  stroke: #ff9800; stroke-width: 2; }
.lr.active.corrected { fill: rgba(171,71,188,.18); stroke: #ab47bc; stroke-width: 2; }
.lr.active.empty     { fill: rgba(255,255,255,.08); stroke: #90a4ae; stroke-width: 2; }

/* tooltip */
#tooltip {
  position: fixed; background: #1e1e1e; border: 1px solid #3a3a3a;
  border-radius: 6px; padding: 7px 11px; font-size: 12px;
  max-width: 480px; pointer-events: none; z-index: 300;
  display: none; box-shadow: 0 6px 20px rgba(0,0,0,.6);
  white-space: pre-wrap; word-break: break-word; line-height: 1.5; color: #ccc;
}

/* ── sidebar ── */
#sidebar {
  width: 400px; flex-shrink: 0;
  background: #131313; border-left: 1px solid #222;
  display: flex; flex-direction: column; overflow: hidden;
}
#sidebar-header {
  padding: 8px 12px 6px; border-bottom: 1px solid #1e1e1e;
  font-size: 11px; color: #555; flex-shrink: 0;
  display: flex; justify-content: space-between; align-items: center;
}
#legend {
  display: flex; gap: 10px; flex-wrap: wrap;
  padding: 5px 12px; border-bottom: 1px solid #1e1e1e;
  font-size: 10px; color: #666; flex-shrink: 0;
}
.dot { width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; margin-right: 3px; vertical-align: middle; }

/* zoom strip */
#zoom-strip {
  height: 64px; flex-shrink: 0; background: #0a0a0a;
  border-bottom: 1px solid #1e1e1e;
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}
#zoom-canvas { max-width: 100%; max-height: 62px; display: none; }
#zoom-hint   { font-size: 11px; color: #333; }

/* line list */
#line-list { flex: 1; overflow-y: auto; padding: 2px 0; }

.le {
  display: flex; align-items: flex-start; gap: 6px;
  padding: 4px 10px; cursor: pointer; border-left: 3px solid transparent;
  transition: background .1s;
}
.le:hover { background: #1a1a1a; }
.le.active { background: #1c1c1c; }
.le.active.vllm      { border-left-color: #00bcd4; }
.le.active.tesseract { border-left-color: #4caf50; }
.le.active.odaocr    { border-left-color: #ff9800; }
.le.active.corrected { border-left-color: #ab47bc; }
.le.active.empty     { border-left-color: #607d8b; }
.le-num {
  font-size: 10px; color: #3a3a3a; width: 24px; text-align: right;
  flex-shrink: 0; padding-top: 4px;
}
.le-input {
  flex: 1; background: transparent; border: none; outline: none;
  color: #b0b0b0; font-size: 12px; font-family: inherit; line-height: 1.5;
  resize: none; overflow: hidden; min-height: 22px;
  padding: 2px 5px; border-radius: 3px; transition: background .1s, color .1s;
}
.le-input:focus { background: #1e1e1e; color: #e0e0e0; }
.le-input.dirty { color: #ce93d8; }
.le-input.empty-val { color: #2e2e2e; font-style: italic; }
.le-badge {
  font-size: 9px; padding: 1px 5px; border-radius: 8px;
  flex-shrink: 0; margin-top: 4px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .04em;
}
.le-badge.vllm      { background: rgba(0,188,212,.15);  color: #00bcd4; }
.le-badge.tesseract { background: rgba(76,175,80,.15);  color: #4caf50; }
.le-badge.odaocr    { background: rgba(255,152,0,.15);  color: #ff9800; }
.le-badge.corrected { background: rgba(171,71,188,.15); color: #ce93d8; }
.le-badge.empty     { background: rgba(96,125,139,.15); color: #607d8b; }

/* ── history panel ── */
#hist-panel {
  display: none; flex-shrink: 0;
  border-top: 1px solid #222; background: #0f0f0f;
  max-height: 240px; overflow-y: auto;
}
#hist-panel.open { display: block; }
#hist-header {
  padding: 6px 12px; font-size: 11px; color: #555; position: sticky; top: 0;
  background: #0f0f0f; border-bottom: 1px solid #1e1e1e;
  display: flex; justify-content: space-between;
}
.hist-entry {
  padding: 5px 12px; border-bottom: 1px solid #181818; font-size: 11px;
}
.hist-entry .he-meta { color: #3a3a3a; margin-bottom: 2px; }
.hist-entry .he-orig { color: #555; text-decoration: line-through;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-entry .he-new  { color: #ce93d8;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>

<div id="topbar">
  <select id="page-select" onchange="navigateTo(this.value)">
    {% raw %}<!-- filled by JS -->{% endraw %}
  </select>
  <button class="nav-btn" id="btn-prev" onclick="navStep(-1)">←</button>
  <button class="nav-btn" id="btn-next" onclick="navStep(1)">→</button>
  <span id="page-counter"></span>
  <span id="dirty-badge">unsaved</span>
  <span id="save-status"></span>
  <button id="hist-btn" onclick="toggleHist()">history</button>
  <button id="save-btn" disabled onclick="saveAll()">Save</button>
</div>

<div id="main">
  <div id="img-pane">
    <div id="img-wrap">
      <img id="page-img" src="/img/{{ stem }}"
           width="{{ img_w }}" height="{{ img_h }}" alt="{{ stem }}">
      <svg id="overlay" width="{{ img_w }}" height="{{ img_h }}"></svg>
    </div>
  </div>

  <div id="sidebar">
    <div id="sidebar-header">
      <span id="line-count"></span>
      <span><kbd>Tab</kbd> next &nbsp; <kbd>S-Tab</kbd> prev &nbsp; <kbd>Ctrl+S</kbd> save</span>
    </div>
    <div id="legend">
      <span><span class="dot" style="background:#00bcd4"></span>vllm</span>
      <span><span class="dot" style="background:#4caf50"></span>tesseract</span>
      <span><span class="dot" style="background:#ff9800"></span>odaocr</span>
      <span><span class="dot" style="background:#ab47bc"></span>corrected</span>
      <span><span class="dot" style="background:#607d8b"></span>empty</span>
    </div>
    <div id="zoom-strip">
      <canvas id="zoom-canvas"></canvas>
      <span id="zoom-hint">select a line</span>
    </div>
    <div id="line-list"></div>
    <div id="hist-panel">
      <div id="hist-header">
        <span>recent changes</span>
        <span id="hist-count"></span>
      </div>
      <div id="hist-list"></div>
    </div>
  </div>
</div>

<div id="tooltip"></div>

<script>
const LINES     = {{ lines_json | safe }};
const STEMS     = {{ stems_json | safe }};
const STEM      = {{ stem | tojson }};
const PAGE_IDX  = {{ page_idx }};
const TOTAL     = {{ total }};

const overlay    = document.getElementById('overlay');
const lineList   = document.getElementById('line-list');
const imgPane    = document.getElementById('img-pane');
const saveBtn    = document.getElementById('save-btn');
const dirtyBadge = document.getElementById('dirty-badge');
const saveStatus = document.getElementById('save-status');
const zoomCanvas = document.getElementById('zoom-canvas');
const zoomHint   = document.getElementById('zoom-hint');
const tooltip    = document.getElementById('tooltip');
const pageSelect = document.getElementById('page-select');
const btnPrev    = document.getElementById('btn-prev');
const btnNext    = document.getElementById('btn-next');

// populate page selector
STEMS.forEach((s, i) => {
  const opt = document.createElement('option');
  opt.value = s; opt.textContent = s;
  if (s === STEM) opt.selected = true;
  pageSelect.appendChild(opt);
});
document.getElementById('page-counter').textContent = `${PAGE_IDX+1} / ${TOTAL}`;
btnPrev.disabled = PAGE_IDX === 0;
btnNext.disabled = PAGE_IDX === TOTAL - 1;

function navigateTo(stem) {
  if (dirtySet.size > 0 &&
      !confirm('You have unsaved changes. Leave anyway?')) {
    pageSelect.value = STEM; return;
  }
  window.location = `/page/${stem}`;
}
function navStep(d) {
  const idx = PAGE_IDX + d;
  if (idx >= 0 && idx < TOTAL) navigateTo(STEMS[idx]);
}

// original image for zoom
const origImg = new Image();
origImg.src = `/img/${STEM}`;

const dirtySet = new Set();
let activeIdx = null;

// ── build overlay + sidebar ───────────────────────────────────────────────────
LINES.forEach((ln, i) => {
  const hasText  = ln.text && ln.text.length > 0;
  const engClass = hasText ? (ln.engine || 'vllm') : 'empty';

  // polygon
  const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  poly.setAttribute('points', ln.pts.map(p => p[0]+','+p[1]).join(' '));
  poly.setAttribute('class', 'lr ' + engClass);
  poly.dataset.idx = i;
  poly.addEventListener('click',      () => activate(i));
  poly.addEventListener('mouseenter', e => showTooltip(i, e));
  poly.addEventListener('mousemove',  e => moveTooltip(e));
  poly.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  overlay.appendChild(poly);

  // sidebar row
  const row = document.createElement('div');
  row.className = `le ${engClass}`;
  row.dataset.idx = i;
  row.innerHTML = `
    <span class="le-num">${i+1}</span>
    <textarea class="le-input${hasText ? '' : ' empty-val'}"
              rows="1" spellcheck="false"
              placeholder="(empty)">${escHtml(ln.text)}</textarea>
    <span class="le-badge ${engClass}">${engClass}</span>`;

  const ta = row.querySelector('textarea');
  autoResize(ta);

  ta.addEventListener('input',  () => { autoResize(ta); markDirty(i, ta); });
  ta.addEventListener('focus',  () => activate(i, false));
  ta.addEventListener('keydown', e => {
    if (e.key === 'Tab') {
      e.preventDefault();
      activate(e.shiftKey ? i-1 : i+1);
    }
  });
  lineList.appendChild(row);
});

document.getElementById('line-count').textContent =
  `${LINES.length} lines · ${LINES.filter(l=>l.text).length} with text`;

// ── tooltip ───────────────────────────────────────────────────────────────────
function showTooltip(idx, e) {
  const ta = getRow(idx)?.querySelector('textarea');
  tooltip.textContent = (ta ? ta.value : LINES[idx].text) || '(empty)';
  tooltip.style.display = 'block';
  moveTooltip(e);
}
function moveTooltip(e) {
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 8) + 'px';
}

// ── activation ────────────────────────────────────────────────────────────────
function activate(idx, focusTa = true) {
  if (idx < 0 || idx >= LINES.length) return;
  if (activeIdx !== null) {
    getPoly(activeIdx)?.classList.remove('active');
    getRow(activeIdx)?.classList.remove('active');
  }
  activeIdx = idx;
  getPoly(idx)?.classList.add('active');
  const row = getRow(idx);
  row?.classList.add('active');
  row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

  if (focusTa) {
    const ta = row?.querySelector('textarea');
    if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
  }

  const ln = LINES[idx];
  imgPane.scrollTo({ top: Math.max(0, ln.y - 120), behavior: 'smooth' });
  drawZoom(idx);
}

// ── zoom strip ────────────────────────────────────────────────────────────────
origImg.onload = () => { if (activeIdx !== null) drawZoom(activeIdx); };

function drawZoom(idx) {
  const ln  = LINES[idx];
  const sc  = (origImg.naturalWidth || {{ img_w }}) / {{ img_w }};
  const pad = 12;
  const sx  = Math.max(0, Math.round(ln.x * sc) - pad);
  const sy  = Math.max(0, Math.round(ln.y * sc) - pad * 3);
  const sw  = Math.min((origImg.naturalWidth||{{ img_w }}) - sx, Math.round(ln.w*sc) + pad*2);
  const sh  = Math.min((origImg.naturalHeight||{{ img_h }}) - sy, Math.round(ln.h*sc) + pad*6);
  const STRIP_H = 60;
  const dw  = Math.max(1, Math.round(sw * (STRIP_H / sh)));
  zoomCanvas.width = dw; zoomCanvas.height = STRIP_H;
  zoomCanvas.getContext('2d').drawImage(origImg, sx, sy, sw, sh, 0, 0, dw, STRIP_H);
  zoomCanvas.style.display = 'block';
  zoomHint.style.display   = 'none';
}

// ── dirty tracking ────────────────────────────────────────────────────────────
function markDirty(idx, ta) {
  const isDirty = ta.value !== LINES[idx].text;
  isDirty ? dirtySet.add(idx) : dirtySet.delete(idx);
  ta.classList.toggle('dirty', isDirty);
  ta.classList.toggle('empty-val', ta.value.length === 0);
  const badge = getRow(idx)?.querySelector('.le-badge');
  if (badge && isDirty) { badge.className = 'le-badge corrected'; badge.textContent = 'corrected'; }
  else if (badge) { badge.className = `le-badge ${LINES[idx].engine||'empty'}`; badge.textContent = LINES[idx].engine||'empty'; }
  const hasDirty = dirtySet.size > 0;
  saveBtn.disabled = !hasDirty;
  dirtyBadge.style.display = hasDirty ? 'inline' : 'none';
}

// ── save ──────────────────────────────────────────────────────────────────────
function saveAll() {
  const edits = {};
  dirtySet.forEach(idx => {
    const ta = getRow(idx)?.querySelector('textarea');
    if (ta) edits[LINES[idx].id] = ta.value;
  });
  if (!Object.keys(edits).length) return;
  saveBtn.disabled = true;
  saveStatus.textContent = 'saving…';
  fetch(`/save/${STEM}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({edits}),
  }).then(r => r.json()).then(data => {
    if (data.ok) {
      dirtySet.forEach(idx => {
        const ta = getRow(idx)?.querySelector('textarea');
        if (ta) {
          LINES[idx].text   = ta.value;
          LINES[idx].engine = 'corrected';
          ta.classList.remove('dirty');
          const poly = getPoly(idx);
          if (poly) {
            const isActive = poly.classList.contains('active');
            poly.className = `lr corrected${isActive ? ' active' : ''}`;
          }
          const row = getRow(idx);
          if (row) {
            const isActive = row.classList.contains('active');
            row.className = `le corrected${isActive ? ' active' : ''}`;
          }
          const badge = getRow(idx)?.querySelector('.le-badge');
          if (badge) { badge.className='le-badge corrected'; badge.textContent='corrected'; }
        }
      });
      dirtySet.clear();
      saveBtn.disabled = true;
      dirtyBadge.style.display = 'none';
      saveStatus.textContent = `✓ ${data.saved} saved`;
      setTimeout(() => { saveStatus.textContent = ''; }, 3000);
      loadHistory();
    }
  }).catch(err => {
    saveStatus.textContent = '✗ error';
    saveBtn.disabled = false;
    console.error(err);
  });
}

// ── history ───────────────────────────────────────────────────────────────────
function toggleHist() {
  const panel = document.getElementById('hist-panel');
  const btn   = document.getElementById('hist-btn');
  const open  = panel.classList.toggle('open');
  btn.classList.toggle('active', open);
  if (open) loadHistory();
}

function loadHistory() {
  fetch('/history').then(r => r.json()).then(entries => {
    const list = document.getElementById('hist-list');
    document.getElementById('hist-count').textContent = `${entries.length} total`;
    list.innerHTML = '';
    [...entries].reverse().forEach(e => {
      const div = document.createElement('div');
      div.className = 'hist-entry';
      const ts = new Date(e.ts).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
      div.innerHTML = `
        <div class="he-meta">${ts} · ${e.stem}</div>
        <div class="he-orig">${escHtml(e.original || '(empty)')}</div>
        <div class="he-new">${escHtml(e.corrected || '(empty)')}</div>`;
      list.appendChild(div);
    });
  });
}

// ── keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    if (!saveBtn.disabled) saveAll();
  }
  // arrow keys for page navigation when no textarea is focused
  if (document.activeElement.tagName !== 'TEXTAREA') {
    if (e.key === 'ArrowRight') navStep(1);
    if (e.key === 'ArrowLeft')  navStep(-1);
  }
});

window.addEventListener('beforeunload', e => {
  if (dirtySet.size > 0) e.returnValue = 'Unsaved changes';
});

// ── utils ─────────────────────────────────────────────────────────────────────
function getRow(idx)  { return lineList.children[idx]; }
function getPoly(idx) { return overlay.children[idx]; }
function autoResize(ta) { ta.style.height='auto'; ta.style.height=ta.scrollHeight+'px'; }
function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="OCR annotation tool — multi-page")
    p.add_argument("--xml-dir", required=True,
                   help="Directory of PageXML files to annotate")
    p.add_argument("--img-dir", required=True,
                   help="Directory containing page images")
    p.add_argument("--port",    type=int, default=5056)
    return p.parse_args()


def main():
    args     = get_args()
    xml_dir  = Path(args.xml_dir).resolve()
    img_dir  = Path(args.img_dir).resolve()
    history_path = xml_dir.parent / "history.jsonl"

    if not xml_dir.exists():
        raise SystemExit(f"ERROR: xml-dir not found: {xml_dir}")
    if not img_dir.exists():
        raise SystemExit(f"ERROR: img-dir not found: {img_dir}")

    stems = sorted(p.stem for p in xml_dir.glob("*.xml"))
    print(f"XML dir : {xml_dir}  ({len(stems)} pages)")
    print(f"Img dir : {img_dir}")
    print(f"History : {history_path}")
    print(f"\nAnnotation tool ready — http://localhost:{args.port}\n")

    app = build_app(xml_dir, img_dir, history_path)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
