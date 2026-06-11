import json
import shutil
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, url_for, send_file

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ============================================================
# Project root
# ============================================================

# Aquest fitxer està dins review_app/app.py.
# Per tant, l'arrel del projecte és el directori pare.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Configuració de rutes
# ============================================================

# Metadata generat per visual_retrieval/crop_objects_from_layout.py.
METADATA_PATH = PROJECT_ROOT / "outputs/object_crops_raw/metadata.jsonl"

# Carpeta amb crops acceptats.
REVIEWED_DIR = PROJECT_ROOT / "outputs/object_crops_reviewed"

# Carpeta amb crops descartats.
REJECTED_DIR = PROJECT_ROOT / "outputs/object_crops_rejected"

# Carpeta per decisions de skip.
SKIPPED_DIR = PROJECT_ROOT / "outputs/object_crops_skipped"

# Log de decisions humanes.
REVIEW_LOG = PROJECT_ROOT / "outputs/review_logs/review_log.jsonl"

# Carpeta temporal per imatges de pàgina amb bbox dibuixat.
PAGE_PREVIEW_DIR = PROJECT_ROOT / "outputs/review_page_previews"
# Configuració editable de classes, qualitat de bbox i atributs.
REVIEW_SCHEMA_PATH = PROJECT_ROOT / "review_app/review_schema.json"


# faiss 
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs/faiss/current/visual_index.faiss"
FAISS_METADATA_PATH = PROJECT_ROOT / "outputs/faiss/current/metadata.jsonl"
SIMILARITY_TOP_K = 5



app = Flask(__name__)


# ============================================================
# Carrega de dades
# ============================================================

def load_review_schema():
    """
    Carrega la configuració de revisió.

    Això evita hardcodejar classes, qualitats i atributs dins el codi.
    Si el fitxer no existeix, retorna una configuració mínima.
    """
    default_schema = {
        "classes": [
            "stamp",
            "handwritten_text",
            "crossout",
            "censorship_block",
            "table_fragment",
            "false_positive",
        ],
        "bbox_quality": [
            "good",
            "partial",
            "too_large",
            "bad_location",
            "unsure",
        ],
        "human_confidence": [
            "high",
            "medium",
            "low",
            "unsure",
        ],
        "attributes": [
            "crossed_out",
            "overlaps_text",
            "faded",
            "low_contrast",
            "fragmented",
            "rotated",
            "background_noise",
        ],
    }

    if not REVIEW_SCHEMA_PATH.exists():
        return default_schema

    with REVIEW_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)



def load_items():
    """
    Carrega totes les deteccions/crops del metadata.jsonl.

    Cada línia conté:
        crop_id
        type
        crop_path
        image_path
        bbox
        confidence
        document_id
        etc.
    """
    items = []

    if not METADATA_PATH.exists():
        return items

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            item = json.loads(line)
            item["_idx"] = idx
            items.append(item)

    return items


def load_review_entries():
    """
    Carrega totes les decisions humanes fetes fins ara.

    Retorna:
        dict crop_id -> última decisió registrada

    Si un crop es revisa més d'una vegada, ens quedem amb l'última.
    """
    entries = {}

    if not REVIEW_LOG.exists():
        return entries

    with REVIEW_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            crop_id = entry.get("crop_id")

            if crop_id:
                entries[crop_id] = entry

    return entries


def get_item_by_index(index):
    """
    Retorna un item pel seu índex dins metadata.jsonl.
    """
    items = load_items()

    if not items:
        return None, 0

    index = max(0, min(index, len(items) - 1))

    return items[index], len(items)


def get_first_unreviewed_index():
    """
    Busca el primer crop que encara no tingui decisió humana.
    """
    items = load_items()
    reviewed = load_review_entries()

    for idx, item in enumerate(items):
        crop_id = item.get("crop_id")

        if crop_id not in reviewed:
            return idx

    return 0


# ============================================================
# Helpers de path
# ============================================================

def project_path(path_string):
    """
    Converteix un path relatiu del metadata a path absolut.

    El metadata guarda paths tipus:
        outputs/object_crops_raw/stamp/stamp_000001.jpg

    Però Flask s'executa des de review_app/, així que cal resoldre
    sempre respecte l'arrel del projecte.
    """
    return PROJECT_ROOT / path_string


def safe_copy_crop(item, target_root, obj_type):
    """
    Copia el crop original a una carpeta de revisió.

    No mou el fitxer original.
    Això preserva outputs/object_crops_raw/ intacte.
    """
    crop_path = project_path(item["crop_path"])

    target_dir = target_root / obj_type
    target_dir.mkdir(parents=True, exist_ok=True)

    target_crop_path = target_dir / crop_path.name

    if crop_path.exists():
        shutil.copy2(crop_path, target_crop_path)

    return target_crop_path


# ============================================================
# Guardar revisió
# ============================================================

def save_review(
    item,
    decision,
    new_type=None,
    notes=None,
    human_confidence=None,
    bbox_quality=None,
    attributes=None,
):
    """
    Guarda una decisió humana.

    decision pot ser:
        accepted
        rejected
        skipped

    Si accepted:
        copia el crop a outputs/object_crops_reviewed/<classe>/

    Si rejected:
        copia el crop a outputs/object_crops_rejected/<classe>/

    Si skipped:
        opcionalment copia el crop a outputs/object_crops_skipped/<classe>/
        i queda marcat com skipped al log.

    No modifica metadata.jsonl original.
    """
    REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)

    obj_type = new_type or item.get("type", "unknown")

    if decision == "accepted":
        target_crop_path = safe_copy_crop(item, REVIEWED_DIR, obj_type)
    elif decision == "rejected":
        target_crop_path = safe_copy_crop(item, REJECTED_DIR, obj_type)
    else:
        target_crop_path = safe_copy_crop(item, SKIPPED_DIR, obj_type)


    if attributes is None:
        attributes = []

    entry = dict(item)
    entry["decision"] = decision
    entry["reviewed"] = decision in {"accepted", "rejected"}
    entry["skipped"] = decision == "skipped"
    entry["reviewed_type"] = obj_type
    entry["review_notes"] = notes
    entry["human_confidence"] = human_confidence
    entry["bbox_quality"] = bbox_quality
    entry["attributes"] = attributes
    entry["reviewed_crop_path"] = str(target_crop_path)

    with REVIEW_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# Vista de pàgina completa amb bbox
# ============================================================

def make_page_preview(item):
    """
    Genera una imatge de la pàgina completa amb el bbox dibuixat.

    Això serveix per comprovar si la caixa detectada:
        - cobreix l'objecte sencer,
        - només cobreix un tros,
        - agafa massa context,
        - és un fals positiu.
    """
    image_path = item.get("image_path")
    bbox = item.get("bbox")

    if not image_path or not bbox:
        return None

    full_image_path = project_path(image_path)

    if not full_image_path.exists():
        return None

    PAGE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    crop_id = item.get("crop_id", "preview")
    preview_path = PAGE_PREVIEW_DIR / f"{crop_id}_page_bbox.jpg"

    with Image.open(full_image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        # Dibuixem rectangle gruixut.
        for offset in range(5):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline="cyan",
            )

        # Etiqueta visual.
        label = f"{item.get('type')} {item.get('confidence')}"
        draw.rectangle([x1, max(0, y1 - 28), x1 + 420, y1], fill="cyan")
        draw.text([x1 + 5, max(0, y1 - 24)], label, fill="black")

        img.save(preview_path, quality=95)

    return preview_path














# ============================================================
# Similarity search amb FAISS
# ============================================================

def l2_normalize(vector, eps=1e-8):
    """
    Normalitza un vector perquè tingui norma 1.

    Això és necessari perquè el nostre índex FAISS usa cosine similarity
    implementada com Inner Product sobre vectors normalitzats.
    """
    norm = np.linalg.norm(vector)

    if norm < eps:
        return vector

    return vector / norm


def image_to_embedding(image_path, thumbnail_size=64, edge_size=32, hist_bins=16):
    """
    Converteix un crop en el mateix embedding visual simple que build_embeddings.py.

    IMPORTANT:
    Aquesta funció ha de coincidir amb visual_retrieval/build_embeddings.py.
    Si canviem l'embedding allà, també l'hem de canviar aquí.
    """
    with Image.open(image_path).convert("RGB") as img:
        # 1) Miniatura RGB.
        thumb = img.resize((thumbnail_size, thumbnail_size))
        thumb_arr = np.asarray(thumb, dtype=np.float32) / 255.0
        thumb_feat = thumb_arr.flatten()

        # 2) Histograma RGB.
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        hist_features = []

        for channel in range(3):
            hist, _ = np.histogram(
                img_arr[:, :, channel],
                bins=hist_bins,
                range=(0.0, 1.0),
                density=True,
            )
            hist_features.append(hist.astype(np.float32))

        hist_feat = np.concatenate(hist_features)

        # 3) Vores/contorns.
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.resize((edge_size, edge_size))
        edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
        edge_feat = edge_arr.flatten()

    embedding = np.concatenate([
        thumb_feat,
        hist_feat,
        edge_feat,
    ]).astype(np.float32)

    return l2_normalize(embedding)


def load_faiss_metadata():
    """
    Carrega el metadata associat a l'índex FAISS.

    L'ordre del metadata ha de coincidir amb l'ordre dels vectors de l'índex.
    """
    if not FAISS_METADATA_PATH.exists():
        return []

    items = []

    with FAISS_METADATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def find_similar_items(item, top_k=5):
    """
    Retorna els top-k crops més semblants al crop actual.

    Si no existeix l'índex FAISS, si faiss no està instal·lat,
    o si el crop no existeix, retorna [] i l'app continua funcionant.

    També elimina el mateix crop dels resultats, perquè FAISS normalment
    retorna el query com a primer veí amb score 1.0.
    """

    if faiss is None:
        return []

    if not FAISS_INDEX_PATH.exists() or not FAISS_METADATA_PATH.exists():
        return []

    crop_path = project_path(item["crop_path"])

    if not crop_path.exists():
        return []

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = load_faiss_metadata()

    if len(metadata) != index.ntotal:
        return []

    query_embedding = image_to_embedding(crop_path)
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

    if query_embedding.shape[1] != index.d:
        return []

    # Demanem més resultats que top_k perquè segurament el primer serà el mateix crop.
    search_k = min(index.ntotal, top_k + 10)
    scores, ids = index.search(query_embedding, search_k)

    results = []

    current_crop_id = item.get("crop_id")
    current_crop_path = str(crop_path.resolve())

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        candidate = metadata[idx]

        candidate_crop_id = candidate.get("crop_id")
        candidate_crop_path_raw = candidate.get("crop_path")

        # Comprovació 1: mateix crop_id.
        if candidate_crop_id == current_crop_id:
            continue

        # Comprovació 2: mateix path real.
        if candidate_crop_path_raw:
            candidate_crop_path = project_path(candidate_crop_path_raw)

            try:
                if str(candidate_crop_path.resolve()) == current_crop_path:
                    continue
            except FileNotFoundError:
                pass

        result = dict(candidate)
        result["rank"] = len(results) + 1
        result["score"] = float(score)
        result["faiss_id"] = int(idx)

        results.append(result)

        if len(results) >= top_k:
            break

    return results






# ============================================================
# Rutes Flask
# ============================================================

@app.route("/")
def index():
    """
    Pantalla principal.

    Permet:
        - veure crop,
        - veure pàgina completa amb bbox,
        - acceptar,
        - rebutjar,
        - skip,
        - anar endavant/enrere.
    """
    requested_index = request.args.get("idx")

    if requested_index is None:
        idx = get_first_unreviewed_index()
    else:
        idx = int(requested_index)

    item, total = get_item_by_index(idx)

    if item is None:
        return """
        <h1>No metadata found</h1>
        <p>Expected metadata at outputs/object_crops_raw/metadata.jsonl</p>
        """

    reviewed_entries = load_review_entries()
    review_schema = load_review_schema()
    crop_id = item.get("crop_id")
    previous_review = reviewed_entries.get(crop_id)
    # Classe efectiva que ha de mostrar la UI.
    # Si el crop ja està revisat, prioritzem la classe humana.
    # Si no, usem la predicció original del model.
    current_type = (
        previous_review.get("reviewed_type")
        if previous_review and previous_review.get("reviewed_type")
        else item.get("type")
    )

    # Review JSON separat del Raw JSON.
    # El Raw JSON és la predicció original; el Review JSON és la decisió humana.
    if previous_review:
        review_json = json.dumps(previous_review, indent=2, ensure_ascii=False)
    else:
        review_json = json.dumps(
            {
                "reviewed": False,
                "message": "This crop has not been reviewed yet.",
            },
            indent=2,
            ensure_ascii=False,
        )

    similar_items = find_similar_items(item, top_k=SIMILARITY_TOP_K)

    for sim in similar_items:
        sim_crop_id = sim.get("crop_id")
        sim_previous_review = reviewed_entries.get(sim_crop_id)
        sim["previous_review"] = sim_previous_review

        # Classe efectiva del similar:
        # revisió humana si existeix; si no, predicció original.
        sim["effective_type"] = (
            sim_previous_review.get("reviewed_type")
            if sim_previous_review and sim_previous_review.get("reviewed_type")
            else sim.get("type")
        )

    prev_idx = max(0, idx - 1)
    next_idx = min(total - 1, idx + 1)

    message = request.args.get("msg")



    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
            <title>Object Crop Review</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 24px;
                    background: #f5f5f5;
                }
                .topbar {
                    margin-bottom: 18px;
                }
                .container {
                    display: grid;
                    grid-template-columns: 360px minmax(520px, 1fr) 360px;
                    gap: 20px;
                    align-items: start;
                }

                .left-panel,
                .center-panel,
                .right-panel {
                    min-width: 0;
                }
                .card {
                    background: white;
                    padding: 18px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }
                .crop-img {
                    max-width: 380px;
                    max-height: 320px;
                    border: 2px solid #333;
                    background: #ddd;
                }
                .page-img {
                    max-width: 900px;
                    max-height: 850px;
                    border: 2px solid #333;
                    background: #ddd;
                }
                pre {
                    background: #eee;
                    padding: 12px;
                    overflow: auto;
                    max-width: 900px;
                    max-height: 280px;
                }
                button, .navlink {
                    font-size: 16px;
                    padding: 9px 16px;
                    margin-right: 8px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    border-radius: 5px;
                }
                .accept {
                    background: #2ecc71;
                    color: white;
                    border: none;
                }
                .reject {
                    background: #e74c3c;
                    color: white;
                    border: none;
                }
                .skip {
                    background: #f39c12;
                    color: white;
                    border: none;
                }
                .navlink {
                    background: #34495e;
                    color: white;
                }
                input, select {
                    padding: 8px;
                    font-size: 15px;
                    margin-bottom: 10px;
                    width: 260px;
                }
                .status {
                    padding: 10px;
                    background: #eef;
                    border-left: 5px solid #66f;
                    margin-bottom: 15px;
                }
                .attributes-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 6px 12px;
                    margin: 8px 0 14px 0;
                    max-width: 100%;
                }

                .attribute-item {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    font-size: 13px;
                    line-height: 1.2;
                    word-break: break-word;
                }

                .attribute-item input {
                    width: auto;
                    margin: 0;
                }

                .similar-card {
                    border: 1px solid #ccc;
                    padding: 8px;
                    background: #fafafa;
                    margin-bottom: 12px;
                }

                .similar-img {
                    max-width: 260px;
                    max-height: 180px;
                    border: 1px solid #333;
                    display: block;
                    margin-bottom: 8px;
                }

                .review-json {
                    background: #eef8ee;
                    border-left: 4px solid #2f8f2f;
                }

                pre {
                    white-space: pre-wrap;
                    word-break: break-word;
                    max-height: 520px;
                    overflow: auto;
                }

            </style>
        </head>
        <body>
            <h1>Manual Review</h1>
            
            {% if message %}
            <div class="status">
                <b>{{ message }}</b>
            </div>
            {% endif %}


            <div class="topbar">
                <a class="navlink" href="{{ url_for('index', idx=prev_idx) }}">← Previous</a>
                <a class="navlink" href="{{ url_for('index', idx=next_idx) }}">Next →</a>
                <span>Item {{ idx + 1 }} / {{ total }}</span>
            </div>

            {% if previous_review %}
            <div class="status">
                <b>Already reviewed:</b>
                {{ previous_review.get("decision") }}
                as {{ previous_review.get("reviewed_type") }}
            </div>
            {% endif %}

            <div class="container">

                <!-- COLUMN 1: crop + metadata + form -->
                <div class="card left-panel">
                    <h2>Crop</h2>
                    <img class="crop-img" src="{{ url_for('crop_image', crop_id=item['crop_id']) }}">

                    <h2>Metadata</h2>
                    <p><b>Crop ID:</b> {{ item.get("crop_id") }}</p>
                    <p><b>Predicted type:</b> {{ item.get("type") }}</p>
                    <p><b>Confidence:</b> {{ item.get("confidence") }}</p>
                    <p><b>Document:</b> {{ item.get("document_id") }}</p>
                    <p><b>BBox:</b> {{ item.get("bbox") }}</p>

                    <form method="post" action="{{ url_for('review') }}">
                        <input type="hidden" name="crop_id" value="{{ item.get('crop_id') }}">
                        <input type="hidden" name="idx" value="{{ idx }}">

                        <label>Correct class:</label><br>
                        <select name="new_type">
                            {% for cls in review_schema.classes %}
                                <option value="{{ cls }}" {% if cls == current_type %}selected{% endif %}>
                                    {{ cls }}
                                </option>
                            {% endfor %}
                        </select>

                        <br>

                        <label>Human confidence:</label><br>
                        <select name="human_confidence">
                            <option value="">not specified</option>
                            {% for conf in review_schema.human_confidence %}
                                <option value="{{ conf }}">{{ conf }}</option>
                            {% endfor %}
                        </select>

                        <br>

                        <label>BBox quality:</label><br>
                        <select name="bbox_quality">
                            <option value="">not specified</option>
                            {% for q in review_schema.bbox_quality %}
                                <option value="{{ q }}">{{ q }}</option>
                            {% endfor %}
                        </select>

                        <br>

                        <label>Attributes:</label><br>
                        <div class="attributes-grid">
                            {% for attr in review_schema.attributes %}
                                <label class="attribute-item">
                                    <input type="checkbox" name="attributes" value="{{ attr }}">
                                    <span>{{ attr }}</span>
                                </label>
                            {% endfor %}
                        </div>

                        <label>Notes:</label><br>
                        <input type="text" name="notes" placeholder="bbox partial, too large, false positive...">

                        <br><br>

                        <button class="accept" name="decision" value="accepted">Accept</button>
                        <button class="reject" name="decision" value="rejected">Reject</button>
                        <button class="skip" name="decision" value="skipped">Skip</button>
                    </form>
                </div>


                <!-- COLUMN 2: full page + raw json -->
                <div class="card center-panel">
                    <h2>Full page with bbox</h2>
                    <img class="page-img" src="{{ url_for('page_preview', crop_id=item['crop_id']) }}">

                    <h3>Review JSON</h3>
                    <pre class="review-json">{{ review_json }}</pre>

                    <h3>Raw prediction JSON</h3>
                    <pre>{{ raw_json }}</pre>
                </div>


                <!-- COLUMN 3: similar crops -->
                <div class="card right-panel">
                    <h2>Similar crops</h2>

                    {% if similar_items %}
                        {% for sim in similar_items %}
                            <div class="similar-card">
                                <p>
                                    <b>#{{ sim.rank }}</b>
                                    score={{ "%.4f"|format(sim.score) }}
                                    <br>
                                    <p>
                                        <b>#{{ sim.rank }}</b>
                                        score={{ "%.4f"|format(sim.score) }}
                                        <br>
                                        <b>Predicted:</b> {{ sim.get("type") }}
                                        <br>
                                        <b>Effective:</b> {{ sim.get("effective_type") }}
                                        <br>
                                        crop={{ sim.get("crop_id") }}
                                        <br>
                                        conf={{ sim.get("confidence") }}

                                        {% if sim.previous_review %}
                                        <br>
                                        <b>Reviewed:</b>
                                        {{ sim.previous_review.get("decision") }}
                                        {% if sim.previous_review.get("reviewed_type") %}
                                            as {{ sim.previous_review.get("reviewed_type") }}
                                        {% endif %}
                                        {% endif %}
                                    </p>
                                </p>

                                <img
                                    class="similar-img"
                                    src="{{ url_for('similar_crop_image', faiss_id=sim.faiss_id) }}"
                                >

                                <form method="post" action="{{ url_for('review_similar') }}">
                                    <input type="hidden" name="faiss_id" value="{{ sim.faiss_id }}">
                                    <input type="hidden" name="idx" value="{{ idx }}">
                                    <label>Class:</label><br>
                                    <select name="new_type">
                                        {% for cls in review_schema.classes %}
                                            <option value="{{ cls }}" {% if cls == sim.get("effective_type") %}selected{% endif %}>
                                                {{ cls }}
                                            </option>
                                        {% endfor %}
                                    </select>

                                    <button class="accept" name="decision" value="accepted">Accept similar</button>
                                    <button class="reject" name="decision" value="rejected">Reject similar</button>
                                </form>
                            </div>
                        {% endfor %}
                    {% else %}
                        <p>No FAISS similar results available.</p>
                    {% endif %}
                </div>

            </div>
        </body>
        </html>
        """,
        item=item,
        raw_json=json.dumps(item, indent=2, ensure_ascii=False),
        review_json=review_json,
        current_type=current_type,
        idx=idx,
        total=total,
        prev_idx=prev_idx,
        next_idx=next_idx,
        previous_review=previous_review,
        similar_items=similar_items,
        message=message,
        review_schema=review_schema,
    )


@app.route("/crop/<crop_id>")
def crop_image(crop_id):
    """
    Serveix la imatge del crop.
    """
    items = load_items()

    for item in items:
        if item.get("crop_id") == crop_id:
            crop_path = project_path(item["crop_path"])

            if crop_path.exists():
                return send_file(crop_path)

    return "Crop not found", 404


@app.route("/page/<crop_id>")
def page_preview(crop_id):
    """
    Serveix la pàgina completa amb bbox dibuixat.
    """
    items = load_items()

    for item in items:
        if item.get("crop_id") == crop_id:
            preview_path = make_page_preview(item)

            if preview_path and preview_path.exists():
                return send_file(preview_path)

    return "Page preview not found", 404


@app.route("/review", methods=["POST"])
def review():
    """
    Rep la decisió del formulari i redirigeix al següent item.
    """
    crop_id = request.form.get("crop_id")
    decision = request.form.get("decision")
    new_type = request.form.get("new_type")
    notes = request.form.get("notes")
    human_confidence = request.form.get("human_confidence")
    bbox_quality = request.form.get("bbox_quality")
    attributes = request.form.getlist("attributes")
    idx = int(request.form.get("idx", 0))

    items = load_items()

    item = None

    for candidate in items:
        if candidate.get("crop_id") == crop_id:
            item = candidate
            break

    if item is None:
        return f"Crop not found: {crop_id}", 404

    save_review(
        item=item,
        decision=decision,
        new_type=new_type,
        notes=notes,
        human_confidence=human_confidence,
        bbox_quality=bbox_quality,
        attributes=attributes,
    )

    # Després de decidir, anem al següent.
    return redirect(url_for("index", idx=idx + 1))



@app.route("/similar_crop/<int:faiss_id>")
def similar_crop_image(faiss_id):
    """
    Serveix la imatge d'un crop similar retornat per FAISS.
    """
    metadata = load_faiss_metadata()

    if faiss_id < 0 or faiss_id >= len(metadata):
        return "Similar crop not found", 404

    crop_path = project_path(metadata[faiss_id]["crop_path"])

    if crop_path.exists():
        return send_file(crop_path)

    return "Similar crop file not found", 404



@app.route("/review_similar", methods=["POST"])
def review_similar():
    """
    Permet acceptar o rebutjar un crop similar directament des de la UI.

    No mou fitxers originals: copia el crop a reviewed/ o rejected/
    i escriu la decisió al review_log.jsonl.
    """
    faiss_id = int(request.form.get("faiss_id"))
    decision = request.form.get("decision")
    new_type = request.form.get("new_type")
    idx = int(request.form.get("idx", 0))

    metadata = load_faiss_metadata()

    if faiss_id < 0 or faiss_id >= len(metadata):
        return f"Invalid FAISS id: {faiss_id}", 404

    item = metadata[faiss_id]

    if decision == "rejected":
        reviewed_type = "false_positive"
    else:
        reviewed_type = new_type

    save_review(
        item=item,
        decision=decision,
        new_type=reviewed_type,
        notes="reviewed from similar crop suggestion",
        human_confidence="medium",
        bbox_quality="good" if decision == "accepted" else "bad_location",
        attributes=["similar_review"],
    )

    crop_id = item.get("crop_id", "unknown")
    if decision == "rejected":
        msg = f"Similar crop {crop_id} rejected. Predicted type was {item.get('type')}"
    else:
        msg = f"Similar crop {crop_id} accepted as {new_type}"

    # Tornem al mateix crop principal, però amb missatge visible.
    return redirect(url_for("index", idx=idx, msg=msg))






if __name__ == "__main__":
    # App local.
    # Obre al navegador: http://127.0.0.1:5000
    app.run(debug=True, host="127.0.0.1", port=5000)