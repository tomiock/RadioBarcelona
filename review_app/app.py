import json
import os
import shutil
import subprocess
import sys
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

# Metadata generat per processing/crop_objects_from_layout.py.
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs/object_crops_raw/metadata.jsonl"
METADATA_PATH = Path(os.environ.get("REVIEW_METADATA", str(DEFAULT_METADATA_PATH))).expanduser()
if not METADATA_PATH.is_absolute():
    METADATA_PATH = (PROJECT_ROOT / METADATA_PATH).resolve()

# Carpeta amb crops acceptats.
REVIEWED_DIR = PROJECT_ROOT / "outputs/object_crops_reviewed"

# Carpeta amb crops descartats.
REJECTED_DIR = PROJECT_ROOT / "outputs/object_crops_rejected"

# Carpeta per decisions de skip.
SKIPPED_DIR = PROJECT_ROOT / "outputs/object_crops_skipped"

# Crops creats manualment des del selector bbox de la review app.
MANUAL_CROPS_DIR = PROJECT_ROOT / "outputs/object_crops_manual"
MANUAL_METADATA_PATH = MANUAL_CROPS_DIR / "metadata.jsonl"

# Log de decisions humanes.
REVIEW_LOG = PROJECT_ROOT / "outputs/review_logs/review_log.jsonl"

# Carpeta temporal per imatges de pàgina amb bbox dibuixat.
PAGE_PREVIEW_DIR = PROJECT_ROOT / "outputs/review_page_previews"
# Configuració editable de classes, qualitat de bbox i atributs.
REVIEW_SCHEMA_PATH = PROJECT_ROOT / "review_app/review_schema.json"

# Índexs JSONL generats per tools/review_tools/build_review_indexes.py.
# Són opcionals: si no existeixen, els filtres funcionen igual en mode dinàmic.
INDEX_DIR = PROJECT_ROOT / "outputs/index"
BUILD_INDEX_SCRIPT = PROJECT_ROOT / "tools/review_tools/build_review_indexes.py"

FILTERS = {
    "all": "All",
    "pending": "Pending",
    "reviewed": "Reviewed",
    "accepted": "Accepted",
    "rejected": "Rejected",
    "skipped": "Skipped",
    "exportable": "Exportable",
    "accepted_not_exportable": "Accepted but not exportable",
}

# Compatibility aliases after renaming typewritten_line -> typewritten_text.
CLASS_ALIASES = {
    "typewritten_line": "typewritten_text",
}


def normalize_class_name(class_name):
    if not class_name:
        return class_name
    return CLASS_ALIASES.get(class_name, class_name)


# FAISS simple visual search
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs/faiss/current/visual_index.faiss"
FAISS_METADATA_PATH = PROJECT_ROOT / "outputs/faiss/current/metadata.jsonl"

# VAE + FAISS visual search. This is built by:
# python visual_search/build_vae_faiss.py --project-root . --metadata outputs/object_crops_raw/metadata.jsonl --review-log outputs/review_logs/review_log.jsonl --model outputs/vae/vae_best.pt --output-dir outputs/faiss/vae/global --by-type
VAE_FAISS_GLOBAL_INDEX_PATH = PROJECT_ROOT / "outputs/faiss/vae/global/visual_index.faiss"
VAE_FAISS_GLOBAL_METADATA_PATH = PROJECT_ROOT / "outputs/faiss/vae/global/metadata.jsonl"
VAE_FAISS_BY_TYPE_DIR = PROJECT_ROOT / "outputs/faiss/vae/by_type"

SIMILARITY_TOP_K = 5

# Directori d'assets reals revisats que pot llegir generator.py.
GENERATOR_ASSETS_DIR = PROJECT_ROOT / "synthetic_docs_aran/assets_real_reviewed"
GENERATOR_ASSET_TARGETS = {
    "stamp": "stamps",
    "handwritten_text": "handwriting",
    "typewritten_text": "typewriting",
    "crossout": "crossouts",
    "censorship_block": "censorship",
    "table_fragment": "tables",
}
GENERATOR_ASSET_MANIFEST = GENERATOR_ASSETS_DIR / "manifest_review_app_assets.jsonl"



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
            "typewritten_text",
            "crossout",
            "censorship_block",
            "table_fragment",
            "false_positive",
        ],
        "bbox_quality": [
            "good",
            "minor_partial",
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
            "mixed",
            "typewritten",
            "handwritten",
            "stamp",
            "table",    
        ],
    }

    if not REVIEW_SCHEMA_PATH.exists():
        return default_schema

    with REVIEW_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    # Compatibility: older review_schema.json files may not include newer generic flags.
    # mixed = the crop contains more than one relevant visual/textual phenomenon
    # and cannot be described cleanly by a single attribute.
    schema.setdefault("classes", [])
    schema["classes"] = [normalize_class_name(cls) for cls in schema["classes"]]
    # De-duplicate while preserving order.
    schema["classes"] = list(dict.fromkeys(schema["classes"]))

    schema.setdefault("attributes", [])
    if "mixed" not in schema["attributes"]:
        schema["attributes"].append("mixed")

    return schema



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
            if "type" in item:
                item["type"] = normalize_class_name(item.get("type"))
            if "reviewed_type" in item:
                item["reviewed_type"] = normalize_class_name(item.get("reviewed_type"))
            item["_idx"] = idx
            items.append(item)

    return items


def load_manual_items():
    """Carrega només els crops creats manualment des del selector bbox."""
    items = []
    if not MANUAL_METADATA_PATH.exists():
        return items

    with MANUAL_METADATA_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            if "type" in item:
                item["type"] = normalize_class_name(item.get("type"))
            if "reviewed_type" in item:
                item["reviewed_type"] = normalize_class_name(item.get("reviewed_type"))
            item["_manual_idx"] = idx
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
            if "type" in entry:
                entry["type"] = normalize_class_name(entry.get("type"))
            if "reviewed_type" in entry:
                entry["reviewed_type"] = normalize_class_name(entry.get("reviewed_type"))
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



def parse_corrected_bbox_from_form(form):
    """
    Parse optional corrected bbox fields from the review form.

    If all fields are empty, returns None.
    If some fields are filled but invalid, returns None to avoid breaking review.
    """
    keys = ["corrected_x1", "corrected_y1", "corrected_x2", "corrected_y2"]
    values = [form.get(k, "").strip() for k in keys]

    if not any(values):
        return None

    if not all(values):
        return None

    try:
        x1 = int(round(float(values[0])))
        y1 = int(round(float(values[1])))
        x2 = int(round(float(values[2])))
        y2 = int(round(float(values[3])))
    except ValueError:
        return None

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    if x2 <= x1 or y2 <= y1:
        return None

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


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
    corrected_bbox=None,
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

    obj_type = normalize_class_name(new_type or item.get("type", "unknown"))

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
    if corrected_bbox:
        entry["corrected_bbox"] = corrected_bbox
    entry["reviewed_crop_path"] = str(target_crop_path.relative_to(PROJECT_ROOT))

    # Guardem paths relatius perquè el review_log.jsonl sigui portable
    # entre ordinadors i carpetes diferents.
    if entry.get("image_path"):
        try:
            entry["image_path"] = str(Path(entry["image_path"]).relative_to(PROJECT_ROOT))
        except ValueError:
            pass

    if entry.get("reviewed_crop_path"):
        try:
            entry["reviewed_crop_path"] = str(Path(entry["reviewed_crop_path"]).relative_to(PROJECT_ROOT))
        except ValueError:
            pass

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
    Aquesta funció ha de coincidir amb visual_search/build_embeddings.py.
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


def load_faiss_metadata(metadata_path=None):
    """
    Carrega el metadata associat a un índex FAISS.

    Per defecte carrega l'índex FAISS simple antic.
    Si es passa metadata_path, pot carregar també metadata VAE global o by_type.
    """
    path = Path(metadata_path) if metadata_path else FAISS_METADATA_PATH

    if not path.exists():
        return []

    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def find_item_by_crop_id(crop_id):
    """Busca un crop_id dins el metadata raw principal."""
    if not crop_id:
        return None

    for item in load_items():
        if item.get("crop_id") == crop_id:
            return item

    return None


def copy_crop_to_generator_assets(item, asset_type, previous_review=None):
    """
    Copia el crop actual a assets_real_reviewed/<folder>/ i afegeix una línia al manifest.

    És una acció explícita de curació: no substitueix Accept/Reject, sinó que marca
    aquest crop com a asset útil per generar futurs documents sintètics.
    """
    if asset_type not in GENERATOR_ASSET_TARGETS:
        raise ValueError(f"Unsupported generator asset type: {asset_type}")

    crop_path_raw = item.get("crop_path")
    if not crop_path_raw:
        raise FileNotFoundError("Crop has no crop_path")

    src = project_path(crop_path_raw)
    if not src.exists():
        raise FileNotFoundError(f"Crop file not found: {src}")

    target_dir = GENERATOR_ASSETS_DIR / GENERATOR_ASSET_TARGETS[asset_type]
    target_dir.mkdir(parents=True, exist_ok=True)

    crop_id = item.get("crop_id", "crop")
    suffix = src.suffix or ".jpg"
    dst = target_dir / f"{asset_type}_{crop_id}{suffix}"

    # Si ja existeix, no dupliquem el fitxer; el manifest pot registrar múltiples decisions.
    if not dst.exists():
        shutil.copy2(src, dst)

    GENERATOR_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_row = {
        "crop_id": crop_id,
        "asset_type": asset_type,
        "asset_path": str(dst.relative_to(PROJECT_ROOT)),
        "source_crop_path": crop_path_raw,
        "document_id": item.get("document_id"),
        "image": item.get("image"),
        "image_path": item.get("image_path"),
        "bbox": item.get("bbox"),
        "predicted_type": item.get("type"),
        "decision": previous_review.get("decision") if previous_review else None,
        "reviewed_type": previous_review.get("reviewed_type") if previous_review else None,
        "bbox_quality": previous_review.get("bbox_quality") if previous_review else None,
        "source": "review_app_send_to_generator_assets",
    }

    with GENERATOR_ASSET_MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    return dst


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_manual_crop_id(asset_type):
    safe_type = asset_type.replace("/", "_")
    existing = 0
    if MANUAL_METADATA_PATH.exists():
        with MANUAL_METADATA_PATH.open("r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
    return f"manual_{safe_type}_{existing:06d}"


def create_manual_crop_from_bbox(source_item, bbox, manual_type, notes=""):
    manual_type = normalize_class_name(manual_type)
    """
    Retalla una bbox manual sobre la pàgina original, guarda crop i metadata.

    Per compatibilitat amb el pipeline actual, la metadata també s'afegeix a
    outputs/object_crops_raw/metadata.jsonl amb source=manual_bbox_review_app.
    També es guarda una còpia neta a outputs/object_crops_manual/metadata.jsonl.
    """
    image_path_raw = source_item.get("image_path")
    if not image_path_raw:
        raise FileNotFoundError("Current item has no image_path")

    image_path = project_path(image_path_raw)
    if not image_path.exists():
        raise FileNotFoundError(f"Source page image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    x1 = int(round(min(float(bbox["x1"]), float(bbox["x2"]))))
    y1 = int(round(min(float(bbox["y1"]), float(bbox["y2"]))))
    x2 = int(round(max(float(bbox["x1"]), float(bbox["x2"]))))
    y2 = int(round(max(float(bbox["y1"]), float(bbox["y2"]))))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))

    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 2 or (y2 - y1) < 2:
        raise ValueError(f"Invalid manual bbox: {(x1, y1, x2, y2)}")

    crop = img.crop((x1, y1, x2, y2))

    crop_id = make_manual_crop_id(manual_type)
    out_dir = MANUAL_CROPS_DIR / manual_type
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / f"{crop_id}.jpg"
    crop.save(crop_path, quality=95)

    row = {
        "crop_id": crop_id,
        "type": manual_type,
        "subtype": "manual_bbox",
        "document_id": source_item.get("document_id"),
        "image": source_item.get("image"),
        "image_path": source_item.get("image_path"),
        "layout_path": source_item.get("layout_path"),
        "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "original_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": 1.0,
        "source": "manual_bbox_review_app",
        "source_crop_id": source_item.get("crop_id"),
        "reviewed": True,
        "decision": "accepted",
        "reviewed_type": manual_type,
        "bbox_quality": "good",
        "human_confidence": "high",
        "review_notes": notes or "manual bbox crop created in review app",
        "attributes": ["manual_bbox"],
    }

    append_jsonl(MANUAL_METADATA_PATH, row)
    append_jsonl(METADATA_PATH, row)

    save_review(
        item=row,
        decision="accepted",
        new_type=manual_type,
        notes=notes or "manual bbox crop created in review app",
        human_confidence="high",
        bbox_quality="good",
        attributes=["manual_bbox"],
    )

    return row, crop_path


def enrich_similar_items_with_review(similar_items, reviewed_entries):
    """Afegeix previous_review i effective_type als resultats de similitud."""
    for sim in similar_items:
        sim_crop_id = sim.get("crop_id")
        sim_previous_review = reviewed_entries.get(sim_crop_id)
        sim["previous_review"] = sim_previous_review
        sim["effective_type"] = (
            sim_previous_review.get("reviewed_type")
            if sim_previous_review and sim_previous_review.get("reviewed_type")
            else sim.get("type")
        )
    return similar_items


def resolve_vae_index_paths_for_item(item):
    """
    Prioritza l'índex VAE per tipus quan existeix.
    Si no hi ha índex per aquell tipus, cau al VAE global.
    """
    crop_type = (
        item.get("effective_type")
        or item.get("reviewed_type")
        or item.get("type")
    )

    if crop_type:
        type_dir = VAE_FAISS_BY_TYPE_DIR / crop_type
        type_index = type_dir / "visual_index.faiss"
        type_metadata = type_dir / "metadata.jsonl"
        if type_index.exists() and type_metadata.exists():
            return type_index, type_metadata, f"vae_by_type:{crop_type}"

    return VAE_FAISS_GLOBAL_INDEX_PATH, VAE_FAISS_GLOBAL_METADATA_PATH, "vae_global"


def find_similar_items_from_existing_faiss_vector(item, index_path, metadata_path, top_k=5, source_label="vae"):
    """
    Cerca similars en un índex FAISS ja construït usant el vector existent del query.

    Això evita carregar el model VAE dins Flask: l'índex VAE ja conté vectors per crop_id.
    Si el crop no és dins l'índex per tipus, es retorna [] i l'app continua funcionant.
    """
    if faiss is None:
        return []

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists() or not metadata_path.exists():
        return []

    metadata = load_faiss_metadata(metadata_path)
    if not metadata:
        return []

    index = faiss.read_index(str(index_path))

    if len(metadata) != index.ntotal:
        return []

    current_crop_id = item.get("crop_id")
    query_idx = None

    for i, candidate in enumerate(metadata):
        if candidate.get("crop_id") == current_crop_id:
            query_idx = i
            break

    if query_idx is None:
        return []

    try:
        query_embedding = index.reconstruct(int(query_idx)).reshape(1, -1).astype(np.float32)
    except Exception:
        return []

    search_k = min(index.ntotal, top_k + 10)
    scores, ids = index.search(query_embedding, search_k)

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        candidate = metadata[idx]
        candidate_crop_id = candidate.get("crop_id")

        if candidate_crop_id == current_crop_id:
            continue

        result = dict(candidate)
        result["rank"] = len(results) + 1
        result["score"] = float(score)
        result["faiss_id"] = int(idx)
        result["similarity_source"] = source_label

        results.append(result)

        if len(results) >= top_k:
            break

    return results


def find_vae_similar_items(item, top_k=5):
    """Retorna similars VAE, preferentment dins el mateix tipus efectiu."""
    index_path, metadata_path, source_label = resolve_vae_index_paths_for_item(item)
    return find_similar_items_from_existing_faiss_vector(
        item=item,
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=top_k,
        source_label=source_label,
    )


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





def compute_review_stats():
    """
    Calcula estadístiques globals de la revisió manual.

    Important:
    - metadata.jsonl conté totes les deteccions/crops originals.
    - review_log.jsonl conté les decisions humanes.
    - Si un crop s'ha revisat més d'una vegada, load_review_entries()
      ja retorna només l'última decisió per crop_id.

    Criteri explícit:
    - Reviewed = accepted + rejected. Skip no compta com reviewed net.
    - Exportable assets = accepted + bbox_quality good + classe exportable.
    - False positives = rejected o reviewed_type false_positive.
      Això fa que el crop principal i els similar crops tinguin una lògica coherent.
    """
    items = load_items()
    reviewed_entries = load_review_entries()

    total_crops = len(items)

    decisions = {}
    reviewed_types = {}
    predicted_types = {}
    effective_types = {}
    bbox_qualities = {}
    attributes = {}

    exportable_types = {
        "stamp",
        "handwritten_text",
        "typewritten_text",
        "crossout",
        "censorship_block",
        "table_fragment",
    }

    accepted_total = 0
    rejected_total = 0
    skipped_total = 0
    reviewed_total = 0
    exportable_candidates = 0
    false_positives = 0
    accepted_not_exportable = 0

    for item in items:
        predicted_type = item.get("type") or "unknown"
        crop_id = item.get("crop_id")
        review = reviewed_entries.get(crop_id) if crop_id else None
        effective_type = (review.get("reviewed_type") if review and review.get("reviewed_type") else predicted_type)
        predicted_types[predicted_type] = predicted_types.get(predicted_type, 0) + 1
        effective_types[effective_type] = effective_types.get(effective_type, 0) + 1

    for entry in reviewed_entries.values():
        decision = entry.get("decision") or "unknown"
        reviewed_type = entry.get("reviewed_type") or "unknown"
        bbox_quality = entry.get("bbox_quality") or "unspecified"

        decisions[decision] = decisions.get(decision, 0) + 1
        reviewed_types[reviewed_type] = reviewed_types.get(reviewed_type, 0) + 1
        bbox_qualities[bbox_quality] = bbox_qualities.get(bbox_quality, 0) + 1

        for attr in entry.get("attributes", []) or []:
            attributes[attr] = attributes.get(attr, 0) + 1

        if decision == "accepted":
            accepted_total += 1
        elif decision == "rejected":
            rejected_total += 1
        elif decision == "skipped":
            skipped_total += 1

        if decision in {"accepted", "rejected"}:
            reviewed_total += 1

        # Rebutjar un crop vol dir que no és un asset aprofitable.
        # Si a més la classe humana és false_positive, també queda explícit.
        if decision == "rejected" or reviewed_type == "false_positive":
            false_positives += 1

        is_exportable = (
            decision == "accepted"
            and bbox_quality in {"good", "minor_partial"}
            and reviewed_type in exportable_types
        )

        if is_exportable:
            exportable_candidates += 1
        elif decision == "accepted":
            accepted_not_exportable += 1

    # Pending = crops que encara no tenen una decisió final accept/reject.
    # Els skipped continuen sense ser assets finals, però ja apareixen separats.
    pending_total = max(0, total_crops - reviewed_total)

    return {
        "total_crops": total_crops,
        "reviewed_total": reviewed_total,
        "pending_total": pending_total,
        "accepted_total": accepted_total,
        "rejected_total": rejected_total,
        "skipped_total": skipped_total,
        "decisions": decisions,
        "reviewed_types": reviewed_types,
        "predicted_types": predicted_types,
        "effective_types": effective_types,
        "bbox_qualities": bbox_qualities,
        "attributes": attributes,
        "exportable_candidates": exportable_candidates,
        "accepted_not_exportable": accepted_not_exportable,
        "false_positives": false_positives,
    }




# ============================================================
# Filtres i export package
# ============================================================

def is_exportable_review_entry(entry):
    """Mateix criteri que compute_review_stats(), reutilitzat pels filtres."""
    exportable_types = {
        "stamp",
        "handwritten_text",
        "typewritten_text",
        "crossout",
        "censorship_block",
        "table_fragment",
    }
    return (
        entry.get("decision") == "accepted"
        and entry.get("bbox_quality") in {"good", "minor_partial"}
        and entry.get("reviewed_type") in exportable_types
    )


def filter_index_path(filter_name):
    """Retorna el JSONL filtrat si build_review_indexes.py ja l'ha generat."""
    mapping = {
        "all": INDEX_DIR / "all.jsonl",
        "pending": INDEX_DIR / "by_status/pending.jsonl",
        "reviewed": INDEX_DIR / "by_status/reviewed.jsonl",
        "accepted": INDEX_DIR / "by_status/accepted.jsonl",
        "rejected": INDEX_DIR / "by_status/rejected.jsonl",
        "skipped": INDEX_DIR / "by_status/skipped.jsonl",
        "exportable": INDEX_DIR / "by_status/exportable.jsonl",
        "accepted_not_exportable": INDEX_DIR / "by_status/accepted_not_exportable.jsonl",
    }
    return mapping.get(filter_name, mapping["all"])


def load_jsonl_item_by_index(path, index):
    """Carrega només una línia d'un JSONL per índex.

    Per milions d'elements, el pas següent seria afegir byte-offset indexes (.idx).
    """
    if not path.exists():
        return None, 0

    item = None
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i == index:
                item = json.loads(line)
            total += 1

    if total == 0:
        return None, 0

    if item is None:
        index = max(0, min(index, total - 1))
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == index and line.strip():
                    item = json.loads(line)
                    break

    return item, total


def enrich_item_for_filter(item, reviewed_entries):
    crop_id = item.get("crop_id")
    review = reviewed_entries.get(crop_id)
    row = dict(item)
    row["predicted_type"] = item.get("type")

    if review:
        row.update({
            "decision": review.get("decision"),
            "reviewed_type": review.get("reviewed_type"),
            "bbox_quality": review.get("bbox_quality"),
            "human_confidence": review.get("human_confidence"),
            "attributes": review.get("attributes", []),
        })
    else:
        row.update({
            "decision": None,
            "reviewed_type": None,
            "bbox_quality": None,
            "human_confidence": None,
            "attributes": [],
        })

    row["effective_type"] = row.get("reviewed_type") or row.get("type")
    row["is_exportable"] = is_exportable_review_entry(row)
    row["is_accepted_not_exportable"] = row.get("decision") == "accepted" and not row["is_exportable"]
    return row


def item_matches_filter(row, filter_name, type_field=None, type_value=None):
    decision = row.get("decision")

    if filter_name == "all":
        status_ok = True
    elif filter_name == "pending":
        status_ok = decision is None
    elif filter_name == "reviewed":
        status_ok = decision in {"accepted", "rejected"}
    elif filter_name == "accepted":
        status_ok = decision == "accepted"
    elif filter_name == "rejected":
        status_ok = decision == "rejected"
    elif filter_name == "skipped":
        status_ok = decision == "skipped"
    elif filter_name == "exportable":
        status_ok = bool(row.get("is_exportable"))
    elif filter_name == "accepted_not_exportable":
        status_ok = bool(row.get("is_accepted_not_exportable"))
    else:
        status_ok = True

    if not status_ok:
        return False

    if type_value:
        if type_field == "predicted":
            return (row.get("predicted_type") or row.get("type")) == type_value
        if type_field == "effective":
            return (row.get("effective_type") or row.get("reviewed_type") or row.get("type")) == type_value

    return True


def get_filtered_item_by_index(index, filter_name, type_field=None, type_value=None):
    """Retorna un item filtrat, preservant el comportament antic si no hi ha índexs."""
    filter_name = filter_name if filter_name in FILTERS else "all"
    # Mode dinàmic: la UI reflecteix immediatament accept/reject/skip.
    # Els JSONL precomputats continuen servint per export/package i escala offline,
    # però no els usem aquí per evitar filtres desactualitzats mentre revisem.
    items = load_items()
    reviewed_entries = load_review_entries()
    filtered = []

    for item in items:
        row = enrich_item_for_filter(item, reviewed_entries)
        if item_matches_filter(row, filter_name, type_field=type_field, type_value=type_value):
            filtered.append(item)

    if not filtered:
        return None, 0

    index = max(0, min(index, len(filtered) - 1))
    return filtered[index], len(filtered)



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
    filter_name = request.args.get("filter", "all")
    if filter_name not in FILTERS:
        filter_name = "all"

    type_field = request.args.get("type_field") or ""
    type_value = request.args.get("type_value") or ""
    if type_field not in {"", "predicted", "effective"}:
        type_field = ""
        type_value = ""

    if requested_index is None:
        idx = 0 if filter_name != "all" else get_first_unreviewed_index()
    else:
        idx = int(requested_index)

    item, total = get_filtered_item_by_index(idx, filter_name, type_field=type_field, type_value=type_value)

    if item is None:
        if not METADATA_PATH.exists():
            return f"""
            <h1>No metadata found</h1>
            <p>Expected metadata at {METADATA_PATH}</p>
            """

        clear_url = url_for("index", filter="all", idx=0)
        return f"""
        <h1>No crops match the current filter</h1>
        <p><b>Filter:</b> {filter_name}</p>
        <p><b>Type filter:</b> {type_field or '-'} = {type_value or '-'}</p>
        <p>This usually means that a status filter and a type filter are being combined and the intersection is empty.</p>
        <p><a href="{clear_url}">Clear filters and return to all crops</a></p>
        """

    reviewed_entries = load_review_entries()
    review_schema = load_review_schema()
    review_stats = compute_review_stats()
    crop_id = item.get("crop_id")
    previous_review = reviewed_entries.get(crop_id)
    # Classe efectiva mostrada al formulari:
    # si ja hi ha revisió humana, usem reviewed_type;
    # si no, usem la predicció original.
    current_type = normalize_class_name(
        previous_review.get("reviewed_type")
        if previous_review and previous_review.get("reviewed_type")
        else item.get("type")
    )

    current_human_confidence = (
        previous_review.get("human_confidence")
        if previous_review and previous_review.get("human_confidence")
        else ""
    )

    current_bbox_quality = (
        previous_review.get("bbox_quality")
        if previous_review and previous_review.get("bbox_quality")
        else ""
    )

    current_notes = (
        previous_review.get("review_notes")
        if previous_review and previous_review.get("review_notes")
        else ""
    )

    current_attributes = (
        previous_review.get("attributes", [])
        if previous_review
        else []
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

    item_for_similarity = dict(item)
    item_for_similarity["effective_type"] = current_type

    similar_items = enrich_similar_items_with_review(
        find_similar_items(item_for_similarity, top_k=SIMILARITY_TOP_K),
        reviewed_entries,
    )

    vae_similar_items = enrich_similar_items_with_review(
        find_vae_similar_items(item_for_similarity, top_k=SIMILARITY_TOP_K),
        reviewed_entries,
    )

    prev_idx = max(0, idx - 1)
    next_idx = min(total - 1, idx + 1)

    message = request.args.get("msg")

    last_manual_crop_id = request.args.get("last_manual_crop_id")
    last_manual_crop = find_item_by_crop_id(last_manual_crop_id) if last_manual_crop_id else None



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
                    color: #111;
                }
                .topbar {
                    margin-bottom: 18px;
                }
                .container {
                    display: grid;
                    grid-template-columns: minmax(300px, 0.75fr) minmax(520px, 1.30fr) minmax(320px, 0.85fr) minmax(300px, 0.75fr) minmax(300px, 0.75fr);
                    gap: 18px;
                    align-items: start;
                    max-width: 2400px;
                }                

                .left-panel,
                .page-panel,
                .json-panel,
                .right-panel,
                .vae-panel {
                    min-width: 0;
                }

                
                .json-panel pre {
                    max-height: 100%;
                    overflow: auto;
                    white-space: pre-wrap;
                    word-break: break-word;
                }

                .review-json {
                    background: #eef8ee;
                    border-left: 4px solid #2f8f2f;
                }
                                
                .card {
                    background: white;
                    padding: 18px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }


                * {
                    box-sizing: border-box;
                }                

                .crop-img {
                    width: 100%;
                    max-width: 100%;
                    max-height: 260px;
                    object-fit: contain;
                    border: 2px solid #333;
                    background: #ddd;
                }

                .page-img {
                    width: 100%;
                    max-width: 100%;
                    max-height: 82vh;
                    object-fit: contain;
                    border: 2px solid #333;
                    background: #ddd;
                }

                input,
                select,
                textarea {
                    padding: 8px;
                    font-size: 15px;
                    margin-bottom: 10px;
                    width: 100%;
                    max-width: 100%;
                }

                @media (max-width: 1650px) {
                    .container {
                        grid-template-columns: minmax(320px, 0.9fr) minmax(520px, 1.5fr) minmax(340px, 1fr);
                    }

                    .right-panel {
                        grid-column: 1 / -1;
                    }
                }

                @media (max-width: 1150px) {
                    body {
                        margin: 14px;
                    }

                    .container {
                        grid-template-columns: 1fr;
                    }

                    .page-img {
                        max-height: none;
                    }
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
                input, select, textarea {
                    padding: 8px;
                    font-size: 15px;
                    margin-bottom: 10px;
                    width: 100%;
                    max-width: 100%;
                }
                .status {
                    padding: 10px;
                    background: #eef;
                    border-left: 5px solid #66f;
                    margin-bottom: 15px;
                }
                .attributes-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 8px 12px;
                    margin: 8px 0 14px 0;
                    max-width: 100%;
                }

                .attribute-item {
                    display: flex;
                    align-items: flex-start;
                    gap: 7px;
                    font-size: 14px;
                    line-height: 1.25;
                    overflow-wrap: anywhere;
                    hyphens: auto;
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
                    width: 100%;
                    max-width: 100%;
                    max-height: 180px;
                    border: 1px solid #333;
                    display: block;
                    margin-bottom: 8px;
                }          

                .review-badge {
                    display: inline-block;
                    margin-top: 6px;
                    padding: 4px 6px;
                    border-radius: 4px;
                    font-weight: bold;
                }

                .meta-badge,
                .decision-badge {
                    display: inline-block;
                    padding: 3px 7px;
                    border-radius: 999px;
                    font-size: 13px;
                    font-weight: 700;
                    border: 1px solid rgba(0,0,0,0.15);
                    background: #eee;
                    color: #222;
                }

                .type-stamp { background: #e3f2fd; color: #0d47a1; border-color: #64b5f6; }
                .type-handwritten_text { background: #f3e5f5; color: #6a1b9a; border-color: #ba68c8; }
                .type-typewritten_text { background: #eceff1; color: #263238; border-color: #90a4ae; }
                .type-typewritten_text { background: #eceff1; color: #263238; border-color: #90a4ae; }
                .type-crossout { background: #ffebee; color: #b71c1c; border-color: #ef9a9a; }
                .type-censorship_block { background: #ede7f6; color: #311b92; border-color: #9575cd; }
                .type-table_fragment { background: #e8f5e9; color: #1b5e20; border-color: #81c784; }
                .type-false_positive { background: #212121; color: white; border-color: #000; }
                .type-unknown { background: #fff3e0; color: #e65100; border-color: #ffb74d; }

                .decision-accepted { background: #d8f5df; color: #176b2c; border-color: #2ecc71; }
                .decision-rejected { background: #fde0dc; color: #9f241b; border-color: #e74c3c; }
                .decision-skipped { background: #fff1cc; color: #8a5a00; border-color: #f39c12; }
                .decision-unknown { background: #eef; color: #333; border-color: #99f; }

                .review-status {
                    border-left: 6px solid #999;
                    font-weight: 600;
                    color: #555;}

                .review-accepted {
                    background: #d8f5df;
                    color: #176b2c;
                    border: 1px solid #2ecc71;
                }

                .review-rejected {
                    background: #fde0dc;
                    color: #9f241b;
                    border: 1px solid #e74c3c;
                }

                .review-skipped {
                    background: #fff1cc;
                    color: #8a5a00;
                    border: 1px solid #f39c12;
                }

                .button-row {
                    display: flex;
                    gap: 8px;
                    align-items: center;
                    flex-wrap: nowrap;
                }

                .button-row button {
                    margin-right: 0;
                    white-space: nowrap;
                }

                .stats-bar {
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                    margin: 12px 0 16px 0;
                }

                .stat-card {
                    background: white;
                    padding: 10px 14px;
                    border-radius: 8px;
                    min-width: 120px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.15);
                    text-align: center;
                }

                .stat-link {
                    text-decoration: none;
                    color: inherit;
                    display: block;
                }

                .stat-link:hover {
                    outline: 2px solid #34495e;
                }

                .active-filter {
                    background: #1abc9c !important;
                    color: white !important;
                    font-weight: bold;
                }

                .filter-panel {
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                    align-items: center;
                }

                .filter-panel form {
                    display: inline-flex;
                    gap: 8px;
                    flex-wrap: wrap;
                    align-items: center;
                    margin: 0 0 0 8px;
                }

                .stat-action {
                    border: none;
                    font: inherit;
                }

                .stat-action-button {
                    background: transparent;
                    color: inherit;
                    border: none;
                    font-weight: 800;
                    font-size: 15px;
                    cursor: pointer;
                    padding: 0;
                    margin: 0;
                    width: 100%;
                }

                .manual-page-wrap {
                    position: relative;
                    display: inline-block;
                    max-width: 100%;
                }

                .manual-page-wrap img {
                    user-select: none;
                    -webkit-user-drag: none;
                }

                .manual-selection-box {
                    position: absolute;
                    border: 3px solid #00a8ff;
                    background: rgba(0, 168, 255, 0.16);
                    pointer-events: none;
                    display: none;
                    z-index: 10;
                }

                .manual-crop-panel {
                    margin-top: 14px;
                    background: #f4fbff;
                    border-left: 5px solid #00a8ff;
                }

                .coord-grid {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(60px, 1fr));
                    gap: 8px;
                }

                .coord-grid input {
                    font-size: 13px;
                }

                .zoomable {
                    cursor: zoom-in;
                }

                .mini-link {
                    display: inline-block;
                    margin: 6px 0 8px 0;
                    font-size: 13px;
                    color: #34495e;
                    font-weight: 700;
                }

                .badge-link {
                    text-decoration: none;
                }

                .type-filter-panel {
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                    align-items: center;
                }

                .type-filter-title {
                    font-weight: 800;
                    margin-right: 4px;
                }

                .similar-card .button-row {
                    flex-wrap: wrap;
                }

                .similar-card .button-row button {
                    flex: 1 1 90px;
                    padding-left: 8px;
                    padding-right: 8px;
                }

                .good-stat {
                    border-left: 5px solid #2ecc71;
                }

                .bad-stat {
                    border-left: 5px solid #e74c3c;
                }

                .stats-details {
                    background: white;
                    padding: 10px 14px;
                    border-radius: 8px;
                    margin-bottom: 18px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.12);
                }

                .stats-details summary {
                    cursor: pointer;
                    font-weight: bold;
                }

                .stats-columns {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(160px, 1fr));
                    gap: 12px;
                }

                .stats-columns ul {
                    padding-left: 18px;
                    margin-top: 4px;
                }



                .stat-subtext {
                    display: block;
                    margin-top: 2px;
                    font-size: 12px;
                    color: #555;
                    font-weight: normal;
                }

                .instructions {
                    margin-top: 16px;
                    background: #fffdf4;
                    border-left: 5px solid #f1c40f;
                }

                .instructions h3 {
                    margin-top: 0;
                }

                .instructions-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 12px;
                }

                .instructions ul {
                    margin-top: 6px;
                    padding-left: 18px;
                }

                .similar-list {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                    gap: 12px;
                }

                .vae-note {
                    margin-top: -4px;
                    color: #555;
                    font-size: 13px;
                    line-height: 1.35;
                }

                @media (max-width: 1650px) {
                    .container {
                        grid-template-columns: minmax(320px, 0.9fr) minmax(520px, 1.5fr) minmax(340px, 1fr);
                    }
                    .right-panel,
                    .vae-panel {
                        grid-column: 1 / -1;
                    }
                }


                @media (max-width: 1000px) {
                    .stats-columns {
                        grid-template-columns: 1fr 1fr;
                    }
                }

                @media (max-width: 650px) {
                    .stats-columns {
                        grid-template-columns: 1fr;
                    }
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


            

            <div class="stats-bar">
                <a class="stat-card stat-link {% if filter_name == 'all' %}active-filter{% endif %}" href="{{ url_for('index', filter='all', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Total crops</b><br>
                    {{ review_stats.total_crops }}
                </a>

                <a class="stat-card stat-link {% if filter_name == 'reviewed' %}active-filter{% endif %}" href="{{ url_for('index', filter='reviewed', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Reviewed</b><br>
                    {{ review_stats.reviewed_total }}
                    <span class="stat-subtext">accepted + rejected</span>
                </a>

                <a class="stat-card stat-link {% if filter_name == 'accepted' %}active-filter{% endif %}" href="{{ url_for('index', filter='accepted', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Accepted</b><br>
                    {{ review_stats.accepted_total }}
                </a>

                <a class="stat-card stat-link {% if filter_name == 'rejected' %}active-filter{% endif %}" href="{{ url_for('index', filter='rejected', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Rejected</b><br>
                    {{ review_stats.rejected_total }}
                </a>

                <a class="stat-card stat-link {% if filter_name == 'skipped' %}active-filter{% endif %}" href="{{ url_for('index', filter='skipped', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Skipped</b><br>
                    {{ review_stats.skipped_total }}
                </a>

                <a class="stat-card stat-link {% if filter_name == 'pending' %}active-filter{% endif %}" href="{{ url_for('index', filter='pending', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Pending</b><br>
                    {{ review_stats.pending_total }}
                </a>

                <a class="stat-card stat-link" href="{{ url_for('index', filter='all', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Progress</b><br>
                    {{ "%.1f"|format((review_stats.reviewed_total / review_stats.total_crops * 100) if review_stats.total_crops else 0) }}%
                </a>

                <a class="stat-card stat-link good-stat {% if filter_name == 'exportable' %}active-filter{% endif %}" href="{{ url_for('index', filter='exportable', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Exportable assets</b><br>
                    {{ review_stats.exportable_candidates }}
                    <span class="stat-subtext">accepted + good/minor bbox</span>
                </a>

                <a class="stat-card stat-link {% if filter_name == 'accepted_not_exportable' %}active-filter{% endif %}" href="{{ url_for('index', filter='accepted_not_exportable', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Accepted, not exportable</b><br>
                    {{ review_stats.accepted_not_exportable }}
                    <span class="stat-subtext">partial/unsure/bad bbox</span>
                </a>

                <a class="stat-card stat-link bad-stat {% if filter_name == 'rejected' %}active-filter{% endif %}" href="{{ url_for('index', filter='rejected', idx=0, type_field=type_field, type_value=type_value) }}">
                    <b>Rejected / false positives</b><br>
                    {{ review_stats.false_positives }}
                </a>

                <form class="stat-card stat-action" method="post" action="{{ url_for('build_review_indexes_route') }}">
                    <button class="stat-action-button" type="submit" name="export_package" value="0">Rebuild indexes</button>
                    <span class="stat-subtext">save filtered JSONL</span>
                </form>

                <form class="stat-card stat-action good-stat" method="post" action="{{ url_for('build_review_indexes_route') }}">
                    <button class="stat-action-button" type="submit" name="export_package" value="1">Export package</button>
                    <span class="stat-subtext">save retraining package</span>
                </form>
            </div>

            <div class="stats-details type-filter-panel">
                <span class="type-filter-title">Predicted types:</span>
                {% for key, value in review_stats.predicted_types.items() %}
                    <a class="navlink {% if type_field == 'predicted' and type_value == key %}active-filter{% endif %}"
                       href="{{ url_for('index', filter='all', type_field='predicted', type_value=key, idx=0) }}">{{ key }} ({{ value }})</a>
                {% endfor %}

                <span class="type-filter-title">Effective types:</span>
                {% for key, value in review_stats.effective_types.items() %}
                    <a class="navlink {% if type_field == 'effective' and type_value == key %}active-filter{% endif %}"
                       href="{{ url_for('index', filter='all', type_field='effective', type_value=key, idx=0) }}">{{ key }} ({{ value }})</a>
                {% endfor %}

                {% if type_value %}
                    <a class="navlink" href="{{ url_for('index', filter=filter_name, idx=0) }}">Clear type filter</a>
                {% endif %}
            </div>

            <div class="stats-details">
                <details>
                    <summary>Review statistics</summary>

                    <div class="stats-columns">
                        <div>
                            <h4>Decisions</h4>
                            <ul>
                                {% for key, value in review_stats.decisions.items() %}
                                <li>{{ key }}: {{ value }}</li>
                                {% endfor %}
                            </ul>
                        </div>

                        <div>
                            <h4>Reviewed types</h4>
                            <ul>
                                {% for key, value in review_stats.reviewed_types.items() %}
                                <li>{{ key }}: {{ value }}</li>
                                {% endfor %}
                            </ul>
                        </div>

                        <div>
                            <h4>BBox quality</h4>
                            <ul>
                                {% for key, value in review_stats.bbox_qualities.items() %}
                                <li>{{ key }}: {{ value }}</li>
                                {% endfor %}
                            </ul>
                        </div>

                        <div>
                            <h4>Attributes</h4>
                            <ul>
                                {% for key, value in review_stats.attributes.items() %}
                                <li>{{ key }}: {{ value }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                    </div>
                </details>
            </div>


            <div class="topbar">
                <a class="navlink" href="{{ url_for('index', idx=prev_idx, filter=filter_name, type_field=type_field, type_value=type_value) }}">← Previous</a>
                <a class="navlink" href="{{ url_for('index', idx=next_idx, filter=filter_name, type_field=type_field, type_value=type_value) }}">Next →</a>
                <span>Item {{ idx + 1 }} / {{ total }} · Filter: {{ filters[filter_name] }}</span>
            </div>

      
            <div class="container">

                <!-- COLUMN 1: crop + metadata + form -->
                <div class="card left-panel">                    
                
                    <h2>Crop</h2>
                    <a href="{{ url_for('crop_image', crop_id=item['crop_id']) }}" target="_blank" title="Open crop full size">
                        <img class="crop-img zoomable" src="{{ url_for('crop_image', crop_id=item['crop_id']) }}">
                    </a>
                    {% if previous_review %}
                    <div class="status review-status review-{{ previous_review.get('decision') }}">
                        <b>Already reviewed:</b>
                        {{ previous_review.get("decision") }}
                        as {{ previous_review.get("reviewed_type") }}
                    </div>
                    {% endif %}
                    

                    <h2>Metadata</h2>
                    <p><b>Crop ID:</b> {{ item.get("crop_id") }}</p>
                    <p><b>Predicted type:</b> <a class="badge-link" href="{{ url_for('index', filter=filter_name, type_field='predicted', type_value=item.get('type'), idx=0) }}"><span class="meta-badge type-{{ item.get('type') or 'unknown' }}">{{ item.get("type") }}</span></a></p>

                    {% if previous_review %}
                    <p><b>Review decision:</b> <span class="decision-badge decision-{{ previous_review.get('decision') or 'unknown' }}">{{ previous_review.get("decision") }}</span></p>
                    <p><b>Reviewed type:</b> <span class="meta-badge type-{{ previous_review.get('reviewed_type') or 'unknown' }}">{{ previous_review.get("reviewed_type") }}</span></p>
                    <p><b>BBox quality:</b> {{ previous_review.get("bbox_quality") }}</p>
                    <p><b>Human confidence:</b> {{ previous_review.get("human_confidence") }}</p>

                    {% if previous_review.get("review_notes") %}
                    <p><b>Review notes:</b> {{ previous_review.get("review_notes") }}</p>
                    {% endif %}
                    {% endif %}

                    <p><b>Effective type:</b> <a class="badge-link" href="{{ url_for('index', filter=filter_name, type_field='effective', type_value=current_type, idx=0) }}"><span class="meta-badge type-{{ current_type or 'unknown' }}">{{ current_type }}</span></a></p>
                    <p><b>Confidence:</b> {{ item.get("confidence") }}</p>
                    <p><b>Document:</b> {{ item.get("document_id") }}</p>
                    <p><b>BBox:</b> {{ item.get("bbox") }}</p>
                    {% set corrected_bbox = previous_review.get("corrected_bbox") if previous_review else None %}
                    {% if corrected_bbox %}
                    <p><b>Corrected BBox:</b> {{ corrected_bbox }}</p>
                    {% endif %}

                    <details class="manual-crop-panel">
                        <summary>Optional corrected bbox for YOLO export</summary>
                        <p class="vae-note">
                            Fill these only if the original bbox should be replaced during YOLO export.
                            Empty fields keep the original bbox.
                        </p>
                        <div class="manual-bbox-fields">
                            <label>x1<input name="corrected_x1" value="{{ corrected_bbox.get('x1', '') if corrected_bbox else '' }}"></label>
                            <label>y1<input name="corrected_y1" value="{{ corrected_bbox.get('y1', '') if corrected_bbox else '' }}"></label>
                            <label>x2<input name="corrected_x2" value="{{ corrected_bbox.get('x2', '') if corrected_bbox else '' }}"></label>
                            <label>y2<input name="corrected_y2" value="{{ corrected_bbox.get('y2', '') if corrected_bbox else '' }}"></label>
                        </div>
                    </details>

                    <form method="post" action="{{ url_for('review') }}">
                        <input type="hidden" name="crop_id" value="{{ item.get('crop_id') }}">
                        <input type="hidden" name="idx" value="{{ idx }}">
                        <input type="hidden" name="filter" value="{{ filter_name }}">
                        <input type="hidden" name="type_field" value="{{ type_field }}">
                        <input type="hidden" name="type_value" value="{{ type_value }}">

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
                            <option value="" {% if current_human_confidence == "" %}selected{% endif %}>not specified</option>
                            {% for conf in review_schema.human_confidence %}
                                <option value="{{ conf }}" {% if conf == current_human_confidence %}selected{% endif %}>
                                    {{ conf }}
                                </option>
                            {% endfor %}
                        </select>

                        <br>

                        <label>BBox quality:</label><br>
                        <select name="bbox_quality">
                            <option value="" {% if current_bbox_quality == "" %}selected{% endif %}>not specified</option>
                            {% for q in review_schema.bbox_quality %}
                                <option value="{{ q }}" {% if q == current_bbox_quality %}selected{% endif %}>
                                    {{ q }}
                                </option>
                            {% endfor %}
                        </select>

                        <br>

                        <label>Attributes:</label><br>
                        <div class="attributes-grid">
                            {% for attr in review_schema.attributes %}
                                <label class="attribute-item">
                                    <input
                                        type="checkbox"
                                        name="attributes"
                                        value="{{ attr }}"
                                        {% if attr in current_attributes %}checked{% endif %}
                                    >
                                    <span>{{ attr }}</span>
                                </label>
                            {% endfor %}
                        </div>

                        <label>Notes:</label><br>
                        <textarea
                            name="notes"
                            rows="3"
                            placeholder="bbox partial, too large, false positive..."
                        >{{ current_notes }}</textarea>

                        <br><br>

                        <div class="button-row">
                            <button class="accept" name="decision" value="accepted">Accept</button>
                            <button class="reject" name="decision" value="rejected">Reject</button>
                            <button class="skip" name="decision" value="skipped">Skip</button>
                        </div>
                    </form>

                    <hr>
                    <h3>Generator assets</h3>
                    <p class="vae-note">
                        Copy this crop into <code>assets_real_reviewed/</code> so <code>generator.py</code> can reuse it
                        when creating future synthetic documents. This does not change the review decision.
                    </p>
                    <form method="post" action="{{ url_for('send_to_generator_assets') }}">
                        <input type="hidden" name="crop_id" value="{{ item.get('crop_id') }}">
                        <input type="hidden" name="idx" value="{{ idx }}">
                        <input type="hidden" name="filter" value="{{ filter_name }}">
                        <input type="hidden" name="type_field" value="{{ type_field }}">
                        <input type="hidden" name="type_value" value="{{ type_value }}">

                        <label>Send as asset type:</label><br>
                        <select name="asset_type">
                            {% for asset_type, folder in generator_asset_targets.items() %}
                                <option value="{{ asset_type }}" {% if asset_type == current_type %}selected{% endif %}>
                                    {{ asset_type }} → assets_real_reviewed/{{ folder }}/
                                </option>
                            {% endfor %}
                        </select>

                        <button class="navlink" type="submit">Send to generator assets</button>
                    </form>
                </div>


                <!-- COLUMN 2: full page -->
                <div class="card page-panel">
                    <h2>Full page with bbox</h2>
                    <p class="vae-note">Draw a rectangle on the page to create a manual crop/bbox.</p>
                    <div class="manual-page-wrap" id="manual-page-wrap">
                        <img id="manual-page-img" class="page-img" src="{{ url_for('page_preview', crop_id=item['crop_id']) }}">
                        <div id="manual-selection-box" class="manual-selection-box"></div>
                    </div>
                    <br>
                    <a class="mini-link" href="{{ url_for('page_preview', crop_id=item['crop_id']) }}" target="_blank">Open full page preview</a>

                    <div class="card manual-crop-panel">
                        <h3>Manual bbox / crop selector</h3>
                        <p class="vae-note">
                            Drag over the page image, choose a class, then save. The crop is saved as accepted/good and can optionally be copied to generator assets.
                        </p>
                        <form method="post" action="{{ url_for('save_manual_crop') }}" id="manual-crop-form">
                            <input type="hidden" name="source_crop_id" value="{{ item.get('crop_id') }}">
                            <input type="hidden" name="idx" value="{{ idx }}">
                            <input type="hidden" name="filter" value="{{ filter_name }}">
                            <input type="hidden" name="type_field" value="{{ type_field }}">
                            <input type="hidden" name="type_value" value="{{ type_value }}">

                            <div class="coord-grid">
                                <label>x1<input id="manual-x1" name="x1" readonly></label>
                                <label>y1<input id="manual-y1" name="y1" readonly></label>
                                <label>x2<input id="manual-x2" name="x2" readonly></label>
                                <label>y2<input id="manual-y2" name="y2" readonly></label>
                            </div>

                            <label>Manual class:</label><br>
                            <select name="manual_type">
                                {% for cls in review_schema.classes %}
                                    {% if cls != "false_positive" %}
                                    <option value="{{ cls }}" {% if cls == current_type %}selected{% endif %}>{{ cls }}</option>
                                    {% endif %}
                                {% endfor %}
                            </select>

                            <label>Notes:</label><br>
                            <textarea name="notes" rows="2" placeholder="manual bbox; clean stamp; block of typewritten text...">manual bbox from review app</textarea>

                            <label class="attribute-item">
                                <input type="checkbox" name="send_to_assets" value="1">
                                <span>Also send to generator assets</span>
                            </label>

                            <div class="button-row">
                                <button class="accept" type="submit">Save manual crop</button>
                                <button class="skip" type="button" id="manual-clear-selection">Clear selection</button>
                            </div>
                        </form>

                        <p>
                            <a class="mini-link" href="{{ url_for('manual_crops_gallery') }}" target="_blank">Open manual crops gallery</a>
                            · folder: <code>outputs/object_crops_manual/</code>
                        </p>

                        {% if last_manual_crop %}
                        <div class="similar-card">
                            <h4>Last manual crop created</h4>
                            <a href="{{ url_for('crop_image_by_id', crop_id=last_manual_crop.get('crop_id')) }}" target="_blank">
                                <img class="similar-img" src="{{ url_for('crop_image_by_id', crop_id=last_manual_crop.get('crop_id')) }}">
                            </a>
                            <p><b>{{ last_manual_crop.get('crop_id') }}</b></p>
                            <p>{{ last_manual_crop.get('type') }} · {{ last_manual_crop.get('crop_path') }}</p>
                        </div>
                        {% endif %}
                    </div>

                    <div class="card instructions">
                        <h3>Review instructions</h3>
                        <div class="instructions-grid">
                            <div>
                                <b>Main crop</b>
                                <ul>
                                    <li><b>Accept</b> only if the class is correct and the bbox is useful.</li>
                                    <li><b>Reject</b> if it is a false positive, a bad/huge bbox, or a mixed region that cannot be reused.</li>
                                    <li><b>Skip</b> if you are not sure and want to decide later.</li>
                                </ul>
                            </div>
                            <div>
                                <b>Export logic</b>
                                <ul>
                                    <li><b>Exportable</b> = accepted + bbox_quality <code>good</code>/<code>minor_partial</code> + valid class.</li>
                                    <li>Accepted crops with <code>partial</code>, <code>too_large</code> or <code>unsure</code> are reviewed but not clean export assets.</li>
                                    <li>Rejected crops count as rejected/false positives.</li>
                                </ul>
                            </div>
                            <div>
                                <b>Attributes</b>
                                <ul>
                                    <li>Use visual flags such as <code>faded</code>, <code>low_contrast</code>, <code>rotated</code>, <code>stain</code> or <code>background_noise</code>.</li>
                                    <li>Use <code>mixed</code> when several phenomena overlap and no single attribute explains the crop well.</li>
                                    <li>Use notes for the reason: bbox partial, too large, false positive, mixed content, etc.</li>
                                </ul>
                            </div>
                            <div>
                                <b>FAISS similar detector crops</b>
                                <ul>
                                    <li>They are quick binary reviews from visual retrieval.</li>
                                    <li>Accept similar is a weak review from crop-only context; it is saved with bbox_quality <code>unsure</code>.</li>
                                    <li>Reject similar marks it directly as false positive.</li>
                                    <li>Use <b>Open page context</b> or Skip similar when the crop needs full-page context.</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- COLUMN 3: JSONs -->
                <div class="card json-panel">
                    <h2>JSON</h2>

                    <h3>Review JSON</h3>
                    <pre class="review-json">{{ review_json }}</pre>

                    <h3>Raw prediction JSON</h3>
                    <pre>{{ raw_json }}</pre>
                </div>


                <!-- COLUMN 4: similar crops -->
                <div class="card right-panel">
                    <h2>FAISS similar detector crops</h2>

                    {% if similar_items %}
                        <div class="similar-list">
                        {% for sim in similar_items %}
                            <div class="similar-card">
                                <p>
                                    <b>#{{ sim.rank }}</b>
                                    score={{ "%.4f"|format(sim.score) }}
                                    <br>
                                    <b>Predicted:</b> <a class="badge-link" href="{{ url_for('index', filter=filter_name, type_field='predicted', type_value=sim.get('type'), idx=0) }}"><span class="meta-badge type-{{ sim.get('type') or 'unknown' }}">{{ sim.get("type") }}</span></a>
                                    <br>
                                    <b>Effective:</b> <a class="badge-link" href="{{ url_for('index', filter=filter_name, type_field='effective', type_value=sim.get('effective_type'), idx=0) }}"><span class="meta-badge type-{{ sim.get('effective_type') or 'unknown' }}">{{ sim.get("effective_type") }}</span></a>
                                    <br>
                                    crop={{ sim.get("crop_id") }}
                                    <br>
                                    conf={{ sim.get("confidence") }}                                    
                                </p>

                                <a href="{{ url_for('similar_crop_image', faiss_id=sim.faiss_id) }}" target="_blank" title="Open similar crop full size">
                                    <img
                                        class="similar-img zoomable"
                                        src="{{ url_for('similar_crop_image', faiss_id=sim.faiss_id) }}"
                                    >
                                </a>
                                <a class="mini-link" href="{{ url_for('page_preview', crop_id=sim.get('crop_id')) }}" target="_blank">Open page context</a>

                                {% if sim.previous_review %}
                                    
                                <span class="review-badge review-{{ sim.previous_review.get('decision') }}">
                                    Reviewed: {{ sim.previous_review.get("decision") }}
                                    {% if sim.previous_review.get("reviewed_type") %}
                                        as {{ sim.previous_review.get("reviewed_type") }}
                                    {% endif %}
                                </span>
                                <br>
                                <br>
                                {% endif %}

                                

                                <form method="post" action="{{ url_for('review_similar') }}">
                                    <input type="hidden" name="faiss_id" value="{{ sim.faiss_id }}">
                                    <input type="hidden" name="idx" value="{{ idx }}">
                                    <input type="hidden" name="filter" value="{{ filter_name }}">
                                    <input type="hidden" name="type_field" value="{{ type_field }}">
                                    <input type="hidden" name="type_value" value="{{ type_value }}">
                                    <label>Class:</label><br>
                                    <select name="new_type">
                                        {% for cls in review_schema.classes %}
                                            <option value="{{ cls }}" {% if cls == sim.get("effective_type") %}selected{% endif %}>
                                                {{ cls }}
                                            </option>
                                        {% endfor %}
                                    </select>

                                    <input type="hidden" name="bbox_quality" value="unsure">

                                    <div class="button-row">
                                        <button class="accept" name="decision" value="accepted">Accept similar</button>
                                        <button class="reject" name="decision" value="rejected">Reject similar</button>
                                        <button class="skip" name="decision" value="skipped">Skip similar</button>
                                    </div>
                                </form>
                            </div>
                        {% endfor %}
                        </div>
                    {% else %}
                        <p>No FAISS similar results available.</p>
                    {% endif %}
                </div>


                <!-- COLUMN 5: VAE similar detector crops -->
                <div class="card vae-panel">
                    <h2>VAE similar detector crops</h2>
                    <p class="vae-note">
                        Uses the trained VAE latent space + FAISS. When available, it searches in the by-type index
                        first, so this column is useful to compare against the simple visual FAISS column.
                    </p>

                    {% if vae_similar_items %}
                        <div class="similar-list">
                        {% for sim in vae_similar_items %}
                            <div class="similar-card">
                                <p>
                                    <b>#{{ sim.rank }}</b>
                                    score={{ "%.4f"|format(sim.score) }}
                                    <br>
                                    <small>{{ sim.get("similarity_source") }}</small>
                                    <br>
                                    <b>Predicted:</b> <a class="badge-link" href="{{ url_for('index', filter=filter_name, type_field='predicted', type_value=sim.get('type'), idx=0) }}"><span class="meta-badge type-{{ sim.get('type') or 'unknown' }}">{{ sim.get("type") }}</span></a>
                                    <br>
                                    <b>Effective:</b> <a class="badge-link" href="{{ url_for('index', filter=filter_name, type_field='effective', type_value=sim.get('effective_type'), idx=0) }}"><span class="meta-badge type-{{ sim.get('effective_type') or 'unknown' }}">{{ sim.get("effective_type") }}</span></a>
                                    <br>
                                    crop={{ sim.get("crop_id") }}
                                    <br>
                                    conf={{ sim.get("confidence") }}
                                </p>

                                <a href="{{ url_for('crop_image_by_id', crop_id=sim.get('crop_id')) }}" target="_blank" title="Open VAE similar crop full size">
                                    <img
                                        class="similar-img zoomable"
                                        src="{{ url_for('crop_image_by_id', crop_id=sim.get('crop_id')) }}"
                                    >
                                </a>
                                <a class="mini-link" href="{{ url_for('page_preview', crop_id=sim.get('crop_id')) }}" target="_blank">Open page context</a>

                                {% if sim.previous_review %}
                                <span class="review-badge review-{{ sim.previous_review.get('decision') }}">
                                    Reviewed: {{ sim.previous_review.get("decision") }}
                                    {% if sim.previous_review.get("reviewed_type") %}
                                        as {{ sim.previous_review.get("reviewed_type") }}
                                    {% endif %}
                                </span>
                                <br>
                                <br>
                                {% endif %}

                                <form method="post" action="{{ url_for('review_similar_crop') }}">
                                    <input type="hidden" name="crop_id" value="{{ sim.get('crop_id') }}">
                                    <input type="hidden" name="idx" value="{{ idx }}">
                                    <input type="hidden" name="filter" value="{{ filter_name }}">
                                    <input type="hidden" name="type_field" value="{{ type_field }}">
                                    <input type="hidden" name="type_value" value="{{ type_value }}">
                                    <label>Class:</label><br>
                                    <select name="new_type">
                                        {% for cls in review_schema.classes %}
                                            <option value="{{ cls }}" {% if cls == sim.get("effective_type") %}selected{% endif %}>
                                                {{ cls }}
                                            </option>
                                        {% endfor %}
                                    </select>

                                    <input type="hidden" name="bbox_quality" value="unsure">

                                    <div class="button-row">
                                        <button class="accept" name="decision" value="accepted">Accept VAE</button>
                                        <button class="reject" name="decision" value="rejected">Reject VAE</button>
                                        <button class="skip" name="decision" value="skipped">Skip VAE</button>
                                    </div>
                                </form>
                            </div>
                        {% endfor %}
                        </div>
                    {% else %}
                        <p>No VAE FAISS similar results available.</p>
                        <p class="vae-note">Run <code>visual_search/build_vae_faiss.py</code> first.</p>
                    {% endif %}
                </div>

            </div>
            <script>
            (function() {
                const img = document.getElementById('manual-page-img');
                const wrap = document.getElementById('manual-page-wrap');
                const box = document.getElementById('manual-selection-box');
                const clearBtn = document.getElementById('manual-clear-selection');
                const form = document.getElementById('manual-crop-form');
                const x1Input = document.getElementById('manual-x1');
                const y1Input = document.getElementById('manual-y1');
                const x2Input = document.getElementById('manual-x2');
                const y2Input = document.getElementById('manual-y2');

                if (!img || !wrap || !box || !form) return;

                let dragging = false;
                let start = null;

                function pointFromEvent(ev) {
                    const rect = img.getBoundingClientRect();
                    const clientX = ev.clientX;
                    const clientY = ev.clientY;
                    const xCss = Math.max(0, Math.min(clientX - rect.left, rect.width));
                    const yCss = Math.max(0, Math.min(clientY - rect.top, rect.height));
                    const scaleX = img.naturalWidth / rect.width;
                    const scaleY = img.naturalHeight / rect.height;
                    return {
                        cssX: xCss,
                        cssY: yCss,
                        x: Math.round(xCss * scaleX),
                        y: Math.round(yCss * scaleY)
                    };
                }

                function drawBox(a, b) {
                    const left = Math.min(a.cssX, b.cssX);
                    const top = Math.min(a.cssY, b.cssY);
                    const width = Math.abs(a.cssX - b.cssX);
                    const height = Math.abs(a.cssY - b.cssY);
                    box.style.left = left + 'px';
                    box.style.top = top + 'px';
                    box.style.width = width + 'px';
                    box.style.height = height + 'px';
                    box.style.display = 'block';
                }

                function setInputs(a, b) {
                    x1Input.value = Math.min(a.x, b.x);
                    y1Input.value = Math.min(a.y, b.y);
                    x2Input.value = Math.max(a.x, b.x);
                    y2Input.value = Math.max(a.y, b.y);
                }

                img.addEventListener('pointerdown', function(ev) {
                    ev.preventDefault();
                    dragging = true;
                    start = pointFromEvent(ev);
                    drawBox(start, start);
                    img.setPointerCapture(ev.pointerId);
                });

                img.addEventListener('pointermove', function(ev) {
                    if (!dragging || !start) return;
                    ev.preventDefault();
                    const current = pointFromEvent(ev);
                    drawBox(start, current);
                    setInputs(start, current);
                });

                img.addEventListener('pointerup', function(ev) {
                    if (!dragging || !start) return;
                    ev.preventDefault();
                    const end = pointFromEvent(ev);
                    drawBox(start, end);
                    setInputs(start, end);
                    dragging = false;
                    start = null;
                    try { img.releasePointerCapture(ev.pointerId); } catch(e) {}
                });

                clearBtn.addEventListener('click', function() {
                    box.style.display = 'none';
                    x1Input.value = '';
                    y1Input.value = '';
                    x2Input.value = '';
                    y2Input.value = '';
                });

                form.addEventListener('submit', function(ev) {
                    if (!x1Input.value || !y1Input.value || !x2Input.value || !y2Input.value) {
                        ev.preventDefault();
                        alert('Draw a bbox on the page before saving a manual crop.');
                    }
                });
            })();
            </script>
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
        vae_similar_items=vae_similar_items,
        message=message,
        review_schema=review_schema,
        generator_asset_targets=GENERATOR_ASSET_TARGETS,
        last_manual_crop=last_manual_crop,
        current_human_confidence=current_human_confidence,
        current_bbox_quality=current_bbox_quality,
        current_notes=current_notes,
        current_attributes=current_attributes,
        review_stats=review_stats,
        filter_name=filter_name,
        filters=FILTERS,
        type_field=type_field,
        type_value=type_value,
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



@app.route("/manual_crop_image/<crop_id>")
def manual_crop_image(crop_id):
    """
    Serve a manual crop image directly from manual crop metadata.

    This avoids depending on the active REVIEW_METADATA, because manual
    crops live in outputs/object_crops_manual/metadata.jsonl.
    """
    for item in load_manual_items():
        if item.get("crop_id") == crop_id:
            crop_path_raw = item.get("crop_path")
            if not crop_path_raw:
                return "Manual crop has no crop_path", 404

            crop_path = project_path(crop_path_raw)

            if crop_path.exists():
                return send_file(crop_path)

            return "Manual crop file not found", 404

    return "Manual crop not found", 404


@app.route("/manual_crops")
def manual_crops_gallery():
    """Galeria simple dels crops creats manualment."""
    selected_type = request.args.get("type") or ""
    items = load_manual_items()
    if selected_type:
        items = [item for item in items if item.get("type") == selected_type]

    types = sorted({item.get("type", "unknown") for item in load_manual_items()})

    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
            <title>Manual crops gallery</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #111; }
                .topbar { margin-bottom: 16px; }
                .navlink { background: #34495e; color: white; text-decoration: none; padding: 8px 12px; border-radius: 5px; display: inline-block; margin: 4px; }
                .active { background: #1abc9c; font-weight: bold; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
                .card { background: white; padding: 12px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.14); }
                img { width: 100%; max-height: 190px; object-fit: contain; background: #ddd; border: 1px solid #333; }
                code { background: #eee; padding: 2px 4px; }
                .small { font-size: 12px; color: #555; word-break: break-all; }
            </style>
        </head>
        <body>
            <h1>Manual crops gallery</h1>
            <p><b>Folder:</b> <code>{{ manual_dir }}</code></p>
            <p><b>Metadata:</b> <code>{{ metadata_path }}</code></p>

            <div class="topbar">
                <a class="navlink" href="{{ url_for('index') }}">← Back to review app</a>
                <a class="navlink {% if not selected_type %}active{% endif %}" href="{{ url_for('manual_crops_gallery') }}">All ({{ all_count }})</a>
                {% for t in types %}
                    <a class="navlink {% if selected_type == t %}active{% endif %}" href="{{ url_for('manual_crops_gallery', type=t) }}">{{ t }}</a>
                {% endfor %}
            </div>

            {% if items %}
            <div class="grid">
                {% for item in items|reverse %}
                <div class="card">
                    <a href="{{ url_for('manual_crop_image', crop_id=item.get('crop_id')) }}" target="_blank">
                        <img src="{{ url_for('manual_crop_image', crop_id=item.get('crop_id')) }}">
                    </a>
                    <p><b>{{ item.get('crop_id') }}</b></p>
                    <p>{{ item.get('type') }} · bbox={{ item.get('bbox') }}</p>
                    <p class="small">{{ item.get('crop_path') }}</p>
                    <a class="navlink" href="{{ url_for('page_preview', crop_id=item.get('crop_id')) }}" target="_blank">Open page context</a>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <p>No manual crops yet.</p>
            {% endif %}
        </body>
        </html>
        """,
        items=items,
        types=types,
        selected_type=selected_type,
        all_count=len(load_manual_items()),
        manual_dir=str(MANUAL_CROPS_DIR.relative_to(PROJECT_ROOT)),
        metadata_path=str(MANUAL_METADATA_PATH.relative_to(PROJECT_ROOT)),
    )


@app.route("/page/<crop_id>")
def page_preview(crop_id):
    """
    Serveix la pàgina completa amb bbox dibuixat.

    Supports both normal review crops and manual crops.
    """
    item = find_item_by_crop_id(crop_id)

    if item:
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
    new_type = normalize_class_name(request.form.get("new_type"))
    notes = request.form.get("notes")
    human_confidence = request.form.get("human_confidence")
    bbox_quality = request.form.get("bbox_quality")
    attributes = request.form.getlist("attributes")
    idx = int(request.form.get("idx", 0))
    filter_name = request.form.get("filter", "all")
    type_field = request.form.get("type_field", "")
    type_value = request.form.get("type_value", "")

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
        corrected_bbox=parse_corrected_bbox_from_form(request.form),
    )

    # Després de decidir, mantenim el filtre. En filtres dinàmics, quedar-se al mateix
    # índex evita saltar un element quan el crop revisat surt del filtre actual.
    next_idx_after_review = idx + 1 if filter_name == "all" else idx
    return redirect(url_for("index", idx=next_idx_after_review, filter=filter_name, type_field=type_field, type_value=type_value))


@app.route("/send_to_generator_assets", methods=["POST"])
def send_to_generator_assets():
    """Copia un crop revisat/manual a assets_real_reviewed/ per al generator.py."""
    crop_id = request.form.get("crop_id")
    asset_type = request.form.get("asset_type")
    idx = int(request.form.get("idx", 0))
    filter_name = request.form.get("filter", "all")
    type_field = request.form.get("type_field", "")
    type_value = request.form.get("type_value", "")

    item = find_item_by_crop_id(crop_id)
    if item is None:
        return f"Crop not found: {crop_id}", 404

    previous_review = load_review_entries().get(crop_id)

    try:
        dst = copy_crop_to_generator_assets(item, asset_type, previous_review=previous_review)
    except Exception as e:
        msg = f"Could not send crop {crop_id} to generator assets: {type(e).__name__}: {e}"
        return redirect(url_for("index", idx=idx, filter=filter_name, type_field=type_field, type_value=type_value, msg=msg))

    msg = f"Crop {crop_id} copied to generator assets: {dst.relative_to(PROJECT_ROOT)}"
    return redirect(url_for("index", idx=idx, filter=filter_name, type_field=type_field, type_value=type_value, msg=msg))


@app.route("/save_manual_crop", methods=["POST"])
def save_manual_crop():
    """Crea un crop manual a partir d'una bbox dibuixada sobre la pàgina."""
    source_crop_id = request.form.get("source_crop_id")
    manual_type = normalize_class_name(request.form.get("manual_type"))
    notes = request.form.get("notes") or "manual bbox from review app"
    send_to_assets = request.form.get("send_to_assets") == "1"
    idx = int(request.form.get("idx", 0))
    filter_name = request.form.get("filter", "all")
    type_field = request.form.get("type_field", "")
    type_value = request.form.get("type_value", "")

    source_item = find_item_by_crop_id(source_crop_id)
    if source_item is None:
        return f"Source crop not found: {source_crop_id}", 404

    try:
        bbox = {
            "x1": request.form.get("x1"),
            "y1": request.form.get("y1"),
            "x2": request.form.get("x2"),
            "y2": request.form.get("y2"),
        }
        manual_item, crop_path = create_manual_crop_from_bbox(
            source_item=source_item,
            bbox=bbox,
            manual_type=manual_type,
            notes=notes,
        )

        asset_msg = ""
        if send_to_assets and manual_type in GENERATOR_ASSET_TARGETS:
            dst = copy_crop_to_generator_assets(manual_item, manual_type, previous_review=manual_item)
            asset_msg = f" | sent to generator assets: {dst.relative_to(PROJECT_ROOT)}"

        msg = f"Manual crop created: {manual_item.get('crop_id')} → {crop_path.relative_to(PROJECT_ROOT)}{asset_msg}"

    except Exception as e:
        msg = f"Could not create manual crop: {type(e).__name__}: {e}"

    kwargs = {
        "idx": idx,
        "filter": filter_name,
        "type_field": type_field,
        "type_value": type_value,
        "msg": msg,
    }
    if "manual_item" in locals() and manual_item:
        kwargs["last_manual_crop_id"] = manual_item.get("crop_id")

    return redirect(url_for("index", **kwargs))



@app.route("/crop_by_id/<crop_id>")
def crop_image_by_id(crop_id):
    """
    Serve crop image by crop_id.

    Supports both active review metadata and manual crop metadata.
    """
    item = find_item_by_crop_id(crop_id)

    if not item:
        return "Crop not found", 404

    crop_path_raw = item.get("crop_path")
    if not crop_path_raw:
        return "Crop has no crop_path", 404

    crop_path = project_path(crop_path_raw)

    if crop_path.exists():
        return send_file(crop_path)

    return "Crop file not found", 404

    crop_path_raw = item.get("crop_path")
    if not crop_path_raw:
        return "Crop path not found", 404

    crop_path = project_path(crop_path_raw)

    if crop_path.exists():
        return send_file(crop_path)

    return "Crop file not found", 404


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
    bbox_quality_selected = request.form.get("bbox_quality") or "unsure"
    idx = int(request.form.get("idx", 0))
    filter_name = request.form.get("filter", "all")
    type_field = request.form.get("type_field", "")
    type_value = request.form.get("type_value", "")

    metadata = load_faiss_metadata()

    if faiss_id < 0 or faiss_id >= len(metadata):
        return f"Invalid FAISS id: {faiss_id}", 404

    item = metadata[faiss_id]
    crop_id = item.get("crop_id", "unknown")

    # Si acceptem, guardem la classe i la qualitat seleccionades.
    # Si rebutgem, ho marquem com a false_positive per no exportar-ho com asset bo.
    # Si fem skip, no el comptem com a exportable i el podem revisar més endavant.
    if decision == "rejected":
        reviewed_type = "false_positive"
        bbox_quality = "bad_location"
        msg = f"Similar crop {crop_id} rejected. Predicted type was {item.get('type')}"
    elif decision == "skipped":
        reviewed_type = new_type
        bbox_quality = bbox_quality_selected
        msg = f"Similar crop {crop_id} skipped for later review"
    else:
        reviewed_type = new_type
        bbox_quality = bbox_quality_selected
        msg = f"Similar crop {crop_id} accepted as {reviewed_type} with bbox_quality={bbox_quality}"

    save_review(
        item=item,
        decision=decision,
        new_type=reviewed_type,
        notes="reviewed from similar crop suggestion",
        human_confidence="medium",
        bbox_quality=bbox_quality,
        attributes=["similar_review"],
    )

    # Tornem al mateix crop principal, però amb missatge visible.
    return redirect(url_for("index", idx=idx, filter=filter_name, type_field=type_field, type_value=type_value, msg=msg))


@app.route("/review_similar_crop", methods=["POST"])
def review_similar_crop():
    """
    Permet acceptar/rebutjar/skip d'un crop similar identificat per crop_id.
    S'utilitza sobretot per resultats VAE, on el faiss_id depèn de l'índex by_type/global.
    """
    crop_id = request.form.get("crop_id")
    decision = request.form.get("decision")
    new_type = request.form.get("new_type")
    bbox_quality_selected = request.form.get("bbox_quality") or "unsure"
    idx = int(request.form.get("idx", 0))
    filter_name = request.form.get("filter", "all")
    type_field = request.form.get("type_field", "")
    type_value = request.form.get("type_value", "")

    item = find_item_by_crop_id(crop_id)

    if item is None:
        return f"Crop not found: {crop_id}", 404

    if decision == "rejected":
        reviewed_type = "false_positive"
        bbox_quality = "bad_location"
        msg = f"VAE similar crop {crop_id} rejected. Predicted type was {item.get('type')}"
    elif decision == "skipped":
        reviewed_type = new_type
        bbox_quality = bbox_quality_selected
        msg = f"VAE similar crop {crop_id} skipped for later review"
    else:
        reviewed_type = new_type
        bbox_quality = bbox_quality_selected
        msg = f"VAE similar crop {crop_id} accepted as {reviewed_type} with bbox_quality={bbox_quality}"

    save_review(
        item=item,
        decision=decision,
        new_type=reviewed_type,
        notes="reviewed from VAE similar crop suggestion",
        human_confidence="medium",
        bbox_quality=bbox_quality,
        attributes=["vae_similar_review"],
    )

    return redirect(url_for("index", idx=idx, filter=filter_name, type_field=type_field, type_value=type_value, msg=msg))


@app.route("/build_review_indexes", methods=["POST"])
def build_review_indexes_route():
    """Genera índexs JSONL filtrats i opcionalment un export package."""
    export_package = request.form.get("export_package") == "1"

    if not BUILD_INDEX_SCRIPT.exists():
        msg = "tools/review_tools/build_review_indexes.py not found. Ensure tools/review_tools exists in the project root."
        return redirect(url_for("index", msg=msg))

    cmd = [
        sys.executable,
        str(BUILD_INDEX_SCRIPT),
        "--project-root",
        str(PROJECT_ROOT),
    ]
    if export_package:
        cmd.append("--export-package")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        msg = "Index/export process timed out after 300 seconds. Run it from terminal for very large datasets."
        return redirect(url_for("index", msg=msg))

    if result.returncode != 0:
        msg = "Index/export failed. Check terminal logs or run tools/review_tools/build_review_indexes.py manually."
    else:
        stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        saved_lines = [line for line in stdout_lines if "written to:" in line]
        if export_package:
            msg = "Export package created. " + " | ".join(saved_lines[-2:])
        else:
            msg = "Review indexes rebuilt. " + " | ".join(saved_lines[-1:])

    return redirect(url_for("index", msg=msg))



if __name__ == "__main__":
    # App local.
    # Obre al navegador: http://127.0.0.1:5000
    app.run(debug=True, host="127.0.0.1", port=5000)