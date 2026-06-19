#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image
from torchvision import transforms

try:
    import faiss
except ImportError as exc:
    raise SystemExit("faiss is not installed. Install faiss-cpu or faiss-gpu first.") from exc

from vae_model import ConvVAE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "outputs/faiss/vae/global/visual_index.faiss"
METADATA_PATH = PROJECT_ROOT / "outputs/faiss/vae/global/metadata.jsonl"
MODEL_PATH = PROJECT_ROOT / "outputs/vae/vae_best.pt"

app = Flask(__name__)
_state = {"model": None, "index": None, "metadata": None, "device": None}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), eps)


def load_model(model_path: Path, device: torch.device) -> ConvVAE:
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = ConvVAE(
        image_size=int(cfg.get("image_size", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        in_channels=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def ensure_loaded():
    if _state["metadata"] is not None:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device
    _state["model"] = load_model(MODEL_PATH, device)
    _state["index"] = faiss.read_index(str(INDEX_PATH))
    _state["metadata"] = read_jsonl(METADATA_PATH)


def encode_crop_path(crop_path: str) -> np.ndarray:
    ensure_loaded()
    model: ConvVAE = _state["model"]
    device = _state["device"]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((model.image_size, model.image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(PROJECT_ROOT / crop_path).convert("L")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
    z = mu.cpu().numpy()[0].astype(np.float32)
    return l2_normalize(z).reshape(1, -1).astype(np.float32)


@app.route("/api/similar-crops/<crop_id>")
def similar_crops(crop_id: str):
    ensure_loaded()
    k = int(request.args.get("k", 10))
    filter_type = request.args.get("type")
    include_false_positives = request.args.get("include_false_positives", "0") == "1"

    metadata = _state["metadata"]
    matches = [row for row in metadata if row.get("crop_id") == crop_id]
    if not matches:
        return jsonify({"error": f"crop_id not found: {crop_id}"}), 404

    query = matches[0]
    q = encode_crop_path(query["crop_path"])
    index = _state["index"]
    scores, ids = index.search(q, min(index.ntotal, k + 100))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(metadata[int(idx)])
        if row.get("crop_id") == crop_id:
            continue
        if filter_type and (row.get("effective_type") or row.get("type")) != filter_type:
            continue
        if not include_false_positives and row.get("is_false_positive"):
            continue
        row["score"] = float(score)
        row["rank"] = len(results) + 1
        results.append(row)
        if len(results) >= k:
            break

    return jsonify({"query": query, "results": results})


@app.route("/api/crop-image/<crop_id>")
def crop_image(crop_id: str):
    ensure_loaded()
    metadata = _state["metadata"]
    matches = [row for row in metadata if row.get("crop_id") == crop_id]
    if not matches:
        return "Crop not found", 404
    path = PROJECT_ROOT / matches[0]["crop_path"]
    if not path.exists():
        return "Crop file not found", 404
    return send_file(path)


@app.route("/api/health")
def health():
    ensure_loaded()
    return jsonify({
        "status": "ok",
        "num_vectors": int(_state["index"].ntotal),
        "metadata_rows": len(_state["metadata"]),
        "index": str(INDEX_PATH),
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)
