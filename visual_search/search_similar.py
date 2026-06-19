#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import faiss
except ImportError as exc:
    raise SystemExit("faiss is not installed. Install faiss-cpu or faiss-gpu first.") from exc

from vae_model import ConvVAE


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


def encode_image(project_root: Path, crop_path: str, model: ConvVAE, device: torch.device) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((model.image_size, model.image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(project_root / crop_path).convert("L")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
    z = mu.cpu().numpy()[0].astype(np.float32)
    return l2_normalize(z).reshape(1, -1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--crop-id", required=True)
    parser.add_argument("--index", default="outputs/faiss/vae/global/visual_index.faiss")
    parser.add_argument("--metadata", default="outputs/faiss/vae/global/metadata.jsonl")
    parser.add_argument("--model", default="outputs/vae/vae_best.pt")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    metadata = read_jsonl(project_root / args.metadata)
    candidates = [r for r in metadata if r.get("crop_id") == args.crop_id]
    if not candidates:
        raise SystemExit(f"crop_id not found: {args.crop_id}")
    query = candidates[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(project_root / args.model, device)
    q = encode_image(project_root, query["crop_path"], model, device)

    index = faiss.read_index(str(project_root / args.index))
    scores, ids = index.search(q, min(index.ntotal, args.top_k + 10))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = metadata[int(idx)]
        if row.get("crop_id") == args.crop_id:
            continue
        out = dict(row)
        out["score"] = float(score)
        out["rank"] = len(results) + 1
        results.append(out)
        if len(results) >= args.top_k:
            break

    print(json.dumps({"query": query, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
