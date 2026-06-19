import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# Extensions d'imatge acceptades.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_image_files(input_dir, classes=None):
    """
    Recorre una carpeta de crops i retorna imatges.

    Pot treballar amb estructura:
        outputs/object_crops_raw/
        ├── stamp/
        ├── handwritten_text/
        └── crossout/

    Si classes és None, agafa totes les imatges.
    Si classes té valors, només agafa imatges dins d'aquestes carpetes.
    """
    input_dir = Path(input_dir)

    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # La classe s'infereix del nom de la carpeta pare.
        # Exemple: outputs/object_crops_raw/stamp/stamp_000001.jpg -> stamp
        class_name = path.parent.name

        if classes is not None and class_name not in classes:
            continue

        yield path


def load_metadata(metadata_path):
    """
    Carrega el metadata.jsonl generat per crop_objects_from_layout.py.

    Retorna un diccionari:
        crop_path -> metadata original

    Això ens permet mantenir la connexió entre embedding, crop, document original,
    bbox, confidence, source, etc.
    """
    if metadata_path is None:
        return {}

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        print(f"WARNING: metadata file not found: {metadata_path}")
        return {}

    metadata_by_crop_path = {}

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            # Guardem tant el path tal com apareix com el path resolt.
            crop_path = item.get("crop_path")
            if crop_path:
                metadata_by_crop_path[crop_path] = item
                metadata_by_crop_path[str(Path(crop_path).resolve())] = item

    return metadata_by_crop_path


def l2_normalize(vector, eps=1e-8):
    """
    Normalitza un vector perquè tingui norma 1.

    Això és important per comparar embeddings amb distància L2 o similitud cosinus.
    """
    norm = np.linalg.norm(vector)

    if norm < eps:
        return vector

    return vector / norm


def image_to_embedding(image_path, thumbnail_size=64, edge_size=32, hist_bins=16):
    """
    Converteix un crop en un embedding visual simple.

    L'embedding combina:
        1. Miniatura RGB normalitzada.
        2. Histograma de color RGB.
        3. Miniatura de vores/contorns.

    No és tan potent com CLIP/DINO/VAE, però és:
        - ràpid,
        - sense entrenament,
        - sense dependències pesades,
        - suficient per començar a buscar similituds entre crops.
    """
    with Image.open(image_path).convert("RGB") as img:
        # 1) Miniatura RGB.
        # Captura forma i distribució visual global.
        thumb = img.resize((thumbnail_size, thumbnail_size))
        thumb_arr = np.asarray(thumb, dtype=np.float32) / 255.0
        thumb_feat = thumb_arr.flatten()

        # 2) Histograma de color per canal.
        # Ajuda a distingir tintes, segells, fons, etc.
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
        # Ajuda a capturar estructura: text manuscrit, formes de stamp, ratllats.
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.resize((edge_size, edge_size))
        edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
        edge_feat = edge_arr.flatten()

    # Concatenem totes les parts.
    embedding = np.concatenate([
        thumb_feat,
        hist_feat,
        edge_feat,
    ]).astype(np.float32)

    # Normalització final.
    embedding = l2_normalize(embedding)

    return embedding


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-crops",
        required=True,
        help="Folder with crop images, usually outputs/object_crops_raw/",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where embeddings.npy and metadata.jsonl will be saved",
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata.jsonl from crop_objects_from_layout.py",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional list of classes to embed, e.g. stamp handwritten_text crossout",
    )

    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=64,
        help="Size of RGB thumbnail used in the embedding",
    )

    parser.add_argument(
        "--edge-size",
        type=int,
        default=32,
        help="Size of edge thumbnail used in the embedding",
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=16,
        help="Number of bins per RGB channel histogram",
    )

    args = parser.parse_args()

    input_crops = Path(args.input_crops)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carreguem metadata original si existeix.
    metadata_by_crop_path = load_metadata(args.metadata)

    embeddings = []
    output_metadata = []

    image_paths = list(iter_image_files(input_crops, classes=args.classes))

    print(f"Input crops found: {len(image_paths)}")

    for embedding_id, image_path in enumerate(image_paths):
        # Calculem embedding visual.
        embedding = image_to_embedding(
            image_path=image_path,
            thumbnail_size=args.thumbnail_size,
            edge_size=args.edge_size,
            hist_bins=args.hist_bins,
        )

        embeddings.append(embedding)

        # Intentem recuperar metadata original del crop.
        original_meta = metadata_by_crop_path.get(str(image_path))
        if original_meta is None:
            original_meta = metadata_by_crop_path.get(str(image_path.resolve()))

        # Si no hi ha metadata original, creem una mínima.
        if original_meta is None:
            original_meta = {
                "crop_path": str(image_path),
                "type": image_path.parent.name,
            }

        # Afegim identificador intern de l'embedding.
        enriched_meta = dict(original_meta)
        enriched_meta["embedding_id"] = embedding_id
        enriched_meta["embedding_source"] = "simple_visual_embedding"
        enriched_meta["embedding_dim"] = int(embedding.shape[0])

        output_metadata.append(enriched_meta)

    if not embeddings:
        print("No embeddings created. Check --input-crops and --classes.")
        return

    embeddings_array = np.vstack(embeddings).astype(np.float32)

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"
    config_path = output_dir / "embedding_config.json"

    # Guardem embeddings en format NumPy.
    np.save(embeddings_path, embeddings_array)

    # Guardem metadata en JSONL, una línia per crop.
    with metadata_path.open("w", encoding="utf-8") as f:
        for item in output_metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Guardem configuració per reproduïbilitat.
    config = {
        "input_crops": str(input_crops),
        "metadata": args.metadata,
        "classes": args.classes,
        "thumbnail_size": args.thumbnail_size,
        "edge_size": args.edge_size,
        "hist_bins": args.hist_bins,
        "embedding_dim": int(embeddings_array.shape[1]),
        "num_embeddings": int(embeddings_array.shape[0]),
        "method": "simple_visual_embedding",
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved metadata:   {metadata_path}")
    print(f"Saved config:     {config_path}")
    print(f"Shape: {embeddings_array.shape}")


if __name__ == "__main__":
    main()
