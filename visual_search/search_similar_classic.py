import argparse
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
from PIL import Image, ImageFilter


# Mateixes extensions acceptades que en build_embeddings.py.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def l2_normalize(vector, eps=1e-8):
    """
    Normalitza un vector perquè tingui norma 1.

    Això és necessari quan l'índex FAISS s'ha creat amb mètrica cosine,
    perquè en realitat usem Inner Product sobre vectors normalitzats.
    """
    norm = np.linalg.norm(vector)

    if norm < eps:
        return vector

    return vector / norm


def image_to_embedding(image_path, thumbnail_size=64, edge_size=32, hist_bins=16):
    """
    Converteix una imatge/crop en un embedding visual simple.

    IMPORTANT:
    Aquesta funció ha de ser equivalent a la de build_embeddings.py.
    Si canviem com construïm embeddings allà, també ho hem de canviar aquí.

    L'embedding combina:
        1. Miniatura RGB.
        2. Histograma de color.
        3. Miniatura de vores/contorns.
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


def load_metadata(metadata_path):
    """
    Carrega metadata.jsonl en una llista.

    L'ordre és important:
    metadata[i] ha de correspondre a l'embedding amb id i.
    """
    items = []

    with Path(metadata_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            items.append(json.loads(line))

    return items


def load_faiss_config(index_dir):
    """
    Carrega faiss_config.json si existeix.

    Això ens permet saber si l'índex es va crear amb cosine o L2.
    """
    config_path = Path(index_dir) / "faiss_config.json"

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_result_images(results, output_dir):
    """
    Copia les imatges dels resultats a una carpeta per revisar-les visualment.

    Això és útil per la UI o per inspecció manual:
        outputs/search_results/query_x/
            query.jpg
            rank_01_stamp_000003.jpg
            rank_02_stamp_000010.jpg
            ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, item in enumerate(results, start=1):
        crop_path = item.get("crop_path")

        if not crop_path:
            continue

        crop_path = Path(crop_path)

        if not crop_path.exists():
            continue

        obj_type = item.get("type", "object")
        crop_id = item.get("crop_id", f"rank_{rank:02d}")

        out_name = f"rank_{rank:02d}_{obj_type}_{crop_id}{crop_path.suffix}"
        shutil.copy2(crop_path, output_dir / out_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
        help="Path to query crop/image",
    )

    parser.add_argument(
        "--index",
        required=True,
        help="Path to FAISS index, e.g. outputs/faiss/real_test25/visual_index.faiss",
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata.jsonl copied next to the FAISS index",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar crops to return",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional folder to copy result crop images for visual inspection",
    )

    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=64,
        help="Must match build_embeddings.py setting",
    )

    parser.add_argument(
        "--edge-size",
        type=int,
        default=32,
        help="Must match build_embeddings.py setting",
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=16,
        help="Must match build_embeddings.py setting",
    )

    args = parser.parse_args()

    query_path = Path(args.query)

    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    # 1. Carreguem l'índex FAISS.
    index = faiss.read_index(str(args.index))

    # 2. Carreguem metadata.
    metadata = load_metadata(args.metadata)

    if len(metadata) != index.ntotal:
        raise ValueError(
            f"Metadata length ({len(metadata)}) does not match "
            f"FAISS index size ({index.ntotal})"
        )

    # 3. Generem embedding del query.
    query_embedding = image_to_embedding(
        image_path=query_path,
        thumbnail_size=args.thumbnail_size,
        edge_size=args.edge_size,
        hist_bins=args.hist_bins,
    )

    # FAISS espera una matriu 2D: (num_queries, embedding_dim).
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dim ({query_embedding.shape[1]}) does not match "
            f"index dim ({index.d})"
        )

    # 4. Fem la cerca.
    # scores: similituds o distàncies segons el tipus d'índex.
    # ids: posicions dels embeddings més propers.
    scores, ids = index.search(query_embedding, args.top_k)

    results = []

    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue

        item = dict(metadata[idx])
        item["rank"] = rank
        item["score"] = float(score)
        item["faiss_id"] = int(idx)

        results.append(item)

    # 5. Mostrem resultats per terminal.
    print(f"Query: {query_path}")
    print(f"Top-{args.top_k} results:")

    for item in results:
        print(
            f"#{item['rank']} "
            f"score={item['score']:.4f} "
            f"type={item.get('type')} "
            f"crop_id={item.get('crop_id')} "
            f"confidence={item.get('confidence')} "
            f"path={item.get('crop_path')}"
        )

    # 6. Guardem resultats, si s'ha demanat.
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copiem el query.
        query_out = output_dir / f"query{query_path.suffix}"
        shutil.copy2(query_path, query_out)

        # Copiem imatges resultants.
        copy_result_images(results, output_dir)

        # Guardem resultats en JSON.
        results_path = output_dir / "search_results.json"

        with results_path.open("w", encoding="utf-8") as f:
            json.dump({
                "query": str(query_path),
                "top_k": args.top_k,
                "results": results,
            }, f, indent=2, ensure_ascii=False)

        print(f"Saved visual results to: {output_dir}")
        print(f"Saved JSON results to:   {results_path}")


if __name__ == "__main__":
    main()
