import argparse
import json
import os
import shutil
import subprocess
import random

from collections import Counter
from datetime import datetime
from pathlib import Path



def run(cmd, cwd=None, env=None):
    """
    Executa una comanda mostrant-la abans per terminal.

    Si alguna comanda falla, el pipeline s'atura.
    """
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def remove_path(path):
    """
    Esborra un fitxer o carpeta si existeix.

    Serveix per començar una pipeline neta sense outputs antics.
    """
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed folder: {path}")
    elif path.exists():
        path.unlink()
        print(f"Removed file: {path}")






def count_predicted_objects(layouts_dir):
    """
    Compta quants objectes detectats hi ha per classe dins els predicted_layout.json.

    Exemple de sortida:
        {
            "stamp": 60,
            "handwritten_text": 148,
            "crossout": 41
        }
    """
    layouts_dir = Path(layouts_dir)
    counts = Counter()

    if not layouts_dir.exists():
        return {}

    for path in layouts_dir.glob("*_predicted_layout.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            counts[obj.get("type", "unknown")] += 1

    return dict(counts)


def count_crops_by_class(crops_dir):
    """
    Compta crops guardats per classe.

    Espera estructura:
        outputs/object_crops_raw/stamp/*.jpg
        outputs/object_crops_raw/handwritten_text/*.jpg
        ...
    """
    crops_dir = Path(crops_dir)
    counts = {}

    if not crops_dir.exists():
        return counts

    for class_dir in sorted(p for p in crops_dir.iterdir() if p.is_dir()):
        n = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
        counts[class_dir.name] = n

    return counts


def count_jsonl_lines(path):
    """
    Compta línies no buides d'un JSONL.
    """
    path = Path(path)

    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_manifest(
    *,
    root,
    run_id,
    args,
    synthetic_root,
    annotations_dir,
    yolo_dataset,
    real_test_dir,
    embeddings_dir,
    faiss_dir,
    search_dir,
    run_name,
    status="completed",
):
    """
    Escriu un manifest JSON amb el resum de l'execució del pipeline.

    Aquest fitxer és important per poder compartir l'experiment i saber:
        - amb quina configuració es va executar,
        - quins outputs va generar,
        - quantes deteccions/crops hi ha,
        - quins pesos i índex FAISS s'han fet servir.
    """
    root = Path(root)

    layouts_dir = root / "outputs/real_predicted_layouts"
    crops_dir = root / "outputs/object_crops_raw"
    review_log = root / "outputs/review_logs/review_log.jsonl"

    embeddings_path = embeddings_dir / "embeddings.npy"
    faiss_index_path = faiss_dir / "visual_index.faiss"
    faiss_metadata_path = faiss_dir / "metadata.jsonl"

    raw_crops_by_type = count_crops_by_class(crops_dir)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "synthetic_generation_detection_review_retrieval",
        "status": status,

        "config": {
            "samples": args.samples,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "real_pages": args.real_pages,
            "real_selection": args.real_selection,
            "seed": args.seed,
            "conf": args.conf,
            "skip_train": args.skip_train,
            "weights": args.weights,
            "classes": args.classes,
            "retrieval_classes": args.retrieval_classes,
        },

        "paths": {
            "synthetic_root": str(synthetic_root),
            "annotations_dir": str(annotations_dir),
            "yolo_dataset": str(yolo_dataset),
            "real_test_dir": str(real_test_dir),
            "real_predicted_layouts": str(layouts_dir),
            "object_crops_raw": str(crops_dir),
            "embeddings_dir": str(embeddings_dir),
            "faiss_dir": str(faiss_dir),
            "search_dir": str(search_dir),
            "review_log": str(review_log),
            "training_run": str(root / f"runs/detect/{run_name}"),
        },

        "counts": {
            "synthetic_layout_jsons": len(list(Path(synthetic_root).glob("sample_*/layout_annotations.json"))),
            "real_pages": len(list(Path(real_test_dir).glob("*.jpg"))),
            "predicted_layout_jsons": len(list(layouts_dir.glob("*_predicted_layout.json"))) if layouts_dir.exists() else 0,
            "predicted_objects_by_type": count_predicted_objects(layouts_dir),
            "raw_crops_by_type": raw_crops_by_type,
            "raw_crops_total": sum(raw_crops_by_type.values()),
            "review_entries": count_jsonl_lines(review_log),
            "embeddings_exists": embeddings_path.exists(),
            "faiss_index_exists": faiss_index_path.exists(),
        },

        "files": {
            "embeddings": str(embeddings_path) if embeddings_path.exists() else None,
            "faiss_index": str(faiss_index_path) if faiss_index_path.exists() else None,
            "faiss_metadata": str(faiss_metadata_path) if faiss_metadata_path.exists() else None,
        },
    }

    manifests_dir = root / "outputs/manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / f"{run_id}_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSaved manifest: {manifest_path}")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))






def main():
    parser = argparse.ArgumentParser(
        description="Run the synthetic generation + detection + retrieval pipeline."
    )

    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--real-pages", type=int, default=25)
    parser.add_argument(
        "--real-selection",
        choices=["first", "random"],
        default="random",
        help="How to select real pages: first sorted pages or random sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --real-selection random.",
    )
    parser.add_argument("--conf", type=float, default=0.35)

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated outputs before running.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip YOLO training and reuse existing weights.",
    )

    parser.add_argument(
        "--weights",
        default="runs/detect/visual_marks_detector_pipeline_test/weights/best.pt",
        help="YOLO weights to use for real inference.",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=["stamp", "crossout", "censorship_block", "handwritten_text", "table_fragment"],
        help="Classes used for masks and YOLO dataset.",
    )

    parser.add_argument(
        "--retrieval-classes",
        nargs="+",
        default=["stamp", "handwritten_text", "crossout", "table_fragment"],
        help="Classes used for crops, embeddings and FAISS.",
    )

    parser.add_argument(
        "--run-id",
        default="current",
        help="Name for output experiment folders, e.g. current, real_test25, real_test290",
    )

    parser.add_argument(
        "--real-pages-dir",
        default=None,
        help="Directory with real JPG pages used in the experiment.",
    )
        


    args = parser.parse_args()

    root = Path.cwd()

    synthetic_root = root / "synthetic_docs_aran/output_dataset_pro_try"
    annotations_dir = root / "synthetic_docs_aran/annotations"
    yolo_dataset = root / "visual_marks_dataset"
    outputs_dir = root / "outputs"
    real_test_dir = root / f"real_test_{args.run_id}_pages_{args.real_pages}"
    run_name = "visual_marks_detector_pipeline_test"
    # Output folders depending on the selected run id.
    # Example:
    #   --run-id current      -> outputs/faiss/current/
    #   --run-id real_test290 -> outputs/faiss/real_test290/
    embeddings_dir = root / f"outputs/embeddings/{args.run_id}"
    faiss_dir = root / f"outputs/faiss/{args.run_id}"
    search_dir = root / f"outputs/search_results/{args.run_id}_query_stamp_001"

    if args.clean:
        remove_path(synthetic_root)
        remove_path(annotations_dir)
        remove_path(yolo_dataset)
        remove_path(outputs_dir)
        remove_path(real_test_dir)
        remove_path(root / f"runs/detect/{run_name}")

    # ------------------------------------------------------------------
    # 1. Generate synthetic documents
    # ------------------------------------------------------------------
    env = os.environ.copy()
    env["USE_GEMINI"] = "false"
    env["USE_OPENAI"] = "false"
    env["TOTAL_SAMPLES"] = str(args.samples)

    run(
        ["python", "generator.py"],
        cwd=root / "synthetic_docs_aran",
        env=env,
    )

    # ------------------------------------------------------------------
    # 2. Build JSONL annotations
    # ------------------------------------------------------------------
    run([
        "python", "annotation_tools/build_annotations_jsonl.py",
        "--synthetic-root", str(synthetic_root),
        "--output-dir", str(annotations_dir),
    ])

    # ------------------------------------------------------------------
    # 3. Build masks for GAN / segmentation
    # ------------------------------------------------------------------
    run([
        "python", "annotation_tools/layout_to_masks.py",
        "--synthetic-root", str(synthetic_root),
        "--output-dir", "outputs/layout_masks",
        "--classes", *args.classes,
    ])

    # ------------------------------------------------------------------
    # 4. Convert layout annotations to YOLO format
    # ------------------------------------------------------------------
    run([
        "python", "annotation_tools/layout_to_yolo.py",
        "--synthetic-root", str(synthetic_root),
        "--output", str(yolo_dataset),
        "--classes", *args.classes,
        "--min-box-size", "40",
    ])

    # ------------------------------------------------------------------
    # 5. Train YOLO detector
    # ------------------------------------------------------------------
    if not args.skip_train:
        run([
            "python", "stamp_detection_module/stamp_detection/train_yolo_stamps.py",
            "--data", str(yolo_dataset / "data.yaml"),
            "--model", "yolo11n.pt",
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--name", run_name,
        ])

    # ------------------------------------------------------------------
    # 6. Copy real pages for testing
    # ------------------------------------------------------------------
    real_test_dir.mkdir(parents=True, exist_ok=True)

    real_pages_source = root / "synthetic_docs_aran/data/pages"
    all_real_pages = sorted(real_pages_source.glob("*.jpg"))

    if not all_real_pages:
        raise RuntimeError(f"No real pages found in {real_pages_source}")

    if args.real_pages > len(all_real_pages):
        raise RuntimeError(
            f"Requested {args.real_pages} real pages, "
            f"but only found {len(all_real_pages)} in {real_pages_source}"
        )

    if args.real_selection == "random":
        rng = random.Random(args.seed)
        real_pages = sorted(rng.sample(all_real_pages, args.real_pages))
    else:
        real_pages = all_real_pages[:args.real_pages]

    for page in real_pages:
        shutil.copy2(page, real_test_dir / page.name)

    print(f"Copied real pages: {len(real_pages)}")
    print(f"Real page selection: {args.real_selection}")
    print(f"Seed: {args.seed}")
    print(f"Real test dir: {real_test_dir}")

    # Guardem quines pàgines s'han seleccionat.
    # Això fa que l'experiment sigui reproduïble i auditable.
    selected_pages_path = real_test_dir / "selected_real_pages.json"

    with selected_pages_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "real_pages_source": str(real_pages_source),
                "real_selection": args.real_selection,
                "seed": args.seed,
                "requested_real_pages": args.real_pages,
                "selected_pages": [p.name for p in real_pages],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # 7. Run YOLO inference on real pages and export predicted_layout.json
    # ------------------------------------------------------------------
    run([
        "python", "stamp_detection_module/stamp_detection/infer_yolo_layout_json.py",
        "--weights", args.weights,
        "--input", str(real_test_dir),
        "--output", "outputs/real_predicted_layouts",
        "--conf", str(args.conf),
        "--imgsz", str(args.imgsz),
        "--document-prefix", "rb",
    ])

    # ------------------------------------------------------------------
    # 8. Crop predicted objects
    # ------------------------------------------------------------------
    run([
        "python", "processing/crop_objects_from_layout.py",
        "--layouts", "outputs/real_predicted_layouts",
        "--images", str(real_test_dir),
        "--output", "outputs/object_crops_raw",
        "--classes", *args.retrieval_classes,
        "--padding", "10",
        "--min-width", "25",
        "--min-height", "25",
    ])


    # ------------------------------------------------------------------
    # Stop retrieval steps if no crops were produced
    # ------------------------------------------------------------------
    crop_files = list((root / "outputs/object_crops_raw").glob("*/*.jpg"))

    if not crop_files:
        print("\nNo crops were produced from real predictions.")
        print("Skipping embeddings, FAISS and similarity search.")
        print("Suggestions:")
        print("  - lower --conf, e.g. --conf 0.10")
        print("  - train more epochs, e.g. --epochs 30 or 50")
        print("  - use more synthetic samples, e.g. --samples 150 or 200")
        print("  - use real pages with visible stamps/handwriting/crossouts")
        print("\nPipeline finished partially.")
        print("Review app will not have crops to review yet.")

        write_manifest(
            root=root,
            run_id=args.run_id,
            args=args,
            synthetic_root=synthetic_root,
            annotations_dir=annotations_dir,
            yolo_dataset=yolo_dataset,
            real_test_dir=real_test_dir,
            embeddings_dir=embeddings_dir,
            faiss_dir=faiss_dir,
            search_dir=search_dir,
            run_name=run_name,
            status="partial_no_crops",
        )

        return



    # ------------------------------------------------------------------
    # 9. Build embeddings
    # ------------------------------------------------------------------
    run([
        "python", "visual_search/build_embeddings.py",
        "--input-crops", "outputs/object_crops_raw",
        "--output-dir", str(embeddings_dir),
        "--metadata", "outputs/object_crops_raw/metadata.jsonl",
        "--classes", *args.retrieval_classes,
    ])

    # ------------------------------------------------------------------
    # 10. Build FAISS index
    # ------------------------------------------------------------------
    run([
        "python", "visual_search/build_faiss_index.py",
        "--embeddings", str(embeddings_dir / "embeddings.npy"),
        "--metadata", str(embeddings_dir / "metadata.jsonl"),
        "--output-dir", str(faiss_dir),
        "--metric", "cosine",
    ])

    # ------------------------------------------------------------------
    # 11. Optional similarity query if a stamp crop exists
    # ------------------------------------------------------------------
    stamp_dir = root / "outputs/object_crops_raw/stamp"
    stamp_crops = sorted(stamp_dir.glob("*.jpg")) if stamp_dir.exists() else []

    if stamp_crops:
        run([
            "python", "visual_search/search_similar_classic.py",
            "--query", str(stamp_crops[0]),
            "--index", str(faiss_dir / "visual_index.faiss"),
            "--metadata", str(faiss_dir / "metadata.jsonl"),
            "--top-k", "5",
            "--output-dir", str(search_dir),
        ])
    else:
        print("No stamp crops found. Skipping similarity query.")


    write_manifest(
        root=root,
        run_id=args.run_id,
        args=args,
        synthetic_root=synthetic_root,
        annotations_dir=annotations_dir,
        yolo_dataset=yolo_dataset,
        real_test_dir=real_test_dir,
        embeddings_dir=embeddings_dir,
        faiss_dir=faiss_dir,
        search_dir=search_dir,
        run_name=run_name,
    )

    print("\nPipeline finished successfully.")
    print("Review app can be launched with:")
    print("  python review_app/app.py")


if __name__ == "__main__":
    main()
