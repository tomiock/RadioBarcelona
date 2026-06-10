import argparse
import os
import shutil
import subprocess
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the synthetic generation + detection + retrieval pipeline."
    )

    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--real-pages", type=int, default=25)
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

    args = parser.parse_args()

    root = Path.cwd()

    synthetic_root = root / "synthetic_docs_aran/output_dataset_pro_try"
    annotations_dir = root / "synthetic_docs_aran/annotations"
    yolo_dataset = root / "visual_marks_dataset"
    outputs_dir = root / "outputs"
    real_test_dir = root / "real_test_pages_25"
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
    real_pages = sorted(real_pages_source.glob("*.jpg"))[:args.real_pages]

    if not real_pages:
        raise RuntimeError(f"No real pages found in {real_pages_source}")

    for page in real_pages:
        shutil.copy2(page, real_test_dir / page.name)

    print(f"Copied real pages: {len(real_pages)}")

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
        "python", "visual_retrieval/crop_objects_from_layout.py",
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
        return



    # ------------------------------------------------------------------
    # 9. Build embeddings
    # ------------------------------------------------------------------
    run([
        "python", "visual_retrieval/build_embeddings.py",
        "--input-crops", "outputs/object_crops_raw",
        "--output-dir", str(embeddings_dir),
        "--metadata", "outputs/object_crops_raw/metadata.jsonl",
        "--classes", *args.retrieval_classes,
    ])

    # ------------------------------------------------------------------
    # 10. Build FAISS index
    # ------------------------------------------------------------------
    run([
        "python", "visual_retrieval/build_faiss_index.py",
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
            "python", "visual_retrieval/search_similar.py",
            "--query", str(stamp_crops[0]),
            "--index", str(faiss_dir / "visual_index.faiss"),
            "--metadata", str(faiss_dir / "metadata.jsonl"),
            "--top-k", "5",
            "--output-dir", str(search_dir),
        ])
    else:
        print("No stamp crops found. Skipping similarity query.")

    print("\nPipeline finished successfully.")
    print("Review app can be launched with:")
    print("  python review_app/app.py")


if __name__ == "__main__":
    main()
