import argparse
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


def copy_if_exists(src, dst):
    """
    Copia un fitxer o carpeta si existeix.
    Si no existeix, no falla: simplement ho ignora.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        print(f"Skip missing: {src}")
        return False

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print(f"Copied: {src} -> {dst}")
    return True


def count_files(root, patterns):
    """
    Compta fitxers dins una carpeta segons una llista de patrons.
    """
    root = Path(root)

    if not root.exists():
        return 0

    total = 0
    for pattern in patterns:
        total += len(list(root.rglob(pattern)))

    return total


def load_json(path):
    """
    Carrega un JSON si existeix.
    """
    path = Path(path)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_readme(package_dir, run_id):
    """
    Escriu un README curt dins el paquet.
    """
    text = f"""# Experiment package: {run_id}

This package contains outputs from the visual layout detection/review pipeline.

## Main contents

- manifest.json: original pipeline manifest.
- package_manifest.json: package summary.
- weights/: YOLO weights.
- real_pages/: real JPG pages used for validation.
- real_predicted_layouts/: predicted layout JSON files.
- object_crops_raw/: raw crops and metadata.
- review/: human review logs and reviewed/rejected/skipped crops.
- embeddings/: visual embeddings.
- faiss/: FAISS similarity index.
- synthetic_annotations/: JSONL train/val/test annotations.
- layout_masks/: masks generated from layout annotations.

Raw predictions and human review decisions are intentionally separated.

Raw metadata:
object_crops_raw/metadata.jsonl

Human review log:
review/review_logs/review_log.jsonl
"""

    with (package_dir / "README_EXPERIMENT.md").open("w", encoding="utf-8") as f:
        f.write(text)


def make_archive(package_dir, archive_path):
    """
    Crea un .tar.gz del paquet.
    """
    package_dir = Path(package_dir)
    archive_path = Path(archive_path)

    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)

    print(f"Created archive: {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Package a pipeline experiment for sharing."
    )

    parser.add_argument("--run-id", default="current")
    parser.add_argument("--output-dir", default="shared_experiments")
    parser.add_argument(
        "--weights",
        default="runs/detect/visual_marks_detector/weights/best.pt",
    )
    parser.add_argument("--include-yolo-dataset", action="store_true")
    parser.add_argument("--include-synthetic-output", action="store_true")
    parser.add_argument("--make-archive", action="store_true")

    args = parser.parse_args()

    project_root = Path.cwd()
    package_root = Path(args.output_dir)
    package_dir = package_root / args.run_id

    if package_dir.exists():
        shutil.rmtree(package_dir)

    package_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = project_root / f"outputs/manifests/{args.run_id}_manifest.json"

    copied = {}

    copied["manifest"] = copy_if_exists(
        source_manifest,
        package_dir / "manifest.json",
    )

    copied["review_schema"] = copy_if_exists(
        project_root / "review_app/review_schema.json",
        package_dir / "review_schema.json",
    )

    copied["weights"] = copy_if_exists(
        project_root / args.weights,
        package_dir / "weights/best.pt",
    )

    copied["real_pages"] = copy_if_exists(
        project_root / "real_test_pages_25",
        package_dir / "real_pages",
    )

    copied["real_predicted_layouts"] = copy_if_exists(
        project_root / "outputs/real_predicted_layouts",
        package_dir / "real_predicted_layouts",
    )

    copied["object_crops_raw"] = copy_if_exists(
        project_root / "outputs/object_crops_raw",
        package_dir / "object_crops_raw",
    )

    copied["review_logs"] = copy_if_exists(
        project_root / "outputs/review_logs",
        package_dir / "review/review_logs",
    )

    copied["object_crops_reviewed"] = copy_if_exists(
        project_root / "outputs/object_crops_reviewed",
        package_dir / "review/object_crops_reviewed",
    )

    copied["object_crops_rejected"] = copy_if_exists(
        project_root / "outputs/object_crops_rejected",
        package_dir / "review/object_crops_rejected",
    )

    copied["object_crops_skipped"] = copy_if_exists(
        project_root / "outputs/object_crops_skipped",
        package_dir / "review/object_crops_skipped",
    )

    copied["embeddings"] = copy_if_exists(
        project_root / f"outputs/embeddings/{args.run_id}",
        package_dir / "embeddings",
    )

    copied["faiss"] = copy_if_exists(
        project_root / f"outputs/faiss/{args.run_id}",
        package_dir / "faiss",
    )

    copied["search_results"] = copy_if_exists(
        project_root / f"outputs/search_results/{args.run_id}_query_stamp_001",
        package_dir / "search_results",
    )

    copied["synthetic_annotations"] = copy_if_exists(
        project_root / "synthetic_docs_aran/annotations",
        package_dir / "synthetic_annotations",
    )

    copied["layout_masks"] = copy_if_exists(
        project_root / "outputs/layout_masks",
        package_dir / "layout_masks",
    )

    if args.include_yolo_dataset:
        copied["yolo_dataset"] = copy_if_exists(
            project_root / "visual_marks_dataset",
            package_dir / "yolo_dataset",
        )

    if args.include_synthetic_output:
        copied["synthetic_output"] = copy_if_exists(
            project_root / "synthetic_docs_aran/output_dataset_pro_try",
            package_dir / "synthetic_output",
        )

    source_manifest_json = load_json(source_manifest)

    package_manifest = {
        "package_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(source_manifest),
        "source_manifest_data": source_manifest_json,
        "copied_sections": copied,
        "counts": {
            "real_pages": count_files(package_dir / "real_pages", ["*.jpg", "*.jpeg", "*.png"]),
            "predicted_layouts": count_files(package_dir / "real_predicted_layouts", ["*_predicted_layout.json"]),
            "raw_crops": count_files(package_dir / "object_crops_raw", ["*.jpg", "*.png"]),
            "reviewed_crops": count_files(package_dir / "review/object_crops_reviewed", ["*.jpg", "*.png"]),
            "rejected_crops": count_files(package_dir / "review/object_crops_rejected", ["*.jpg", "*.png"]),
            "skipped_crops": count_files(package_dir / "review/object_crops_skipped", ["*.jpg", "*.png"]),
            "layout_masks": count_files(package_dir / "layout_masks", ["*.png"]),
        },
    }

    with (package_dir / "package_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(package_manifest, f, indent=2, ensure_ascii=False)

    write_readme(package_dir, args.run_id)

    print("\nPackage created:")
    print(package_dir)
    print(json.dumps(package_manifest["counts"], indent=2, ensure_ascii=False))

    if args.make_archive:
        archive_path = package_root / f"{args.run_id}.tar.gz"
        make_archive(package_dir, archive_path)


if __name__ == "__main__":
    main()