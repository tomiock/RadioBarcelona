import argparse
import json
import shutil
from pathlib import Path


# Mapegem classes internes a carpetes d'assets per al generator.
# Això manté la sortida ordenada i clara.
CLASS_TO_ASSET_DIR = {
    "stamp": "stamps",
    "handwritten_text": "handwriting",
    "crossout": "crossouts",
    "censorship_block": "censorship",
    "table_fragment": "tables",
}


def iter_review_log(review_log_path):
    """
    Llegeix el review_log.jsonl línia a línia.

    Cada línia és una decisió humana:
        accepted / rejected / skipped

    Només exportarem les decisions accepted.
    """
    review_log_path = Path(review_log_path)

    if not review_log_path.exists():
        raise FileNotFoundError(f"Review log not found: {review_log_path}")

    with review_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            yield json.loads(line)


def choose_source_crop(entry):
    """
    Decideix quin fitxer de crop copiar.

    Preferim reviewed_crop_path perquè és el fitxer ja copiat per la review app.
    Si no existeix, fem fallback a crop_path original.
    """
    reviewed_crop_path = entry.get("reviewed_crop_path")
    crop_path = entry.get("crop_path")

    candidates = []

    if reviewed_crop_path:
        candidates.append(Path(reviewed_crop_path))

    if crop_path:
        candidates.append(Path(crop_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def safe_name(value):
    """
    Neteja un string perquè sigui útil com a nom de fitxer.
    """
    value = str(value or "unknown")
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = value.replace(" ", "_")
    return value


def export_reviewed_assets(review_log_path, output_dir, min_human_confidence=None):
    """
    Exporta només els crops acceptats a una carpeta d'assets reals revisats.

    Entrada:
        outputs/review_logs/review_log.jsonl

    Sortida:
        synthetic_docs_aran/assets_real_reviewed/
        ├── stamps/
        ├── handwriting/
        ├── crossouts/
        ├── censorship/
        ├── tables/
        └── reviewed_assets_metadata.jsonl
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "reviewed_assets_metadata.jsonl"

    exported = 0
    skipped = 0
    seen_crop_ids = set()

    with metadata_path.open("w", encoding="utf-8") as meta_out:
        for entry in iter_review_log(review_log_path):
            # Només exportem acceptats.
            if entry.get("decision") != "accepted":
                skipped += 1
                continue

            reviewed_type = entry.get("reviewed_type") or entry.get("type")

            # Ignorem falsos positius o classes no útils com assets.
            if reviewed_type not in CLASS_TO_ASSET_DIR:
                skipped += 1
                continue

            # Opcional: filtrar per confiança humana.
            if min_human_confidence:
                if entry.get("human_confidence") != min_human_confidence:
                    skipped += 1
                    continue

            crop_id = entry.get("crop_id")

            # Evitem exportar el mateix crop_id múltiples vegades.
            # Si un crop s'ha revisat més d'una vegada, aquí ens quedem amb la primera aparició acceptada.
            if crop_id in seen_crop_ids:
                skipped += 1
                continue

            seen_crop_ids.add(crop_id)

            source_crop = choose_source_crop(entry)

            if source_crop is None:
                skipped += 1
                continue

            asset_subdir = CLASS_TO_ASSET_DIR[reviewed_type]
            target_dir = output_dir / asset_subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            document_id = safe_name(entry.get("document_id"))
            crop_id_safe = safe_name(crop_id)
            suffix = source_crop.suffix.lower() or ".jpg"

            target_name = f"{reviewed_type}_{document_id}_{crop_id_safe}{suffix}"
            target_path = target_dir / target_name

            shutil.copy2(source_crop, target_path)

            asset_meta = dict(entry)
            asset_meta["asset_path"] = str(target_path)
            asset_meta["asset_class"] = reviewed_type
            asset_meta["asset_subdir"] = asset_subdir
            asset_meta["asset_source"] = "human_reviewed_real_crop"

            meta_out.write(json.dumps(asset_meta, ensure_ascii=False) + "\n")

            exported += 1

    print(f"Exported assets: {exported}")
    print(f"Skipped entries: {skipped}")
    print(f"Output dir: {output_dir}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-log",
        required=True,
        help="Path to outputs/review_logs/review_log.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for reviewed real assets",
    )

    parser.add_argument(
        "--min-human-confidence",
        default=None,
        choices=[None, "high", "medium", "low"],
        help="Optional filter. Example: only export high-confidence human reviews.",
    )

    args = parser.parse_args()

    export_reviewed_assets(
        review_log_path=args.review_log,
        output_dir=args.output_dir,
        min_human_confidence=args.min_human_confidence,
    )


if __name__ == "__main__":
    main()
