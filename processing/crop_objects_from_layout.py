import argparse
import json
from pathlib import Path

from PIL import Image


def iter_layout_jsons(layout_dir):
    layout_dir = Path(layout_dir)
    return sorted(layout_dir.glob("*_predicted_layout.json")) + sorted(layout_dir.glob("sample_*/layout_annotations.json"))


def clamp_bbox(bbox, width, height, padding=0):
    x1 = int(bbox["x1"]) - padding
    y1 = int(bbox["y1"]) - padding
    x2 = int(bbox["x2"]) + padding
    y2 = int(bbox["y2"]) + padding

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def find_image_path(images_dir, image_name):
    images_dir = Path(images_dir)

    direct = images_dir / image_name
    if direct.exists():
        return direct

    matches = list(images_dir.rglob(image_name))
    if matches:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", required=True, help="Folder with predicted_layout.json or layout_annotations.json files")
    parser.add_argument("--images", required=True, help="Folder with source images")
    parser.add_argument("--output", required=True, help="Output folder for crops")
    parser.add_argument("--classes", nargs="+", default=["stamp"])
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--min-width", type=int, default=20)
    parser.add_argument("--min-height", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"

    n_crops = 0
    n_docs = 0

    with metadata_path.open("w", encoding="utf-8") as meta_out:
        for layout_path in iter_layout_jsons(args.layouts):
            with layout_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            image_name = data.get("image")
            image_path = find_image_path(args.images, image_name)

            if image_path is None:
                print(f"WARNING image not found for {layout_path}: {image_name}")
                continue

            with Image.open(image_path).convert("RGB") as img:
                width, height = img.size

                n_docs += 1

                for obj_idx, obj in enumerate(data.get("objects", []), start=1):
                    obj_type = obj.get("type")

                    if obj_type not in args.classes:
                        continue

                    bbox = obj.get("bbox")
                    if not bbox:
                        continue

                    clamped = clamp_bbox(bbox, width, height, padding=args.padding)
                    if clamped is None:
                        continue

                    x1, y1, x2, y2 = clamped

                    if (x2 - x1) < args.min_width or (y2 - y1) < args.min_height:
                        continue

                    class_dir = output_dir / obj_type
                    class_dir.mkdir(parents=True, exist_ok=True)

                    crop_id = f"{obj_type}_{n_crops:06d}"
                    crop_path = class_dir / f"{crop_id}.jpg"

                    crop = img.crop((x1, y1, x2, y2))
                    crop.save(crop_path, quality=95)

                    metadata = {
                        "crop_id": crop_id,
                        "type": obj_type,
                        "subtype": obj.get("subtype"),
                        "document_id": data.get("document_id"),
                        "image": image_name,
                        "image_path": str(image_path),
                        "layout_path": str(layout_path),
                        "crop_path": str(crop_path),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "original_bbox": bbox,
                        "confidence": obj.get("confidence"),
                        "source": obj.get("source"),
                        "reviewed": obj.get("reviewed", False),
                    }

                    meta_out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    n_crops += 1

    print(f"Processed documents: {n_docs}")
    print(f"Saved crops: {n_crops}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

