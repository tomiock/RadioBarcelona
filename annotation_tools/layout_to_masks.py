import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_CLASSES = [
    "stamp",
    "crossout",
    "censorship_block",
    "handwritten_text",
    "table_fragment",
]


def iter_layout_jsons(synthetic_root):
    synthetic_root = Path(synthetic_root)
    for path in sorted(synthetic_root.glob("sample_*/layout_annotations.json")):
        yield path


def clamp_bbox(bbox, width, height):
    x1 = max(0, min(int(bbox["x1"]), width - 1))
    y1 = max(0, min(int(bbox["y1"]), height - 1))
    x2 = max(0, min(int(bbox["x2"]), width - 1))
    y2 = max(0, min(int(bbox["y2"]), height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def create_empty_mask(width, height):
    return Image.new("L", (width, height), 0)


def save_masks_for_layout(layout_path, output_dir, classes):
    output_dir = Path(output_dir)

    with open(layout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_id = data.get("document_id", layout_path.parent.name)
    width = int(data["image_width"])
    height = int(data["image_height"])

    sample_output_dir = output_dir / document_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    masks = {cls: create_empty_mask(width, height) for cls in classes}
    all_mask = create_empty_mask(width, height)

    drawers = {cls: ImageDraw.Draw(mask) for cls, mask in masks.items()}
    all_draw = ImageDraw.Draw(all_mask)

    counts = {cls: 0 for cls in classes}

    for obj in data.get("objects", []):
        obj_type = obj.get("type")

        if obj_type not in classes:
            continue

        bbox = obj.get("bbox")
        if not bbox:
            continue

        clamped = clamp_bbox(bbox, width, height)
        if clamped is None:
            continue

        drawers[obj_type].rectangle(clamped, fill=255)
        all_draw.rectangle(clamped, fill=255)
        counts[obj_type] += 1

    for cls, mask in masks.items():
        mask.save(sample_output_dir / f"{cls}_mask.png")

    all_mask.save(sample_output_dir / "all_visual_marks_mask.png")

    metadata = {
        "document_id": document_id,
        "image": data.get("image"),
        "image_width": width,
        "image_height": height,
        "classes": classes,
        "counts": counts,
        "source_layout": str(layout_path),
    }

    with open(sample_output_dir / "mask_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    args = parser.parse_args()

    total_counts = {cls: 0 for cls in args.classes}
    n_docs = 0

    for layout_path in iter_layout_jsons(args.synthetic_root):
        counts = save_masks_for_layout(
            layout_path=layout_path,
            output_dir=args.output_dir,
            classes=args.classes,
        )

        for cls, count in counts.items():
            total_counts[cls] += count

        n_docs += 1

    print(f"Processed documents: {n_docs}")
    print("Total objects used for masks:")
    for cls, count in total_counts.items():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
