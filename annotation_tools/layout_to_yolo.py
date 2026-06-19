import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image


# ============================================================
# Detector class policy
# ============================================================
# This file is DETECTOR-side: it converts generator annotations
# into YOLO labels. The generator may create more annotation
# types than the detector should learn.
#
# Example:
#   - typewritten_word can be useful for OCR/word metadata.
#   - typewritten_text is the region/line-level class we want
#     the YOLO layout detector to learn.
DEFAULT_DETECTOR_CLASSES = [
    "stamp",
    "handwritten_text",
    "crossout",
    "typewritten_text",
]

# Compatibility aliases for old or generator-specific names.
TYPE_ALIASES = {
    "typewritten_line": "typewritten_text",
    "typed_line": "typewritten_text",
    "typed_text": "typewritten_text",
    "official_stamp": "stamp",
    "synthetic_stamp": "stamp",
}

# These annotation types should not become YOLO boxes.
# Word-level boxes are too dense for the visual layout detector
# and are better handled by OCR/line-specific pipelines.
IGNORE_TYPES = {
    "typewritten_word",
    "word",
    "text_word",
}


def normalize_annotation_type(raw_type):
    """Map generator annotation types to detector classes."""
    if not raw_type:
        return None

    obj_type = TYPE_ALIASES.get(raw_type, raw_type)

    if obj_type in IGNORE_TYPES:
        return None

    return obj_type


def iter_layout_jsons(synthetic_root):
    synthetic_root = Path(synthetic_root)
    for path in sorted(synthetic_root.glob("sample_*/layout_annotations.json")):
        yield path


def yolo_line_from_bbox(bbox, img_w, img_h, class_id):
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])

    box_w = x2 - x1
    box_h = y2 - y1

    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = box_w / img_w
    bh = box_h / img_h

    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def convert_one(layout_path, synthetic_root, output_dir, split, class_to_id, min_box_size):
    synthetic_root = Path(synthetic_root)
    output_dir = Path(output_dir)

    with layout_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data["image"]
    img_w = int(data.get("image_width", 0))
    img_h = int(data.get("image_height", 0))

    image_path = synthetic_root / "all_final_images" / image_name

    if not image_path.exists():
        print(f"WARNING: image not found: {image_path}")
        return 0

    if img_w <= 0 or img_h <= 0:
        with Image.open(image_path) as img:
            img_w, img_h = img.size

    lines = []

    for obj in data.get("objects", []):
        # Generator-side type, normalized into detector-side class.
        raw_type = obj.get("type")
        obj_type = normalize_annotation_type(raw_type)

        if obj_type is None or obj_type not in class_to_id:
            continue

        bbox = obj.get("bbox")
        if not bbox:
            continue

        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < min_box_size or box_h < min_box_size:
            continue

        # Clamp per evitar coordenades fora de la imatge
        bbox = {
            "x1": max(0, min(int(x1), img_w - 1)),
            "y1": max(0, min(int(y1), img_h - 1)),
            "x2": max(0, min(int(x2), img_w - 1)),
            "y2": max(0, min(int(y2), img_h - 1)),
        }

        if bbox["x2"] <= bbox["x1"] or bbox["y2"] <= bbox["y1"]:
            continue

        class_id = class_to_id[obj_type]
        lines.append(yolo_line_from_bbox(bbox, img_w, img_h, class_id))

    if not lines:
        return 0

    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    out_image = images_out / image_path.name
    out_label = labels_out / f"{image_path.stem}.txt"

    shutil.copy2(image_path, out_image)

    with out_label.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return len(lines)


def write_data_yaml(output_dir, classes):
    output_dir = Path(output_dir)
    yaml_path = output_dir / "data.yaml"

    names = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])

    yaml_text = f"""path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
{names}
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_DETECTOR_CLASSES,
        help="YOLO detector classes to export. Defaults to layout detector classes.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-box-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    synthetic_root = Path(args.synthetic_root)
    output_dir = Path(args.output)

    # Normalize requested class names too, so old names like
    # typewritten_line are mapped to the detector class typewritten_text.
    classes = []
    for name in args.classes:
        norm = normalize_annotation_type(name)
        if norm and norm not in classes:
            classes.append(norm)

    class_to_id = {name: idx for idx, name in enumerate(classes)}

    layout_paths = list(iter_layout_jsons(synthetic_root))

    random.seed(args.seed)
    random.shuffle(layout_paths)

    n = len(layout_paths)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)

    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    n_train = max(0, n - n_val - n_test)

    splits = (
        [("train", p) for p in layout_paths[:n_train]]
        + [("val", p) for p in layout_paths[n_train:n_train + n_val]]
        + [("test", p) for p in layout_paths[n_train + n_val:]]
    )

    total_boxes = 0
    used_images = 0

    for split, layout_path in splits:
        n_boxes = convert_one(
            layout_path=layout_path,
            synthetic_root=synthetic_root,
            output_dir=output_dir,
            split=split,
            class_to_id=class_to_id,
            min_box_size=args.min_box_size,
        )

        if n_boxes > 0:
            used_images += 1
            total_boxes += n_boxes

    write_data_yaml(output_dir, classes)

    print(f"Input layout files: {n}")
    print(f"Used images: {used_images}")
    print(f"Total YOLO boxes: {total_boxes}")
    print(f"Classes: {class_to_id}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
