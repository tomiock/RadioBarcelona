import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# Fallback detector classes.
# In normal use, class names should come from model.names,
# which are saved inside the YOLO weights from data.yaml.
DEFAULT_CLASS_NAMES = {
    0: "stamp",
    1: "handwritten_text",
    2: "crossout",
    3: "typewritten_text",
}

# Compatibility aliases for older detector/generator names.
TYPE_ALIASES = {
    "typewritten_line": "typewritten_text",
    "typed_line": "typewritten_text",
    "typed_text": "typewritten_text",
    "official_stamp": "stamp",
    "synthetic_stamp": "stamp",
}


def normalize_prediction_type(name):
    """Normalize predicted class names for downstream review/export."""
    if name is None:
        return None
    name = str(name)
    return TYPE_ALIASES.get(name, name)


def get_model_class_names(model):
    """
    Return class names stored in YOLO weights.

    Detector-side rule:
    - Prefer model.names from the trained model.
    - Fallback to DEFAULT_CLASS_NAMES only if unavailable.
    """
    names = getattr(model, "names", None)

    if isinstance(names, dict) and names:
        return {int(k): normalize_prediction_type(v) for k, v in names.items()}

    if isinstance(names, list) and names:
        return {i: normalize_prediction_type(v) for i, v in enumerate(names)}

    return DEFAULT_CLASS_NAMES


def image_files(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in exts
    )


def xyxy_to_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return {
        "x1": int(round(float(x1))),
        "y1": int(round(float(y1))),
        "x2": int(round(float(x2))),
        "y2": int(round(float(y2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--output", required=True, help="Output folder for predicted layout JSONs")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--document-prefix", default="real")
    args = parser.parse_args()

    model = YOLO(args.weights)
    class_names = get_model_class_names(model)
    print(f"Model classes: {class_names}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    imgs = image_files(args.input)

    print(f"Input images: {len(imgs)}")

    for img_idx, img_path in enumerate(imgs, start=1):
        with Image.open(img_path) as im:
            width, height = im.size

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )

        objects = []

        for result in results:
            if result.boxes is None:
                continue

            for det_idx, box in enumerate(result.boxes, start=1):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                obj_type = class_names.get(cls_id, f"class_{cls_id}")

                objects.append({
                    "id": f"pred_{img_idx:04d}_{det_idx:04d}",
                    "type": obj_type,
                    "subtype": None,
                    "bbox": xyxy_to_bbox(xyxy),
                    "text": None,
                    "layer": "real_document",
                    "source": "yolo_prediction",
                    "confidence": round(conf, 4),
                    "reviewed": False,
                    "validated_by": None,
                })

        document_id = f"{args.document_prefix}_{img_path.stem}"

        data = {
            "document_id": document_id,
            "page": 1,
            "image": img_path.name,
            "image_width": width,
            "image_height": height,
            "objects": objects,
        }

        out_path = output_dir / f"{img_path.stem}_predicted_layout.json"

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"{img_path.name}: {len(objects)} objects -> {out_path}")


if __name__ == "__main__":
    main()
