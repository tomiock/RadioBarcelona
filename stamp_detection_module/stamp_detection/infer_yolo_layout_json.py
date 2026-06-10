import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


DEFAULT_CLASS_NAMES = {
    0: "stamp",
    1: "crossout",
    2: "censorship_block",
    3: "handwritten_text",
    4: "table_fragment",
}


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

                obj_type = DEFAULT_CLASS_NAMES.get(cls_id, f"class_{cls_id}")

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
