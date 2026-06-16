import argparse
import json
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def iter_images(path: Path):
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path
    else:
        for p in sorted(path.rglob('*')):
            if p.suffix.lower() in IMG_EXTS:
                yield p


def crop_and_save(img_path: Path, bbox, crop_dir: Path, stamp_id: str):
    crop_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert('RGB')
    x1, y1, x2, y2 = map(int, bbox)
    crop = img.crop((x1, y1, x2, y2))
    crop_path = crop_dir / f'{stamp_id}.png'
    crop.save(crop_path)
    return str(crop_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--input', required=True, help='Image file or folder')
    parser.add_argument('--output', required=True, help='Folder for JSON outputs')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--document-prefix', default='radio_barcelona')
    args = parser.parse_args()

    model = YOLO(args.weights)
    input_path = Path(args.input)
    out_dir = Path(args.output)
    crop_dir = out_dir / 'crops'
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(iter_images(input_path), start=1):
        result = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        detections = []

        if result.boxes is not None:
            for j, box in enumerate(result.boxes, start=1):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = model.names.get(cls_id, 'stamp') if hasattr(model, 'names') else 'stamp'
                stamp_id = f'stamp_{idx:04d}_{j:03d}'
                crop_path = crop_and_save(img_path, xyxy, crop_dir, stamp_id)
                x1, y1, x2, y2 = map(lambda v: int(round(v)), xyxy)
                detections.append({
                    'id': stamp_id,
                    'class': cls_name,
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'confidence': round(conf, 4),
                    'crop_path': crop_path,
                    'mask_path': None,
                    'ocr_text': None,
                    'notes': 'detected by YOLO stamp detector'
                })

        payload = {
            'document_id': f'{args.document_prefix}_{img_path.stem}',
            'page': idx,
            'image_path': str(img_path),
            'stamp_detections': detections
        }

        json_path = out_dir / f'{img_path.stem}_stamps.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'Wrote {json_path} ({len(detections)} stamps)')


if __name__ == '__main__':
    main()
