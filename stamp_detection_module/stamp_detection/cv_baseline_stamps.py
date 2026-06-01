import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def iter_images(path: Path):
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path
    else:
        for p in sorted(path.rglob('*')):
            if p.suffix.lower() in IMG_EXTS:
                yield p


def detect_stamp_candidates(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Blavós / morat / vermellós: habitual en tampons. Ajustable.
    blue = cv2.inRange(hsv, (85, 25, 30), (145, 255, 230))
    red1 = cv2.inRange(hsv, (0, 25, 30), (15, 255, 230))
    red2 = cv2.inRange(hsv, (160, 25, 30), (179, 255, 230))
    mask = cv2.bitwise_or(blue, cv2.bitwise_or(red1, red2))

    # També agafem tinta fosca amb saturació mitjana per segells negres/grisos.
    dark = cv2.inRange(hsv, (0, 10, 0), (179, 180, 100))
    mask = cv2.bitwise_or(mask, dark)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bgr.shape[:2]
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.0008 * w * h or area > 0.25 * w * h:
            continue
        aspect = bw / max(1, bh)
        if aspect < 0.25 or aspect > 5.0:
            continue
        # Evitem línies massa fines.
        if bw < 30 or bh < 20:
            continue
        candidates.append((x, y, x + bw, y + bh, float(area) / (w * h)))

    return candidates, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    crop_dir = out_dir / 'crops'
    mask_dir = out_dir / 'masks'
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, img_path in enumerate(iter_images(in_path), start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        candidates, mask = detect_stamp_candidates(bgr)
        Image.fromarray(mask).save(mask_dir / f'{img_path.stem}_mask.png')

        detections = []
        pil = Image.open(img_path).convert('RGB')
        for j, (x1, y1, x2, y2, score) in enumerate(candidates, start=1):
            stamp_id = f'cv_stamp_{page_idx:04d}_{j:03d}'
            crop_path = crop_dir / f'{stamp_id}.png'
            pil.crop((x1, y1, x2, y2)).save(crop_path)
            detections.append({
                'id': stamp_id,
                'class': 'stamp_candidate',
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'confidence': round(min(0.99, 0.35 + score * 10), 4),
                'crop_path': str(crop_path),
                'mask_path': str(mask_dir / f'{img_path.stem}_mask.png'),
                'ocr_text': None,
                'notes': 'candidate from OpenCV color/dark-ink heuristic; requires validation'
            })

        payload = {
            'document_id': f'radio_barcelona_{img_path.stem}',
            'page': page_idx,
            'image_path': str(img_path),
            'stamp_detections': detections
        }
        json_path = out_dir / f'{img_path.stem}_stamps_cv.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'Wrote {json_path} ({len(detections)} candidates)')


if __name__ == '__main__':
    main()
