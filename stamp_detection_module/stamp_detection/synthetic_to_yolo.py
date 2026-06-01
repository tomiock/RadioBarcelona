import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image
import yaml


def yolo_line_from_bbox(bbox, w, h, cls_id=0):
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f'{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-root', required=True, help='Folder with sample_XXXX and all_final_images')
    parser.add_argument('--output', required=True, help='YOLO dataset output folder')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()

    root = Path(args.synthetic_root)
    out = Path(args.output)
    samples = sorted(root.glob('sample_*'))
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio)) if samples else 0

    for split in ['train', 'val']:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    used = 0
    for i, sample_dir in enumerate(samples):
        stamp_json = sample_dir / 'stamp_detections.json'
        if not stamp_json.exists():
            continue
        with open(stamp_json, encoding='utf-8') as f:
            data = json.load(f)

        final_name = data.get('image') or f'final_merged_{sample_dir.name.split("_")[-1]}.jpg'
        img_path = root / 'all_final_images' / final_name
        if not img_path.exists():
            # fallback: try common names
            candidates = list((root / 'all_final_images').glob(f'*{sample_dir.name.split("_")[-1]}*'))
            if not candidates:
                continue
            img_path = candidates[0]

        im = Image.open(img_path)
        w, h = im.size
        detections = data.get('stamp_detections', [])
        lines = []
        for det in detections:
            bbox = det.get('bbox')
            if isinstance(bbox, dict):
                bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
            if bbox:
                lines.append(yolo_line_from_bbox(bbox, w, h, 0))
        if not lines:
            continue

        split = 'val' if i < n_val else 'train'
        dst_img = out / 'images' / split / img_path.name
        dst_label = out / 'labels' / split / (img_path.stem + '.txt')
        shutil.copy2(img_path, dst_img)
        dst_label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        used += 1

    data_yaml = {
        'path': str(out.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'official_stamp'}
    }
    with open(out / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f'Created YOLO dataset in {out} with {used} images')


if __name__ == '__main__':
    main()
