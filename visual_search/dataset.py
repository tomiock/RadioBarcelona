from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_latest_reviews(review_log: Optional[Path]) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if not review_log or not review_log.exists():
        return latest
    for row in read_jsonl(review_log) or []:
        crop_id = row.get("crop_id")
        if crop_id:
            latest[crop_id] = row
    return latest


def collect_crop_rows(project_root: Path, metadata_path: Path, review_log: Optional[Path] = None, include_reviewed_paths: bool = True) -> List[dict]:
    """Collect rows for VAE training/indexing.

    - Always includes raw crop paths from metadata.jsonl.
    - Optionally adds reviewed_crop_path entries from review_log.jsonl. This helps train
      the VAE on accepted/rejected copies without changing the raw dataset.
    """
    rows: List[dict] = []
    seen_paths = set()
    reviews = load_latest_reviews(review_log)

    for item in read_jsonl(metadata_path) or []:
        row = dict(item)
        crop_path = row.get("crop_path")
        if not crop_path:
            continue
        full = project_root / crop_path
        if not full.exists():
            continue
        review = reviews.get(row.get("crop_id"))
        row["predicted_type"] = row.get("type")
        row["decision"] = review.get("decision") if review else None
        row["reviewed_type"] = review.get("reviewed_type") if review else None
        row["effective_type"] = row.get("reviewed_type") or row.get("type")
        row["bbox_quality"] = review.get("bbox_quality") if review else None
        row["source_kind"] = "raw"
        key = str(full.resolve())
        if key not in seen_paths:
            rows.append(row)
            seen_paths.add(key)

    if include_reviewed_paths and review_log and review_log.exists():
        for review in read_jsonl(review_log) or []:
            reviewed_path = review.get("reviewed_crop_path")
            if not reviewed_path:
                continue
            full = project_root / reviewed_path
            if not full.exists():
                continue
            row = dict(review)
            row["crop_path"] = reviewed_path
            row["predicted_type"] = review.get("type")
            row["effective_type"] = review.get("reviewed_type") or review.get("type")
            row["source_kind"] = "reviewed_copy"
            key = str(full.resolve())
            if key not in seen_paths:
                rows.append(row)
                seen_paths.add(key)

    return rows


class CropImageDataset(Dataset):
    def __init__(self, project_root: Path, rows: List[dict], image_size: int = 128):
        self.project_root = project_root
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.project_root / row["crop_path"]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.new("L", (128, 128), color=255)
        return self.transform(img), row


def collate_images_and_rows(batch):
    images = torch.stack([item[0] for item in batch])
    rows = [item[1] for item in batch]
    return images, rows
