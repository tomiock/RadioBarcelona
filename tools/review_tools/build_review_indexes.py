#!/usr/bin/env python3
"""
Build review indexes and export packages from metadata.jsonl + review_log.jsonl.

This script is intentionally file-based: no database is required.
It creates JSONL shards by status/type and an optional export package that can be
used by the current retraining pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

EXPORTABLE_TYPES = {
    "stamp",
    "handwritten_text",
    "typewritten_text",
    "crossout",
    "censorship_block",
    "table_fragment",
}
EXPORTABLE_BBOX_QUALITIES = {"good", "minor_partial"}


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_latest_reviews(review_log: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for entry in read_jsonl(review_log) or []:
        crop_id = entry.get("crop_id")
        if crop_id:
            latest[crop_id] = entry
    return latest


def is_exportable(entry: dict) -> bool:
    return (
        entry.get("decision") == "accepted"
        and entry.get("bbox_quality") in EXPORTABLE_BBOX_QUALITIES
        and entry.get("reviewed_type") in EXPORTABLE_TYPES
    )


def enrich_item(item: dict, review: Optional[dict]) -> dict:
    row = dict(item)
    row["predicted_type"] = item.get("type")

    if review:
        row["decision"] = review.get("decision")
        row["reviewed_type"] = review.get("reviewed_type")
        row["bbox_quality"] = review.get("bbox_quality")
        row["human_confidence"] = review.get("human_confidence")
        row["attributes"] = review.get("attributes", [])
        row["review_notes"] = review.get("review_notes")
        row["reviewed_crop_path"] = review.get("reviewed_crop_path")
        row["corrected_bbox"] = review.get("corrected_bbox")
    else:
        row["decision"] = None
        row["reviewed_type"] = None
        row["bbox_quality"] = None
        row["human_confidence"] = None
        row["attributes"] = []
        row["review_notes"] = None
        row["reviewed_crop_path"] = None
        row["corrected_bbox"] = None

    row["effective_type"] = row.get("reviewed_type") or row.get("type")
    row["is_reviewed"] = row.get("decision") in {"accepted", "rejected"}
    row["is_pending"] = row.get("decision") is None
    row["is_exportable"] = is_exportable(row)
    row["is_accepted_not_exportable"] = row.get("decision") == "accepted" and not row["is_exportable"]
    row["is_false_positive"] = row.get("decision") == "rejected" or row.get("reviewed_type") == "false_positive"
    return row


def status_name(row: dict) -> str:
    if row.get("is_exportable"):
        return "exportable"
    if row.get("is_accepted_not_exportable"):
        return "accepted_not_exportable"
    if row.get("decision") == "accepted":
        return "accepted"
    if row.get("decision") == "rejected":
        return "rejected"
    if row.get("decision") == "skipped":
        return "skipped"
    return "pending"


def build_rows(metadata_path: Path, review_log: Path) -> List[dict]:
    latest_reviews = load_latest_reviews(review_log)
    rows: List[dict] = []
    for item in read_jsonl(metadata_path) or []:
        crop_id = item.get("crop_id")
        review = latest_reviews.get(crop_id) if crop_id else None
        rows.append(enrich_item(item, review))
    return rows


def compute_stats(rows: List[dict]) -> dict:
    stats = {
        "total_crops": len(rows),
        "pending": 0,
        "reviewed": 0,
        "accepted": 0,
        "rejected": 0,
        "skipped": 0,
        "exportable": 0,
        "accepted_not_exportable": 0,
        "false_positive": 0,
        "by_type": {},
        "by_effective_type": {},
        "by_bbox_quality": {},
        "by_attribute": {},
    }

    for row in rows:
        decision = row.get("decision")
        if decision in {"accepted", "rejected"}:
            stats["reviewed"] += 1
        if decision == "accepted":
            stats["accepted"] += 1
        elif decision == "rejected":
            stats["rejected"] += 1
        elif decision == "skipped":
            stats["skipped"] += 1
        elif decision is None:
            stats["pending"] += 1

        if row.get("is_exportable"):
            stats["exportable"] += 1
        if row.get("is_accepted_not_exportable"):
            stats["accepted_not_exportable"] += 1
        if row.get("is_false_positive"):
            stats["false_positive"] += 1

        for key, value in [
            ("by_type", row.get("type") or "unknown"),
            ("by_effective_type", row.get("effective_type") or "unknown"),
            ("by_bbox_quality", row.get("bbox_quality") or "unspecified"),
        ]:
            stats[key][value] = stats[key].get(value, 0) + 1

        for attr in row.get("attributes", []) or []:
            stats["by_attribute"][attr] = stats["by_attribute"].get(attr, 0) + 1

    return stats


def copy_crop(row: dict, project_root: Path, target_root: Path) -> None:
    src_rel = row.get("reviewed_crop_path") or row.get("crop_path")
    if not src_rel:
        return
    src = project_root / src_rel
    if not src.exists():
        return
    cls = row.get("effective_type") or row.get("type") or "unknown"
    target_dir = target_root / cls
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target_dir / src.name)


def build_indexes(project_root: Path, metadata_path: Path, review_log: Path, output_dir: Path, export_package: bool) -> Path:
    rows = build_rows(metadata_path, review_log)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main index files.
    write_jsonl(output_dir / "all.jsonl", rows)
    write_jsonl(output_dir / "by_status" / "pending.jsonl", (r for r in rows if status_name(r) == "pending"))
    write_jsonl(output_dir / "by_status" / "reviewed.jsonl", (r for r in rows if r.get("is_reviewed")))
    write_jsonl(output_dir / "by_status" / "accepted.jsonl", (r for r in rows if r.get("decision") == "accepted"))
    write_jsonl(output_dir / "by_status" / "rejected.jsonl", (r for r in rows if r.get("decision") == "rejected"))
    write_jsonl(output_dir / "by_status" / "skipped.jsonl", (r for r in rows if r.get("decision") == "skipped"))
    write_jsonl(output_dir / "by_status" / "exportable.jsonl", (r for r in rows if r.get("is_exportable")))
    write_jsonl(output_dir / "by_status" / "accepted_not_exportable.jsonl", (r for r in rows if r.get("is_accepted_not_exportable")))

    by_type = {}
    for row in rows:
        cls = row.get("effective_type") or row.get("type") or "unknown"
        by_type.setdefault(cls, []).append(row)
    for cls, cls_rows in by_type.items():
        safe_cls = str(cls).replace("/", "_")
        write_jsonl(output_dir / "by_type" / f"{safe_cls}.jsonl", cls_rows)

    stats = compute_stats(rows)
    stats_path = output_dir / "review_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "metadata_path": str(metadata_path.relative_to(project_root) if metadata_path.is_relative_to(project_root) else metadata_path),
        "review_log": str(review_log.relative_to(project_root) if review_log.is_relative_to(project_root) else review_log),
        "output_dir": str(output_dir.relative_to(project_root) if output_dir.is_relative_to(project_root) else output_dir),
        "stats": stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if export_package:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = project_root / "outputs" / "review_exports" / f"export_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        exportable_rows = [r for r in rows if r.get("is_exportable")]
        rejected_rows = [r for r in rows if r.get("is_false_positive")]
        accepted_not_exportable_rows = [r for r in rows if r.get("is_accepted_not_exportable")]

        write_jsonl(export_dir / "exportable_metadata.jsonl", exportable_rows)
        write_jsonl(export_dir / "rejected_metadata.jsonl", rejected_rows)
        write_jsonl(export_dir / "accepted_not_exportable_metadata.jsonl", accepted_not_exportable_rows)
        (export_dir / "review_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        if review_log.exists():
            shutil.copy2(review_log, export_dir / "review_log_snapshot.jsonl")

        for row in exportable_rows:
            copy_crop(row, project_root, export_dir / "accepted_crops")
        for row in rejected_rows:
            copy_crop(row, project_root, export_dir / "rejected_crops")

        (export_dir / "manifest.json").write_text(json.dumps({
            "created_at": timestamp,
            "source_index_dir": str(output_dir.relative_to(project_root)),
            "num_exportable": len(exportable_rows),
            "num_rejected": len(rejected_rows),
            "num_accepted_not_exportable": len(accepted_not_exportable_rows),
            "note": "Use exportable_metadata.jsonl and accepted_crops/ for current retraining. Rejected crops can be used as false-positive hard negatives if the training pipeline supports them.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return export_dir

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata", default="outputs/object_crops_raw/metadata.jsonl")
    parser.add_argument("--review-log", default="outputs/review_logs/review_log.jsonl")
    parser.add_argument("--output-dir", default="outputs/index")
    parser.add_argument("--export-package", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    metadata_path = (project_root / args.metadata).resolve()
    review_log = (project_root / args.review_log).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    result = build_indexes(project_root, metadata_path, review_log, output_dir, args.export_package)
    print(f"✅ Review indexes written to: {output_dir}")
    if args.export_package:
        print(f"✅ Export package written to: {result}")


if __name__ == "__main__":
    main()
