#!/usr/bin/env python3
"""
Sample random pages from UAB DDD Radio Barcelona search results.

Purpose:
    - Download a limited number of PDFs from DDD search/record pages.
    - Convert a random subset of pages to JPG.
    - Write a manifest.jsonl with record URL, PDF URL, local paths and page numbers.

Default search URL targets: "Guions de Ràdio Barcelona" in the UAB DDD.

Requirements:
    pip install requests beautifulsoup4
    sudo apt install poppler-utils

Typical use:
    python tools/data_tools/sample_ddd_random_pages.py --num-records 3 --pages-per-record 5 --seed 42

Dry run:
    python tools/data_tools/sample_ddd_random_pages.py --num-records 3 --pages-per-record 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_SEARCH_URL = (
    "https://ddd.uab.cat/search?cc=fonper&ln=ca&jrec=1"
    "&p=Guions+de+R%C3%A0dio+Barcelona&f=title"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RadioBarcelonaResearchSampler/1.0; educational project)"
}


@dataclass
class Record:
    record_id: str
    record_url: str
    title: str | None = None
    year: str | None = None
    pdf_url: str | None = None


def fetch_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def record_id_from_url(url: str) -> str:
    m = re.search(r"/record/(\d+)", url)
    if m:
        return m.group(1)
    parsed = urlparse(url)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", parsed.path.strip("/") or "record")
    return safe[:80]


def guess_year(text: str | None) -> str | None:
    if not text:
        return None
    years = re.findall(r"\b(19[0-9]{2}|20[0-9]{2})\b", text)
    return years[0] if years else None


def discover_record_links(search_url: str, max_search_pages: int = 5) -> list[Record]:
    """Collect unique /record/<id> links from the search results pages."""
    records: dict[str, Record] = {}

    # DDD search pages often use jrec=1, 11, 21... for pagination. We try a few offsets.
    urls = [search_url]
    if "jrec=" in search_url:
        for i in range(1, max_search_pages):
            urls.append(re.sub(r"jrec=\d+", f"jrec={1 + 10 * i}", search_url))

    for url in urls:
        try:
            html = fetch_text(url)
        except Exception as e:
            print(f"[WARN] Could not fetch search page {url}: {e}", file=sys.stderr)
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/record/" not in href:
                continue
            full_url = urljoin(url, href).split("#", 1)[0]
            # Normalize record URL to /record/<id> if possible.
            rid = record_id_from_url(full_url)
            base_record = re.sub(r"(/record/\d+).*", r"\1", full_url)
            title = " ".join(a.get_text(" ", strip=True).split()) or None
            if rid not in records:
                records[rid] = Record(
                    record_id=rid,
                    record_url=base_record,
                    title=title,
                    year=guess_year(title),
                )

    return list(records.values())


def find_pdf_url(record: Record) -> Record:
    try:
        html = fetch_text(record.record_url)
    except Exception as e:
        print(f"[WARN] Could not fetch record {record.record_url}: {e}", file=sys.stderr)
        return record

    soup = BeautifulSoup(html, "html.parser")

    title_text = soup.get_text(" ", strip=True)
    if not record.year:
        record.year = guess_year(title_text)

    # Prefer real DDD document PDFs.
    # IMPORTANT: DDD record pages also include footer/logo links to external PDF files
    # such as Recolecta/Fecyt repository certificates. We must ignore those.
    record_host = urlparse(record.record_url).netloc
    pdf_candidates = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(record.record_url, href).split("#", 1)[0]
        parsed = urlparse(full_url)
        lowered = full_url.lower()

        if not lowered.endswith(".pdf") and ".pdf" not in lowered:
            continue

        # Keep only PDFs hosted by ddd.uab.cat, not external footer links.
        if parsed.netloc != record_host:
            continue

        # Guions de Ràdio Barcelona PDFs are normally under /pub/guiradbcn/.
        # Keep this as a strong preference/filter to avoid unrelated site PDFs.
        if "/pub/" not in parsed.path:
            continue

        pdf_candidates.append(full_url)

    pdf_candidates = sorted(set(pdf_candidates))

    if pdf_candidates:
        # Random but reproducible because main() sets random.seed().
        record.pdf_url = random.choice(pdf_candidates)

    return record


def download_pdf(record: Record, raw_dir: Path, max_pdf_mb: float, dry_run: bool) -> Path | None:
    if not record.pdf_url:
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / f"ddd_record_{record.record_id}.pdf"

    if dry_run:
        print(f"[DRY] Would download {record.pdf_url} -> {pdf_path}")
        return pdf_path

    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    with requests.get(record.pdf_url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        size = int(r.headers.get("content-length") or 0)
        if size and size > max_pdf_mb * 1024 * 1024:
            print(f"[SKIP] PDF too large ({size/1024/1024:.1f} MB): {record.pdf_url}")
            return None

        tmp_path = pdf_path.with_suffix(".pdf.part")
        downloaded = 0
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_pdf_mb * 1024 * 1024:
                    f.close()
                    tmp_path.unlink(missing_ok=True)
                    print(f"[SKIP] PDF exceeded max size while downloading: {record.pdf_url}")
                    return None
                f.write(chunk)
        tmp_path.rename(pdf_path)

    return pdf_path


def require_binary(name: str) -> bool:
    return shutil.which(name) is not None


def get_pdf_page_count(pdf_path: Path) -> int | None:
    if require_binary("pdfinfo"):
        result = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.lower().startswith("pages:"):
                    try:
                        return int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

    # Fallback to PyMuPDF if installed.
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return None


def convert_page_to_jpg(pdf_path: Path, page_number: int, output_path: Path, dpi: int, dry_run: bool) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY] Would render page {page_number}: {pdf_path} -> {output_path}")
        return True

    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    if require_binary("pdftoppm"):
        prefix = output_path.with_suffix("")
        cmd = [
            "pdftoppm",
            "-jpeg",
            "-r",
            str(dpi),
            "-f",
            str(page_number),
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            return True
        print(f"[WARN] pdftoppm failed for {pdf_path} page {page_number}: {result.stderr}", file=sys.stderr)

    # Fallback to PyMuPDF.
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(output_path))
        doc.close()
        return output_path.exists()
    except Exception as e:
        print(f"[ERROR] Could not render {pdf_path} page {page_number}: {e}", file=sys.stderr)
        return False


def append_manifest(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--output-dir", default="data/ddd_random")
    parser.add_argument("--num-records", type=int, default=3)
    parser.add_argument("--pages-per-record", type=int, default=5)
    parser.add_argument("--max-search-pages", type=int, default=5)
    parser.add_argument("--max-pdf-mb", type=float, default=200.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_pdfs"
    pages_dir = output_dir / "pages"
    manifest_path = output_dir / "manifest.jsonl"

    if not require_binary("pdftoppm"):
        print("[WARN] pdftoppm not found. Install poppler-utils or PyMuPDF for rendering.", file=sys.stderr)
    if not require_binary("pdfinfo"):
        print("[WARN] pdfinfo not found. Install poppler-utils or PyMuPDF for page counts.", file=sys.stderr)

    print(f"Search URL: {args.search_url}")
    records = discover_record_links(args.search_url, max_search_pages=args.max_search_pages)
    print(f"Discovered records: {len(records)}")

    random.shuffle(records)
    selected_records = records[: args.num_records]

    if not selected_records:
        print("No records found. Check the search URL or DDD HTML structure.")
        return 1

    rendered = 0
    for record in selected_records:
        record = find_pdf_url(record)
        if not record.pdf_url:
            print(f"[SKIP] No PDF found for record {record.record_id}: {record.record_url}")
            continue

        pdf_path = download_pdf(record, raw_dir=raw_dir, max_pdf_mb=args.max_pdf_mb, dry_run=args.dry_run)
        if not pdf_path:
            continue

        if args.dry_run:
            # Cannot know page count without download; record enough info.
            continue

        page_count = get_pdf_page_count(pdf_path)
        if not page_count:
            print(f"[SKIP] Could not determine page count: {pdf_path}")
            continue

        n_pages = min(args.pages_per_record, page_count)
        page_numbers = sorted(random.sample(range(1, page_count + 1), k=n_pages))

        for page_number in page_numbers:
            page_path = pages_dir / f"ddd_record_{record.record_id}_page_{page_number:04d}.jpg"
            ok = convert_page_to_jpg(pdf_path, page_number, page_path, dpi=args.dpi, dry_run=args.dry_run)
            if not ok:
                continue

            row = {
                "source": "ddd_uab_random_sampler",
                "record_id": record.record_id,
                "record_url": record.record_url,
                "file_url": record.pdf_url,
                "search_url": args.search_url,
                "title": record.title,
                "year": record.year,
                "local_pdf_path": str(pdf_path),
                "local_page_path": str(page_path),
                "page_number": page_number,
                "page_count": page_count,
                "sample_seed": args.seed,
                "dpi": args.dpi,
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            }
            append_manifest(manifest_path, row)
            rendered += 1
            print(f"[OK] {page_path}")

    print()
    print(f"Rendered pages: {rendered}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
