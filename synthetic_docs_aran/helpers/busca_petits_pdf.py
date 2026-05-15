import re
import csv
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://ddd.uab.cat"
SEARCH_URL = "https://ddd.uab.cat/search?cc=fonper&ln=ca&p=Guions+de+R%C3%A0dio+Barcelona&f=title&jrec={}"

OUT_DIR = Path("data/raw_pdfs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_record_urls():
    urls = set()

    for jrec in [1, 11, 21]:
        print(f"Llegint pàgina de cerca jrec={jrec}...")
        html = requests.get(SEARCH_URL.format(jrec), headers=HEADERS, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "record/" in href:
                full = urljoin(BASE, href).split("?")[0] + "?ln=ca"
                urls.add(full)

    return sorted(urls)


def extract_pdfs_from_record(record_url):
    print(f"\nEntrant a registre: {record_url}")

    html = requests.get(record_url, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    text_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in soup.get_text("\n").split("\n")
        if line.strip()
    ]

    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower():
            pdf_links.append(urljoin(BASE, href).split("?")[0])

    pdf_links = sorted(set(pdf_links))

    items = []

    for pdf_url in pdf_links:
        filename = pdf_url.split("/")[-1]

        m_day = re.search(r"a(19\d{2})m(\d{1,2})d(\d{1,2})(?:-\d{1,2})?", filename)
        m_month = re.search(r"a(19\d{2})m(\d{1,2})", filename)

        if m_day:
            year, month, day = m_day.groups()
            date_key = f"{year}-{int(month):02d}-{int(day):02d}"
        elif m_month:
            year, month = m_month.groups()
            date_key = f"{year}-{int(month):02d}"
        else:
            continue

        pages = find_pages_for_pdf(filename, text_lines)

        items.append({
            "year": year,
            "date": date_key,
            "pages": pages,
            "url": pdf_url,
            "filename": filename,
        })

    if items:
        years = sorted(set(i["year"] for i in items))
        print(f"  PDFs trobats: {len(items)} | anys: {', '.join(years)}")
    else:
        print("  Cap PDF trobat.")

    return items


def find_pages_for_pdf(filename, lines):
    """
    Busca la línia on apareix el PDF i mira les línies properes
    per trobar coses com '2 p', '36 p', etc.
    """
    for i, line in enumerate(lines):
        if filename in line:
            window = " ".join(lines[max(0, i-3): min(len(lines), i+6)])
            m = re.search(r"(\d+)\s*p\b", window)
            if m:
                return int(m.group(1))

    # fallback: mirar tot el text, però només si hi ha una aparició clara
    candidates = []
    for line in lines:
        m = re.search(r"(\d+)\s*p\b", line)
        if m:
            candidates.append(int(m.group(1)))

    return 9999


def download(item, rank):
    year = item["year"]
    date = item["date"]
    pages = item["pages"]
    url = item["url"]
    filename = item["filename"]

    out_name = f"{year}_{rank:02d}_{date}_{pages}p_{filename}"
    out_path = OUT_DIR / out_name

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  Ja existeix: {out_name}")
        return str(out_path)

    print(f"  Descarregant {date} ({pages} p): {filename}")

    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    out_path.write_bytes(r.content)
    return str(out_path)


def main():
    records = get_record_urls()
    print(f"\nRegistres anuals trobats: {len(records)}")

    all_items = []

    for idx, record_url in enumerate(records, start=1):
        print(f"\n[{idx}/{len(records)}]")
        items = extract_pdfs_from_record(record_url)
        all_items.extend(items)

    by_year = {}

    for item in all_items:
        by_year.setdefault(item["year"], []).append(item)

    selected = []

    print("\n==============================")
    print("Seleccionant 2 més petits per any")
    print("==============================")

    for year in sorted(by_year.keys()):
        items = by_year[year]
        items_sorted = sorted(items, key=lambda x: (x["pages"], x["date"]))
        chosen = items_sorted[:2]

        print(f"\n{year}:")
        for rank, item in enumerate(chosen, start=1):
            print(f"  #{rank}: {item['date']} — {item['pages']} p")
            local_path = download(item, rank)

            selected.append({
                "year": item["year"],
                "date": item["date"],
                "pages": item["pages"],
                "url": item["url"],
                "local_path": local_path,
            })

    csv_path = OUT_DIR / "selected_2_per_year.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["year", "date", "pages", "url", "local_path"]
        )
        writer.writeheader()
        writer.writerows(selected)

    print("\nFet!")
    print(f"PDFs guardats a: {OUT_DIR}")
    print(f"Metadata guardada a: {csv_path}")


if __name__ == "__main__":
    main()