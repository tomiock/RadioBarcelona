import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

def load_lines(xml_dir: Path) -> dict[tuple[str, str], dict]:
    """
    Parse every XML in xml_dir and return a dict keyed by (stem, line_id).
    Each value has keys: text, engine, stem.
    """
    lines = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for tl in root.findall(f".//{{{NS}}}TextLine"):
            lid = tl.attrib.get("id", "")
            te  = tl.find(f"{{{NS}}}TextEquiv")
            u   = te.find(f"{{{NS}}}Unicode") if te is not None else None
            lines[(xml_path.stem, lid)] = {
                "text":   (u.text or "").strip() if u is not None else "",
                "engine": te.attrib.get("engine", "") if te is not None else "",
                "stem":   xml_path.stem,
            }
    return lines


def levenshtein(a: list, b: list) -> int:
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for ca in a:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j-1])
    return dp[len(b)]


def cer_pair(hyp: str, ref: str) -> tuple[int, int]:
    """(edit_distance_chars, ref_char_count)"""
    return levenshtein(list(hyp), list(ref)), len(ref)


def wer_pair(hyp: str, ref: str) -> tuple[int, int]:
    """(edit_distance_words, ref_word_count)"""
    return levenshtein(hyp.split(), ref.split()), len(ref.split())


class Bucket:
    def __init__(self):
        self.ced = self.rl = self.wed = self.wl = self.n = 0

    def add(self, hyp: str, ref: str):
        cd, rl = cer_pair(hyp, ref)
        wd, wl = wer_pair(hyp, ref)
        self.ced += cd; self.rl  += rl
        self.wed += wd; self.wl  += wl
        self.n   += 1

    @property
    def cer(self): return self.ced / self.rl  * 100 if self.rl  else 0.0
    @property
    def wer(self): return self.wed / self.wl  * 100 if self.wl  else 0.0


def bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_table(rows: list[tuple], headers: list[str], col_w: list[int]):
    sep = "  "
    hdr = sep.join(f"{h:{w}}" for h, w in zip(headers, col_w))
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for row in rows:
        print("  " + sep.join(f"{str(v):{w}}" for v, w in zip(row, col_w)))


def get_args():
    p = argparse.ArgumentParser(
        description="OCR benchmark: hypothesis PageXML vs ground-truth PageXML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gt",  required=True,
                   help="Ground-truth PageXML directory")
    p.add_argument("--hyp", required=True,
                   help="Hypothesis PageXML directory to evaluate")
    p.add_argument("--out", default="",
                   help="Optional path to write a JSON report")
    p.add_argument("--show-errors", type=int, default=0, metavar="N",
                   help="Print the N lines with the highest CER")
    p.add_argument("--skip-empty-ref", action="store_true",
                   help="Exclude lines where the GT text is empty")
    p.add_argument("--filter-engine", default="", metavar="ENGINE",
                   help="Only evaluate lines where the *hypothesis* engine matches "
                        "this value (e.g. tesseract, odaocr, vllm, corrected). "
                        "Comma-separate multiple values.")
    return p.parse_args()


def main():
    args    = get_args()
    gt_dir  = Path(args.gt)
    hyp_dir = Path(args.hyp)

    if not gt_dir.exists():
        raise SystemExit(f"GT dir not found: {gt_dir}")
    if not hyp_dir.exists():
        raise SystemExit(f"Hyp dir not found: {hyp_dir}")

    filter_engines = {e.strip() for e in args.filter_engine.split(",") if e.strip()}

    print(f"GT  : {gt_dir}")
    print(f"Hyp : {hyp_dir}")
    if filter_engines:
        print(f"Filter (hyp engine) : {', '.join(sorted(filter_engines))}")

    gt_lines  = load_lines(gt_dir)
    hyp_lines = load_lines(hyp_dir)

    gt_keys  = set(gt_lines)
    hyp_keys = set(hyp_lines)
    matched  = gt_keys & hyp_keys
    only_gt  = gt_keys - hyp_keys
    only_hyp = hyp_keys - gt_keys

    print(f"\nLines in GT      : {len(gt_keys)}")
    print(f"Lines in Hyp     : {len(hyp_keys)}")
    print(f"Matched by id    : {len(matched)}")
    if only_gt:
        print(f"Only in GT (missing from hyp, counted as empty): {len(only_gt)}")
    if only_hyp:
        print(f"Only in Hyp (no GT reference, ignored): {len(only_hyp)}")

    # lines only in GT are treated as hyp="" (model produced nothing)
    # optionally restrict to lines whose hypothesis engine matches the filter
    if filter_engines:
        eval_keys = {k for k in gt_keys
                     if (hyp_lines.get(k, {}).get("engine") or "none") in filter_engines}
        print(f"\nLines after engine filter : {len(eval_keys)}")
    else:
        eval_keys = gt_keys

    overall   = Bucket()
    by_page   = defaultdict(Bucket)
    by_engine = defaultdict(Bucket)
    per_line  = []

    for key in sorted(eval_keys):
        ref_entry = gt_lines[key]
        ref       = ref_entry["text"]
        hyp_entry = hyp_lines.get(key)
        hyp       = hyp_entry["text"] if hyp_entry else ""
        gt_eng    = ref_entry["engine"] or "none"
        hyp_eng   = (hyp_entry["engine"] if hyp_entry else "") or "none"
        stem      = ref_entry["stem"]

        if args.skip_empty_ref and not ref:
            continue

        overall.add(hyp, ref)
        by_page[stem].add(hyp, ref)
        by_engine[gt_eng].add(hyp, ref)

        cd, rl = cer_pair(hyp, ref)
        line_cer = cd / rl * 100 if rl else 0.0
        per_line.append({
            "stem": stem, "line_id": key[1],
            "ref": ref, "hyp": hyp,
            "cer": round(line_cer, 2),
            "gt_engine": gt_eng,
            "hyp_engine": hyp_eng,
        })

    print(f"\n{'─'*52}")
    print(f"  {'':24}  {'CER':>8}  {'WER':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Overall':24}  {overall.cer:>7.2f}%  {overall.wer:>7.2f}%")
    print(f"  (n={overall.n} lines, {overall.rl} ref chars, {overall.wl} ref words)")
    print(f"{'─'*52}")

    print("\nBy page:")
    rows = []
    for stem in sorted(by_page):
        b = by_page[stem]
        rows.append((stem, b.n, f"{b.cer:>6.1f}%", f"{b.wer:>6.1f}%",
                     bar(b.cer, 18)))
    print_table(rows,
                ["page", "n", "CER", "WER", ""],
                [16, 5, 8, 8, 20])

    print("\nBy GT engine:")
    rows = []
    for eng in sorted(by_engine):
        b = by_engine[eng]
        rows.append((eng or "(none)", b.n, f"{b.cer:>6.1f}%", f"{b.wer:>6.1f}%"))
    print_table(rows,
                ["engine", "n", "CER", "WER"],
                [12, 6, 8, 8])

    if args.show_errors:
        worst = sorted(per_line, key=lambda x: x["cer"], reverse=True)
        print(f"\nWorst {args.show_errors} lines by CER:")
        for i, e in enumerate(worst[:args.show_errors], 1):
            print(f"  {i:>3}. [{e['stem']}]  CER={e['cer']:.1f}%")
            print(f"       ref: {repr(e['ref'])}")
            print(f"       hyp: {repr(e['hyp'])}")

    if args.out:
        report = {
            "gt_dir":  str(gt_dir),
            "hyp_dir": str(hyp_dir),
            "overall": {
                "n_lines": overall.n,
                "cer":     round(overall.cer, 4),
                "wer":     round(overall.wer, 4),
                "ref_chars": overall.rl,
                "ref_words": overall.wl,
            },
            "by_page": {
                stem: {"n": b.n, "cer": round(b.cer, 4), "wer": round(b.wer, 4)}
                for stem, b in sorted(by_page.items())
            },
            "by_engine": {
                eng: {"n": b.n, "cer": round(b.cer, 4), "wer": round(b.wer, 4)}
                for eng, b in sorted(by_engine.items())
            },
            "lines": per_line,
        }
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nJSON report written to: {args.out}")


if __name__ == "__main__":
    main()
