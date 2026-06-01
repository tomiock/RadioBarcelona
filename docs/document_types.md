# Radio Barcelona — Document Types by Era

The archive contains several visually distinct document types whose character changes significantly across decades. Understanding this variety is essential for any OCR or classification pipeline.

## Summary table

| Era | Document type | Text style | Annotation style |
|-----|--------------|------------|------------------|
| 1924–1935 | Pre-printed form / free-form | Handwritten ink | Corrections, additions in margins |
| 1936–1939 | Typewriter base + heavy annotation | Mixed | Strikethroughs, cross-links, multi-ink |
| 1940–1942 | Daily programme sheet | Typewriter | Sparse tick/X marks, stamps |
| 1943–1953 | Programme sheet + script transcripts | Typewriter | Official stamps, red annotations |

---

## 1924–1935 — Handwritten era

### 1925: Pre-printed table forms
The earliest documents use **pre-printed structured forms** with column headers (Hora, Autor título, Asunto, Ejecutantes, Observaciones). The actual programme content is **entirely handwritten in ink** — cursive, dense, with corrections and additions at the bottom under a "MODIFICACIONES" section. Paper quality is good; no degradation.

### 1931: Two sub-types already diverging
- **Free-form working notes**: clean, sparse cursive handwriting with wide line spacing. Disc reference numbers written in the left margin (e.g. `333g`, `58`). Personal and informal.
- **Carbon copies / denser documents**: heavier handwriting, darker background (carbon paper or different scan exposure). **First appearance of strikethroughs and cross-outs** — entire lines struck diagonally, corrections written over text.

Key OCR challenge: pure HTR required; stroke width and letter connectivity vary significantly between scribes.

---

## 1936–1939 — Mixed transition era

### 1936: First typewritten documents with handwritten annotation
The first typewritten base appears, but pages carry **heavy handwritten annotation on top**: underlines, interlinear insertions, marginal notes in multiple ink colours. The visual layering of typewriter + cursive closely matches what the synthetic document generator (`synthetic_docs_aran/generator.py`) is designed to replicate.

### 1938: Peak annotation complexity
The most visually complex documents in the sample set. Typewritten programme entries are connected by **diagonal lines linking programme slots to performers** — a unique annotation pattern that is neither a strikethrough nor a censorship mark, but a production cross-reference. Multiple ink colours (black, blue, red). Marginal stamps. Dense interlinear notes.

Key OCR challenge: at the line level, any given crop may be:
- Pure typewriter text
- Pure handwriting
- Typewriter with handwritten strikethrough
- Handwriting partially obscuring typewriter

The H/T classifier must be trained on crops from this era to avoid failing on the hardest cases.

---

## 1940–1942 — Early Franco-era standardisation

### 1940: Institutional reset
Clean typewriter on good paper. Sparse layout. The **UAB digitisation watermark** (bottom-right) appears consistently from this point. Hand annotations are now sparse: occasional margin notes, a few tick marks. `1940_2` shows the most complete daily programme format seen to this point — listing every slot from 8:00 to 23:00.

Key visual marker: documents become significantly lighter and more uniform; scan contrast improves.

---

## 1943–1953 — Mature institutional format

### 1943–1944: Fully standardised programme sheets
Official header: *"PROGRAMA DE 'RADIO BARCELONA' E.A.J.-1 / SOCIEDAD ESPAÑOLA DE RADIODIFUSIÓN"*. Date centred. Hourly slots with em-dash notation (`8h.--`). Hand annotations shift to **marginal tick marks and X symbols** — likely approved/rejected indicators — rather than full-text rewrites.

Two parallel document sub-types emerge and coexist:
1. **Guia-índice**: pre-printed column grid (Hora, Emisión, Título, Autores, Ejecutante). Stamped **"ORIGINAL LOCUTORIO"** in red — this is the studio broadcast copy.
2. **Relay/news feed pages**: sparse, often just a headline and a few lines. Headed `CF – ALFIL – Hoja número N`. Short, one per news item.

### 1944–1945: Script transcripts appear
Full radio scripts with **role labels** (Locutor / Locutora alternating) and dense typewriter text. Structurally different from programme sheets: no time slots, dialogue-format, can run to many pages. First examples of advertising scripts (sponsored programme inserts).

### 1949: Formal script format
Complete script format with section markers (SONIDO / LOCUTOR / LOCUTORA), **red official stamps** (circular seal + rectangular censor/review stamp), page numbering (`-1-`), and red pen annotations.

### 1950–1953: Full daily package
A typical day in 1950+ produces:
- 1× Guia-índice programme sheet (1 page)
- N× script pages (variable, can be 50–150 pages for a drama or variety show)
- Occasional relay feed pages

The `1950_1` Guia-índice is the most complete programme sheet in the sample — running from 7h30 to 22h with full cast and disc listings.

### 1953: Advertising scripts
`1953_3` is a **radio advertising script** for *Almacenes Ruiz* department store — actor/actress dialogue, clearly a sponsored programme insert. Suggests the later corpus contains significant non-editorial content.

**Note — 1953 duplicates**: `1953_1` and `1953_2` are byte-for-byte identical scans (8 792 905 bytes each). Deduplication required before training set construction.

---

## Implications for the OCR pipeline

### Line-level classification (H vs T)
The key boundary is at the **line crop level**, not the page level. A single 1938 page may contain both handwritten and typewritten lines. The H/T classifier must be trained with crops sampled from all eras, including mixed 1936–1939 pages.

### OCR engine routing
| Line type | Engine | Notes |
|-----------|--------|-------|
| Handwritten (H) | Loghi HTR | Needs a model fine-tuned or pre-trained on Spanish/Catalan historical cursive |
| Typewritten (T) | Tesseract | `spa+cat` language pack, PSM 7 (single line mode) |

### Stamp and strikethrough detection
These are **visual features independent of OCR text** and should run as separate detectors on the page image or line crop before OCR routing. The 1938 diagonal cross-reference pattern and the post-1949 red circle stamps are the two most distinct mark types not well-captured by a simple binary strikethrough classifier.
