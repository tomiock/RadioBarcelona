# Àlex Update — Synthetic Documents, Layout Annotations i Detection Pipeline

Branca: `stamp-detection-alex`

Aquest document resumeix la nova pipeline perquè el generador creï **documents sintètics + anotacions posicionals reutilitzables** per entrenar detectors, ajudar l’OCR, alimentar el prototype i preparar separació de capes / GAN / U-Net / segmentació.

---

## 1. Idea general

```text
generator.py genera imatges sintètiques
↓
guarda layout_annotations.json amb la posició de cada element
↓
annotation_tools converteix aquestes anotacions a JSONL o YOLO
↓
YOLO aprèn a detectar visualment stamps, crossouts, handwriting, tables...
```

Durant la generació sabem on és cada element perquè el generador l’ha col·locat. Després, YOLO aprèn a detectar aquests elements en imatges noves.

---

## 2. Directoris principals

```text
RadioBarcelona-main/
├── synthetic_docs_aran/
├── annotation_tools/
├── detection/
├── visual_marks_dataset/       # generat, no pujar normalment
├── stamp_dataset_layout/       # generat, no pujar normalment
├── stamp_dataset/              # generat, pipeline antiga de stamps
├── outputs/                    # generat, no pujar normalment
├── runs/                       # generat, no pujar normalment
├── yolo11n.pt                  # pes base YOLO, no pujar normalment
└── .gitignore
```

---

## 3. Generació sintètica

Directori principal:

```text
synthetic_docs_aran/
├── generator.py
├── assets/
├── iam_samples/
├── fonts/
├── typewriter_fonts/
├── output_dataset_pro_try/     # generat
└── annotations/                # generat
```

### `generator.py`

Genera documents sintètics amb:

```text
text mecanografiat
stamps / segells
crossouts / tatxadures
censorship blocks
handwritten text
table fragments
textures de paper
capes separades
imatge final
layout_annotations.json
```

Ordre per executar:

```bash
cd RadioBarcelona-main
source .venv/bin/activate

cd synthetic_docs_aran
USE_GEMINI=false USE_OPENAI=false TOTAL_SAMPLES=200 python generator.py
cd ..
```

Opcions importants:

```text
TOTAL_SAMPLES=200       nombre de documents
USE_GEMINI=false        no usar Gemini
USE_OPENAI=false        no usar OpenAI
```

Si Gemini/OpenAI estan activats, el text pot venir d’un LLM. Si no, es fa servir el generador local.

---

## 4. Assets utilitzats

```text
synthetic_docs_aran/assets/
├── stamps/
├── censorship/
├── erasures/
├── patches/
├── stains/
├── paper_textures/
└── tables/
```

Ús actual:

```text
assets/stamps/        → type: stamp, subtype: official_stamp
assets/censorship/    → type: crossout o censorship_block
assets/erasures/      → type: crossout, subtype: erasure_asset
assets/tables/        → type: table_fragment
assets/patches/       → augmentació visual
assets/stains/        → augmentació visual
paper_textures/       → fons del document
```

També:

```text
iam_samples/          → handwritten_text, subtype: iam_sample
fonts/                → handwritten_text, subtype: procedural_font
typewriter_fonts/     → typewritten_word i typewritten_text
```

Nota: si els assets d’`erasures/` són tatxadures o guixades, els tractem com `crossout`. Si algun dia hi ha corrector blanc o raspats reals, es podrien tractar com `erasure` o `occlusion`, però ara no és prioritari.

---

## 5. Output del generador

El generador crea:

```text
synthetic_docs_aran/output_dataset_pro_try/
├── all_final_images/
│   ├── final_merged_0000.jpg
│   ├── final_merged_0001.jpg
│   └── ...
├── sample_0000/
│   ├── text_original.txt
│   ├── text_annotations.json
│   ├── stamp_detections.json
│   ├── layout_annotations.json
│   ├── image_layer0_clean.png
│   └── image_layer1_clean.png
└── sample_0001/
    └── ...
```

Fitxers importants:

```text
final_merged_XXXX.jpg       imatge final sintètica
image_layer0_clean.png      capa de text mecanografiat
image_layer1_clean.png      capa d’anotacions, stamps i censura
layout_annotations.json     anotacions posicionals generals
stamp_detections.json       compatibilitat amb el mòdul inicial de stamps
```

---

## 6. `layout_annotations.json`

És el fitxer principal nou. Exemple:

```json
{
  "document_id": "synthetic_0000",
  "page": 1,
  "image": "final_merged_0000.jpg",
  "image_width": 1654,
  "image_height": 2339,
  "objects": [
    {
      "id": "stamp_0000_001",
      "type": "stamp",
      "subtype": "official_stamp",
      "bbox": {"x1": 300, "y1": 500, "x2": 600, "y2": 700},
      "text": null,
      "layer": "layer1_annotations",
      "source": "synthetic_generator",
      "confidence": 1.0
    }
  ]
}
```

Classes actuals:

```text
typewritten_word
typewritten_text
handwritten_text
stamp
crossout
censorship_block
table_fragment
```

Subtypes habituals:

```text
official_stamp
synthetic_stamp
iam_sample
procedural_font
long_note
erasure_asset
censorship_asset
table_asset
```

Diferència important:

```text
type     = categoria funcional principal
subtype  = origen, variant o estil
```

Exemple:

```text
type: stamp
subtype: official_stamp / synthetic_stamp
```

---

## 7. Annotation tools

Directori:

```text
annotation_tools/
├── build_annotations_jsonl.py
└── layout_to_yolo.py
```

### `build_annotations_jsonl.py`

Converteix molts `layout_annotations.json` en fitxers `.jsonl`.

Ordre:

```bash
python annotation_tools/build_annotations_jsonl.py \
  --synthetic-root synthetic_docs_aran/output_dataset_pro_try \
  --output-dir synthetic_docs_aran/annotations
```

Crea:

```text
synthetic_docs_aran/annotations/
├── annotations_train.jsonl
├── annotations_val.jsonl
└── annotations_test.jsonl
```

Això és important per datasets grans perquè permet llegir línia a línia, sense carregar tots els JSONs en memòria.

### `layout_to_yolo.py`

Converteix `layout_annotations.json` a format YOLO.

Ordre actual recomanada:

```bash
python annotation_tools/layout_to_yolo.py \
  --synthetic-root synthetic_docs_aran/output_dataset_pro_try \
  --output visual_marks_dataset \
  --classes stamp crossout censorship_block handwritten_text table_fragment \
  --min-box-size 40
```

Genera:

```text
visual_marks_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Classes YOLO actuals:

```text
0: stamp
1: crossout
2: censorship_block
3: handwritten_text
4: table_fragment
```

---

## 8. Entrenar YOLO

Script:

```text
detection/stamp_detection/train_yolo_stamps.py
```

Ordre:

```bash
python detection/stamp_detection/train_yolo_stamps.py \
  --data visual_marks_dataset/data.yaml \
  --model yolo11n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 4 \
  --name visual_marks_detector
```

Sortida:

```text
runs/detect/visual_marks_detector/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── val_batch*_pred.jpg
```

Tot i que el script es diu `train_yolo_stamps.py`, ara també pot entrenar detectors multi-classe si el `data.yaml` té més classes.

---

## 9. Utilitat per OCR

El format ajuda l’OCR perquè indica quines zones són:

```text
typewritten_text      text mecanografiat, útil per OCR principal
handwritten_text      text manuscrit, pot requerir OCR separat
stamp                 segell, pot ignorar-se o processar-se separat
crossout              zona tatxada, OCR no fiable
censorship_block      zona censurada, OCR no fiable
table_fragment        possible estructura tabular
```

Això evita barrejar text mecanografiat amb segells, ratllades o manuscrit.

---

## 10. Utilitat per GAN / separació de capes

El generador dona parelles útils:

```text
final_merged.jpg         entrada combinada/degradada
image_layer0_clean.png   text mecanografiat net
image_layer1_clean.png   anotacions visuals
layout_annotations.json  bboxes i classes
```

Pot servir per entrenar o avaluar:

```text
GAN
U-Net
segmentació
inpainting
separació de capes
restauració documental
```

No cal que sigui estrictament una GAN. Les mateixes dades poden servir per segmentació, restauració o separació de capes.

---

## 11. Outputs que normalment no s’han de pujar al Git

```text
synthetic_docs_aran/output_dataset_pro_try/
synthetic_docs_aran/annotations/
visual_marks_dataset/
stamp_dataset_layout/
stamp_dataset/
runs/
outputs/
*.pt
yolo11n.pt
```

Són artefactes generats i es poden regenerar amb les ordres anteriors.

El Git hauria de contenir principalment:

```text
synthetic_docs_aran/generator.py
annotation_tools/
detection/
.gitignore
README.md
README_STAMP_AND_LAYOUT.md
demo petita si cal
```

---

## 12. Següent pas: supervisió manual real

Validar sobre documents reals.

Estructura recomanada:

```text
real_reviewed_dataset/
├── all_final_images/
│   ├── rb_0001.jpg
│   ├── rb_0002.jpg
│   └── ...
├── sample_0001/
│   └── layout_annotations.json
├── sample_0002/
│   └── layout_annotations.json
└── README.md
```

Mateix format que el sintètic, però amb:

```json
{
  "type": "stamp",
  "subtype": "manual_real",
  "source": "manual_annotation",
  "reviewed": true,
  "validated_by": "human"
}
```

Pla mínim:

```text
20 pàgines reals anotades manualment
classes: stamp, crossout, censorship_block, handwritten_text, table_fragment
validar 5-10 pàgines per una segona persona
entrenar amb sintètic
avaluar sobre real
fer fine-tuning amb real
comparar abans/després
```

---

## 13. Resum final

Ara el sistema:

```text
genera documents sintètics
sap on posa cada element
desa aquesta informació a layout_annotations.json
converteix les anotacions a JSONL o YOLO
entrena models per detectar aquests elements
prepara el camí per OCR, prototype i supervisió real
```

La idea final és combinar:

```text
moltes dades sintètiques anotades automàticament
+
poques dades reals anotades manualment
=
detector més útil per documents reals de Radio Barcelona
```
