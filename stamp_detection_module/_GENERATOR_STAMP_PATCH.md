# Patch necessari a synthetic_docs_aran/generator.py

El generator actual enganxa segells però no en desa la bbox. Per entrenar YOLO necessitem que cada sample guardi `stamp_detections.json`.

## Canvi 1: `_paste_asset_random` ha de retornar metadata

Substituir la signatura:

```python
def _paste_asset_random(... ) -> bool:
```

per:

```python
def _paste_asset_random(..., asset_class: str = "asset") -> dict | None:
```

I al final, en comptes de:

```python
canvas.paste(asset, (x, y), asset)
return True
```

fer:

```python
canvas.paste(asset, (x, y), asset)
return {
    "class": asset_class,
    "bbox": {"x1": x, "y1": y, "x2": x + asset.width, "y2": y + asset.height},
    "confidence": 1.0,
    "source": "synthetic_generator"
}
```

Els `return False` passen a ser `return None`.

## Canvi 2: `render_layer1` ha de acumular deteccions de stamps

Al principi de `render_layer1`:

```python
stamp_detections = []
```

Quan enganxes un stamp real:

```python
used_stamp = self._paste_asset_random(
    img,
    self.stamp_paths,
    min_scale=0.35,
    max_scale=0.85,
    rotation=45.0,
    margin=80,
    asset_class="official_stamp"
)
if used_stamp:
    stamp_detections.append(used_stamp)
    continue
```

Quan enganxes stamps extra:

```python
used_stamp = self._paste_asset_random(..., asset_class="official_stamp")
if used_stamp:
    stamp_detections.append(used_stamp)
```

En els stamps generats manualment, després de `img.paste(txt_img, (x, y), txt_img)` afegir:

```python
stamp_detections.append({
    "class": "synthetic_stamp",
    "bbox": {"x1": x, "y1": y, "x2": x + txt_img.width, "y2": y + txt_img.height},
    "confidence": 1.0,
    "source": "synthetic_generator"
})
```

I canviar el return de `render_layer1`:

```python
return img, stamp_detections
```

## Canvi 3: guardar `stamp_detections.json`

A `render_and_save_sync`, canviar:

```python
l1_img = self.renderer.render_layer1(word_boxes, anotaciones)
```

per:

```python
l1_img, stamp_detections = self.renderer.render_layer1(word_boxes, anotaciones)
```

I després de guardar `text_annotations.json`:

```python
with open(os.path.join(sample_dir, "stamp_detections.json"), "w", encoding="utf-8") as f:
    json.dump({
        "document_id": f"synthetic_{i:04d}",
        "page": 1,
        "image": f"final_merged_{i:04d}.jpg",
        "stamp_detections": [
            {
                "id": f"stamp_{i:04d}_{j:03d}",
                "class": det.get("class", "official_stamp"),
                "bbox": det["bbox"],
                "confidence": det.get("confidence", 1.0),
                "crop_path": None,
                "mask_path": None,
                "ocr_text": None,
                "notes": "synthetic ground truth"
            }
            for j, det in enumerate(stamp_detections, start=1)
        ]
    }, f, indent=2, ensure_ascii=False)
```
