# Stamp Detection module

Objectiu:
1. Crear dataset inicial per a detecció de segells.
2. Entrenar YOLO.
3. Exportar resultats en JSON compatible amb OCR/UI/LLM.

Estratègia recomanada:
- Primer: synthetic_docs_aran amb ground truth de segells guardat pel generator.
- Després: afegir 20-50 pàgines reals de Radio Barcelona anotades manualment.
- Finalment: entrenar/fine-tune YOLO i exportar JSON.

## Estructura esperada

```
stamp_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Cada label YOLO:

```
0 x_center y_center width height
```

Totes les coordenades normalitzades entre 0 i 1.

## Entrenar

```
pip install ultralytics opencv-python pillow pyyaml
python stamp_detection/train_yolo_stamps.py --data stamp_dataset/data.yaml --model yolo11n.pt --epochs 50
```

## Inferència i JSON

```
python stamp_detection/infer_yolo_stamps_json.py \
  --weights runs/detect/stamp_detector/weights/best.pt \
  --input data/raw_pages \
  --output outputs/stamp_json
```

## Baseline OpenCV

```
python stamp_detection/cv_baseline_stamps.py \
  --input data/raw_pages \
  --output outputs/cv_stamp_baseline
```

