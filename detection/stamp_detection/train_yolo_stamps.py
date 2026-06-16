import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default="stamp_detector")
    parser.add_argument("--project", default="runs/detect", help="Output folder for YOLO runs")
    parser.add_argument("--device", default=None, help="cpu, cuda, 0, 1... Leave empty for auto")
    args = parser.parse_args()

    # Convert project to absolute path to avoid duplicated paths like:
    # runs/detect/runs/detect/stamp_detector
    project_path = Path(args.project).resolve()

    model = YOLO(args.model)

    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
        "project": str(project_path),
        "patience": 15,
        "cache": False,
        "exist_ok": True,
    }

    if args.device is not None:
        train_args["device"] = args.device

    model.train(**train_args)


if __name__ == "__main__":
    main()