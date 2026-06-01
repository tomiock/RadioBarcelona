import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to YOLO data.yaml')
    parser.add_argument('--model', default='yolo11n.pt', help='Base YOLO model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--name', default='stamp_detector')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project='runs/detect',
        patience=15,
        cache=False,
    )


if __name__ == '__main__':
    main()
