
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")

    parser.add_argument("--weights", type=str, default="/app/weights/yolov8n.pt")
    parser.add_argument("--data", type=str, default="/app/data/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="/app/runs")
    parser.add_argument("--name", type=str, default="baseline_yolov8n")

    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
