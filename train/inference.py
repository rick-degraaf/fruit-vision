import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")

    parser.add_argument("--weights", type=str, default="/app/runs/baseline_yolov8n/weights/best.pt")
    parser.add_argument("--data", type=str, default="/app/data/images/val")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument('--save_txt', action='store_true', default=False)
    parser.add_argument('--save_conf', action='store_true', default=False)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="val_preds")

    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    model.predict(source=args.data, 
                conf=args.conf, 
                save_txt=args.save_txt, 
                save_conf=args.save_conf,
                project=args.project, 
                name=args.name)

if __name__ == "__main__":
    main()

