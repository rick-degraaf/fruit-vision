from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/app/data/data.yaml",
    imgsz=640, batch=8, workers=2,
    epochs=40,
    seed=0,
    project="runs", name="baseline_yolov8n",
)