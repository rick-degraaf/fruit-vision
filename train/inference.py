from ultralytics import YOLO

model = YOLO("/app/runs/detect/runs/baseline_yolov8n/weights/best.pt")
model.predict(
    source="/app/assets/no-label.jpeg",
    conf=0.8,  # Note: 0.001 will dump up to 300 noisy boxes into the txt file
    save_txt=True,
    save_conf=True,  # Outputs 5 columns: class x_center y_center width height
    project="runs",
    name="val_preds",
)