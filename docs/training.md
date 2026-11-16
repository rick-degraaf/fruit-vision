# Model Training

This project uses **YOLOv8n**, a pretrained object-detection model from Ultralytics. The model is initialized with weights pretrained on the COCO 2017 dataset and fine-tuned on the custom five-class fruit dataset.

## Training Configuration

The training process uses the prepared training and validation datasets.

- **Model:** YOLOv8n
- **Pretrained weights:** COCO 2017
- **Dataset:** Custom five-class fruit dataset
- **Training scenes:** Scenes 1 and 3
- **Validation scene:** Scene 2

## Run Training

Open a terminal inside the development container and navigate to the training directory:

```bash
cd /app/train
```

Start the training process:

```bash
python train.py
```

The training script loads the pretrained YOLOv8n weights and fine-tunes the model on the custom dataset.

## Training Outputs

The training run saves its results under:

```text
/app/runs/baseline_yolov8n/
```

The main output files include:

```text
baseline_yolov8n/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
└── results.png
```

- `best.pt` contains the model weights from the best-performing training epoch.
- `last.pt` contains the weights from the final training epoch.
- `results.csv` contains the training metrics recorded during training.
- `results.png` provides a visual overview of the training results.

The `best.pt` weights are used for the subsequent inference and evaluation steps.

## Next Step

Once training is complete, use the best-performing weights to generate predictions on the validation dataset.

See [Inference](inference.md) for instructions on running the model.