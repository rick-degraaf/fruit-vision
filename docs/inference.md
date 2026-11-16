# Model Inference

After training, the best-performing YOLOv8n weights are used to generate predictions on the validation dataset.

The inference script loads the trained model, runs object detection, and saves the predictions for the evaluation process.

## Run Inference

Open a terminal inside the development container and navigate to the training directory:

```bash
cd /app/train
```

Run the inference script:

```bash
python inference.py --save_txt --save_conf
```

By default, the script uses the following trained weights:

```text
/app/runs/baseline_yolov8n/weights/best.pt
```

## Confidence Threshold

The default confidence threshold is set to `0.001`.

This is intentionally low. Rather than filtering out detections too early, it retains a larger set of candidate predictions with different confidence scores.

This is useful when calculating Precision-Recall curves and Average Precision, as the evaluation process can consider predictions across different confidence thresholds.

## Prediction Outputs

The predictions are saved to:

```text
/app/runs/val_preds/
```

The `--save_txt` argument saves the predicted bounding boxes as text files, while `--save_conf` includes the confidence score for each detection.

These prediction files are used by the custom evaluation scripts to compare model predictions against the ground-truth annotations.

## Visualize Predictions

To visualize the ground-truth bounding boxes alongside the model predictions, run:

```bash
python /app/utils/visualize_bbox_GT_vs_Pred.py
```

This generates a visual comparison that can be used to inspect correct detections, missed objects, and false positives.

## Next Step

After generating the predictions, proceed to [Model Evaluation](evaluation.md) to calculate the detection metrics and inspect the Precision-Recall curves.