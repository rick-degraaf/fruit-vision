
import sys
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.metrics import iou

gt_labels_path = "/app/data/labels/val"
pred_labels_path = "/app/runs/val_preds/labels"

# Store all matched labels
all_data = {}

# --------------------------------------------------
# Load all GT and predictions
# --------------------------------------------------

for file in tqdm(os.listdir(gt_labels_path), desc="Loading Images"):

    gt_file = os.path.join(gt_labels_path, file)
    pred_file = os.path.join(pred_labels_path, file)

    if not os.path.isfile(pred_file):
        print(f"Missing prediction file: {file}")
        continue

    gt = []
    pred = []

    # Load GT labels
    with open(gt_file, "r") as f:
        for line in f:
            values = line.split()

            if not values:
                continue

            c, x_center, y_center, width, height = map(float, values[:5])

            np_box = np.array([c, x_center, y_center, width, height])
            gt.append(np_box)

    # Load predictions
    with open(pred_file, "r") as f:
        for line in f:
            values = line.split()

            if not values:
                continue

            c, x_center, y_center, width, height, confidence = map(float, values[:6])

            np_box = np.array([
                c, x_center, y_center, width, height, confidence
            ])

            pred.append(np_box)

    all_data[file] = {
        "gt": gt,
        "pred": pred
    }



# --------------------------------------------------
# Identify all classes
# --------------------------------------------------

classes = sorted(set(
    int(gt_box[0])
    for image in all_data.values()
    for gt_box in image["gt"]
))

# Standard COCO IoU thresholds: 0.50 to 0.95
iou_thresholds = np.arange(0.50, 0.96, 0.05)

class_results = {}
data_pr = {}
map_results = {}

# --------------------------------------------------
# Calculate AP for every IoU threshold and class
# --------------------------------------------------

for iou_threshold in tqdm(iou_thresholds, desc="Processing IoU thresholds"):

    class_results[iou_threshold] = {}
    data_pr[iou_threshold] = {}

    for c in classes:

        TP = []
        FP = []

        predictions = []

        # Collect predictions for this class across all images
        for file, image in all_data.items():

            for prediction in image["pred"]:

                if int(prediction[0]) == c:

                    predictions.append({
                        "file": file,
                        "prediction": prediction
                    })

        # Sort predictions by confidence (highest first)
        predictions.sort(
            key=lambda x: x["prediction"][5],
            reverse=True
        )

        # Initialize GT trackers for every image
        gt_taken = {
            file: [False] * len(image["gt"])
            for file, image in all_data.items()
        }

        # --------------------------------------------------
        # Match predictions
        # --------------------------------------------------

        for prediction_data in predictions:

            file = prediction_data["file"]
            prediction = prediction_data["prediction"]

            gt = all_data[file]["gt"]

            # Find GT boxes of the same class
            matching_gt = [
                idx for idx, gt_box in enumerate(gt)
                if int(gt_box[0]) == c
            ]

            # Remove already matched GT boxes
            matching_gt_taken = [
                idx for idx in matching_gt
                if not gt_taken[file][idx]
            ]

            gt_iou = []

            for match_idx in matching_gt_taken:
                gt_iou.append(
                    iou(prediction[:5], gt[match_idx])
                )

            if len(gt_iou) > 0:

                max_idx = gt_iou.index(max(gt_iou))

                if gt_iou[max_idx] >= iou_threshold:

                    TP.append(1)
                    FP.append(0)

                    gt_taken[file][matching_gt_taken[max_idx]] = True

                else:

                    TP.append(0)
                    FP.append(1)

            else:

                TP.append(0)
                FP.append(1)

        # --------------------------------------------------
        # Calculate total GT objects for this class
        # --------------------------------------------------

        total_gt = sum(
            1
            for image in all_data.values()
            for gt_box in image["gt"]
            if int(gt_box[0]) == c
        )

        # --------------------------------------------------
        # Calculate cumulative TP and FP
        # --------------------------------------------------

        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        # Calculate precision and recall
        if len(TP) > 0:

            precision_list = TP / (TP + FP)
            recall_list = TP / total_gt

        else:

            precision_list = np.array([])
            recall_list = np.array([])

        # --------------------------------------------------
        # Calculate interpolated AP
        # --------------------------------------------------

        mrec = np.concatenate(([0.0], recall_list, [1.0]))
        mpre = np.concatenate(([0.0], precision_list, [0.0]))

        # Precision envelope
        for idx in range(len(mpre) - 1, 0, -1):
            mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

        # Calculate area
        AP = np.sum(
            (mrec[1:] - mrec[:-1]) * mpre[1:]
        )

        class_results[iou_threshold][c] = AP

        data_pr[iou_threshold][c] = {
            "precision": precision_list,
            "recall": recall_list
        }

    # Calculate mAP for this IoU threshold
    map_results[iou_threshold] = np.mean(
        list(class_results[iou_threshold].values())
    )

    print(
        f"IoU {iou_threshold:.2f}: "
        f"mAP = {map_results[iou_threshold]:.4f}"
    )


# --------------------------------------------------
# Calculate mAP@0.5 and mAP@0.5:0.95
# --------------------------------------------------

mAP50 = map_results[0.5]

mAP50_95 = np.mean(list(map_results.values()))

print(f"\nmAP@0.5 = {mAP50:.4f}")
print(f"mAP@0.5:0.95 = {mAP50_95:.4f}")


# --------------------------------------------------
# Plot separate PR curves for each class
# --------------------------------------------------

num_classes = len(classes)

cols = 2
rows = int(np.ceil(num_classes / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(8 * cols, 6 * rows)
)

axes = np.array(axes).flatten()

for idx, c in enumerate(classes):

    ax = axes[idx]

    for iou_threshold in iou_thresholds:

        precision_list = data_pr[iou_threshold][c]["precision"]
        recall_list = data_pr[iou_threshold][c]["recall"]

        ax.plot(
            recall_list,
            precision_list,
            label=(
                f"IoU {iou_threshold:.2f} "
                f"(AP={class_results[iou_threshold][c]:.3f})"
            )
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Class {c}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.grid()
    ax.legend(fontsize=8)

# Hide unused subplots
for idx in range(num_classes, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f"Precision-Recall Curves per Class\n"
    f"mAP@0.5 = {mAP50:.3f} | "
    f"mAP@0.5:0.95 = {mAP50_95:.3f}",
    fontsize=15
)

plt.tight_layout()
plt.show()