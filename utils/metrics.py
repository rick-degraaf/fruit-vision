import numpy as np

def iou(boxA, boxB):
    """
    Calculates IoU (0-1) between two YOLO-format bounding boxes.
    Format: [class, x_center, y_center, width, height]
    """

    # Extract YOLO coordinates
    xA, yA, wA, hA = boxA[1:5]
    xB, yB, wB, hB = boxB[1:5]

    # Convert box A to corner format
    x_min_a = xA - wA / 2
    y_min_a = yA - hA / 2
    x_max_a = xA + wA / 2
    y_max_a = yA + hA / 2

    # Convert box B to corner format
    x_min_b = xB - wB / 2
    y_min_b = yB - hB / 2
    x_max_b = xB + wB / 2
    y_max_b = yB + hB / 2

    # Calculate individual areas
    areaA = wA * hA
    areaB = wB * hB

    # Calculate intersection
    x_overlap = max(0, min(x_max_a, x_max_b) - max(x_min_a, x_min_b))
    y_overlap = max(0, min(y_max_a, y_max_b) - max(y_min_a, y_min_b))

    intersection = x_overlap * y_overlap

    # Calculate union
    union = areaA + areaB - intersection

    # Avoid division by zero
    if union <= 0:
        return 0.0

    return intersection / union

def nms(boxes, scores, thr):
    """Determine box indices to keep based on confidence scores and IoU threshold."""
    if len(boxes) == 0:
        return []

    # Sort box indices from highest confidence score to lowest
    candidates = np.argsort(scores)[::-1]
    keep = []

    for i in candidates:
        # Compare candidate box i against every higher-scoring box already kept
        suppress = False
        for kept_idx in keep:
            if iou(boxes[i], boxes[kept_idx]) > thr:
                suppress = True
                break

        if not suppress:
            keep.append(int(i))

    return keep

