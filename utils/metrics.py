import numpy as np

def iou(boxA, boxB):
    """ Calculates iou value (0-1) of Bounding Boxes A & B in Corner Format [x1, y1, x2, y2]"""

    x_min_a, y_min_a, x_max_a, y_max_a = boxA[0], boxA[1], boxA[2], boxA[3]
    x_min_b, y_min_b, x_max_b, y_max_b = boxB[0], boxB[1], boxB[2], boxB[3]

    areaA = (x_max_a - x_min_a) * (y_max_a - y_min_a)   
    areaB = (x_max_b - x_min_b) * (y_max_b - y_min_b)

    x_overlap = max(0, min(x_max_a, x_max_b) - max(x_min_a, x_min_b))
    y_overlap = max(0, min(y_max_a, y_max_b) - max(y_min_a, y_min_b))
    overlap = x_overlap * y_overlap
    union = areaA + areaB - overlap
    iou_metric = overlap / union

    return iou_metric

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

