
import cv2
import argparse
import os
from pathlib import Path

CLASS_NAMES = ["Banana", "Kiwi", "Orange", "Pepper", "Tomato"]

colors = [
    (0, 255, 255),
    (0, 255, 0),
    (0, 165, 255),
    (180, 105, 255),
    (0, 0, 255)
]


def plot_bbox(image, label_path):
    """
    Draw YOLO bounding boxes on an image.
    Keeps the original plotting style and colors.
    """

    h, w, _ = image.shape
    overlay = image.copy()

    if not os.path.exists(label_path):
        return image

    with open(label_path) as f:
        for line in f:
            values = line.split()

            if len(values) < 5:
                continue

            c, x_center, y_center, width, height = values[:5]
            confidence = values[5] if len(values) > 5 else None

            c = int(c)

            x_center = float(x_center) * w
            y_center = float(y_center) * h
            width = float(width) * w
            height = float(height) * h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Keep bounding boxes within image boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            color = colors[c % len(colors)]

            label = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)

            # Transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            image[y1:y2, x1:x2] = cv2.addWeighted(
                overlay[y1:y2, x1:x2], 0.25,
                image[y1:y2, x1:x2], 0.75, 0
            )

            # Outline and class name
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            cv2.putText(
                image, label,
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2
            )

            # Confidence at top-right (predictions only)
            if confidence is not None:
                conf_label = f"Conf: {float(confidence):.3f}"

                (text_width, text_height), _ = cv2.getTextSize(
                    conf_label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 2
                )

                conf_x = x2 - text_width
                conf_y = max(y1 - 5, 20)

                cv2.putText(
                    image, conf_label,
                    (conf_x, conf_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2
                )

    return image


def visualize_dataset(image_dir, gt_dir, pred_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_paths = sorted([
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print("No images found!")
        return

    for idx, image_path in enumerate(image_paths):

        image_name = image_path.stem

        gt_path = os.path.join(gt_dir, image_name + ".txt")
        pred_path = os.path.join(pred_dir, image_name + ".txt")

        # Load original image
        original = cv2.imread(str(image_path))

        if original is None:
            print(f"Could not load image: {image_path}")
            continue

        # Create independent copies
        gt_image = original.copy()
        pred_image = original.copy()

        # Draw GT and Pred bounding boxes
        gt_image = plot_bbox(gt_image, gt_path)
        pred_image = plot_bbox(pred_image, pred_path)

        # Add titles
        cv2.putText(
            gt_image, "Ground Truth",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 3
        )

        cv2.putText(
            pred_image, "Predictions",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 3
        )

        # Side-by-side visualization
        combined = cv2.hconcat([gt_image, pred_image])

        # Display at half resolution
        resized = cv2.resize(
            combined, None,
            fx=0.5, fy=0.5
        )


        # Save visualization
        output_path = os.path.join(
            output_dir, image_name + "_comparison.jpg"
        )

        cv2.imwrite(output_path, combined)

        print(
            f"[{idx + 1}/{len(image_paths)}] "
            f"Saved: {output_path}"
        )

        # Press any key for next image, ESC to exit
        key = cv2.waitKey(0)

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLO Ground Truth vs Predictions Visualization"
    )

    parser.add_argument(
        "--image_dir", required=True,
        help="Folder containing images"
    )

    parser.add_argument(
        "--gt_dir", required=True,
        help="Folder containing ground truth labels"
    )

    parser.add_argument(
        "--pred_dir", required=True,
        help="Folder containing predicted labels"
    )

    parser.add_argument(
        "--output_dir", required=True,
        help="Folder to save comparison visualizations"
    )

    args = parser.parse_args()

    visualize_dataset(
        args.image_dir,
        args.gt_dir,
        args.pred_dir,
        args.output_dir
    )
