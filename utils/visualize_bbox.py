
import cv2
import argparse

CLASS_NAMES = ["Banana", "Kiwi", "Orange", "Pepper", "Tomato"]

colors = [
    (0, 255, 255),
    (0, 255, 0),
    (0, 165, 255),
    (180, 105, 255),
    (0, 0, 255)
]

def plot_bbox(image_path, label_path, output_path):

    image = cv2.imread(image_path)
    h, w, _ = image.shape
    overlay = image.copy()

    with open(label_path) as f:
        for line in f:
            values = line.split()

            c, x_center, y_center, width, height = values[:5]
            confidence = values[5] if len(values) > 5 else None

            c = int(c)
            x_center, y_center = float(x_center) * w, float(y_center) * h
            width, height = float(width) * w, float(height) * h

            x1, y1 = int(x_center - width/2), int(y_center - height/2)
            x2, y2 = int(x_center + width/2), int(y_center + height/2)

            color = colors[c % len(colors)]
            label = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            # Class name at top-left
            label = f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else c}"

            # Transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            image[y1:y2, x1:x2] = cv2.addWeighted(
                overlay[y1:y2, x1:x2], 0.25,
                image[y1:y2, x1:x2], 0.75, 0
            )

            # Outline and class name
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, label, (x1, max(y1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Confidence at top-right, only if present in TXT
            if confidence is not None:
                conf_label = f"Conf: {float(confidence):.3f}"

                # Calculate text width
                (text_width, text_height), _ = cv2.getTextSize(
                    conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                # Position at top-right of bounding box
                conf_x = x2 - text_width
                conf_y = max(y1 - 5, 20)

                cv2.putText(image, conf_label, (conf_x, conf_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    


    # Display at half resolution
    resized_image = cv2.resize(image, None, fx=0.5, fy=0.5)

    cv2.imshow("Displayed Image", resized_image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLO Bounding Boxes")
    parser.add_argument("--image_path", required=True, help="Path to image")
    parser.add_argument("--label_path", required=True, help="Path to label")
    parser.add_argument("--output_path", required=True, help="Path to output")
    args = parser.parse_args()

    plot_bbox(args.image_path, args.label_path, args.output_path)
