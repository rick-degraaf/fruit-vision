
import os
import argparse
import shutil
from pathlib import Path

# Dataset split
SPLITS = {
    "train": ["S1", "S3"],
    "val": ["S2"],
    "test": ["S4"]
}

def split_dataset(images_path, labels_path, target_dir):

    images_path = Path(images_path)
    labels_path = Path(labels_path)
    target_dir = Path(target_dir)

    # Create output directories
    for split in SPLITS:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Find images and labels
    images = list(images_path.glob("*"))
    labels = {f.stem: f for f in labels_path.glob("*.txt")}

    for image in images:
        if image.suffix.lower() not in [".jpg", ".jpeg"]:
            continue

        # Determine split from filename prefix
        prefix = image.stem.split("_")[0]

        split = next(
            (name for name, groups in SPLITS.items() if prefix in groups),
            None
        )

        if split is None:
            continue

        label = labels.get(image.stem)

        if label is None:
            print(f"Missing label: {image.stem}")
            continue

        # Copy image and corresponding label
        shutil.copy2(image, target_dir / "images" / split / image.name)
        shutil.copy2(label, target_dir / "labels" / split / label.name)

    print("Dataset split complete!")

    for split in SPLITS:
        n_images = len(list((target_dir / "images" / split).glob("*")))
        n_labels = len(list((target_dir / "labels" / split).glob("*")))
        print(f"{split}: {n_images} images, {n_labels} labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Splitter")

    parser.add_argument("--images_path", required=True, help="Input images directory")
    parser.add_argument("--labels_path", required=True, help="Input labels directory")
    parser.add_argument("--target_dir", required=True, help="Output dataset directory")

    args = parser.parse_args()

    split_dataset(args.images_path, args.labels_path, args.target_dir)
