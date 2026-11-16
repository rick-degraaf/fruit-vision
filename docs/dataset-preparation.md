# Dataset Labelling & Preparation

This project uses a custom dataset containing five fruit classes. The images were labelled with bounding boxes and exported in YOLO format.

## Annotation Format

Each image has a corresponding `.txt` annotation file. Every line represents one object and follows this format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

The bounding-box coordinates are normalized relative to the image dimensions, meaning their values range from 0 to 1.

The `class_id` identifies the fruit class, while the remaining values describe the bounding box.

## Dataset Scenes

The dataset contains four different scenes:

| Scene | Dataset purpose |
|---|---|
| Scene 1 | Training |
| Scene 2 | Validation |
| Scene 3 | Training |
| Scene 4 | Testing |

The scenes were kept separate when creating the dataset splits. This helps avoid placing images from the same scene in both the training and validation sets.

Scene 4 was reserved for final testing. It contains a different lighting environment, allowing the model to be evaluated on images it has not seen during training or validation.

## Dataset Split

The dataset is divided as follows:

- **Training:** Scenes 1 and 3
- **Validation:** Scene 2
- **Testing:** Scene 4

The training set is used to fine-tune the model. The validation set is used during development to monitor performance, while the test set is kept separate for the final evaluation.

## Dataset Directory Structure

The original images and labels are stored in:

```text
/app/data/train/
├── images/
└── labels/
```

After generating the dataset split, the target directory is `/app/data`.

## Generate the Dataset Split

The repository includes a script to generate the dataset split.

Run the following command from the project environment:

```bash
python utils/generate_split.py \
    --images_path /app/data/train/images \
    --labels_path /app/data/train/labels \
    --target_dir /app/data
```

The script uses the images and corresponding annotation files to create the dataset splits. Scenes 1 and 3 are used for training, Scene 2 for validation, and Scene 4 is kept separate for final testing.

For the exact split configuration and generated output, refer to the script in `utils/generate_split.py`.