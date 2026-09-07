from pathlib import Path
import cv2
import albumentations as A


SOURCE = Path(r"C:\drishti_clean")
OUTPUT = Path(r"C:\drishti_augmented")

# Sonar-safe augmentations
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(
    	format="yolo",
    	label_fields=["class_labels"],
    	clip=True,
    	filter_invalid_bboxes=True
    )
)


def load_labels(label_file):
    boxes = []
    classes = []

    if not label_file.exists():
        return boxes, classes

    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        classes.append(int(parts[0]))
        boxes.append([float(x) for x in parts[1:]])

    return boxes, classes


def save_labels(label_file, boxes, classes):
    with open(label_file, "w") as f:
        for class_id, box in zip(classes, boxes):

            box = [
                max(0.0, min(1.0, float(x)))
                for x in box
            ]

            if box[2] <= 0 or box[3] <= 0:
                continue

            f.write(
                f"{int(class_id)} "
                f"{box[0]:.6f} "
                f"{box[1]:.6f} "
                f"{box[2]:.6f} "
                f"{box[3]:.6f}\n"
            )


def main():
    source_images = SOURCE / "train" / "images"
    source_labels = SOURCE / "train" / "labels"

    output_images = OUTPUT / "train" / "images"
    output_labels = OUTPUT / "train" / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_path in source_images.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = source_labels / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        boxes, classes = load_labels(label_path)

        result = transform(
            image=image,
            bboxes=boxes,
            class_labels=classes
        )

        output_name = f"{image_path.stem}_aug{image_path.suffix}"

        cv2.imwrite(
            str(output_images / output_name),
            result["image"]
        )

        save_labels(
            output_labels / f"{Path(output_name).stem}.txt",
            result["bboxes"],
            result["class_labels"]
        )

        count += 1

    print("Augmentation complete.")
    print(f"Augmented training images created: {count}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()