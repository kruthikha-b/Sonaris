from pathlib import Path

DATASET = Path(r"C:\drishti_clean")

splits = ["train", "val", "test"]

removed = 0

for split in splits:
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    for image in images_dir.iterdir():
        if not image.is_file():
            continue

        label = labels_dir / f"{image.stem}.txt"

        # Remove image only if its label is completely missing.
        # Empty labels are valid background/negative samples, so keep them.
        if not label.exists():
            print(f"Removing image without label: {image}")
            image.unlink()
            removed += 1

print(f"\nCleaning complete.")
print(f"Images removed: {removed}")