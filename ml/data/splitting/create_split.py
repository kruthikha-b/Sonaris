from pathlib import Path

DATASET = Path(r"C:\drishti_clean")

splits = {
    "train": 0,
    "val": 0,
    "test": 0
}

print("Existing dataset split verification")
print("-" * 40)

for split in splits:
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    images = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    labels = list(labels_dir.glob("*.txt"))

    splits[split] = len(images)

    print(f"{split}:")
    print(f"  Images : {len(images)}")
    print(f"  Labels : {len(labels)}")

print("-" * 40)
print("No new random split was created.")
print("The original train/val/test split is preserved.")