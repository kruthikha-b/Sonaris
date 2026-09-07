from pathlib import Path

DATASET = Path(r"C:\drishti_clean")
VALID_CLASSES = set(range(5))

converted = 0
checked = 0

for split in ["train", "val", "test"]:
    labels_dir = DATASET / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        checked += 1
        lines = label_file.read_text().splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Empty label file = valid background image
            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            if class_id not in VALID_CLASSES:
                raise ValueError(
                    f"Invalid class ID {class_id} in {label_file}"
                )

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_file}: {line}"
                )

            cleaned_lines.append(" ".join(parts))

        new_content = "\n".join(cleaned_lines)

        if cleaned_lines:
            new_content += "\n"

        if label_file.read_text() != new_content:
            label_file.write_text(new_content)
            converted += 1

print("Annotation standardization complete.")
print(f"Label files checked: {checked}")
print(f"Label files standardized: {converted}")