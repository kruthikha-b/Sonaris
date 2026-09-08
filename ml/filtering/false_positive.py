def calculate_bbox_features(bbox):
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    area = width * height
    aspect_ratio = width / height if height > 0 else 0.0

    return {
        "width": width,
        "height": height,
        "area": area,
        "aspect_ratio": aspect_ratio
    }


def filter_false_positives(
    detections,
    min_area=0,
    max_area=None,
    min_aspect_ratio=0,
    max_aspect_ratio=None
):
    filtered = []

    for detection in detections:
        features = calculate_bbox_features(detection["bbox"])

        if features["area"] < min_area:
            continue

        if max_area is not None and features["area"] > max_area:
            continue

        if features["aspect_ratio"] < min_aspect_ratio:
            continue

        if (
            max_aspect_ratio is not None
            and features["aspect_ratio"] > max_aspect_ratio
        ):
            continue

        result = detection.copy()
        result["bbox_area"] = round(features["area"], 2)
        result["aspect_ratio"] = round(features["aspect_ratio"], 4)

        filtered.append(result)

    return filtered
