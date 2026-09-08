def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def remove_duplicates(detections, iou_threshold=0.5):
    detections = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True
    )

    kept = []

    for detection in detections:
        duplicate = False

        for existing in kept:
            if detection["class_id"] != existing["class_id"]:
                continue

            if calculate_iou(
                detection["bbox"],
                existing["bbox"]
            ) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept
