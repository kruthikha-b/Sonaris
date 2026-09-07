def filter_false_positives(detections, min_area=0):
    filtered = []

    for detection in detections:
        bbox = detection["bbox"]

        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        area = width * height

        if area >= min_area:
            filtered.append(detection)

    return filtered
