def filter_by_confidence(detections, threshold=0.25):
    return [
        detection
        for detection in detections
        if detection["confidence"] >= threshold
    ]
