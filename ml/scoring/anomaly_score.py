CLASS_WEIGHTS = {
    0: 0.0,
    1: 0.8,
    2: 1.0,
    3: 1.0,
    4: 0.9
}


def calculate_anomaly_score(detection):
    confidence = detection["confidence"]
    class_id = detection["class_id"]

    class_weight = CLASS_WEIGHTS.get(class_id, 0.5)

    score = confidence * class_weight

    return round(min(score, 1.0), 4)
