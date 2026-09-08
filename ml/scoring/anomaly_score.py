DEFAULT_CLASS_WEIGHTS = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0
}


def calculate_anomaly_score(
    detection,
    class_weights=None
):
    weights = class_weights or DEFAULT_CLASS_WEIGHTS

    confidence = float(detection["confidence"])
    class_id = int(detection["class_id"])

    class_weight = float(weights.get(class_id, 1.0))

    score = confidence * class_weight

    return round(max(0.0, min(score, 1.0)), 4)
