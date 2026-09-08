def assign_priority(
    anomaly_score,
    high_threshold=0.75,
    medium_threshold=0.45
):
    if anomaly_score >= high_threshold:
        return "HIGH"

    if anomaly_score >= medium_threshold:
        return "MEDIUM"

    return "LOW"
