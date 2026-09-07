def assign_priority(anomaly_score):
    if anomaly_score >= 0.75:
        return "HIGH"
    if anomaly_score >= 0.45:
        return "MEDIUM"
    return "LOW"
