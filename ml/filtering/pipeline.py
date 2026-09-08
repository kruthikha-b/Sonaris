from ml.filtering.confidence import filter_by_confidence
from ml.filtering.duplicate import remove_duplicates
from ml.filtering.false_positive import filter_false_positives
from ml.scoring.anomaly_score import calculate_anomaly_score
from ml.scoring.priority import assign_priority


def process_detections(
    detections,
    confidence_threshold=0.25,
    iou_threshold=0.5,
    min_area=0
):
    filtered = filter_by_confidence(
        detections,
        threshold=confidence_threshold
    )

    filtered = remove_duplicates(
        filtered,
        iou_threshold=iou_threshold
    )

    filtered = filter_false_positives(
        filtered,
        min_area=min_area
    )

    final_detections = []

    for detection in filtered:
        result = detection.copy()
        anomaly_score = calculate_anomaly_score(result)
        result["anomaly_score"] = anomaly_score
        result["priority"] = assign_priority(anomaly_score)
        final_detections.append(result)

    return final_detections
