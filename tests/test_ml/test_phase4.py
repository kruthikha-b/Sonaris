from ml.filtering.confidence import filter_by_confidence
from ml.filtering.duplicate import calculate_iou, remove_duplicates
from ml.filtering.false_positive import (
    calculate_bbox_features,
    filter_false_positives,
)
from ml.filtering.pipeline import process_detections
from ml.scoring.anomaly_score import calculate_anomaly_score
from ml.scoring.priority import assign_priority


def sample_detection(confidence=0.8, bbox=None, class_id=3):
    return {
        "image_id": "test_image",
        "class_id": class_id,
        "class": "ghost_net",
        "confidence": confidence,
        "bbox": bbox or [10, 10, 50, 50],
    }


def test_confidence_filter():
    detections = [
        sample_detection(0.9),
        sample_detection(0.2),
    ]

    result = filter_by_confidence(detections, threshold=0.25)

    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_iou():
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]

    iou = calculate_iou(box1, box2)

    assert round(iou, 4) == 0.1429


def test_duplicate_suppression():
    detections = [
        sample_detection(0.9, [0, 0, 100, 100]),
        sample_detection(0.7, [5, 5, 95, 95]),
    ]

    result = remove_duplicates(
        detections,
        iou_threshold=0.5
    )

    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_bbox_features():
    features = calculate_bbox_features([0, 0, 100, 50])

    assert features["width"] == 100
    assert features["height"] == 50
    assert features["area"] == 5000
    assert features["aspect_ratio"] == 2


def test_false_positive_filter():
    detections = [
        sample_detection(0.9, [0, 0, 100, 100]),
        sample_detection(0.8, [0, 0, 10, 10]),
    ]

    result = filter_false_positives(
        detections,
        min_area=500
    )

    assert len(result) == 1
    assert result[0]["bbox_area"] == 10000


def test_anomaly_score():
    detection = sample_detection(0.8)

    score = calculate_anomaly_score(detection)

    assert score == 0.8


def test_priority():
    assert assign_priority(0.8) == "HIGH"
    assert assign_priority(0.5) == "MEDIUM"
    assert assign_priority(0.2) == "LOW"


def test_complete_pipeline():
    detections = [
        sample_detection(
            confidence=0.9,
            bbox=[0, 0, 100, 100]
        ),
        sample_detection(
            confidence=0.2,
            bbox=[200, 200, 250, 250]
        ),
    ]

    result = process_detections(
        detections,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        min_area=0
    )

    assert len(result) == 1
    assert result[0]["class"] == "ghost_net"
    assert result[0]["anomaly_score"] == 0.9
    assert result[0]["priority"] == "HIGH"
