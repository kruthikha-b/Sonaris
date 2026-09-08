# P1 Phase 4 Filtering and Intelligence Report

## 1. Purpose

Phase 4 adds a post-processing and intelligence layer after YOLO detection.

The pipeline converts raw model detections into structured final detections by applying:

1. Confidence filtering
2. Duplicate suppression using IoU
3. Configurable bounding-box based filtering
4. Anomaly scoring
5. Inspection priority assignment

The filtering layer is intentionally configurable. No heuristic is presented as scientifically validated unless supported by an experiment.

---

## 2. Phase 4 Pipeline

The processing flow is:

Raw YOLO detections
→ Confidence filtering
→ Duplicate suppression
→ Configurable false-positive filtering
→ Anomaly scoring
→ Inspection priority
→ Final structured detections

The final detection schema contains at least:

- image_id
- class_id
- class
- confidence
- bbox
- bbox_area
- aspect_ratio
- anomaly_score
- priority

---

## 3. Confidence Filtering

Detections below the configured confidence threshold are removed.

Default:

- confidence threshold = 0.25

The threshold is configurable and is not treated as a universally optimal value.

---

## 4. Duplicate Suppression

Duplicate detections of the same class are suppressed using Intersection over Union (IoU).

Default:

- IoU threshold = 0.50

The higher-confidence detection is retained when overlapping detections are considered duplicates.

---

## 5. Bounding-Box Features

For each detection, the filtering layer calculates:

- Bounding-box width
- Bounding-box height
- Bounding-box area
- Aspect ratio

These features are exposed for analysis and configurable filtering.

The implementation supports:

- minimum area
- maximum area
- minimum aspect ratio
- maximum aspect ratio

The default minimum area is 0, meaning no area-based rejection is enabled by default.

---

## 6. False-Positive Filtering

The current false-positive filtering layer provides configurable geometric rules.

These rules are treated as heuristics rather than scientifically validated marine-sonar false-positive classifiers.

No class-specific area thresholds or weights are claimed to be validated.

Further validation would require a larger controlled experiment across representative sonar conditions.

---

## 7. Anomaly Score

The current anomaly score is:

anomaly_score = confidence × class_weight

For Phase 4 evaluation, all class weights are set to 1.0.

Therefore:

anomaly_score = confidence

This avoids introducing arbitrary class-specific importance weights without supporting evidence.

The scoring function remains configurable so validated class-specific weighting can be introduced later if sufficient evaluation data becomes available.

---

## 8. Inspection Priority

The current priority system converts anomaly score into three review levels:

- HIGH: score >= 0.75
- MEDIUM: score >= 0.45
- LOW: score < 0.45

These thresholds are configurable and should be treated as operational review thresholds rather than scientifically validated risk thresholds.

---

## 9. Filtering Experiment

A filtering experiment was performed using predictions from the trained YOLO11n baseline model on 100 unseen test images.

Ground-truth bounding boxes were converted from normalized YOLO coordinates to image-pixel coordinates before comparison with predictions.

The experiment compared the baseline post-processing pipeline with an area-based filtering rule.

### Baseline

Configuration:

- Confidence threshold: 0.25
- IoU threshold: 0.50
- Minimum area: 0

Results:

- Detections after filtering: 48
- True positives: 20
- False positives: 28
- False negatives: 51
- Precision: 0.4167
- Recall: 0.2817

### Area Filtering

Configuration:

- Confidence threshold: 0.25
- IoU threshold: 0.50
- Minimum area: 2000 pixels²

Results:

- Detections after filtering: 12
- True positives: 5
- False positives: 7
- False negatives: 66
- Precision: 0.4167
- Recall: 0.0704

---

## 10. Experiment Interpretation

The minimum-area threshold of 2000 pixels² did not improve precision.

It substantially reduced recall from 0.2817 to 0.0704 on this 100-image evaluation sample.

Therefore, the area threshold is NOT enabled as a default filtering rule.

This experiment demonstrates why geometric heuristics should not be presented as validated improvements without measurement.

The current implementation keeps geometric filtering configurable so that future experiments can test thresholds under different sonar conditions and classes.

---

## 11. Prediction Distribution Analysis

A separate analysis of raw predictions at a low confidence threshold was used to understand the detection distribution.

On a 100-image sample:

- Raw detections: 904
- Confidence median: approximately 0.0269
- Confidence maximum: approximately 0.9623
- Bounding-box area median: approximately 2035 pixels²
- Aspect-ratio median: approximately 1.018

Class distribution of raw detections:

- Shipwreck: 612
- Mine cylinder: 211
- Submarine pipeline: 48
- Ghost net: 33

This analysis was exploratory and was not used to claim model performance.

---

## 12. Phase 4 Unit Tests

Eight unit tests were implemented covering:

- Confidence filtering
- IoU calculation
- Duplicate suppression
- Bounding-box feature calculation
- False-positive filtering
- Anomaly scoring
- Priority assignment
- Complete filtering pipeline

Test result:

8 passed in 0.01s

---

## 13. Current Limitations

Phase 4 does not claim that geometric filtering solves marine-sonar false positives.

Important limitations include:

- Limited evaluation sample for filtering experiments
- Dataset-specific behavior
- Class imbalance
- Sonar artifacts and acoustic shadows
- Variation in image resolution
- Potential boundary effects
- Lack of independently validated risk labels
- No validated class-specific priority weights

These limitations should be addressed in future experiments.

---

## 14. Integration Boundary

P1 owns:

- Detection
- Confidence filtering
- Duplicate suppression
- Filtering heuristics
- Anomaly scoring
- Inspection priority
- Detection evaluation

P3 owns:

- Sonar preprocessing
- Denoising
- Normalization
- Contrast enhancement
- Sonar-specific artifact handling
- Geolocation

P1 consumes P3's processed sonar output and does not implement a second preprocessing pipeline.

---

## 15. Phase 4 Conclusion

Phase 4 establishes a deterministic and configurable post-processing pipeline from YOLO predictions to structured inspection detections.

The current implementation provides a foundation for:

- Reducing duplicate detections
- Applying configurable confidence and geometric filters
- Ranking detections for human review
- Passing structured detections to downstream geolocation, backend, reporting, and frontend components

No unsupported claim is made that the current heuristic false-positive filtering improves model accuracy.
