# P1 Phase 5 Final ML Report

## 1. Purpose

Phase 5 finalizes the P1 AI/ML detection work by consolidating:

- Final unseen-test evaluation
- Per-class performance
- Inference benchmarking
- Confidence-threshold sensitivity analysis
- Phase 4 filtering findings
- Current limitations
- Deployment considerations

The objective is to provide reproducible and defensible evidence for the Sonaris marine sonar anomaly detection pipeline.

---

## 2. Final Model

Model:

- YOLO11n
- Object detection
- 2,583,127 parameters
- 6.4 GFLOPs

Training configuration:

- Image size: 640
- Batch size: 16
- Epochs: 12
- Seed: 42
- Deterministic training: enabled
- Device during training: Apple M4 MPS

Best weights:

`runs/detect/models/baseline/yolo11n_baseline/weights/best.pt`

---

## 3. Final Unseen Test Evaluation

The final evaluation was performed on the unseen test split.

Test set:

- Images: 600
- Ground-truth instances: 748

Overall results:

| Metric | Result |
|---|---:|
| Precision | 0.6816 |
| Recall | 0.6883 |
| mAP@0.50 | 0.6717 |
| mAP@0.50:0.95 | 0.5173 |

These metrics are the final baseline detection results for the current YOLO11n model.

---

## 4. Per-Class Performance

| Class | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| submarine_pipeline | 0.9434 | 0.9885 | 0.9835 | 0.7580 |
| shipwreck | 0.4656 | 0.4355 | 0.4171 | 0.2285 |
| ghost_net | 0.9920 | 1.0000 | 0.9950 | 0.9614 |
| mine_cylinder | 0.3253 | 0.3293 | 0.2913 | 0.1214 |

The `crab_pot` class has no test-set instances and therefore does not have a meaningful per-class test metric.

The model performs strongly on submarine pipelines and ghost nets, while shipwreck and mine-cylinder detection remain challenging.

---

## 5. Inference Benchmark

The final evaluation was performed on an Apple M4 CPU.

Measured inference speed:

- Preprocessing: approximately 0.35 ms/image
- Inference: approximately 123.31 ms/image
- Postprocessing: approximately 0.18 ms/image
- Total measured latency: approximately 123.84 ms/image
- Throughput: approximately 8.1 FPS

These numbers are a local CPU benchmark and should not be interpreted as universal deployment performance.

The model is small enough to support future optimization for edge deployment.

---

## 6. Confidence-Threshold Sensitivity

A confidence-threshold comparison was performed on the same 600-image test split.

| Confidence | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---:|---:|---:|---:|---:|
| 0.10 | 0.682 | 0.688 | 0.656 | 0.509 |
| 0.25 | 0.682 | 0.688 | 0.672 | 0.517 |
| 0.40 | 0.840 | 0.609 | 0.595 | 0.484 |
| 0.50 | 0.878 | 0.581 | 0.574 | 0.475 |

### Interpretation

Increasing the confidence threshold increases precision while reducing recall.

Among the tested operating points, 0.25 produced the highest observed mAP values while retaining the highest recall.

A threshold of 0.40 or 0.50 can therefore be useful for a precision-oriented review mode where the operator prefers fewer, more confident detections.

Because this comparison was performed on the test split, it is treated as an observed sensitivity experiment rather than a formally held-out threshold-selection procedure.

The current default confidence threshold remains configurable and is not claimed to be universally optimal.

---

## 7. Phase 4 Filtering Findings

Phase 4 tested confidence filtering, duplicate suppression, and configurable geometric filtering.

A 100-image filtering experiment compared the baseline post-processing configuration with a minimum bounding-box area threshold of 2000 pixels².

### Baseline

- Detections: 48
- True positives: 20
- False positives: 28
- False negatives: 51
- Precision: 0.4167
- Recall: 0.2817

### Minimum Area = 2000 pixels²

- Detections: 12
- True positives: 5
- False positives: 7
- False negatives: 66
- Precision: 0.4167
- Recall: 0.0704

### Conclusion

The tested area threshold did not improve precision and substantially reduced recall.

Therefore, area-based rejection is not enabled as a default validated filtering rule.

The geometric filtering implementation remains configurable for future controlled experiments.

---

## 8. Final P1 Processing Pipeline

The current P1 pipeline is:

SSS image
→ Sonar preprocessing from P3
→ YOLO11n detection
→ Confidence filtering
→ Duplicate suppression
→ Configurable geometric filtering
→ Anomaly scoring
→ Inspection priority
→ Structured detection output
→ Geolocation / backend / reporting

P1 does not implement a second sonar preprocessing pipeline.

---

## 9. Anomaly Score

The current anomaly score is:

`anomaly_score = confidence × class_weight`

All class weights are currently 1.0.

Therefore, the current score is equivalent to detection confidence.

No unsupported class-specific risk weights are used.

The scoring mechanism is configurable so validated class-specific weighting can be introduced later if sufficient evidence becomes available.

---

## 10. Inspection Priority

The current operational priority thresholds are:

- HIGH: anomaly score >= 0.75
- MEDIUM: anomaly score >= 0.45
- LOW: anomaly score < 0.45

These are configurable operational review thresholds.

They are not claimed to represent scientifically validated environmental or safety risk levels.

---

## 11. Final Detection Output

P1 inference produces structured detections containing:

- image_id
- class_id
- class
- confidence
- bbox
- inference time

The Phase 4 intelligence pipeline additionally produces:

- bbox_area
- aspect_ratio
- anomaly_score
- priority

This structure is designed for downstream integration with geolocation, backend services, reports, and the frontend dashboard.

---

## 12. Limitations

The current results have several limitations:

- Performance varies substantially between classes.
- Mine-cylinder and shipwreck detection require further improvement.
- The dataset contains synthetic and heterogeneous sources.
- The filtering experiment used a limited 100-image sample.
- Confidence-threshold sensitivity was evaluated on the test split and therefore should not be treated as formal model-selection evidence.
- No independently validated environmental risk labels are available for the anomaly score.
- Geolocation accuracy depends on the quality and availability of sonar navigation metadata.
- No claim is made that the current model is ready for safety-critical autonomous decisions.

Human review remains part of the intended operational workflow.

---

## 13. Edge Deployment Consideration

The current YOLO11n model is relatively lightweight.

The intended deployment path is:

PyTorch model
→ ONNX export
→ ONNX Runtime
→ optional TensorRT optimization

Actual edge-device performance has not yet been measured.

Therefore, the current system should be described as designed for eventual edge deployment rather than claiming validated onboard AUV/ROV performance.

---

## 14. Reproducibility

The final test evaluation can be reproduced using:

- The committed YOLO11n best weights
- The cleaned DRISHTI-SSS test split
- The Mac-compatible dataset configuration
- `ml/detection/evaluate.py`

The evaluation output is saved as:

`runs/detect/phase5_final_test.json`

The evaluation script records:

- Overall precision
- Overall recall
- mAP@0.50
- mAP@0.50:0.95
- Processing speed
- Per-class metrics

---

## 15. Phase 5 Conclusion

Phase 5 establishes the final measured baseline for the current Sonaris P1 detection system.

The YOLO11n model achieves:

- 68.16% precision
- 68.83% recall
- 67.17% mAP@0.50
- 51.73% mAP@0.50:0.95

on 600 unseen test images.

The system provides a complete detection-to-intelligence interface with configurable filtering, anomaly scoring, inspection priority, and structured outputs.

The strongest current detection classes are ghost nets and submarine pipelines. Shipwreck and mine-cylinder detection remain the primary areas for future model improvement.

The results are presented as measured experimental results rather than claims of production-level or scientifically validated marine-survey accuracy.
