from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


class Detector:
    def __init__(self, model_path, device=None):
        self.model = YOLO(model_path)
        self.device = device

    def _parse_result(self, result):
        image_path = Path(result.path)
        image_id = image_path.stem
        detections = []

        if result.boxes is None:
            return image_id, detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = [round(float(value), 2) for value in box.xyxy[0].tolist()]

            detections.append({
                "image_id": image_id,
                "class_id": class_id,
                "class": self.model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": bbox
            })

        return image_id, detections

    def predict(self, image_path, confidence=0.25):
        start = perf_counter()

        results = self.model.predict(
            source=image_path,
            conf=confidence,
            device=self.device,
            verbose=False
        )

        image_id, detections = self._parse_result(results[0])
        elapsed = perf_counter() - start

        return {
            "image_id": image_id,
            "detections": detections,
            "inference_time_ms": round(elapsed * 1000, 2)
        }

    def predict_batch(self, image_paths, confidence=0.25):
        start = perf_counter()

        results = self.model.predict(
            source=image_paths,
            conf=confidence,
            device=self.device,
            verbose=False
        )

        predictions = []

        for result in results:
            image_id, detections = self._parse_result(result)

            predictions.append({
                "image_id": image_id,
                "detections": detections
            })

        elapsed = perf_counter() - start

        return {
            "predictions": predictions,
            "total_images": len(predictions),
            "total_inference_time_ms": round(elapsed * 1000, 2),
            "average_inference_time_ms": round(
                (elapsed * 1000) / len(predictions), 2
            ) if predictions else 0.0
        }
