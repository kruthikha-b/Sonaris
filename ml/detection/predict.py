from ultralytics import YOLO


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path, confidence=0.25):
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "confidence": confidence_score,
                    "bbox": coordinates
                })

        return detections
