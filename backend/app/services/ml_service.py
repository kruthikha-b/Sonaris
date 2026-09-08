from ml.detection.predict import Detector

detector = Detector("models/best.pt")


def predict(image_path: str):
    """
    Run real P1 model inference.
    """

    result = detector.predict(image_path)

    return result