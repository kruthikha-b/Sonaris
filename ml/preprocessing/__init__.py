"""
ml/preprocessing/__init__.py
============================
P3 — Sonar Processing + Geolocation | SONARIS Project
Phase 1: Sonar Preprocessing Foundation

Public API for the sonar preprocessing package.
Import preprocess_image directly from this package:

    from ml.preprocessing import preprocess_image
    result = preprocess_image("path/to/sonar.png")
"""

from .pipeline import preprocess_image

__all__ = ["preprocess_image"]
