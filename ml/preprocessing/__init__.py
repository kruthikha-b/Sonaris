"""
ml/preprocessing/__init__.py
============================
P3 — Sonar Processing + Geolocation | SONARIS Project

Public API for the sonar preprocessing package.

Phase 1 exports:
    preprocess_image    — denoise + CLAHE normalize → float32 [0, 1]

Phase 2 exports:
    preprocess_image_full  — Phase 1 + unsharp enhancement → float32 [0, 1]
    tile_image             — decompose image into 640×640 overlapping tiles
    tile_filename          — generate standard tile filename
    transform_annotations  — transform YOLO labels to tile-local coordinates
    load_yolo_labels       — read a YOLO .txt label file

Usage:
    from ml.preprocessing import preprocess_image
    from ml.preprocessing import preprocess_image_full, tile_image
"""

# Phase 1
from .pipeline import preprocess_image

# Phase 2
from .pipeline import preprocess_image_full
from .tiler import tile_image, tile_filename
from .tile_annotations import transform_annotations, load_yolo_labels

__all__ = [
    # Phase 1
    "preprocess_image",
    # Phase 2
    "preprocess_image_full",
    "tile_image",
    "tile_filename",
    "transform_annotations",
    "load_yolo_labels",
]
