"""
tests/test_integration_e2e.py
=============================
P3 — Sonar Processing + Geolocation | SONARIS Project
Phase 4: Final Integration, Testing, Compatibility & E2E Validation

Test suite covering:
  - Dataset configuration and environment variable resolver (SONARIS_DATA_DIR)
  - P1 (AI/YOLO) interface compatibility
  - P2 (Data Engineering) read-only guarantee and schema alignment
  - Full End-to-End P3 pipeline:
      Raw Sonar Image -> Preprocessing -> Enhancement -> Tiling ->
      Detection Interface -> Geolocation -> P4/P5 Structured Result
  - Edge cases (uniform, small, non-divisible, missing GPS, invalid coords)
  - Performance sanity checks
"""

import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from geolocation import (
    DRISHTI_CLASS_NAMES,
    Detection,
    GeoLocation,
    SonarMetadata,
    create_default_metadata,
    geolocate_detection,
    geolocate_detections,
)
from ml.preprocessing import (
    get_dataset_dir,
    get_output_dir,
    is_dataset_available,
    preprocess_image_full,
    tile_image,
    transform_annotations,
)
from ml.preprocessing.config import ENV_SONARIS_DATA_DIR, ENV_SONARIS_OUTPUT_DIR


# ===========================================================================
# Fixture: Synthetic Representative SSS Image
# ===========================================================================

@pytest.fixture
def representative_sss_image():
    """
    Generate a representative side-scan sonar image (1280x640) with:
    - Port side: column 0 to 639 (intensity gradient from nadir to edge)
    - Starboard side: column 640 to 1279
    - Nadir dead-zone: low reflection strip in the center
    - Realistic Rayleigh/speckle noise
    - Acoustic anomaly targets (bright highlight + acoustic shadow)
    """
    H, W = 640, 1280
    rng = np.random.default_rng(42)

    # Base seabed intensity gradient from center nadir to outer edges
    dist_from_nadir = np.abs(np.arange(W) - W / 2.0) / (W / 2.0)
    base_intensity = 180.0 - 90.0 * dist_from_nadir  # [90, 180]
    img = np.tile(base_intensity.astype(np.float32), (H, 1))

    # Dark nadir stripe (column 620 to 660)
    img[:, 625:655] *= 0.25

    # Simulated target 1: Starboard shipwreck at x=950..990, y=280..320 (bright + shadow)
    img[280:310, 950:980] = 245.0  # acoustic highlight
    img[280:310, 985:1020] = 15.0  # acoustic shadow behind target

    # Simulated target 2: Port pipeline at x=200..230, y=100..500
    img[100:500, 200:215] = 230.0  # pipe specular reflection
    img[100:500, 216:230] = 20.0   # shadow

    # Multiplicative speckle noise
    noise = rng.normal(1.0, 0.12, (H, W)).astype(np.float32)
    noisy_img = np.clip(img * noise, 0, 255).astype(np.uint8)

    return noisy_img


# ===========================================================================
# 1. Test Dataset Configuration & Resolver
# ===========================================================================

class TestDatasetConfiguration:
    def test_get_dataset_dir_with_explicit_valid_path(self, tmp_path):
        custom = tmp_path / "my_external_drive_dataset"
        custom.mkdir()
        resolved = get_dataset_dir(custom_path=custom)
        assert resolved == custom

    def test_get_dataset_dir_with_explicit_missing_path(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Specified dataset path does not exist"):
            get_dataset_dir(custom_path=nonexistent)

    def test_get_dataset_dir_from_env_var(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "gdrive_synced_dataset"
        env_dir.mkdir()
        monkeypatch.setenv(ENV_SONARIS_DATA_DIR, str(env_dir))

        resolved = get_dataset_dir()
        assert resolved == env_dir
        assert is_dataset_available() is True

    def test_get_dataset_dir_missing_env_var(self, monkeypatch):
        monkeypatch.setenv(ENV_SONARIS_DATA_DIR, "Z:\\nonexistent_network_path_12345")
        with pytest.raises(FileNotFoundError, match="is set to 'Z:"):
            get_dataset_dir()

    def test_get_output_dir_creation(self, tmp_path, monkeypatch):
        out = tmp_path / "preprocessed_out"
        monkeypatch.setenv(ENV_SONARIS_OUTPUT_DIR, str(out))
        resolved = get_output_dir()
        assert resolved == out
        assert out.exists()


# ===========================================================================
# 2. Test P1 Interface Compatibility
# ===========================================================================

class TestP1Compatibility:
    def test_tile_dimensions_match_yolo_standard(self, representative_sss_image):
        """P1 YOLO expects 640x640 input tiles."""
        processed = preprocess_image_full(representative_sss_image)
        tiles = tile_image(processed, tile_h=640, tile_w=640, overlap=64)

        assert len(tiles) > 0
        for t in tiles:
            assert t.tile.shape == (640, 640), f"Tile shape {t.tile.shape} must be 640x640"
            assert t.tile.dtype == np.float32
            assert 0.0 <= t.tile.min() <= t.tile.max() <= 1.0

    def test_tiles_convertible_to_uint8_bgr(self, representative_sss_image):
        """Standard Ultralytics YOLO models consume 3-channel uint8 (640x640x3)."""
        processed = preprocess_image_full(representative_sss_image)
        tiles = tile_image(processed, tile_h=640, tile_w=640)

        # Convert float32 tile to 3-channel uint8 BGR
        tile_uint8 = (tiles[0].tile * 255.0).astype(np.uint8)
        tile_bgr = cv2.cvtColor(tile_uint8, cv2.COLOR_GRAY2BGR)

        assert tile_bgr.shape == (640, 640, 3)
        assert tile_bgr.dtype == np.uint8

    def test_p1_detection_schema_ingestion(self):
        """Simulate P1 YOLO detection row and verify Detection model accepts it."""
        # Simulated YOLO detection: class_id=2 (shipwreck), conf=0.92, normalized coords
        yolo_detection = {
            "image_id": "survey_tile_001",
            "class_id": 2,
            "cx_norm": 0.54,
            "cy_norm": 0.48,
            "w_norm": 0.08,
            "h_norm": 0.05,
            "confidence": 0.92,
        }

        det = Detection.from_yolo_row(**yolo_detection)
        assert det.class_name == "shipwreck"
        assert det.confidence == 0.92
        assert det.bbox_cx_norm == 0.54


# ===========================================================================
# 3. Test P2 Interface Compatibility
# ===========================================================================

class TestP2Compatibility:
    def test_all_drishti_class_ids_aligned(self):
        """Ensure class IDs 0-4 match DRISHTI-SSS benchmark dataset."""
        expected_classes = {
            0: "crab_pot",
            1: "submarine_pipeline",
            2: "shipwreck",
            3: "ghost_net",
            4: "mine_cylinder",
        }
        assert DRISHTI_CLASS_NAMES == expected_classes

    def test_read_only_guarantee_on_input_image(self, representative_sss_image):
        """P3 preprocessing must never mutate source image array."""
        original_copy = representative_sss_image.copy()
        _ = preprocess_image_full(representative_sss_image)
        assert np.array_equal(representative_sss_image, original_copy)


# ===========================================================================
# 4. Test End-to-End P3 Pipeline
# ===========================================================================

class TestEndToEndPipeline:
    def test_e2e_pipeline_relative_mode(self, representative_sss_image):
        """
        End-to-End Pipeline in Relative Sonar mode (no GPS in dataset):
        Raw SSS -> preprocess_image_full -> tile -> P1 detection ->
        geolocate_detection -> structured result.
        """
        # 1. Preprocess full SSS image
        enhanced = preprocess_image_full(representative_sss_image)
        assert enhanced.shape == (640, 1280)
        assert enhanced.dtype == np.float32

        # 2. Decompose into overlapping tiles
        tiles = tile_image(enhanced, tile_h=640, tile_w=640, overlap=64)
        assert len(tiles) >= 2

        # 3. Sensor metadata (swath range = 75m)
        meta = create_default_metadata(image_width_px=1280, image_height_px=640, range_m=75.0)
        assert not meta.has_gps()

        # 4. Simulated P1 detection on the starboard side (target at x=965, y=295 in 1280x640)
        # cx_norm = 965 / 1280 = 0.7539
        det = Detection.from_yolo_row(
            image_id="sonar_scan_001",
            class_id=2,  # shipwreck
            cx_norm=965.0 / 1280.0,
            cy_norm=295.0 / 640.0,
            w_norm=40.0 / 1280.0,
            h_norm=30.0 / 640.0,
            confidence=0.94,
        )

        # 5. Geolocation
        result = geolocate_detection(det, meta)

        # 6. Verify result
        assert isinstance(result, GeoLocation)
        assert result.coordinate_system == "local_sonar"
        assert result.status == "relative"
        assert result.latitude is None
        assert result.longitude is None
        assert result.uncertainty_m is None
        assert result.bearing_deg == 90.0  # Starboard side
        assert result.local_x_m > 0.0      # Positive cross-track
        assert pytest.approx(result.range_m, rel=1e-2) == (965.0 - 640.0) * (75.0 / 640.0)

        # 7. Verify P4/P5 structured dictionary contract
        data_dict = result.to_dict()
        required_keys = {
            "image_id", "class_name", "confidence", "range_m", "bearing_deg",
            "local_x_m", "local_y_m", "coordinate_system", "status",
            "uncertainty_m", "latitude", "longitude"
        }
        assert required_keys.issubset(data_dict.keys())
        assert data_dict["class_name"] == "shipwreck"
        assert data_dict["confidence"] == 0.94

    def test_e2e_pipeline_geographic_mode(self, representative_sss_image):
        """
        End-to-End Pipeline in Geographic Mode (WGS84 with navigation input):
        Raw SSS -> preprocess_image_full -> tile -> P1 detection ->
        geolocate_detection -> WGS84 coordinates & uncertainty.
        """
        enhanced = preprocess_image_full(representative_sss_image)

        # Vessel heading East (90.0°), location in Goa coastal waters
        meta = SonarMetadata(
            image_width_px=1280,
            image_height_px=640,
            range_m=75.0,
            latitude=15.4989,
            longitude=73.8278,
            heading_deg=90.0,
            gps_accuracy_m=2.0,
        )
        assert meta.has_gps()

        # Starboard detection (heading 90° + bearing +90° = 180° South)
        det = Detection.from_yolo_row(
            image_id="sonar_scan_002",
            class_id=1,  # pipeline
            cx_norm=0.75,  # 37.5m starboard
            cy_norm=0.5,
            w_norm=0.04,
            h_norm=0.04,
            confidence=0.88,
        )

        result = geolocate_detection(det, meta)

        assert result.coordinate_system == "WGS84"
        assert result.status == "estimated"
        assert result.latitude is not None
        assert result.longitude is not None
        assert result.uncertainty_m is not None
        assert result.uncertainty_m > 2.0  # Includes GPS + resolution + pixel + heading error

        # Heading East + Starboard = South: Latitude must decrease, Longitude approx unchanged
        assert result.latitude < 15.4989
        assert pytest.approx(result.longitude, abs=1e-5) == 73.8278


# ===========================================================================
# 5. Test Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_uniform_black_image(self):
        """Completely black sonar image (zero echo) must process without crashing."""
        black = np.zeros((512, 512), dtype=np.uint8)
        out = preprocess_image_full(black)
        assert out.shape == (512, 512)
        assert np.all(out == 0.0)

    def test_empty_uniform_white_image(self):
        """Completely saturated image must process safely."""
        white = np.full((512, 512), 255, dtype=np.uint8)
        out = preprocess_image_full(white)
        assert out.shape == (512, 512)
        assert np.all(out == 0.0)

    def test_image_smaller_than_tile_size(self):
        """Small image (e.g. 200x300) padded to tile size with reflect padding."""
        small = np.random.randint(50, 200, (200, 300), dtype=np.uint8)
        processed = preprocess_image_full(small)
        tiles = tile_image(processed, tile_h=640, tile_w=640)
        assert len(tiles) == 1
        assert tiles[0].tile.shape == (640, 640)

    def test_image_non_divisible_dimensions(self):
        """Image dimensions not divisible by tile size (e.g. 1025x513)."""
        odd = np.random.randint(50, 200, (513, 1025), dtype=np.uint8)
        processed = preprocess_image_full(odd)
        tiles = tile_image(processed, tile_h=640, tile_w=640, overlap=64)
        assert len(tiles) >= 2
        for t in tiles:
            assert t.tile.shape == (640, 640)

    def test_empty_detections_batch(self):
        """Empty list of detections should return empty list without error."""
        meta = create_default_metadata(1000, 500, 50.0)
        results = geolocate_detections([], meta)
        assert results == []

    def test_partial_gps_metadata_falls_back_to_relative(self):
        """If GPS coordinates are present but heading is missing, fall back to relative."""
        meta = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            latitude=15.0,
            longitude=73.0,
            heading_deg=None,  # Missing heading
        )
        det = Detection.from_yolo_row("img", 0, 0.7, 0.5, 0.05, 0.05)
        result = geolocate_detection(det, meta)
        assert result.coordinate_system == "local_sonar"
        assert result.status == "relative"
        assert result.latitude is None

    def test_corrupt_invalid_input_raises_type_error(self):
        """Invalid types passed to geolocate_detection raise TypeError."""
        meta = create_default_metadata(1000, 500, 50.0)
        with pytest.raises(TypeError, match="Expected Detection instance"):
            geolocate_detection({"not": "a detection"}, meta)


# ===========================================================================
# 6. Performance Sanity Check
# ===========================================================================

class TestPerformanceSanity:
    def test_processing_latency_benchmarks(self, representative_sss_image):
        """
        Verify that processing per image satisfies reasonable throughput:
        - Full preprocessing (denoise + CLAHE + unsharp) < 1.0 second on 1280x640
        - Tiling < 0.1 second
        - Geolocation for 100 detections < 0.05 second
        """
        # Preprocessing time
        t0 = time.perf_counter()
        processed = preprocess_image_full(representative_sss_image)
        t_pre = time.perf_counter() - t0

        # Tiling time
        t1 = time.perf_counter()
        tiles = tile_image(processed, tile_h=640, tile_w=640, overlap=64)
        t_tile = time.perf_counter() - t1

        # Geolocation time (100 detections)
        meta = SonarMetadata(
            image_width_px=1280,
            image_height_px=640,
            range_m=75.0,
            latitude=15.0,
            longitude=73.0,
            heading_deg=45.0,
            gps_accuracy_m=2.5,
        )
        detections = [
            Detection.from_yolo_row(
                image_id=f"det_{i}",
                class_id=i % 5,
                cx_norm=0.2 + 0.6 * (i / 100.0),
                cy_norm=0.5,
                w_norm=0.05,
                h_norm=0.03,
            )
            for i in range(100)
        ]

        t2 = time.perf_counter()
        results = geolocate_detections(detections, meta)
        t_geo = time.perf_counter() - t2

        assert len(results) == 100
        assert t_pre < 1.0, f"Preprocessing too slow: {t_pre:.3f}s"
        assert t_tile < 0.1, f"Tiling too slow: {t_tile:.3f}s"
        assert t_geo < 0.05, f"Geolocation too slow: {t_geo:.3f}s"
