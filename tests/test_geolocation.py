"""
tests/test_geolocation.py
==========================
P3 — Sonar Processing + Geolocation | SONARIS Project
Phase 3: Geolocation Foundation Tests

Test suite covering:
  - SonarMetadata data model & validation
  - Detection data model & YOLO conversion
  - Sonar geometry (nadir-relative conversions, starboard/port bearings, slant range)
  - Coordinate transformations (local sonar -> WGS84 latitude/longitude)
  - Positional uncertainty estimation
  - Metadata loaders (dict & JSON)
  - Complete locator pipeline (single & batch geolocations)
"""

import json
import math
import pytest

from geolocation.coordinate_transform import (
    METRES_PER_DEGREE_LAT,
    local_to_wgs84,
)
from geolocation.locator import (
    geolocate_detection,
    geolocate_detections,
)
from geolocation.metadata_loader import (
    create_default_metadata,
    load_metadata_from_dict,
    load_metadata_from_json,
)
from geolocation.models import (
    DRISHTI_CLASS_NAMES,
    Detection,
    GeoLocation,
    SonarMetadata,
)
from geolocation.sonar_geometry import (
    pixel_localization_uncertainty_m,
    pixel_to_sonar_coords,
)
from geolocation.uncertainty import (
    DEFAULT_GPS_ACCURACY_M,
    estimate_uncertainty,
)


# ===========================================================================
# 1. Test SonarMetadata
# ===========================================================================

class TestSonarMetadata:
    def test_metadata_creation_minimal(self):
        meta = SonarMetadata(image_width_px=1000, image_height_px=500, range_m=50.0)
        assert meta.image_width_px == 1000
        assert meta.image_height_px == 500
        assert meta.range_m == 50.0
        assert not meta.has_gps()
        assert meta.latitude is None
        assert meta.longitude is None
        assert meta.heading_deg is None

    def test_metadata_creation_full_gps(self):
        meta = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=75.0,
            latitude=15.4989,
            longitude=73.8278,
            heading_deg=45.0,
            altitude_m=10.0,
            gps_accuracy_m=2.5,
            range_resolution_m=0.1,
        )
        assert meta.has_gps()
        assert meta.latitude == 15.4989
        assert meta.longitude == 73.8278
        assert meta.heading_deg == 45.0
        assert meta.altitude_m == 10.0
        assert meta.gps_accuracy_m == 2.5
        assert meta.range_resolution_m == 0.1

    def test_metadata_partial_gps_has_gps_is_false(self):
        # Lat and lon present, but heading missing
        meta1 = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            latitude=15.0,
            longitude=73.0,
            heading_deg=None,
        )
        assert not meta1.has_gps()

        # Heading present, lat/lon missing
        meta2 = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            heading_deg=90.0,
        )
        assert not meta2.has_gps()

    def test_effective_range_resolution(self):
        # Derived: range_m / (image_width_px / 2.0) = 50.0 / 500.0 = 0.1 m/px
        meta1 = SonarMetadata(image_width_px=1000, image_height_px=500, range_m=50.0)
        assert pytest.approx(meta1.effective_range_resolution(), rel=1e-5) == 0.1

        # Explicit override
        meta2 = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            range_resolution_m=0.08,
        )
        assert meta2.effective_range_resolution() == 0.08


# ===========================================================================
# 2. Test Detection
# ===========================================================================

class TestDetection:
    def test_detection_from_yolo_row_valid(self):
        det = Detection.from_yolo_row(
            image_id="sonar_sample_01",
            class_id=2,
            cx_norm=0.7,
            cy_norm=0.4,
            w_norm=0.1,
            h_norm=0.05,
            confidence=0.88,
        )
        assert det.image_id == "sonar_sample_01"
        assert det.class_id == 2
        assert det.class_name == "shipwreck"
        assert det.confidence == 0.88
        assert det.bbox_cx_norm == 0.7
        assert det.bbox_cy_norm == 0.4
        assert det.bbox_w_norm == 0.1
        assert det.bbox_h_norm == 0.05

    def test_detection_all_drishti_classes(self):
        for cid, name in DRISHTI_CLASS_NAMES.items():
            det = Detection.from_yolo_row(
                image_id="img",
                class_id=cid,
                cx_norm=0.5,
                cy_norm=0.5,
                w_norm=0.1,
                h_norm=0.1,
            )
            assert det.class_name == name

    def test_detection_invalid_class_raises(self):
        with pytest.raises(ValueError, match="Unknown class_id"):
            Detection.from_yolo_row(
                image_id="img",
                class_id=99,
                cx_norm=0.5,
                cy_norm=0.5,
                w_norm=0.1,
                h_norm=0.1,
            )


# ===========================================================================
# 3. Test Sonar Geometry
# ===========================================================================

class TestSonarGeometry:
    @pytest.fixture
    def meta_50m(self):
        # 1000px wide, range_m = 50.0 -> nadir at 500px, 0.1 m/px
        return SonarMetadata(image_width_px=1000, image_height_px=600, range_m=50.0)

    def test_nadir_center_detection(self, meta_50m):
        # Center cx = 0.5 (exactly at nadir)
        rng, bearing, lx, ly, ext = pixel_to_sonar_coords(0.5, 0.5, 0.04, meta_50m)
        assert pytest.approx(rng, abs=1e-5) == 0.0
        assert pytest.approx(lx, abs=1e-5) == 0.0
        assert ly == 0.0
        assert bearing == 90.0
        # extent: half width is 0.02 * 1000 = 20px -> 20 * 0.1 = 2.0m
        assert pytest.approx(ext, abs=1e-5) == 2.0

    def test_starboard_detection(self, meta_50m):
        # cx = 0.75 -> 750px -> 250px right of nadir
        # 250px * 0.1 m/px = 25.0m
        rng, bearing, lx, ly, _ = pixel_to_sonar_coords(0.75, 0.5, 0.02, meta_50m)
        assert pytest.approx(rng, abs=1e-5) == 25.0
        assert pytest.approx(lx, abs=1e-5) == 25.0
        assert bearing == 90.0

    def test_port_detection(self, meta_50m):
        # cx = 0.25 -> 250px -> 250px left of nadir (-250px)
        # -250px * 0.1 m/px = -25.0m
        rng, bearing, lx, ly, _ = pixel_to_sonar_coords(0.25, 0.5, 0.02, meta_50m)
        assert pytest.approx(rng, abs=1e-5) == 25.0
        assert pytest.approx(lx, abs=1e-5) == -25.0
        assert bearing == -90.0

    def test_outer_edges(self, meta_50m):
        # Far right edge cx = 1.0 -> 50.0m starboard
        rng_sb, bearing_sb, lx_sb, _, _ = pixel_to_sonar_coords(1.0, 0.5, 0.01, meta_50m)
        assert pytest.approx(rng_sb, abs=1e-5) == 50.0
        assert pytest.approx(lx_sb, abs=1e-5) == 50.0
        assert bearing_sb == 90.0

        # Far left edge cx = 0.0 -> 50.0m port
        rng_port, bearing_port, lx_port, _, _ = pixel_to_sonar_coords(0.0, 0.5, 0.01, meta_50m)
        assert pytest.approx(rng_port, abs=1e-5) == 50.0
        assert pytest.approx(lx_port, abs=1e-5) == -50.0
        assert bearing_port == -90.0

    def test_invalid_coordinates_raise(self, meta_50m):
        with pytest.raises(ValueError, match="cx_norm must be in"):
            pixel_to_sonar_coords(1.2, 0.5, 0.1, meta_50m)
        with pytest.raises(ValueError, match="cx_norm must be in"):
            pixel_to_sonar_coords(-0.1, 0.5, 0.1, meta_50m)
        with pytest.raises(ValueError, match="bbox_w_norm must be in"):
            pixel_to_sonar_coords(0.5, 0.5, 0.0, meta_50m)
        with pytest.raises(ValueError, match="bbox_w_norm must be in"):
            pixel_to_sonar_coords(0.5, 0.5, 1.5, meta_50m)

    def test_invalid_metadata_raises(self):
        meta_bad = SonarMetadata(image_width_px=0, image_height_px=100, range_m=50.0)
        with pytest.raises(ValueError, match="image_width_px must be > 0"):
            pixel_to_sonar_coords(0.5, 0.5, 0.1, meta_bad)

        meta_bad_range = SonarMetadata(image_width_px=100, image_height_px=100, range_m=-10.0)
        with pytest.raises(ValueError, match="range_m must be > 0"):
            pixel_to_sonar_coords(0.5, 0.5, 0.1, meta_bad_range)

    def test_pixel_localization_uncertainty(self, meta_50m):
        # bbox width 0.05 on 1000px = 50px width -> half is 25px
        # 25px * 0.1 m/px = 2.5m
        unc = pixel_localization_uncertainty_m(0.05, meta_50m)
        assert pytest.approx(unc, abs=1e-5) == 2.5


# ===========================================================================
# 4. Test Coordinate Transformations
# ===========================================================================

class TestCoordinateTransform:
    def test_heading_north_starboard(self):
        # Heading 0° (North), target on starboard (+90°) -> True bearing 90° (East)
        # Should move directly East: lat unchanged, lon increases
        lat0, lon0 = 0.0, 0.0
        lat1, lon1 = local_to_wgs84(
            latitude=lat0,
            longitude=lon0,
            heading_deg=0.0,
            slant_range_m=111.32,
            bearing_deg=90.0,
        )
        assert pytest.approx(lat1, abs=1e-6) == lat0
        assert lon1 > lon0
        assert pytest.approx(lon1, rel=1e-4) == 111.32 / METRES_PER_DEGREE_LAT

    def test_heading_north_port(self):
        # Heading 0° (North), target on port (-90°) -> True bearing 270° (West)
        # Should move directly West: lat unchanged, lon decreases
        lat0, lon0 = 0.0, 0.0
        lat1, lon1 = local_to_wgs84(
            latitude=lat0,
            longitude=lon0,
            heading_deg=0.0,
            slant_range_m=111.32,
            bearing_deg=-90.0,
        )
        assert pytest.approx(lat1, abs=1e-6) == lat0
        assert lon1 < lon0
        assert pytest.approx(lon1, rel=1e-4) == -111.32 / METRES_PER_DEGREE_LAT

    def test_heading_east_starboard(self):
        # Heading 90° (East), target on starboard (+90°) -> True bearing 180° (South)
        # Should move directly South: lat decreases, lon unchanged
        lat0, lon0 = 10.0, 20.0
        lat1, lon1 = local_to_wgs84(
            latitude=lat0,
            longitude=lon0,
            heading_deg=90.0,
            slant_range_m=111.32,
            bearing_deg=90.0,
        )
        assert lat1 < lat0
        assert pytest.approx(lon1, abs=1e-6) == lon0

    def test_heading_east_port(self):
        # Heading 90° (East), target on port (-90°) -> True bearing 0° (North)
        # Should move directly North: lat increases, lon unchanged
        lat0, lon0 = 10.0, 20.0
        lat1, lon1 = local_to_wgs84(
            latitude=lat0,
            longitude=lon0,
            heading_deg=90.0,
            slant_range_m=111.32,
            bearing_deg=-90.0,
        )
        assert lat1 > lat0
        assert pytest.approx(lon1, abs=1e-6) == lon0

    def test_longitude_scaling_with_latitude(self):
        # At lat = 60°, cos(60°) = 0.5. A displacement of 100m East covers twice
        # as many degrees of longitude as it does at the equator.
        lat_eq, lon_eq = local_to_wgs84(0.0, 0.0, 0.0, 100.0, 90.0)
        lat_60, lon_60 = local_to_wgs84(60.0, 0.0, 0.0, 100.0, 90.0)
        delta_lon_eq = lon_eq - 0.0
        delta_lon_60 = lon_60 - 0.0
        assert pytest.approx(delta_lon_60, rel=1e-3) == delta_lon_eq * 2.0

    def test_invalid_geodetic_inputs(self):
        with pytest.raises(ValueError, match="latitude must be in"):
            local_to_wgs84(95.0, 0.0, 0.0, 10.0, 90.0)
        with pytest.raises(ValueError, match="longitude must be in"):
            local_to_wgs84(0.0, -190.0, 0.0, 10.0, 90.0)
        with pytest.raises(ValueError, match="heading_deg must be in"):
            local_to_wgs84(0.0, 0.0, 360.0, 10.0, 90.0)
        with pytest.raises(ValueError, match="slant_range_m must be >= 0.0"):
            local_to_wgs84(0.0, 0.0, 0.0, -5.0, 90.0)


# ===========================================================================
# 5. Test Uncertainty Estimation
# ===========================================================================

class TestUncertainty:
    def test_uncertainty_none_without_gps(self):
        meta = SonarMetadata(image_width_px=1000, image_height_px=500, range_m=50.0)
        det = Detection.from_yolo_row("img", 0, 0.6, 0.5, 0.04, 0.04)
        assert estimate_uncertainty(det, meta, 20.0) is None

    def test_uncertainty_calculated_with_gps(self):
        meta = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            latitude=15.0,
            longitude=73.0,
            heading_deg=0.0,
            gps_accuracy_m=3.0,
        )
        det = Detection.from_yolo_row("img", 0, 0.6, 0.5, 0.04, 0.04)
        unc = estimate_uncertainty(det, meta, 20.0)
        assert unc is not None
        assert unc > 3.0  # Must exceed GPS-only error due to resolution, pixel loc & heading

    def test_uncertainty_increases_with_larger_bbox(self):
        meta = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            latitude=15.0,
            longitude=73.0,
            heading_deg=0.0,
        )
        det_small = Detection.from_yolo_row("img", 0, 0.6, 0.5, 0.01, 0.01)
        det_large = Detection.from_yolo_row("img", 0, 0.6, 0.5, 0.10, 0.10)

        unc_small = estimate_uncertainty(det_small, meta, 20.0)
        unc_large = estimate_uncertainty(det_large, meta, 20.0)
        assert unc_large > unc_small


# ===========================================================================
# 6. Test Metadata Loader
# ===========================================================================

class TestMetadataLoader:
    def test_load_from_dict_minimal(self):
        payload = {
            "image_width_px": 1280,
            "image_height_px": 640,
            "range_m": 75.0,
        }
        meta = load_metadata_from_dict(payload)
        assert meta.image_width_px == 1280
        assert meta.image_height_px == 640
        assert meta.range_m == 75.0
        assert not meta.has_gps()

    def test_load_from_dict_full(self):
        payload = {
            "image_width_px": 1280,
            "image_height_px": 640,
            "range_m": 75.0,
            "latitude": 12.34,
            "longitude": 56.78,
            "heading_deg": 120.0,
            "gps_accuracy_m": 1.5,
        }
        meta = load_metadata_from_dict(payload)
        assert meta.has_gps()
        assert meta.latitude == 12.34
        assert meta.longitude == 56.78
        assert meta.heading_deg == 120.0
        assert meta.gps_accuracy_m == 1.5

    def test_load_from_dict_missing_mandatory(self):
        with pytest.raises(KeyError, match="Missing mandatory metadata field"):
            load_metadata_from_dict({"image_width_px": 100, "range_m": 50.0})

    def test_load_from_dict_invalid_values(self):
        with pytest.raises(ValueError, match="range_m must be positive"):
            load_metadata_from_dict({"image_width_px": 100, "image_height_px": 100, "range_m": -5.0})

    def test_load_from_json_file(self, tmp_path):
        json_file = tmp_path / "metadata.json"
        json_file.write_text(
            json.dumps({
                "image_width_px": 800,
                "image_height_px": 400,
                "range_m": 40.0,
                "latitude": 15.1,
                "longitude": 73.2,
                "heading_deg": 90.0,
            })
        )
        meta = load_metadata_from_json(json_file)
        assert meta.image_width_px == 800
        assert meta.has_gps()

    def test_load_from_json_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_metadata_from_json("nonexistent_path_xyz.json")

    def test_create_default_metadata(self):
        meta = create_default_metadata(1024, 512, 60.0)
        assert meta.image_width_px == 1024
        assert meta.image_height_px == 512
        assert meta.range_m == 60.0
        assert not meta.has_gps()


# ===========================================================================
# 7. Test Locator
# ===========================================================================

class TestLocator:
    def test_geolocate_detection_without_gps(self):
        meta = SonarMetadata(image_width_px=1000, image_height_px=500, range_m=50.0)
        det = Detection.from_yolo_row(
            image_id="scan_01",
            class_id=2,  # shipwreck
            cx_norm=0.8,
            cy_norm=0.5,
            w_norm=0.06,
            h_norm=0.04,
            confidence=0.92,
        )
        res = geolocate_detection(det, meta)
        assert isinstance(res, GeoLocation)
        assert res.coordinate_system == "local_sonar"
        assert res.status == "relative"
        assert res.latitude is None
        assert res.longitude is None
        assert res.uncertainty_m is None
        assert pytest.approx(res.local_x_m, abs=1e-5) == 30.0  # (0.8 - 0.5)*1000 = 300px * 0.1m = 30m
        assert res.bearing_deg == 90.0

        # to_dict verification
        d = res.to_dict()
        assert d["image_id"] == "scan_01"
        assert d["class_name"] == "shipwreck"
        assert d["confidence"] == 0.92
        assert d["coordinate_system"] == "local_sonar"
        assert d["status"] == "relative"
        assert d["latitude"] is None

    def test_geolocate_detection_with_gps(self):
        meta = SonarMetadata(
            image_width_px=1000,
            image_height_px=500,
            range_m=50.0,
            latitude=15.0,
            longitude=73.0,
            heading_deg=0.0,
            gps_accuracy_m=2.0,
        )
        det = Detection.from_yolo_row(
            image_id="scan_02",
            class_id=1,  # submarine_pipeline
            cx_norm=0.8,
            cy_norm=0.5,
            w_norm=0.04,
            h_norm=0.02,
            confidence=0.85,
        )
        res = geolocate_detection(det, meta)
        assert res.coordinate_system == "WGS84"
        assert res.status == "estimated"
        assert res.latitude is not None
        assert res.longitude is not None
        assert res.uncertainty_m is not None
        assert res.uncertainty_m > 2.0

        d = res.to_dict()
        assert d["coordinate_system"] == "WGS84"
        assert d["status"] == "estimated"
        assert isinstance(d["latitude"], float)
        assert isinstance(d["longitude"], float)

    def test_geolocate_detections_batch(self):
        meta = SonarMetadata(image_width_px=1000, image_height_px=500, range_m=50.0)
        detections = [
            Detection.from_yolo_row("scan_03", 0, 0.2, 0.3, 0.05, 0.05),
            Detection.from_yolo_row("scan_03", 3, 0.7, 0.8, 0.04, 0.04),
        ]
        results = geolocate_detections(detections, meta)
        assert len(results) == 2
        assert results[0].bearing_deg == -90.0  # port
        assert results[1].bearing_deg == 90.0   # starboard
