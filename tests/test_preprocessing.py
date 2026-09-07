"""
tests/test_preprocessing.py
============================
P3 — Sonar Processing + Geolocation | SONARIS Project
Phase 1: Sonar Preprocessing Foundation — Test Suite

Tests for ml/preprocessing/ (denoise, normalize, pipeline).

All tests use synthetic numpy arrays — no real dataset access required.
The real DRISHTI-SSS dataset (C:\\drishti_clean) is NOT touched.

Run with:
    python -m pytest tests/test_preprocessing.py -v

    # Or from the project root:
    python -m pytest tests/test_preprocessing.py -v --tb=short
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing ml.preprocessing from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.denoise import denoise
from ml.preprocessing.normalize import (
    normalize_clahe,
    normalize_float,
    normalize_uint8,
)
from ml.preprocessing.pipeline import preprocess_image, preprocess_image_full
from ml.preprocessing.enhance import enhance
from ml.preprocessing.tiler import tile_image, tile_filename, TileInfo
from ml.preprocessing.tile_annotations import transform_annotations, load_yolo_labels



# ---------------------------------------------------------------------------
# Fixtures — reusable synthetic sonar images
# ---------------------------------------------------------------------------

@pytest.fixture
def sonar_gray_small():
    """64×64 grayscale uint8 image with noise."""
    rng = np.random.default_rng(seed=42)
    base = rng.integers(50, 150, (64, 64), dtype=np.uint8)
    noise = rng.integers(0, 30, (64, 64), dtype=np.uint8)
    return np.clip(base.astype(np.int32) + noise.astype(np.int32), 0, 255).astype(np.uint8)


@pytest.fixture
def sonar_gray_medium():
    """256×512 grayscale uint8 image simulating SSS swath."""
    rng = np.random.default_rng(seed=7)
    gradient = np.linspace(200, 30, 512, dtype=np.float32)  # near→far gradient
    image = np.tile(gradient, (256, 1)).astype(np.float32)
    noise = rng.normal(0, 15, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def sonar_bgr():
    """128×128×3 BGR uint8 image (sonar stored as colour)."""
    rng = np.random.default_rng(seed=99)
    gray = rng.integers(30, 220, (128, 128), dtype=np.uint8)
    # Stack identical channels (common SSS storage format)
    return np.stack([gray, gray, gray], axis=2)


@pytest.fixture
def uniform_black():
    """128×128 all-zeros image — edge case for normalization."""
    return np.zeros((128, 128), dtype=np.uint8)


@pytest.fixture
def uniform_white():
    """128×128 all-255 image — edge case for normalization."""
    return np.full((128, 128), 255, dtype=np.uint8)


# ===========================================================================
# 1. DENOISING TESTS
# ===========================================================================

class TestDenoise:
    """Tests for ml.preprocessing.denoise.denoise()"""

    def test_valid_grayscale_returns_same_shape(self, sonar_gray_small):
        """Denoised output must have the same spatial dimensions."""
        result = denoise(sonar_gray_small)
        assert result.shape == sonar_gray_small.shape, (
            f"Shape mismatch: input {sonar_gray_small.shape}, output {result.shape}"
        )

    def test_valid_grayscale_returns_uint8(self, sonar_gray_small):
        """Denoised output must be uint8 (same as input)."""
        result = denoise(sonar_gray_small)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_denoised_values_in_uint8_range(self, sonar_gray_small):
        """All pixel values must remain in [0, 255]."""
        result = denoise(sonar_gray_small)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_denoising_changes_image(self):
        """Denoising with sufficient strength should alter a heavily-noised image.

        The sonar_gray_small fixture has moderate noise (0-30 range) which NLM
        at default h=10 may not visibly change on a small 64×64 patch. This
        test uses a deliberately noisy image and h=15 to ensure measurable
        denoising effect.
        """
        rng = np.random.default_rng(seed=0)
        # Base uniform signal + strong Gaussian noise
        base = np.full((128, 128), 128, dtype=np.uint8)
        noise = rng.integers(0, 60, (128, 128), dtype=np.uint8)
        noisy = np.clip(
            base.astype(np.int32) + noise.astype(np.int32) - 30, 0, 255
        ).astype(np.uint8)

        result = denoise(noisy, h=15.0)

        # Denoised image must differ from input in at least 1% of pixels
        diff_pixels = int(np.sum(result != noisy))
        total_pixels = noisy.size
        assert diff_pixels > total_pixels * 0.01, (
            f"Expected denoising to change >1% of pixels, "
            f"but only {diff_pixels}/{total_pixels} pixels changed."
        )

    def test_valid_bgr_input(self, sonar_bgr):
        """Denoising should handle 3-channel BGR input without error."""
        result = denoise(sonar_bgr)
        assert result.shape == sonar_bgr.shape
        assert result.dtype == np.uint8

    def test_medium_image_shape_preserved(self, sonar_gray_medium):
        """Shape preserved for non-square SSS swath dimensions."""
        result = denoise(sonar_gray_medium)
        assert result.shape == sonar_gray_medium.shape

    def test_raises_on_none_input(self):
        """None input must raise TypeError."""
        with pytest.raises(TypeError):
            denoise(None)  # type: ignore[arg-type]

    def test_raises_on_empty_array(self):
        """Empty array must raise ValueError."""
        with pytest.raises(ValueError):
            denoise(np.array([], dtype=np.uint8))

    def test_raises_on_wrong_ndim(self):
        """1D array must raise ValueError."""
        with pytest.raises(ValueError):
            denoise(np.ones(128, dtype=np.uint8))

    def test_raises_on_wrong_dtype(self, sonar_gray_small):
        """float32 input must raise ValueError (expects uint8)."""
        float_img = sonar_gray_small.astype(np.float32)
        with pytest.raises(ValueError, match="uint8"):
            denoise(float_img)

    def test_configurable_h_parameter(self, sonar_gray_medium):
        """Different h values should produce different denoising strengths."""
        result_low = denoise(sonar_gray_medium, h=3.0)
        result_high = denoise(sonar_gray_medium, h=20.0)
        # Higher h = more smoothing = lower std dev
        assert result_high.std() <= result_low.std(), (
            "Higher h should produce more smoothing (lower std dev)."
        )


# ===========================================================================
# 2. NORMALIZATION TESTS
# ===========================================================================

class TestNormalize:
    """Tests for ml.preprocessing.normalize functions."""

    # --- normalize_clahe ---

    def test_clahe_returns_same_shape(self, sonar_gray_small):
        """CLAHE output must match input shape."""
        result = normalize_clahe(sonar_gray_small)
        assert result.shape == sonar_gray_small.shape

    def test_clahe_returns_uint8(self, sonar_gray_small):
        """CLAHE output must be uint8."""
        result = normalize_clahe(sonar_gray_small)
        assert result.dtype == np.uint8

    def test_clahe_raises_on_none(self):
        """None input must raise TypeError."""
        with pytest.raises(TypeError):
            normalize_clahe(None)  # type: ignore[arg-type]

    def test_clahe_raises_on_empty(self):
        """Empty array must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_clahe(np.zeros((0, 0), dtype=np.uint8))

    def test_clahe_raises_on_non_grayscale(self, sonar_bgr):
        """3D BGR input to CLAHE must raise ValueError (expects 2D)."""
        with pytest.raises(ValueError):
            normalize_clahe(sonar_bgr)

    def test_clahe_raises_on_float_input(self, sonar_gray_small):
        """float32 input must raise ValueError."""
        with pytest.raises(ValueError, match="uint8"):
            normalize_clahe(sonar_gray_small.astype(np.float32))

    # --- normalize_float ---

    def test_float_returns_float32(self, sonar_gray_small):
        """normalize_float must return float32 array."""
        result = normalize_float(sonar_gray_small)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_float_output_in_range(self, sonar_gray_small):
        """normalize_float output must be in [0.0, 1.0]."""
        result = normalize_float(sonar_gray_small)
        assert float(result.min()) >= 0.0, f"Min < 0: {result.min()}"
        assert float(result.max()) <= 1.0, f"Max > 1: {result.max()}"

    def test_float_shape_preserved(self, sonar_gray_medium):
        """Output shape must match input shape (non-square)."""
        result = normalize_float(sonar_gray_medium)
        assert result.shape == sonar_gray_medium.shape

    def test_float_uniform_black_returns_zeros(self, uniform_black):
        """All-black image should return all-zeros (no div by zero crash)."""
        result = normalize_float(uniform_black)
        assert result.dtype == np.float32
        assert float(result.max()) == 0.0

    def test_float_uniform_white_handled(self, uniform_white):
        """All-white image should not crash and produce valid float32."""
        result = normalize_float(uniform_white)
        assert result.dtype == np.float32
        # CLAHE on uniform-255 may produce a uniform image → zeros after norm
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    # --- normalize_uint8 ---

    def test_uint8_output_is_uint8(self, sonar_gray_small):
        """normalize_uint8 must return uint8."""
        result = normalize_uint8(sonar_gray_small)
        assert result.dtype == np.uint8

    def test_uint8_values_in_range(self, sonar_gray_small):
        """normalize_uint8 values must be in [0, 255]."""
        result = normalize_uint8(sonar_gray_small)
        assert result.min() >= 0
        assert result.max() <= 255


# ===========================================================================
# 3. PIPELINE TESTS
# ===========================================================================

class TestPipeline:
    """Tests for ml.preprocessing.pipeline.preprocess_image()"""

    def test_pipeline_from_array_returns_float32(self, sonar_gray_small):
        """Pipeline from numpy array must return float32."""
        result = preprocess_image(sonar_gray_small)
        assert result.dtype == np.float32

    def test_pipeline_output_in_range(self, sonar_gray_small):
        """Pipeline output must be in [0.0, 1.0]."""
        result = preprocess_image(sonar_gray_small)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_pipeline_shape_preserved(self, sonar_gray_small):
        """Output shape must match input spatial dimensions."""
        result = preprocess_image(sonar_gray_small)
        assert result.shape == sonar_gray_small.shape

    def test_pipeline_from_bgr_array(self, sonar_bgr):
        """Pipeline from BGR array must auto-convert to grayscale."""
        result = preprocess_image(sonar_bgr)
        assert result.dtype == np.float32
        # Output should be 2D (grayscale)
        assert result.ndim == 2
        assert result.shape == (128, 128)

    def test_pipeline_medium_image(self, sonar_gray_medium):
        """Pipeline must handle non-square (256×512) SSS swath images."""
        result = preprocess_image(sonar_gray_medium)
        assert result.shape == sonar_gray_medium.shape
        assert result.dtype == np.float32

    def test_pipeline_deterministic(self, sonar_gray_small):
        """Same input must always produce the same output."""
        result_a = preprocess_image(sonar_gray_small)
        result_b = preprocess_image(sonar_gray_small)
        np.testing.assert_array_equal(result_a, result_b)

    def test_pipeline_from_file_path(self, sonar_gray_small, tmp_path):
        """Pipeline must accept a file path and produce valid output."""
        import cv2

        # Save synthetic image to a temp PNG file
        temp_img = tmp_path / "test_sonar.png"
        cv2.imwrite(str(temp_img), sonar_gray_small)

        result = preprocess_image(temp_img)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_pipeline_saves_output_file(self, sonar_gray_small, tmp_path):
        """Pipeline with output_path must create a file on disk."""
        output_file = tmp_path / "output" / "sonar_preprocessed.png"
        preprocess_image(sonar_gray_small, output_path=output_file)

        assert output_file.exists(), f"Expected output file not found: {output_file}"
        assert output_file.stat().st_size > 0, "Output file is empty."

    def test_pipeline_does_not_modify_input(self, sonar_gray_small):
        """Input numpy array must not be modified in-place."""
        original = sonar_gray_small.copy()
        preprocess_image(sonar_gray_small)
        np.testing.assert_array_equal(sonar_gray_small, original)

    def test_pipeline_raises_on_none(self):
        """None input must raise TypeError."""
        with pytest.raises(TypeError):
            preprocess_image(None)  # type: ignore[arg-type]

    def test_pipeline_raises_on_empty_array(self):
        """Empty array must raise ValueError."""
        with pytest.raises(ValueError):
            preprocess_image(np.array([], dtype=np.uint8))

    def test_pipeline_raises_on_missing_file(self, tmp_path):
        """Non-existent file path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocess_image(tmp_path / "does_not_exist.png")

    def test_pipeline_raises_on_wrong_extension(self, tmp_path):
        """Unsupported file extension must raise ValueError."""
        fake_file = tmp_path / "sonar.tiff"
        fake_file.write_bytes(b"fake data")
        with pytest.raises(ValueError, match="Unsupported file format"):
            preprocess_image(fake_file)


# ===========================================================================
# 4. ENHANCEMENT TESTS (Phase 2)
# ===========================================================================

class TestEnhance:
    """Tests for ml.preprocessing.enhance.enhance()"""

    def test_valid_grayscale_returns_same_shape(self, sonar_gray_small):
        """Enhanced output must have the same spatial dimensions."""
        result = enhance(sonar_gray_small)
        assert result.shape == sonar_gray_small.shape

    def test_valid_grayscale_returns_uint8(self, sonar_gray_small):
        """Enhanced output must be uint8."""
        result = enhance(sonar_gray_small)
        assert result.dtype == np.uint8

    def test_enhanced_values_in_uint8_range(self, sonar_gray_small):
        """All pixel values must be in [0, 255]."""
        result = enhance(sonar_gray_small)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_zero_amount_returns_copy(self, sonar_gray_small):
        """amount=0.0 should return a copy identical to input (no-op)."""
        result = enhance(sonar_gray_small, amount=0.0)
        np.testing.assert_array_equal(result, sonar_gray_small)

    def test_enhancement_changes_image(self, sonar_gray_medium):
        """With amount > 0, enhanced image must differ from input."""
        result = enhance(sonar_gray_medium, amount=1.0)
        assert not np.array_equal(result, sonar_gray_medium), (
            "enhance() with amount=1.0 should alter the image."
        )

    def test_stronger_amount_increases_contrast(self, sonar_gray_medium):
        """Higher amount should produce more extreme pixel values (wider std)."""
        result_mild = enhance(sonar_gray_medium, amount=0.5)
        result_strong = enhance(sonar_gray_medium, amount=2.0)
        assert result_strong.std() >= result_mild.std(), (
            "Stronger sharpening should increase standard deviation."
        )

    def test_medium_image_shape_preserved(self, sonar_gray_medium):
        """Shape preserved for non-square SSS swath images."""
        result = enhance(sonar_gray_medium)
        assert result.shape == sonar_gray_medium.shape

    def test_raises_on_none(self):
        """None input must raise TypeError."""
        with pytest.raises(TypeError):
            enhance(None)  # type: ignore[arg-type]

    def test_raises_on_empty_array(self):
        """Empty array must raise ValueError."""
        with pytest.raises(ValueError):
            enhance(np.zeros((0, 0), dtype=np.uint8))

    def test_raises_on_non_grayscale(self, sonar_bgr):
        """3D BGR input must raise ValueError (expects 2D)."""
        with pytest.raises(ValueError):
            enhance(sonar_bgr)

    def test_raises_on_float_input(self, sonar_gray_small):
        """float32 input must raise ValueError (expects uint8)."""
        with pytest.raises(ValueError, match="uint8"):
            enhance(sonar_gray_small.astype(np.float32))

    def test_raises_on_negative_amount(self, sonar_gray_small):
        """Negative amount must raise ValueError."""
        with pytest.raises(ValueError, match="amount"):
            enhance(sonar_gray_small, amount=-0.1)

    def test_raises_on_zero_sigma(self, sonar_gray_small):
        """sigma=0 must raise ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            enhance(sonar_gray_small, sigma=0.0)

    def test_does_not_modify_input(self, sonar_gray_small):
        """Input array must not be modified in-place."""
        original = sonar_gray_small.copy()
        enhance(sonar_gray_small)
        np.testing.assert_array_equal(sonar_gray_small, original)


# ===========================================================================
# 5. TILING TESTS (Phase 2)
# ===========================================================================

class TestTiler:
    """Tests for ml.preprocessing.tiler.tile_image()"""

    @pytest.fixture
    def preprocessed_large(self):
        """800×1200 float32 image simulating a preprocessed SSS swath."""
        rng = np.random.default_rng(seed=55)
        return rng.random((800, 1200)).astype(np.float32)

    @pytest.fixture
    def preprocessed_small(self):
        """300×400 float32 image — smaller than one tile."""
        rng = np.random.default_rng(seed=66)
        return rng.random((300, 400)).astype(np.float32)

    @pytest.fixture
    def preprocessed_exact(self):
        """640×640 float32 image — exactly one tile, no padding needed."""
        rng = np.random.default_rng(seed=77)
        return rng.random((640, 640)).astype(np.float32)

    def test_all_tiles_correct_shape(self, preprocessed_large):
        """Every tile must have the requested tile_h × tile_w shape."""
        tiles = tile_image(preprocessed_large, tile_h=640, tile_w=640, overlap=64)
        assert len(tiles) > 0
        for t in tiles:
            assert t.tile.shape == (640, 640), (
                f"Tile ({t.row},{t.col}) has shape {t.tile.shape}, expected (640,640)"
            )

    def test_all_tiles_float32(self, preprocessed_large):
        """Every tile must be float32."""
        tiles = tile_image(preprocessed_large)
        for t in tiles:
            assert t.tile.dtype == np.float32

    def test_all_tiles_in_range(self, preprocessed_large):
        """Every tile must have values in [0, 1]."""
        tiles = tile_image(preprocessed_large)
        for t in tiles:
            assert float(t.tile.min()) >= 0.0
            assert float(t.tile.max()) <= 1.0

    def test_image_smaller_than_tile(self, preprocessed_small):
        """Image smaller than tile size → exactly 1 tile (padded)."""
        tiles = tile_image(preprocessed_small, tile_h=640, tile_w=640, overlap=64)
        assert len(tiles) == 1
        assert tiles[0].tile.shape == (640, 640)

    def test_image_exactly_tile_size(self, preprocessed_exact):
        """Image exactly matching tile size → exactly 1 tile, no padding."""
        tiles = tile_image(preprocessed_exact, tile_h=640, tile_w=640, overlap=64)
        assert len(tiles) == 1
        assert tiles[0].tile.shape == (640, 640)

    def test_non_divisible_dimensions(self):
        """Image dimensions not divisible by stride → still all tiles are (640,640)."""
        # 700×900 → not divisible by stride (640-64=576)
        rng = np.random.default_rng(seed=88)
        img = rng.random((700, 900)).astype(np.float32)
        tiles = tile_image(img, tile_h=640, tile_w=640, overlap=64)
        for t in tiles:
            assert t.tile.shape == (640, 640)

    def test_zero_overlap(self):
        """overlap=0 should work and produce correct tile count."""
        rng = np.random.default_rng(seed=11)
        img = rng.random((640, 1280)).astype(np.float32)
        tiles = tile_image(img, tile_h=640, tile_w=640, overlap=0)
        # 640×1280 with no overlap → 1 row × 2 cols = 2 tiles
        assert len(tiles) == 2

    def test_tile_row_col_monotonic(self, preprocessed_large):
        """Tiles should be in row-major order (row index non-decreasing)."""
        tiles = tile_image(preprocessed_large)
        rows = [t.row for t in tiles]
        assert rows == sorted(rows)

    def test_offsets_are_non_negative(self, preprocessed_large):
        """x_offset and y_offset must be >= 0."""
        tiles = tile_image(preprocessed_large)
        for t in tiles:
            assert t.x_offset >= 0
            assert t.y_offset >= 0

    def test_raises_on_non_float32(self, sonar_gray_small):
        """uint8 input must raise ValueError (expects float32)."""
        with pytest.raises(ValueError, match="float32"):
            tile_image(sonar_gray_small)

    def test_raises_on_empty(self):
        """Empty array must raise ValueError."""
        with pytest.raises(ValueError):
            tile_image(np.zeros((0, 0), dtype=np.float32))

    def test_raises_on_none(self):
        """None must raise TypeError."""
        with pytest.raises(TypeError):
            tile_image(None)  # type: ignore[arg-type]

    def test_raises_on_overlap_too_large(self):
        """overlap >= tile_h must raise ValueError."""
        img = np.zeros((640, 640), dtype=np.float32)
        with pytest.raises(ValueError, match="overlap"):
            tile_image(img, tile_h=640, tile_w=640, overlap=640)

    def test_tile_filename_format(self):
        """tile_filename() must follow the documented naming convention."""
        name = tile_filename("survey_001", row=2, col=15)
        assert name == "survey_001_tile_002_015.png"


# ===========================================================================
# 6. ANNOTATION TRANSFORMATION TESTS (Phase 2)
# ===========================================================================

class TestTileAnnotations:
    """Tests for ml.preprocessing.tile_annotations.transform_annotations()"""

    @pytest.fixture
    def centered_tile(self):
        """A 640×640 tile starting at offset (0, 0) in a 640×640 full image."""
        return TileInfo(
            tile=np.zeros((640, 640), dtype=np.float32),
            row=0, col=0,
            x_offset=0, y_offset=0,
            tile_h=640, tile_w=640,
        )

    @pytest.fixture
    def offset_tile(self):
        """A 640×640 tile starting at offset (200, 100) for a 1280×960 image."""
        return TileInfo(
            tile=np.zeros((640, 640), dtype=np.float32),
            row=0, col=1,
            x_offset=200, y_offset=100,
            tile_h=640, tile_w=640,
        )

    def test_annotation_fully_inside_tile(self, centered_tile):
        """Annotation fully inside tile → kept with correct normalized coords."""
        # Object at center of a 640×640 image → also center of this tile
        labels = ["2 0.5 0.5 0.4 0.3"]  # shipwreck centered
        result = transform_annotations(labels, 640, 640, centered_tile)
        assert len(result) == 1
        parts = result[0].split()
        assert int(parts[0]) == 2  # class_id preserved
        cx = float(parts[1])
        cy = float(parts[2])
        assert abs(cx - 0.5) < 1e-4
        assert abs(cy - 0.5) < 1e-4

    def test_annotation_outside_tile(self, offset_tile):
        """Annotation entirely outside tile → excluded."""
        # Object at top-left corner of full image (0,0) won't be in offset tile
        labels = ["1 0.05 0.05 0.08 0.08"]
        result = transform_annotations(labels, 960, 1280, offset_tile)
        assert len(result) == 0

    def test_empty_labels_returns_empty(self, centered_tile):
        """Empty label list → empty result (valid negative tile)."""
        result = transform_annotations([], 640, 640, centered_tile)
        assert result == []

    def test_blank_lines_skipped(self, centered_tile):
        """Blank lines in labels are skipped safely."""
        labels = ["", "  ", "2 0.5 0.5 0.4 0.3", ""]
        result = transform_annotations(labels, 640, 640, centered_tile)
        assert len(result) == 1

    def test_partial_annotation_above_threshold(self, centered_tile):
        """Annotation partially crossing tile boundary, area > 25% → included."""
        # Box spans x: [0.0, 0.6], y: [0.0, 0.6] in a 640×640 image
        # cx=0.3, cy=0.3, w=0.6, h=0.6
        labels = ["3 0.3 0.3 0.6 0.6"]
        # tile covers entire image → full box kept
        result = transform_annotations(labels, 640, 640, centered_tile, min_area_ratio=0.25)
        assert len(result) == 1

    def test_partial_annotation_below_threshold(self):
        """Annotation mostly in left tile, tiny sliver in right tile → excluded.

        Full image: 1280×640.
        Right tile covers x=[640, 1280], y=[0, 640].
        Object: x=[0, 672], y=[100, 200] — only 32px (of 672px width) is in right tile.
        Intersection area ratio = 32/672 ≈ 4.8% < 25% threshold → excluded.
        """
        right_tile = TileInfo(
            tile=np.zeros((640, 640), dtype=np.float32),
            row=0, col=1,
            x_offset=640, y_offset=0,
            tile_h=640, tile_w=640,
        )
        # Box: x1=0, x2=672, y1=100, y2=200 in absolute pixels (1280×640 image)
        # cx = (0+672)/(2*1280) = 0.2625
        # cy = (100+200)/(2*640) = 0.234375
        # w  = 672/1280 = 0.525
        # h  = 100/640  = 0.15625
        labels = ["1 0.2625 0.234375 0.525 0.15625"]
        result = transform_annotations(
            labels, 640, 1280, right_tile, min_area_ratio=0.25
        )
        assert len(result) == 0, (
            f"Expected annotation excluded (area ratio ~4.8% < 25%), "
            f"but got: {result}"
        )

    def test_multiple_annotations_correct_count(self, centered_tile):
        """Multiple annotations → all inside tile are included."""
        labels = [
            "2 0.2 0.2 0.1 0.1",  # shipwreck top-left area
            "4 0.8 0.8 0.1 0.1",  # mine bottom-right area
        ]
        result = transform_annotations(labels, 640, 640, centered_tile)
        assert len(result) == 2

    def test_class_id_preserved(self, centered_tile):
        """Class IDs must be preserved exactly."""
        labels = ["0 0.5 0.5 0.2 0.2", "4 0.3 0.7 0.1 0.1"]
        result = transform_annotations(labels, 640, 640, centered_tile)
        class_ids = [int(r.split()[0]) for r in result]
        assert 0 in class_ids
        assert 4 in class_ids

    def test_output_coords_in_unit_range(self, centered_tile):
        """All transformed coordinates must be in [0, 1]."""
        labels = ["2 0.5 0.5 0.8 0.6"]
        result = transform_annotations(labels, 640, 640, centered_tile)
        for line in result:
            parts = line.split()
            for val in parts[1:]:
                assert 0.0 <= float(val) <= 1.0, f"Coord out of range: {val}"

    def test_raises_on_invalid_image_dims(self, centered_tile):
        """image_h=0 must raise ValueError."""
        with pytest.raises(ValueError):
            transform_annotations(["2 0.5 0.5 0.2 0.2"], 0, 640, centered_tile)

    def test_raises_on_invalid_min_area_ratio(self, centered_tile):
        """min_area_ratio > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="min_area_ratio"):
            transform_annotations(["2 0.5 0.5 0.2 0.2"], 640, 640, centered_tile, min_area_ratio=1.5)

    def test_malformed_line_skipped(self, centered_tile):
        """Malformed annotation lines are skipped, valid ones are processed."""
        labels = ["not_valid data", "2 0.5 0.5 0.3 0.3"]
        result = transform_annotations(labels, 640, 640, centered_tile)
        assert len(result) == 1


# ===========================================================================
# 7. FULL PHASE 1 + PHASE 2 PIPELINE TESTS
# ===========================================================================

class TestFullPipeline:
    """Tests for ml.preprocessing.pipeline.preprocess_image_full()"""

    def test_returns_float32(self, sonar_gray_small):
        """Full pipeline must return float32."""
        result = preprocess_image_full(sonar_gray_small)
        assert result.dtype == np.float32

    def test_output_in_range(self, sonar_gray_small):
        """Full pipeline output must be in [0.0, 1.0]."""
        result = preprocess_image_full(sonar_gray_small)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_shape_preserved(self, sonar_gray_small):
        """Output shape must match input spatial dimensions."""
        result = preprocess_image_full(sonar_gray_small)
        assert result.shape == sonar_gray_small.shape

    def test_medium_image(self, sonar_gray_medium):
        """Full pipeline handles non-square SSS swath images."""
        result = preprocess_image_full(sonar_gray_medium)
        assert result.shape == sonar_gray_medium.shape
        assert result.dtype == np.float32

    def test_bgr_input_handled(self, sonar_bgr):
        """Full pipeline auto-converts BGR input to grayscale."""
        result = preprocess_image_full(sonar_bgr)
        assert result.ndim == 2
        assert result.dtype == np.float32

    def test_deterministic(self, sonar_gray_small):
        """Same input always produces same output."""
        r1 = preprocess_image_full(sonar_gray_small)
        r2 = preprocess_image_full(sonar_gray_small)
        np.testing.assert_array_equal(r1, r2)

    def test_differs_from_phase1_output(self, sonar_gray_medium):
        """Phase 2 output should differ from Phase 1 output (enhancement applied)."""
        p1 = preprocess_image(sonar_gray_medium)
        p2 = preprocess_image_full(sonar_gray_medium)
        assert not np.array_equal(p1, p2), (
            "Phase 2 pipeline should produce different output than Phase 1 "
            "due to the unsharp masking enhancement step."
        )

    def test_zero_enhancement_matches_phase1_output(self, sonar_gray_medium):
        """With enhance_amount=0.0, full pipeline should match Phase 1 output."""
        p1 = preprocess_image(sonar_gray_medium)
        p2 = preprocess_image_full(sonar_gray_medium, enhance_amount=0.0)
        np.testing.assert_array_almost_equal(p1, p2, decimal=5)

    def test_from_file_path(self, sonar_gray_small, tmp_path):
        """Full pipeline accepts file path input."""
        import cv2
        temp_img = tmp_path / "sonar_full.png"
        cv2.imwrite(str(temp_img), sonar_gray_small)
        result = preprocess_image_full(temp_img)
        assert result.dtype == np.float32

    def test_saves_output_file(self, sonar_gray_small, tmp_path):
        """Full pipeline with output_path saves a PNG file."""
        out = tmp_path / "enhanced" / "sonar_p2.png"
        preprocess_image_full(sonar_gray_small, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_does_not_modify_input(self, sonar_gray_small):
        """Input array must not be modified in-place."""
        original = sonar_gray_small.copy()
        preprocess_image_full(sonar_gray_small)
        np.testing.assert_array_equal(sonar_gray_small, original)

    def test_raises_on_none(self):
        """None input must raise TypeError."""
        with pytest.raises(TypeError):
            preprocess_image_full(None)  # type: ignore[arg-type]

    def test_raises_on_missing_file(self, tmp_path):
        """Non-existent file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocess_image_full(tmp_path / "no_such_file.png")

    def test_pipeline_then_tile(self, sonar_gray_medium):
        """End-to-end: full pipeline → tile_image → all tiles valid."""
        preprocessed = preprocess_image_full(sonar_gray_medium)
        tiles = tile_image(preprocessed, tile_h=128, tile_w=128, overlap=16)
        assert len(tiles) > 0
        for t in tiles:
            assert t.tile.dtype == np.float32
            assert t.tile.shape == (128, 128)
            assert float(t.tile.min()) >= 0.0
            assert float(t.tile.max()) <= 1.0

