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
from ml.preprocessing.pipeline import preprocess_image


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
