# P3 Phase 1 — Sonar Preprocessing Foundation

**SONARIS Project** | P3 — Sonar Processing + Geolocation  
**Branch**: `feature/p3-sonar-geo`  
**Phase**: 1 — Preprocessing Foundation

---

## Overview

This document describes the P3 Phase 1 implementation of the sonar image
preprocessing pipeline for the SONARIS project.

The preprocessing pipeline transforms raw sonar images from the DRISHTI-SSS
dataset into clean, normalized arrays suitable for downstream AI detection
(P1 YOLO pipeline) and geolocation (P3 Phase 2+).

---

## P2 Dataset — What We Work With

P2 (Data Engineering) produced the clean working copy of the DRISHTI-SSS dataset.

| Property | Value |
|---|---|
| Original dataset | `C:\drishti-sss` (read-only, never modified) |
| Clean working copy | `C:\drishti_clean` |
| Directory structure | `{train,val,test}/{images,labels}/` |
| Image formats | `.jpg`, `.jpeg`, `.png` (mixed) |
| Image colour | Grayscale (may be stored as 3-channel identical BGR) |
| Annotation format | YOLO normalized: `class_id x_center y_center width height` |
| Classes | 5: crab_pot(0), submarine_pipeline(1), shipwreck(2), ghost_net(3), mine_cylinder(4) |
| Total images | 5,198 (Train: 3870 / Val: 628 / Test: 700) |

**What P2 did:**
- Removed 7 duplicate images
- Validated all annotations (0 invalid)
- Confirmed image-label pairing (0 orphans)
- Standardized YOLO label formatting

**What P2 did NOT do** (handled by P3):
- Pixel-level normalization
- Denoising / speckle reduction
- Image enhancement

---

## P1 Detection Pipeline — Integration Context

P1 (AI/ML) is implementing a YOLO-based detection pipeline in `ml/detection/`.  
At the time of P3 Phase 1 implementation, `train.py`, `predict.py`, and `evaluate.py` are empty placeholder files.

**P1 integration assumption:**  
Ultralytics YOLOv8/v11 internally handles image resizing to 640×640. P3 does
not resize images. P3 provides preprocessed images; P1 can load them from a
`preprocessed/` output directory (to be configured in Phase 2).

P3 Phase 1 output (float32 [0,1]) will be converted to uint8 BGR in Phase 2
for direct YOLO compatibility.

---

## Implementation — Files Created

```
ml/preprocessing/__init__.py    Public API: exposes preprocess_image()
ml/preprocessing/denoise.py     Denoising: Non-Local Means
ml/preprocessing/normalize.py   Normalization: CLAHE + min-max
ml/preprocessing/pipeline.py    Full pipeline: load → denoise → normalize
tests/test_preprocessing.py     Test suite (33 tests)
docs/P3_PHASE1.md               This document
```

**Files NOT modified:**
- `ml/data/**` (P2 — untouched)
- `ml/detection/**` (P1 — untouched)
- All `.gitkeep` placeholders
- `README.md`, `datasets/`, `backend/`, `frontend/`, `geolocation/`, `models/`, `reports/`

---

## Pipeline Flow

```
Input (file path or numpy array)
    │
    ▼
_load_as_grayscale()
    │  - Path input: cv2.imread(..., IMREAD_GRAYSCALE)
    │  - Array input: BGR → grayscale conversion if needed
    │  - Output: uint8 (H, W)
    │
    ▼
denoise()           [ml/preprocessing/denoise.py]
    │  - Method: cv2.fastNlMeansDenoising (Non-Local Means)
    │  - h=10.0, template_win=7, search_win=21
    │  - Output: uint8 (H, W)
    │
    ▼
normalize_float()   [ml/preprocessing/normalize.py]
    │  - Step 1: CLAHE (clip_limit=2.0, tile_grid=(8,8))
    │  - Step 2: Min-max scale to float32 [0.0, 1.0]
    │  - Output: float32 (H, W)
    │
    ▼
preprocess_image() returns float32 (H, W) array
    │
    └─ Optional: save as uint8 PNG to output_path
```

---

## Methods

### Denoising — Non-Local Means (NLM)

**Function:** `denoise(image, h=10.0, template_win_size=7, search_win_size=21)`  
**OpenCV call:** `cv2.fastNlMeansDenoising()`

NLM averages pixels weighted by patch similarity across the whole image.
Unlike Gaussian blur, it preserves edge-like sonar structures (shipwreck
outlines, pipeline shadows) while reducing speckle noise from coherent
imaging artifacts.

| Parameter | Default | Effect |
|---|---|---|
| `h` | 10.0 | Filter strength. Higher = more smoothing. Range 5–15 for SSS. |
| `template_win_size` | 7 | Patch size for similarity comparison (pixels) |
| `search_win_size` | 21 | Search window size (pixels) |

### Normalization — CLAHE + Min-Max

**Function:** `normalize_float(image, clip_limit=2.0, tile_grid_size=(8,8))`

**Step 1: CLAHE** (`cv2.createCLAHE`)  
Contrast Limited Adaptive Histogram Equalization operates on local tiles,
enhancing contrast in shadow zones and highlight areas without over-amplifying
noise. Essential for SSS images with strong near-to-far intensity gradients.

**Step 2: Min-Max scaling**  
Maps CLAHE output to float32 [0.0, 1.0] with epsilon guard (1e-6) for
uniform-intensity images.

| Parameter | Default | Effect |
|---|---|---|
| `clip_limit` | 2.0 | Max contrast amplification per tile. Higher = stronger. |
| `tile_grid_size` | (8, 8) | Number of tiles across the image |

---

## Public API

```python
from ml.preprocessing import preprocess_image

# From file path (P2 dataset image)
result = preprocess_image("C:/drishti_clean/train/images/example.jpg")
# result: float32 numpy array, shape (H, W), values in [0.0, 1.0]

# From numpy array
import numpy as np
img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
result = preprocess_image(img)

# With output path (saves PNG, does NOT modify dataset)
result = preprocess_image(
    "C:/drishti_clean/train/images/example.jpg",
    output_path="C:/drishti_preprocessed/train/images/example.png"
)

# Custom parameters
result = preprocess_image(
    img,
    denoise_h=8.0,           # lighter denoising
    clahe_clip_limit=3.0,    # stronger contrast enhancement
    clahe_tile_grid=(4, 4),  # larger tiles
)
```

---

## Running the Pipeline

```bash
# Navigate to project root
cd "c:\Drives\New folder (2)\SIH\Sonaris"

# Quick import test
python -c "from ml.preprocessing import preprocess_image; print('OK')"

# Run all tests
python -m pytest tests/test_preprocessing.py -v

# Run tests with coverage (if pytest-cov installed)
python -m pytest tests/test_preprocessing.py -v --tb=short
```

---

## Testing

The test suite (`tests/test_preprocessing.py`) covers:

| Class | Tests | Description |
|---|---|---|
| `TestDenoise` | 11 | Shape, dtype, value range, change detection, error handling, configurable params |
| `TestNormalize` | 10 | CLAHE, float32 output, [0,1] range, edge cases (uniform black/white), uint8 |
| `TestPipeline` | 12 | End-to-end, file path, save file, determinism, in-place mutation, error cases |
| **Total** | **33** | All synthetic images, no real dataset access |

---

## Assumptions

1. Sonar images are effectively grayscale (single meaningful channel)
2. P2's `C:\drishti_clean` may not be accessible on all development machines — tests use synthetic numpy images
3. P1 (YOLO) handles final resizing internally; P3 does not resize
4. `albumentations` is not required or used by P3
5. The DRISHTI-SSS dataset has not been pixel-normalized by P2

## Limitations (Phase 1)

- No batch processing (Phase 2)
- No tiling / patch extraction (Phase 2)
- No geolocation (separate P3 module, future phase)
- Output is float32 [0,1]; P1 YOLO expects uint8 BGR — conversion will be added in Phase 2 integration
- CLAHE may slightly alter relative intensities across images (acceptable trade-off for SSS enhancement)

## Integration Concerns

- P1 should load preprocessed images from a separate directory (e.g., `C:\drishti_preprocessed\`)
- P3 never modifies `C:\drishti_clean` or `C:\drishti-sss`
- When P1 implements training, a Phase 2 batch script will run the pipeline over the full dataset
- The `preprocess_image()` API is stable and can be imported directly by P1 when needed
