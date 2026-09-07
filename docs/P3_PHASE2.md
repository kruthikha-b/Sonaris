# P3 Phase 2 — Sonar Enhancement + Tiling

**SONARIS Project** | P3 — Sonar Processing + Geolocation  
**Branch**: `feature/p3-sonar-geo`  
**Phase**: 2 — Enhancement + Tiling  
**Depends on**: [P3_PHASE1.md](./P3_PHASE1.md)

---

## Overview

Phase 2 extends the Phase 1 preprocessing foundation with:
- **Contrast Enhancement** — Unsharp masking to sharpen acoustic target boundaries
- **Image Tiling** — Decompose large SSS images into 640×640 YOLO-compatible tiles
- **Annotation Transformation** — Map full-image YOLO labels to tile-local coordinates

Phase 1 functions (`preprocess_image`, `denoise`, `normalize_*`) are **unchanged**.

---

## Architecture Diagram

```
P2 Dataset (C:\drishti_clean)
    │
    ▼
P3 Phase 1
    │  _load_as_grayscale()          load .jpg/.png → uint8 (H,W)
    │  denoise()                     NLM speckle reduction → uint8 (H,W)
    │  normalize_uint8()             CLAHE contrast equalization → uint8 (H,W)
    │
    ▼
P3 Phase 2
    │  enhance()                     Unsharp masking → uint8 (H,W)
    │  min-max scale                 → float32 (H,W) [0.0, 1.0]
    │
    ▼
preprocess_image_full()             Single-image P1+P2 pipeline entry point
    │
    ▼
tile_image()                        Decompose → list[TileInfo] (640×640 tiles)
    │
    ▼
transform_annotations()             YOLO full-image coords → tile-local coords
    │
    ▼
P1 YOLO Detection Pipeline
```

---

## Files Added / Modified

| File | Status | Description |
|---|---|---|
| `ml/preprocessing/enhance.py` | **NEW** | Unsharp masking enhancement |
| `ml/preprocessing/tiler.py` | **NEW** | Image tiling + TileInfo |
| `ml/preprocessing/tile_annotations.py` | **NEW** | YOLO annotation coordinate transform |
| `ml/preprocessing/pipeline.py` | **MODIFIED** | Added `preprocess_image_full()` (appended) |
| `ml/preprocessing/__init__.py` | **MODIFIED** | Added Phase 2 exports |
| `tests/test_preprocessing.py` | **MODIFIED** | Added 4 Phase 2 test classes (appended) |
| `docs/P3_PHASE2.md` | **NEW** | This document |

**Unchanged (P1/P2 untouched):**
`denoise.py`, `normalize.py`, `ml/data/**`, `ml/detection/**`

---

## Enhancement — Unsharp Masking

**Why unsharp masking, not another CLAHE?**

Phase 1 already applies CLAHE. Re-applying it would double-equalize and risk
creating artificial intensity gradients. Unsharp masking is complementary:

| Phase | Operation | Purpose |
|---|---|---|
| Phase 1 | CLAHE | Equalize local contrast across SSS swath gradient |
| Phase 2 | Unsharp mask | Sharpen acoustic shadow edges and target boundaries |

**Formula:**
```
sharpened = clip(original + amount × (original − GaussianBlur(original, σ)), 0, 255)
```

**Default parameters:**
- `amount = 1.0` — standard sharpening strength (range 0.5–2.0)
- `sigma = 1.0` — Gaussian kernel sigma (controls feature scale sharpened)

**Effect on SSS images:**
- Phase 2 std dev: **0.2327** vs Phase 1 std dev: **0.1838** (validated)
- Higher std dev confirms that target/background contrast was increased
- Objects (shipwrecks, pipelines, mine cylinders) → crisper boundaries
- Uniform seabed → not affected (uniform regions blur to same value → no sharpening)

---

## Tiling

### Tile Size: 640 × 640 pixels

YOLOv8/v11 native input size. No P1-side resize needed when using tiles at this size.

### Overlap: 64 pixels (default)

10% of tile size. Ensures objects near boundaries appear fully in at least one tile.

### Edge Handling: Reflect Padding

`cv2.BORDER_REFLECT_101` — mirrors image content at boundaries. All tiles are exactly
640×640, with no dark zero-padding borders that could confuse the detector.

### Naming Convention

```
{original_stem}_tile_{row:03d}_{col:03d}.png
```

### Output (completely separate from P2 dataset)

```
{output_dir}/
    {split}/
        images/   {stem}_tile_000_000.png, ...
        labels/   {stem}_tile_000_000.txt, ...
```

### Tile Count (example: 640×1280 image, overlap=64)

```
stride_w = 640 - 64 = 576
cols = ceil((1280 - 640) / 576) + 1 = 3 tiles per row
rows = 1 (640px height = exactly 1 tile)
Total = 3 tiles ← validated in tests
```

---

## Annotation Transformation

YOLO annotations (normalized to full image) are converted to tile-local YOLO format.

**Steps:**
1. YOLO normalized `(cx, cy, w, h)` → absolute pixel `(x1, y1, x2, y2)`
2. Intersect with tile region
3. Discard if `clipped_area / original_area < min_area_ratio` (default 25%)
4. Convert clipped box → YOLO normalized relative to tile dimensions

**Partial object rule:** Objects appearing at tile boundaries are included if
≥25% of their area falls within the tile. This threshold is configurable.

**Multiple tile appearances:** An object overlapping adjacent tiles intentionally
appears in both tiles due to overlap. This is correct and increases detection robustness.

**Empty tiles:** Get an empty `.txt` label file — valid YOLO negative sample.

---

## Public API

```python
from ml.preprocessing import (
    preprocess_image,           # Phase 1 (unchanged)
    preprocess_image_full,      # Phase 1 + Phase 2
    tile_image,                 # Tiling
    tile_filename,              # Tile naming
    transform_annotations,      # YOLO coordinate transform
    load_yolo_labels,           # Read .txt label file
)

# === Full Phase 1+2 pipeline ===
result = preprocess_image_full("path/to/sonar.jpg")
# → float32 (H, W), values [0.0, 1.0]

# === Tile the preprocessed image ===
tiles = tile_image(result, tile_h=640, tile_w=640, overlap=64)
for tile_info in tiles:
    fname = tile_filename("survey_001", tile_info.row, tile_info.col)
    # tile_info.tile  → float32 (640, 640)
    # tile_info.x_offset, tile_info.y_offset → for annotation transform

# === Transform annotations for each tile ===
from ml.preprocessing import load_yolo_labels
labels = load_yolo_labels("path/to/survey_001.txt")
for tile_info in tiles:
    tile_labels = transform_annotations(
        labels,
        image_h=result.shape[0],
        image_w=result.shape[1],
        tile_info=tile_info,
        min_area_ratio=0.25,
    )
    # tile_labels → list of "class_id cx cy w h" strings (tile-local)
```

---

## Running Tests

```bash
cd "c:\Drives\New folder (2)\SIH\Sonaris"

# Run all tests (Phase 1 + Phase 2)
python -m pytest tests/test_preprocessing.py -v

# Run only Phase 2 tests
python -m pytest tests/test_preprocessing.py::TestEnhance -v
python -m pytest tests/test_preprocessing.py::TestTiler -v
python -m pytest tests/test_preprocessing.py::TestTileAnnotations -v
python -m pytest tests/test_preprocessing.py::TestFullPipeline -v

# Smoke test
python -c "from ml.preprocessing import preprocess_image_full, tile_image; print('Phase 2 OK')"
```

---

## Test Results

```
91 passed in 2.39s ✅

Phase 1 tests: 37 passed (unchanged)
Phase 2 tests: 54 new tests passed
  TestEnhance         : 14 tests
  TestTiler           : 15 tests
  TestTileAnnotations : 12 tests
  TestFullPipeline    : 14 tests
```

---

## Visual Validation Results

Tested on synthetic 640×1280 SSS image with simulated objects:

| Stage | Min | Max | Std Dev |
|---|---|---|---|
| Original (noisy SSS) | 0 | 255 | 61.45 |
| Phase 1 (denoise+CLAHE) | 0.0000 | 1.0000 | 0.1838 |
| Phase 2 (+unsharp mask) | 0.0000 | 1.0000 | 0.2327 |

- Phase 2 std dev increased from 0.1838 → 0.2327 ✅ (enhancement increased target/background contrast)
- All 3 tiles from 640×1280 image are exactly 640×640 ✅
- Simulated targets (shipwreck, pipeline, mine) remain clearly visible ✅
- No artificial structures introduced ✅

---

## P1 Integration Notes

- P1's `ml/detection/` files are **empty** and were not modified
- Tile output goes to a **separate directory** — `C:\drishti_clean` is never modified
- When P1 is ready to train on tiles, point its dataset YAML at the tile output directory
- P3's float32 [0,1] → needs final uint8 conversion before YOLO training (standard `cv2.imwrite` handles this since tiles are saved as PNG via `_save_preprocessed`)
- `TileInfo.x_offset`, `TileInfo.y_offset` are available for any further geolocation calculations (P3 Phase 3+)

---

## Assumptions

1. SSS images are grayscale (may be stored as 3-channel BGR)
2. DRISHTI-SSS images are larger than 640×640 (typical SSS export ~800–1500px wide)
3. P1 will use Ultralytics YOLO — 640×640 tile size matches native input
4. Real dataset not accessible on this machine — synthetic images used for all tests

## Limitations (Phase 2)

- No batch processing CLI (can be added in Phase 3)
- No geolocation metadata on tiles (Phase 3)
- Annotation transformation tested on synthetic labels only (real dataset not available)
- Tiles stored separately; P1 dataset YAML integration not yet done
