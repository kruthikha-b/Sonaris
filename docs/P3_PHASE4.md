# P3 Phase 4 — Final P3 Integration, Testing, and Validation

**SONARIS Project** | P3 — Sonar Processing + Geolocation  
**Branch**: `feature/p3-sonar-geo`  
**Phase**: 4 — Final Integration, Testing & Validation  
**Depends on**: [P3_PHASE1.md](./P3_PHASE1.md), [P3_PHASE2.md](./P3_PHASE2.md), [P3_PHASE3.md](./P3_PHASE3.md)

---

## 1. Overview & Architecture

Phase 4 validates the end-to-end integration of all P3 components with the upstream P2 dataset artifacts and downstream P1 detection / P4 backend contracts.

### Complete Data Flow

```
P2 External Dataset (DRISHTI-SSS on Google Drive)
        │
        ▼ (read-only, resolved via SONARIS_DATA_DIR)
P3 Sonar Preprocessing
   ├── Denoising: Non-Local Means (NLM) filter
   ├── Normalization: CLAHE + min-max scaling to float32 [0.0, 1.0]
   └── Enhancement: Unsharp Masking for acoustic boundary sharpening
        │
        ▼
P3 Image Tiling
   ├── Decomposes into 640×640 overlapping tiles (overlap=64px)
   ├── Reflect-edge padding (BORDER_REFLECT_101)
   └── Annotation transformation (maps YOLO normalized coordinates to tile)
        │
        ▼
P1 AI / YOLO Model Interface
   ├── Consumes 640×640 tiles (float32 or uint8 BGR)
   └── Produces bounding-box detections (cx_norm, cy_norm, w_norm, h_norm)
        │
        ▼
P3 Geolocation Foundation
   ├── Reads SonarMetadata (range_m, optional GPS + heading)
   ├── Calculates cross-track distance from nadir & bearing (±90°)
   ├── Transforms to WGS84 coordinates when GPS is present
   └── Estimates 1-sigma positional uncertainty in quadrature
        │
        ▼
P4/P5 Structured Result Contract
   └── JSON-serializable dictionary with standardized schema
```

---

## 2. External Dataset Workflow & Configuration

The team stores the primary DRISHTI-SSS dataset (~5,198 images, YOLO annotations) externally on Google Drive. To support diverse local developer setups without hardcoding paths or committing large files to GitHub:

### Golden Rules
1. **Never commit the dataset to GitHub**: `.gitignore` strictly protects `datasets/`, `data/`, weights (`*.pt`, `*.onnx`), and output caches.
2. **Never hardcode personal paths**: No personal drive paths or sharing URLs are baked into the code.
3. **Read-only access**: Preprocessing never mutates source images or annotations; all generated tiles write to a designated output directory.

### Configuration via Environment Variables

The dataset path is resolved dynamically via `ml/preprocessing/config.py`:

| Environment Variable | Description | Default Fallback |
|---|---|---|
| `SONARIS_DATA_DIR` | Absolute path to the local or synced Google Drive dataset | `datasets/` in workspace, then `C:\drishti_clean` |
| `SONARIS_OUTPUT_DIR` | Path for generated preprocessed images and tiles | `datasets/preprocessed/` |

#### Setup Instructions for Developers

- **PowerShell (Windows)**:
  ```powershell
  $env:SONARIS_DATA_DIR = "D:\GoogleDrive\Sonaris\drishti_clean"
  $env:SONARIS_OUTPUT_DIR = "c:\Drives\New folder (2)\SIH\Sonaris\datasets\preprocessed"
  ```
- **Bash / Linux / macOS**:
  ```bash
  export SONARIS_DATA_DIR="/mnt/gdrive/Sonaris/drishti_clean"
  export SONARIS_OUTPUT_DIR="./datasets/preprocessed"
  ```
- **Python Code**:
  ```python
  from ml.preprocessing import get_dataset_dir, is_dataset_available

  if is_dataset_available():
      data_path = get_dataset_dir()
      print(f"Dataset active at: {data_path}")
  ```

---

## 3. P1 & P2 Compatibility Verification

### P1 (AI/YOLO Detection) Compatibility
- **Tile Dimensions**: Exactly `640×640`, matching standard Ultralytics YOLOv8 / YOLOv11 input specifications.
- **Data Types**: `tile_image()` outputs `float32 [0.0, 1.0]`. Helper conversion `(tile * 255).astype(np.uint8)` replicated to 3-channel BGR allows immediate inference or training on existing YOLO backbones.
- **Coordinate System**: Bounding box coordinates use standard YOLO normalized values `[0.0, 1.0]`. The P3 `Detection` model ingests `(class_id, cx_norm, cy_norm, w_norm, h_norm, confidence)` without schema transformation.

### P2 (Dataset & Data Engineering) Compatibility
- **DRISHTI-SSS Classes**: Strictly aligned with P2 class indices:
  - `0`: `crab_pot`
  - `1`: `submarine_pipeline`
  - `2`: `shipwreck`
  - `3`: `ghost_net`
  - `4`: `mine_cylinder`
- **Immutability Guarantee**: All P3 preprocessing functions operate on copies or return new arrays. Original source inputs are never modified.

---

## 4. Final P3 Output Contract (for P4 Backend & P5 Frontend)

Each geolocated detection produces a `GeoLocation` instance, serializable to a standardized dictionary via `.to_dict()`:

```json
{
  "image_id": "sonar_scan_042_tile_000_001",
  "class_name": "shipwreck",
  "confidence": 0.94,
  "range_m": 37.5,
  "bearing_deg": 90.0,
  "local_x_m": 37.5,
  "local_y_m": 0.0,
  "coordinate_system": "WGS84",
  "status": "estimated",
  "uncertainty_m": 3.15,
  "latitude": 15.498562,
  "longitude": 73.827800
}
```

### Schema Specification
- `image_id` *(str)*: Source image / tile identifier.
- `class_name` *(str)*: Object classification label.
- `confidence` *(float)*: Detection confidence `[0.0, 1.0]`.
- `range_m` *(float)*: Slant range from nadir (metres).
- `bearing_deg` *(float)*: Relative bearing from sensor heading (`+90.0°` = Starboard, `-90.0°` = Port).
- `local_x_m` *(float)*: Signed cross-track distance in metres (`+` = Starboard, `-` = Port).
- `local_y_m` *(float)*: Along-track displacement (`0.0` when timestamp metadata is unavailable).
- `coordinate_system` *(str)*: `"local_sonar"` (relative) or `"WGS84"` (geographic).
- `status` *(str)*: `"relative"` (no GPS) or `"estimated"` (GPS & heading available).
- `uncertainty_m` *(float | null)*: 1-sigma horizontal position uncertainty radius in metres (`null` in relative mode).
- `latitude` *(float | null)*: WGS84 decimal degrees (`null` in relative mode).
- `longitude` *(float | null)*: WGS84 decimal degrees (`null` in relative mode).

---

## 5. Test Suite Verification & Performance

### Automated Test Summary

```bash
pytest tests/ -v
```

| Test Suite | Focus Area | Test Count | Result |
|---|---|---|---|
| `tests/test_preprocessing.py` | Phase 1 & 2: Denoise, CLAHE, Unsharp, Tiling, Annotations | 91 | ✅ PASS |
| `tests/test_geolocation.py` | Phase 3: Sonar geometry, WGS84 transform, uncertainty, locator | 33 | ✅ PASS |
| `tests/test_integration_e2e.py` | Phase 4: E2E pipeline, P1/P2 compatibility, edge cases, benchmarks | 20 | ✅ PASS |
| **Total Test Suite** | **Full P3 System Coverage** | **144** | **✅ 100% PASS** |

### Performance Benchmarks (Empirical Latency)

| Stage | Input Size | Average Latency | Throughput |
|---|---|---|---|
| Full Preprocessing (NLM + CLAHE + Unsharp) | 640×640 | 236.6 ms | ~4.2 fps |
| Full Preprocessing | 1280×640 | 439.7 ms | ~2.3 fps |
| Full Preprocessing (High-Res SSS) | 1920×1080 | 939.9 ms | ~1.1 fps |
| Tiling (to 640×640 patches) | 1280×640 (3 tiles) | 7.3 ms | ~137 fps |
| Geolocation Engine | 1 detection | 0.012 ms | ~83,000 det/s |
| Geolocation Engine | 100 detections | 0.583 ms | ~170,000 det/s |

---

## 6. Edge-Case Validation

The system was tested against rigorous edge conditions:
- **Empty / Uniform Images (Black & White)**: Processed safely without zero-division exceptions.
- **Sub-Tile Images (< 640×640)**: Padded to `640×640` with mirror reflection (`cv2.BORDER_REFLECT_101`).
- **Arbitrary Image Dimensions (e.g. 1025×513)**: Seamlessly divided into tiled grid with boundary padding.
- **Missing / Partial GPS**: If GPS latitude/longitude is provided without heading, the engine gracefully falls back to `"local_sonar"` relative coordinates rather than guessing orientation.
- **Empty Detections Batch**: Gracefully returns `[]` without error.
