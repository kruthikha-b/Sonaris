# P1: Dataset and YOLO Verification Report

**Project**: Sonaris (SIH 2026, Problem Statement 26057)  
**Module**: P1 AI/ML Detection Module  
**Phase**: Phase 1 — Dataset and YOLO Verification  
**Audit Date**: September 7, 2026  
**Auditor**: P1 AI/ML Detection Team  
**Dataset Path**: `~/Library/CloudStorage/GoogleDrive-kruthikha2207@gmail.com/My Drive/drishti_clean`  
**Symlink Target**: `/Users/kruthikhareddy/Library/CloudStorage/GoogleDrive-kruthikha2207@gmail.com/.shortcut-targets-by-id/1QvzLWu4t13Ng2LOOpI71FtnbJa3YHnEp/drishti_clean`  

---

## Executive Summary

As part of **Phase 1: Dataset and YOLO Verification**, the P1 team conducted a non-destructive, empirical audit of the `drishti_clean` dataset and its configuration `drishti.yaml`. All image headers, dimensions, channels, label files, bounding boxes, coordinate bounds, and class distributions were calculated directly from the filesystem.

### Key Audit Highlights:
- **Image Integrity**: **100% passing** (0 corrupted or unreadable images out of 4,348 images).
- **YOLO Annotation Format**: **100% passing** (0 invalid bounding boxes, 0 out-of-bounds coordinates, 0 non-numeric values, 0 negative dimensions out of 5,193 boxes).
- **Class IDs**: **100% valid** (all class IDs $\in \{1, 2, 3, 4\}$, with 0 instances of class 0 `crab_pot` as intentionally omitted upstream).
- **Direct YOLO Training Compatibility**: **FAIL (NOT COMPATIBLE AS-IS)** due to three blocking issues:
  1. **Invalid Root Path in `drishti.yaml`**: Hardcoded Windows path `D:/Sonar-Drishti/ml/data/splits`.
  2. **Empty Validation Split**: The `val` directory is completely empty (0 images, 0 labels).
  3. **217 Orphaned Label Files**: 122 orphaned label files in `train` (missing synthetic PNGs) and 95 orphaned label files in `test` (missing shipwreck JPGs).

Per the established team boundary:
- **P2 owns dataset creation, cleaning, conversion, and augmentation.**
- **P1 must NOT modify P2's dataset.**
- P1 has documented all findings below to facilitate P2's resolution before Phase 2 training begins.

---

## 1. `drishti.yaml` Contents and Paths

The configuration file was inspected at `drishti_clean/drishti.yaml`:

```yaml
path: D:/Sonar-Drishti/ml/data/splits
train: train/images
val: val/images
test: test/images

task: detect

nc: 5
names:
  0: crab_pot
  1: submarine_pipeline
  2: shipwreck
  3: ghost_net
  4: mine_cylinder
```

### Analysis & Issues:
1. **Hardcoded Windows Path (`path`)**:  
   The parameter `path: D:/Sonar-Drishti/ml/data/splits` points to an external Windows workstation drive. When loaded in macOS/POSIX with Ultralytics YOLO, it attempts to resolve `/Users/kruthikhareddy/Desktop/datasets/D:/Sonar-Drishti/ml/data/splits/val/images` and immediately raises:
   ```text
   FileNotFoundError: Dataset 'drishti.yaml' images not found
   ```
2. **Task and Class Specifications**:  
   `task: detect`, `nc: 5`, and the 5 class names are syntactically valid for Ultralytics YOLOv8/YOLO11.

---

## 2. Train / Val / Test Directory Structure

```text
drishti_clean/
├── drishti.yaml
├── README.md
├── .gitattributes
├── .cache/
│   └── huggingface/download/ ...
├── train/
│   ├── images/         [3,748 image files]
│   ├── labels/         [3,870 label files]
│   └── labels.cache    [Pre-computed cache from upstream]
├── test/
│   ├── images/         [600 image files]
│   └── labels/         [695 label files]
└── val/
    └── [EMPTY DIRECTORY — 0 files, 0 subdirectories]
```

### Observations:
- `train` and `test` adhere to standard YOLO structure (`images/` and `labels/`).
- `val` exists as a directory but contains **no files or subdirectories**.

---

## 3. Number of Images in Each Split

Every image was inspected and counted by file format:

| Split | JPEG (`.jpg`) | PNG (`.png`) | Total Images |
|---|---:|---:|---:|
| **train** | 2,620 | 1,128 | 3,748 |
| **val** | 0 | 0 | **0** |
| **test** | 480 | 120 | 600 |
| **Total** | **3,100** | **1,248** | **4,348** |

*Note: Upstream HuggingFace README listed 3,875 train, 630 val, 700 test (total 5,205). The local clean dataset has 3,748 train, 0 val, 600 test (total 4,348).*

---

## 4. Number of Label Files

| Split | Total Labels (`.txt`) | Background Labels (0 objects) | Object Labels ($\ge 1$ object) |
|---|---:|---:|---:|
| **train** | 3,870 | 495 | 3,375 |
| **val** | 0 | 0 | 0 |
| **test** | 695 | 70 | 625 |
| **Total** | **4,565** | **565** | **4,000** |

All background tiles (`bg_...`) contain empty 0-byte `.txt` files, which correctly signal negative background tiles to YOLO without errors.

---

## 5. Images Without Labels

| Split | Total Images | Images Lacking Labels | % Missing |
|---|---:|---:|---:|
| **train** | 3,748 | **0** | 0.00% |
| **val** | 0 | **0** | N/A |
| **test** | 600 | **0** | 0.00% |
| **Total** | **4,348** | **0** | **0.00%** |

Every existing image has an associated label file with identical stem name.

---

## 6. Labels Without Images (Orphaned Labels)

| Split | Total Labels | Labels Lacking Images | Category of Missing Images |
|---|---:|---:|---|
| **train** | 3,870 | **122** | Synthetic PNG images (`synth_mine_cylinder_*`, `synth_ghost_net_*`) |
| **val** | 0 | **0** | N/A |
| **test** | 695 | **95** | Real shipwreck JPG images (`wreckA_*`, `wreckR_*`) |
| **Total** | **4,565** | **217** | 122 train + 95 test |

### Sample Orphaned Files:
- **Train split**: `synth_mine_cylinder_00200.txt`, `synth_mine_cylinder_00210.txt`, `synth_ghost_net_00139.txt`, `synth_mine_cylinder_00158.txt`
- **Test split**: `wreckR_ship-262_png.rf.7f408eca479e6e616f169e7a77e0d00d.txt`, `wreckA_WP_Thew_09_y1600_x320.txt`, `wreckR_ship-237_png.rf.b659d06886c50e3e6dba3b0d719b6ecc.txt`

These 217 orphaned labels contain 270 annotated bounding boxes that currently have no corresponding image in the dataset.

---

## 7. Corrupted / Unreadable Images

All 4,348 image files across `train` and `test` splits were verified using PIL `im.verify()` and read for format validation:

| Split | Verified Images | Corrupted / Unreadable | Pass Rate |
|---|---:|---:|---:|
| **train** | 3,748 | 0 | 100.0% |
| **val** | 0 | 0 | N/A |
| **test** | 600 | 0 | 100.0% |
| **Total** | **4,348** | **0** | **100.0%** |

Zero corrupted, unreadable, or truncated images were detected.

---

## 8. YOLO Annotation Syntax Verification

All 5,193 bounding boxes across 4,565 label files were programmatically inspected:

| Check | Requirement | Failures Found |
|---|---|---:|
| Token count | Exactly 5 tokens per line | **0** |
| Class ID type | Integer parseable | **0** |
| Coordinates type | Float parseable | **0** |
| Coordinate range | $0.0 \le x_c, y_c, w, h \le 1.0$ | **0** |
| Non-positive dimensions | $w > 0$ and $h > 0$ | **0** |
| Severe boundary overflow | $> 5\%$ outside image margins | **0** |
| **Total Invalid Annotations** | | **0** |

All annotations conform strictly to the standard YOLO bounding-box format:
`class_id x_center y_center width height`

---

## 9. Class IDs Verification

- Expected class IDs from `drishti.yaml`: $\{0, 1, 2, 3, 4\}$
- Range check:
  - Minimum class ID encountered: `1`
  - Maximum class ID encountered: `4`
  - Negative class IDs: **0**
  - Class IDs $\ge 5$: **0**
  - Unrecognized class IDs: **0**

---

## 10. Class Names and Class Distribution

### Class Definitions:
- `0`: `crab_pot` (Excluded upstream due to HuggingFace access gating; 0 instances present)
- `1`: `submarine_pipeline`
- `2`: `shipwreck`
- `3`: `ghost_net`
- `4`: `mine_cylinder`

### A. Distribution in Paired Images (Active Dataset: 4,923 Total Bounding Boxes)
These are the bounding boxes where both the image and the label exist:

| ID | Class Name | Train Count | Val Count | Test Count | Total Instances | % of Dataset |
|---|---|---:|---:|---:|---:|---:|
| 0 | `crab_pot` | 0 | 0 | 0 | 0 | 0.00% |
| 1 | `submarine_pipeline` | 1,000 | 0 | 174 | 1,174 | 23.85% |
| 2 | `shipwreck` | 1,554 | 0 | 372 | 1,926 | 39.12% |
| 3 | `ghost_net` | 873 | 0 | 120 | 993 | 20.17% |
| 4 | `mine_cylinder` | 748 | 0 | 82 | 830 | 16.86% |
| **Total** | | **4,175** | **0** | **748** | **4,923** | **100.00%** |

### B. Distribution in Orphaned Labels (270 Bounding Boxes)
| ID | Class Name | Train Orphaned | Test Orphaned | Total Orphaned Boxes |
|---|---|---:|---:|---:|
| 0 | `crab_pot` | 0 | 0 | 0 |
| 1 | `submarine_pipeline` | 0 | 0 | 0 |
| 2 | `shipwreck` | 0 | 148 | 148 |
| 3 | `ghost_net` | 27 | 0 | 27 |
| 4 | `mine_cylinder` | 95 | 0 | 95 |
| **Total** | | **122** | **148** | **270** |

### C. Total Across All Existing Label Files (5,193 Bounding Boxes)
| ID | Class Name | Train Total | Val | Test Total | Total Boxes |
|---|---|---:|---:|---:|---:|
| 0 | `crab_pot` | 0 | 0 | 0 | 0 |
| 1 | `submarine_pipeline` | 1,000 | 0 | 174 | 1,174 |
| 2 | `shipwreck` | 1,554 | 0 | 520 | 2,074 |
| 3 | `ghost_net` | 900 | 0 | 120 | 1,020 |
| 4 | `mine_cylinder` | 843 | 0 | 82 | 925 |
| **Total** | | **4,297** | **0** | **896** | **5,193** |

---

## 11. Image Resolutions and Formats

### Color Mode:
- **100% RGB**: All 4,348 images are 3-channel RGB images (`mode == 'RGB'`). No grayscale (L) or RGBA images.

### Image Encodings:
- **JPEG (`.jpg`)**: 3,100 files (71.30%)
- **PNG (`.png`)**: 1,248 files (28.70%)

### Resolution Breakdown:
| Resolution ($W \times H$) | Train Count | Test Count | Total Count | % of Dataset | Notes |
|---|---:|---:|---:|---:|---|
| `640x640` | 1,674 | 320 | 1,994 | 45.86% | Standard square YOLO tile |
| `640x500` | 1,495 | 244 | 1,739 | 39.99% | Sonar transect tile |
| `1024x1024` | 115 | 18 | 133 | 3.06% | High-res sonar patches |
| `416x416` | 110 | 18 | 128 | 2.94% | Roboflow export tiles |
| Non-standard variations | 354 | 0 | 354 | 8.14% | Real transect crops (`72x52` to `2048x1161`) |
| **Total** | **3,748** | **600** | **4,348** | **100.00%** | |

*Note: Ultralytics YOLO dynamically pads and resizes non-standard resolutions during training and inference via its standard `imgsz=640` letterbox pipeline.*

---

## 12. Ultralytics YOLO Compatibility Assessment

### Compatibility Verdict: **NOT DIRECTLY COMPATIBLE AS-IS**

| Verification Aspect | Status | Detail / Impact |
|---|---|---|
| Annotation format | **PASS** | Valid normalized coordinates, valid class IDs. |
| Image validity | **PASS** | No corrupted images, 100% RGB format. |
| YAML `path` parameter | **FAIL (BLOCKING)** | Hardcoded to `D:/Sonar-Drishti/ml/data/splits`. Ultralytics throws `FileNotFoundError`. |
| Validation split | **FAIL (BLOCKING)** | `val` directory is completely empty. Ultralytics training requires a non-empty validation set and will crash immediately. |
| Label-image parity | **WARNING** | 217 orphaned label files exist. While YOLO training will ignore orphaned labels when scanning images, they represent incomplete data syncing or incomplete cleaning. |
| Pre-existing cache | **WARNING** | `train/labels.cache` was compiled on a different system. It must be regenerated before training to avoid cache mismatch errors. |

---

## Recommendations & Action Items for Dataset Owner (P2)

To enable Phase 2 model training without violating project boundaries:

1. **Populate the `val` split**:
   - Provide the 628 validation tiles (508 JPG, 120 PNG) and 628 validation labels that are referenced in P2's earlier audit, or perform an automated train/val re-split.
2. **Resolve the 217 Orphaned Labels**:
   - Provide the 122 missing synthetic PNGs in `train/images/` and the 95 missing shipwreck JPGs in `test/images/`, OR purge the orphaned `.txt` files if the images were intentionally deleted during cleaning.
3. **Environment-Agnostic `drishti.yaml`**:
   - Update `drishti.yaml` to remove the hardcoded `D:` drive path, or provide a localized version for POSIX environments.
4. **Remove Stale Cache**:
   - Delete `train/labels.cache` so that Ultralytics compiles a clean local cache upon training invocation.

---

*Report generated by P1 AI/ML Detection Module — Phase 1 verification complete. Ready to proceed to Phase 2 upon user instruction and dataset readiness.*
