# DRISHTI-SSS Dataset Audit (P1 Verification)

## Dataset Path

- **Working Copy (Verified)**: `~/Library/CloudStorage/GoogleDrive-kruthikha2207@gmail.com/My Drive/drishti_clean`
- **Symlink Target**: `/Users/kruthikhareddy/Library/CloudStorage/GoogleDrive-kruthikha2207@gmail.com/.shortcut-targets-by-id/1QvzLWu4t13Ng2LOOpI71FtnbJa3YHnEp/drishti_clean`
- **Config**: `drishti_clean/drishti.yaml`

*Note: Per P1 module boundaries, the dataset is strictly read-only and never modified by P1.*

---

## Dataset Split (Actual vs Expected)

| Split | JPG (.jpg) | PNG (.png) | Total Images Present | Total Label Files (.txt) | Orphaned Labels (Missing Images) |
|---|---:|---:|---:|---:|---:|
| **Train** | 2,620 | 1,128 | 3,748 | 3,870 | 122 |
| **Validation** | 0 | 0 | **0** | **0** | 0 |
| **Test** | 480 | 120 | 600 | 695 | 95 |
| **Total** | **3,100** | **1,248** | **4,348** | **4,565** | **217** |

*Note: The upstream HuggingFace release listed 5,205 tiles (3,875 train, 630 val, 700 test). An earlier local audit referenced 5,198 tiles (7 duplicates removed). In the current clean directory, the `val` folder is completely empty, and 217 label files lack matching image files.*

---

## Annotation Checks

- **Invalid annotations**: 0 (all 5,193 bounding boxes conform to YOLO format `class_id x_center y_center width height` with $0 \le x_c, y_c, w, h \le 1.0$)
- **Invalid class IDs**: 0 (all classes strictly $\in \{1, 2, 3, 4\}$)
- **Corrupted / unreadable images**: 0 (all 4,348 images validated with PIL)
- **Images without corresponding labels**: 0 (100% of images have a corresponding `.txt` file)
- **Labels without corresponding images**: **217** (122 in train, 95 in test)
- **Background images (empty label files)**: 565 (495 in train, 70 in test)

---

## Classes & Distribution

| ID | Class Name | Annotated Objects (Paired Images) | Orphaned Annotations | Total Annotated Objects |
|---:|---|---:|---:|---:|
| 0 | `crab_pot` | 0 | 0 | 0 |
| 1 | `submarine_pipeline` | 1,000 | 0 | 1,000 |
| 2 | `shipwreck` | 1,926 | 148 | 2,074 |
| 3 | `ghost_net` | 993 | 27 | 1,020 |
| 4 | `mine_cylinder` | 830 | 95 | 925 |
| **Total** | | **4,923** | **270** | **5,193** |

*Class 0 (`crab_pot`) has 0 instances in this release due to upstream access gating.*

---

## Image Formats and Resolutions

- **Color Modes**: 100% RGB (all 4,348 images)
- **Primary Resolutions**:
  - `640x640`: 1,994 images (45.9%)
  - `640x500`: 1,739 images (40.0%)
  - `1024x1024`: 133 images (3.1%)
  - `416x416`: 128 images (2.9%)
  - Other crops: 354 images (8.1%)

---

## YOLO Compatibility Status: NOT DIRECTLY COMPATIBLE (BLOCKING)

1. **`drishti.yaml` Root Path**: Contains hardcoded Windows path `path: D:/Sonar-Drishti/ml/data/splits`. Ultralytics throws `FileNotFoundError` on macOS.
2. **Missing Validation Split**: `val` is empty; Ultralytics YOLO training fails immediately as validation evaluation is required.
3. **Orphaned Label Files**: 217 labels without images need reconciliation by P2.
4. **Stale Cache**: `train/labels.cache` was compiled on a different machine and must be purged before training.

For full technical breakdown, see [P1_DATASET_VERIFICATION_REPORT.md](file:///Users/kruthikhareddy/Desktop/Sonaris/ml/data/audit/P1_DATASET_VERIFICATION_REPORT.md).
