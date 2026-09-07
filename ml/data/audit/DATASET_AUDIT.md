\# DRISHTI-SSS Dataset Audit



\## Dataset Path



Working copy:



C:\\drishti\_clean



Original dataset:



C:\\drishti-sss



The original dataset is never modified.



\---



\## Dataset Split



| Split | JPG | PNG | Total |

|---|---:|---:|---:|

| Train | 2620 | 1250 | 3870 |

| Validation | 508 | 120 | 628 |

| Test | 580 | 120 | 700 |

| \*\*Total\*\* | \*\*3708\*\* | \*\*1490\*\* | \*\*5198\*\* |



\---



\## Annotation Checks



\- Invalid annotations: 0

\- Invalid class IDs: 0

\- Corrupted images: 0

\- Labels without corresponding images: 0

\- Images without corresponding labels: 0



\---



\## Duplicate Check



7 duplicate images were removed from the clean working copy.



After removal:



\- Duplicate groups remaining: 0

\- Duplicate images remaining: 0



\---



\## Classes



| ID | Class | Annotated Objects |

|---:|---|---:|

| 0 | crab\_pot | 0 |

| 1 | submarine\_pipeline | 1321 |

| 2 | shipwreck | 2623 |

| 3 | ghost\_net | 1140 |

| 4 | mine\_cylinder | 1018 |



\---



\## Annotation Format



Annotations use YOLO object-detection format:



```text

class\_id x\_center y\_center width height

