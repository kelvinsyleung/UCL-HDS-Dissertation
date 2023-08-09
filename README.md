# UCL-HDS-Dissertation
A University College London MSc Health Data Science Dissertation and AstraZeneca Summer Studentship Project

Project: Human-like AI Analysis on Pathologic Images

Dataset Used: https://www.bracs.icar.cnr.it/

## Objectives
- To explore the effects of human-like image transformation including magnification and colour space on the performance of AI and ML analysis of pathologic whole slide images (WSI)
- To design a computationally efficient ML model for WSI analysis combining various ML techniques
- To create reusable assets for pathologic image analysis and discuss implications on study results

## Progress
### Report Writing
| Section | Status | Completion Date (Expected Date) |
|---|---|---|
| Literature Review | WIP 📝 | (10-8-2023) |
| Methodology | Planning 📆 | (16-8-2023) |
| Results | Dependency - Code ⬇️ | (21-8-2023) |
| Discussion | Dependency - Result ⬆️ | (24-8-2023) |

### Experiments
| Module | Status | Completion Date (Expected Date) |
|---|---|---|
| Patch Extraction | Completed 🙂 | 28-7-2023 |
| Data Transformation Pipeline | Completed 🙂 | 21-7-2023 |
| CNN Classifier | WIP 🧑🏻‍💻 | (31-7-2023) |
| Evalutaion Module | Planning 📆 | (2-8-2023) |

### High Performance Cluster Run
| Work | Status | Completion Date (Expected Date) |
|---|---|---|
| Upload Dataset | WIP 🧑🏻‍💻 | (31-7-2023) |
| Job scripts |  Planning 📆 | (4-8-2023) |
| Factorial Design | Planning 📆 | (8-8-2023) |
| Sequential Model | Planning 📆 | (10-8-2023) |
| Adaptive Model | Planning 📆 | (16-8-2023) |

## Data Directories structure
DATA_PATH - BRACS_WSI

    ├── train
    |     ├── Group_AT
    |     |      ├── Type_ADH
    |     |      └── Type_FEA
    |     ├── Group_BT
    |     |      ├── Type_N
    |     |      ├── Type_PB
    |     |      └── Type_UDH
    |     └── Group_MT
    |            ├── Type_DCIS
    |            └── Type_IC
    ├── val
    .    .
    └── test
         .

ANNOT_PATH - BRACS_WSI

    ├── train
    |     ├── Group_AT
    |     |      ├── Type_ADH ─── BRACS_<ID1>.qpdata, BRACS_<ID2>.qpdata, ...
    |     |      └── Type_FEA ...
    |     ├── Group_BT
    |     |      ├── Type_N ...
    |     |      ├── Type_PB ...
    |     |      └── Type_UDH ...
    |     └── Group_MT
    |            ├── Type_DCIS ...
    |            └── Type_IC ...
    ├── val
    .    .
    └── test
         .


## Run
### Download the dataset from BRACS for processing. The directories structure from BRACS should be identical to above by default. Otherwise, format and align with above.

### Annotation preprocessing
open QuPath to edit the `qp_annotations_to_json.groovy` script

specify the `ANNOT_PATH` in line 5 `def outputPath = "path/to/BRACS_WSI_Annotations"`

run and generate `.geojson` files from the original `.qpdata` files into the same `ANNOT_PATH`

### Patch Extraction Module
modify the file `DATA_PATH` and `ANNOT_PATH` to point to the directories with the WSIs and the annotations
```python
python extract_patches.py
```

### Sample patch transform and classifier
use the `-p` or `--project_root` option to indicate where are the patches extracted in the previous step

```python
python patch_transform_and_classifier --project-path <project_path>
```
