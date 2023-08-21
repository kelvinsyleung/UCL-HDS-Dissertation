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
| Literature Review | Completed 🙂 | 17-8-2023 |
| Methodology | Planning 📆 | (23-8-2023) |
| Results | Dependency - Code ⬇️ | (27-8-2023) |
| Discussion | Dependency - Result ⬆️ | (30-8-2023) |

### Experiments
| Module | Status | Completion Date (Expected Date) |
|---|---|---|
| Patch Extraction | Completed 🙂 | 28-7-2023 |
| Data Transformation Pipeline | Completed 🙂 | 21-7-2023 |
| CNN Classifier - Patch Level | Completed 🙂 | 8-8-2023 |
| CNN Classifier - Slide Level | Completed 🙂 | 19-8-2023 |
| Evalutaion Module | WIP 🧑🏻‍💻 | (15-8-2023) |

### High Performance Cluster / Google Colab Run
| Work | Status | Completion Date (Expected Date) |
|---|---|---|
| Upload Dataset | Completed 🙂 | 1-8-2023 |
| Job scripts |  Completed 🙂 | 5-8-2023 |
| Factorial Classifier | WIP 🧑🏻‍💻 | (21-8-2023) |
| Sequential Model | WIP 🧑🏻‍💻 | (23-8-2023) |
| Adaptive Model | Planning 📆 | (26-8-2023) |

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
### Download the dataset from BRACS for processing
The directories structure from BRACS should be identical to above by default. Otherwise, format and align with above.

### QuPath Annotation preprocessing
1. open QuPath to edit the `qp_annotations_to_json.groovy` script
2. specify the `ANNOT_PATH` in line 5 `def outputPath = "path/to/BRACS_WSI_Annotations"`
3. run and generate `.geojson` files from the original `.qpdata` files into the same `ANNOT_PATH`

### Python scripts and Notebooks
Most Python scripts includes required arguments, use `python <script>.py -h` to read the details

#### Patch Level Feature Extraction Module and Slide Level Feature Extraction Module
use the `-p` or `--project_root` flag to indicate where to export the extracted thumbnails with bboxes

use the `-r` or `--raw_data_folder` flag to indicate where are the BRACS raw data files

use the `-a` or `--annot_folder` flag to indicate where are the processed BRACS annotation files

use the `-t` or `--tile_size` flag to indicate the patch tile size in pixels

use the `-s` or `--step_size` flag to indicate the offset in pixels to extract the next tile

```python
python extract_patches.py -p <project_path>  -r /some/path/BRACS/BRACS_WSI/ -a /some/path/BRACS_WSI_Annotations/
python extract_slides.py -p <project_path>  -r /some/path/BRACS/BRACS_WSI/ -a /some/path/BRACS_WSI_Annotations/
```

#### Sample patch transformations and augmentations
use the `-p` or `--project_root` flag to indicate where are the patches extracted in the previous step (required)

```python
python patch_transform_showcase.py -p <project_path>
```

#### Patch level Classifier (Refactoring)
use the `-p` or `--project_root` flag to indicate where are the patches extracted in the previous step (required)

use the `-c` or `--color_space` option to indicate the target colour space for transformation (`RGB`, `CIELAB`, `BW`)

use the `-m` or `--mag` option to indicate the specific magnification of patches used to train the model (`20x`, `40x`)

```python
python train_unet.py --project_root --color_space <color_space> --mag <magnification>
```
