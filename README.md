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
| Methodology | Completed 🙂 | 5-9-2023 |
| Results | Dependency - Code ⬇️ | (27-8-2023) |
| Discussion | WIP 🧑🏻‍💻 | (30-8-2023) |

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
| Factorial Classifier | Completed 🙂 | 31-8-2023 |
| Sequential Model | Completed 🙂 | 3-9-2023 |

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

PROJECT_PATH

    ├── data
    |     ├── norms
    |     |      └── <normalisation_values>.npy ...
    |     ├── roi_patches
    |     |      ├── sample ...
    |     |      ├── train ...
    |     |      └── val ...
    |     ├── roi_test_imgs (ROI only test set images from BRACS)
    |     |      ├── 0_N ...
    |     |      ├── 1_PB ...
    |     |      .
    |     |      └── 6_IC ...
    |     └── slide_patches
    |            ├── sample ...
    |            ├── train ...
    |            └── val ...
    ├── models
    |     ├── train_obj_detector ...
    |     └── train_patch_classifier ...
    |     
    └── output
          └── plots
                ├── evaluation
                .
                ├── train_obj_detector
                └── train_patch_classifier


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
```python
python src/extract_patches.py -p <project_path> -r /some/path/BRACS/BRACS_WSI/ -a /some/path/BRACS_WSI_Annotations/
python src/extract_slides.py -p <project_path> -r /some/path/BRACS/BRACS_WSI/ -a /some/path/BRACS_WSI_Annotations/
```

#### Sample patch transformations and augmentations
```python
python src/patch_transform_showcase.py -p <project_path>
```

#### Slide level ROI detector

```python
python src/train_obj_detector.py --project_root <project_path> --color_space <color_space>
python src/test_obj_detector.py -p <project_path> -r /some/path/BRACS/BRACS_WSI/ -a /some/path/BRACS_WSI_Annotations/ -c <color_space> -k <top_k_boxes> -n <nms_threshold>
```

#### Patch level Classifier

```python
python src/train_patch_classifier.py --project_root <project_path> --color_space <color_space> --mag <magnification>
python src/test_patch_classifier.py -p <project_path> -c <color_space> -m <magnification>
```

#### Inference Model

```python
python src/run_inference.py -p <project_path> -r /some/path/BRACS/BRACS_WSI/ -a /some/path/BRACS_WSI_Annotations/ -s <slide_tile_size> -t <roi_tile_size> -c <classifier_colour_space> -o <obj_detect_colour_space> -m <classifier_magnification> -b <classifier_batch_size> -k <top_k_boxes> -n <nms_threshold>
```
