import argparse
from pathlib import Path
import glob
import random
import time
import logging

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import torch
import torchvision
import geojson


from inference_model import InferenceModel
from class_mapping import NAME2TYPELABELS_MAP, LABELS2TYPE_MAP
from log_utils import setup_logging


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    setup_logging()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-p", "--project_root", help="project root path, e.g. -p /path/to/data", type=str, default=".", required=True)
    argParser.add_argument(
        "-r", "--raw_data_folder",
        help="raw data folder path, e.g. -r /path/to/raw", type=str, required=True)
    argParser.add_argument(
        "-a", "--annot_folder", help="annotation folder path, e.g. -a /path/to/annot", type=str, required=True)
    argParser.add_argument(
        "-s", "--slide_tile_size", help="slide tile size, e.g. -t 512", type=int, default=512)
    argParser.add_argument(
        "-t", "--roi_tile_size", help="slide tile size, e.g. -t 512", type=int, default=512)
    argParser.add_argument(
        "-c", "--classifier_colour_space", help="colour space: RGB, CIELAB e.g. -c RGB", type=str, default="RGB")
    argParser.add_argument(
        "-o", "--obj_detect_colour_space", help="colour space: RGB, CIELAB e.g. -c RGB", type=str, default="RGB")
    argParser.add_argument(
        "-m", "--mag", help="magnification of patches for training: 20x or 40x, e.g. -m 20x", type=str, default="20x")
    argParser.add_argument(
        "-b", "--classifier_batch_size", help="classifier batch size, e.g. -b 128", type=int, default=128)
    args = argParser.parse_args()

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    RAW_DATA_FOLDER_PATH = args.raw_data_folder
    ANNOT_PATH = args.annot_folder
    OUTPUT_PATH = f"{PROJECT_ROOT}/output/"
    OUTPUT_PLOT_PATH = f"{OUTPUT_PATH}/plots/evaluation"
    SLIDE_PATCH_SIZE = args.slide_tile_size
    ROI_PATCH_SIZE = args.roi_tile_size

    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)

    NORM_PATH = f"{PROJECT_ROOT}/data/norms"
    OBJ_DETECT_CSPACE = args.classifier_colour_space
    CLASSIFIER_CSPACE = args.obj_detect_colour_space
    PATCH_MAGNIFICATION = args.mag
    CLASSIFIER_BATCH_SIZE = args.classifier_batch_size

    if OBJ_DETECT_CSPACE == "RGB":
        mean = np.load(f"{NORM_PATH}/rgb_mean.npy")
        std = np.load(f"{NORM_PATH}/rgb_std.npy")
        print("main - RGB mean and std loaded")
    elif OBJ_DETECT_CSPACE == "CIELAB":
        mean = np.load(f"{NORM_PATH}/cielab_mean.npy")
        std = np.load(f"{NORM_PATH}/cielab_std.npy")
        print("main - CIELAB mean and std loaded")

    num_classes = len(LABELS2TYPE_MAP)

    classifier_model = torchvision.models.resnext101_32x8d(
        num_classes=num_classes)
    classifier_model.load_state_dict(torch.load(
        f"./models/train_patch_classifier/weighted_resnext_101_{CLASSIFIER_CSPACE}_{'mixed' if PATCH_MAGNIFICATION == '*' else PATCH_MAGNIFICATION}_best_model.pth")["model_state_dict"])

    object_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        image_mean=mean, image_std=std, num_classes=2, pretrained_backbone=None)
    object_detection_model.load_state_dict(torch.load(
        f"./models/train_obj_detector/fasterrcnn_resnet50_fpn_{OBJ_DETECT_CSPACE}_best_model.pth")["model_state_dict"])

    inference_model = InferenceModel(
        classifier_model=classifier_model,
        object_detection_model=object_detection_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        classifier_cspace=CLASSIFIER_CSPACE,
        classifier_mag=PATCH_MAGNIFICATION,
        obj_detect_cspace=OBJ_DETECT_CSPACE,
        norm_path=NORM_PATH,
        slide_patch_size=SLIDE_PATCH_SIZE,
        roi_patch_size=ROI_PATCH_SIZE,
        num_classes=num_classes,
        classifier_batch_size=CLASSIFIER_BATCH_SIZE
    )

    test_wsi_paths = sorted(
        glob.glob(f"{RAW_DATA_FOLDER_PATH}/test/*/*/*.svs"))
    test_annot_paths = sorted(glob.glob(f"{ANNOT_PATH}/test/*/*/*.geojson"))

    test_wsi_samples = [Path(path).stem for path in test_wsi_paths]
    test_annot_samples = [Path(path).stem for path in test_annot_paths]

    test_set = []
    test_set_gt = []
    for test_wsi_path in test_wsi_paths:
        test_wsi_sample = Path(test_wsi_path).stem
        test_annot_path = None
        if test_wsi_sample in test_annot_samples:
            test_annot_path = test_annot_paths[test_annot_samples.index(
                test_wsi_sample)]

        test_set.append((test_wsi_path, test_annot_path))

        ground_truth = NAME2TYPELABELS_MAP[test_wsi_path.split("/")[-3]]
        test_set_gt.append(ground_truth)

    test_set_pred = []
    test_set_roi_preds = []
    test_set_roi_logits = []
    test_set_roi_probs = []

    start_time = time.time()
    for test_wsi_path, test_annot_path in test_set:
        preds = inference_model.predict(test_wsi_path)
        test_set_pred.append(preds["slide_class"])
        test_set_roi_preds.append(preds["roi_preds"])
        test_set_roi_logits.append(preds["roi_logits"])
        test_set_roi_probs.append(preds["roi_probs"])

    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_set)
    logging.info(f"Time taken for inference: {end_time - start_time} seconds")
    logging.info(
        f"Average time taken for inference per WSI: {avg_time} seconds")

    logging.info(
        f"\n{classification_report(test_set_gt, test_set_pred, digits=4, target_names=list(LABELS2TYPE_MAP.values()))}")
    ConfusionMatrixDisplay.from_predictions(
        test_set_gt, test_set_pred,
        display_labels=list(LABELS2TYPE_MAP.values()),
        xticks_rotation="vertical",
        cmap=plt.cm.Blues
    )
    plt.savefig(f"{OUTPUT_PLOT_PATH}/confusion_matrix.png")

    for i, (test_wsi_path, test_annot_path) in enumerate(test_set):
        if test_annot_path:
            with open(test_annot_path) as f:
                annot = geojson.load(f)
            inference_model.plot_annotations(
                test_wsi_path,
                test_set_roi_preds[i],
                annot,
                roi_patch_size=inference_model.roi_patch_size,
                save_plot_path=f"{OUTPUT_PLOT_PATH}/{Path(test_wsi_path).stem}_pseudo_annot.png",
                is_prob=False
            )
            inference_model.plot_annotations(
                test_wsi_path,
                test_set_roi_probs[i],
                annot,
                roi_patch_size=inference_model.roi_patch_size,
                save_plot_path=f"{OUTPUT_PLOT_PATH}/{Path(test_wsi_path).stem}_class_prob.png",
                is_prob=True
            )
