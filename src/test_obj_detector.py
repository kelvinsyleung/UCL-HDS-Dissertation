import argparse
import glob
from pathlib import Path
import logging
import time

import numpy as np

import torch
import torchvision

import geojson

from log_utils import setup_logging
from inference_model import InferenceModel

if __name__ == "__main__":
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
        "-c", "--colour_space", help="colour space: RGB, CIELAB e.g. -c RGB", type=str, default="RGB")
    argParser.add_argument(
        "-k", "--top_k_boxes", help="number of boxes retrieve per tile", type=int, default=10)
    argParser.add_argument(
        "-n", "--nms_threshold", help="non-max suppression threshold", type=float, default=0.3)
    args = argParser.parse_args()

    # colour space
    COLOUR_SPACE = args.colour_space
    logging.info(f"main - COLOUR_SPACE: {COLOUR_SPACE}")

    if COLOUR_SPACE not in ["RGB", "CIELAB"]:
        raise ValueError("Invalid colour space")

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    RAW_DATA_FOLDER_PATH = args.raw_data_folder
    ANNOT_PATH = args.annot_folder
    DATA_PATH = f"{PROJECT_ROOT}/data"
    NORM_PATH = f"{DATA_PATH}/norms"

    TOP_K_BOXES = args.top_k_boxes
    NMS_THRESHOLD = args.nms_threshold

    # relative to script execution path
    OUTPUT_PLOT_PATH = f"{PROJECT_ROOT}/output/plots/test_obj_classifier"

    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"pytorch device using: {device}")

    if COLOUR_SPACE == "RGB":
        mean = np.load(f"{NORM_PATH}/rgb_mean.npy")
        std = np.load(f"{NORM_PATH}/rgb_std.npy")
        logging.info("main - RGB mean and std loaded")
    elif COLOUR_SPACE == "CIELAB":
        mean = np.load(f"{NORM_PATH}/cielab_mean.npy")
        std = np.load(f"{NORM_PATH}/cielab_std.npy")
        logging.info("main - CIELAB mean and std loaded")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        image_mean=mean, image_std=std, num_classes=2, pretrained_backbone=None)
    model.load_state_dict(torch.load(
        f"./models/train_obj_detector/fasterrcnn_resnet50_fpn_{COLOUR_SPACE}_best_model.pth")["model_state_dict"])

    inference_model = InferenceModel(
        None, model, device,
        obj_detect_cspace=COLOUR_SPACE,
        top_k_boxes=TOP_K_BOXES, box_nms_threshold=NMS_THRESHOLD
    )

    test_wsi_paths = sorted(
        glob.glob(f"{RAW_DATA_FOLDER_PATH}/test/*/*/*.svs"))
    test_annot_paths = sorted(glob.glob(f"{ANNOT_PATH}/test/*/*/*.geojson"))

    test_wsi_samples = [Path(path).stem for path in test_wsi_paths]
    test_annot_samples = [Path(path).stem for path in test_annot_paths]

    test_set = []
    for test_wsi_path in test_wsi_paths:
        test_wsi_sample = Path(test_wsi_path).stem
        test_annot_path = None
        if test_wsi_sample in test_annot_samples:
            test_annot_path = test_annot_paths[test_annot_samples.index(
                test_wsi_sample)]

        test_set.append((test_wsi_path, test_annot_path))

    start_time = time.time()
    macro_specificity = []
    macro_sensitivity = []
    for test_wsi_path, test_annot_path in test_set:
        if test_annot_path is not None:
            slide, scaled_bboxes = inference_model.get_roi_bboxes(
                test_wsi_path)
            with open(test_annot_path) as f:
                annot = geojson.load(f)

            metrics = inference_model.calculate_roi_metrics(
                scaled_bboxes=scaled_bboxes,
                slide_total_area=slide.level_dimensions[0][0] *
                slide.level_dimensions[0][1],
                gt_annotations=annot
            )
            macro_specificity.append(metrics["specificity"])
            macro_sensitivity.append(metrics["sensitivity"])

    end_time = time.time()
    logging.info(
        f"main - average inference time: {(end_time - start_time)/len(test_set)}")
    logging.info(f"main - macro specificity: {np.mean(macro_specificity)}")
    logging.info(f"main - macro sensitivity: {np.mean(macro_sensitivity)}")
