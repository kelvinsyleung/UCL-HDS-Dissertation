import argparse
from typing import Tuple, List, Dict, Union
import os
from pathlib import Path
import glob
import gc
from datetime import datetime
import logging

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import geojson
from patchify import patchify
import cv2

from coord_utils import get_relative_coordinates, pad_roi_coordinates
from log_utils import setup_logging

OPENSLIDE_PATH  = r"C:/openslide/openslide-win64/bin"
if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def get_bbox_by_shape(coordinates: List, shape_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of coordinates, return the bounding box

    returns: (x_min, y_min), (x_max, y_max)
    """

    coordinates_arr = np.array([])
    if shape_type == "Polygon":
        # first coordinate is the outer polygon, the rest are holes
        coordinates_arr = np.array(coordinates[0]).squeeze()
    elif shape_type == "MultiPolygon":
        for polygon in coordinates:
            coordinates_arr = np.array(polygon[0]).squeeze()
            if coordinates_arr.shape[0] > 4:
                break
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    min_coord = np.floor(np.min(coordinates_arr, axis=0)).astype(int)
    max_coord = np.ceil(np.max(coordinates_arr, axis=0)).astype(int)
    return tuple(min_coord), tuple(max_coord)

def create_bbox_list(annotations: Dict) -> List[Dict[str, Union[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Given a dictionary of annotations, return a list of bounding boxes and their classifications

    returns: list of bounding boxes and their classifications
    """
    bboxes = []
    for annotation in annotations["features"]:
        bboxes.append({
            "classification": annotation["properties"]["classification"]["name"],
            "coordinates": get_bbox_by_shape(
                annotation["geometry"]["coordinates"],
                annotation["geometry"]["type"]
            )
        })
    return bboxes

def get_annotated_rois(
        slide: openslide.OpenSlide,
        bboxes: List[Dict[str, Union[str, Tuple[np.ndarray, np.ndarray]]]],
        patch_size: int, step_size: int
    ) -> List[Dict[str, Union[str, np.ndarray]]]:
    """
    Given a list of bounding boxes, return a list of ROI images and their classifications

    returns: list of ROI images and their classifications
    """
    rois = []
    for bbox in bboxes:
        min_coord, max_coord = bbox["coordinates"][0], bbox["coordinates"][1]
        min_coord, max_coord = pad_roi_coordinates(min_coord, max_coord, slide.dimensions, patch_size, step_size)

        roi = slide.read_region(
            min_coord,
            level=0,
            size=tuple(np.subtract(max_coord, min_coord))
        )
        rois.append({
            "class": bbox["classification"],
            "min_coord": min_coord,
            "max_coord": max_coord,
            "image": roi
        })
    return rois

def display_rois(rois: List[Dict[str, Union[str, Tuple[int, int], Image.Image]]], output_path: str):
    """
    Given a list of ROIs, display them
    """
    num_of_rois = len(rois)
    logging.info(f"display_rois - total number of ROIs: {num_of_rois}")
    plt.figure(figsize=(15, int(np.ceil(num_of_rois))))
    for num, roi in enumerate(rois):
        plt.subplot(int(np.ceil(num_of_rois/3)), 3, num+1)
        plt.imshow(np.array(roi["image"]))
        plt.title(roi["class"])
        plt.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"display_rois - saved to {output_path}")

def create_masks(
        rois: List[Dict[str, Union[str, Tuple[int, int], Image.Image]]],
        annotations: Dict
    ) -> List[np.ndarray]:
    """
    Given a dictionary of annotations, return a list of masks of each ROI

    returns: masks of each ROI
    """
    masks = []

    for i, annotation in enumerate(annotations["features"]):
        coordinates = annotation["geometry"]["coordinates"]
        min_coord, max_coord = rois[i]["min_coord"], rois[i]["max_coord"]
        width, height = np.subtract(max_coord, min_coord)
        mask = np.zeros((height, width), dtype=np.int32)

        # create blank mask if type is not polygon
        if annotation["geometry"]["type"] == "Polygon":
        # first list of coordinate is the outer polygon, the rest are holes
            coordinates_arr = np.array(coordinates[0]).squeeze()
            relative_coordinates = get_relative_coordinates(coordinates_arr, min_coord)
            mask = cv2.fillPoly(mask, pts=np.array([relative_coordinates], dtype=np.int32), color=(255, 255, 255))
        elif annotation["geometry"]["type"] == "MultiPolygon":
            # loop through each polygon, stop at the first polygon with more than 4 coordinates
            for polygon in coordinates:
                # first list of coordinate is the outer polygon, the rest are holes
                coordinates_arr = np.array(polygon[0]).squeeze()
                # take only polygon with more than 4 coordinates
                if coordinates_arr.shape[0] > 4:
                    relative_coordinates = get_relative_coordinates(coordinates_arr, min_coord)
                    mask = cv2.fillPoly(mask, pts=np.array([relative_coordinates], dtype=np.int32), color=(255, 255, 255))
                    break
        
        masks.append(mask)
    return masks

def display_patch(patch: np.ndarray, mask_patch: Union[np.ndarray, None] = None, is_overlay: bool = False):
    plt.imshow(patch, vmin=0, vmax=255)
    if is_overlay:
        masked_arr = np.ma.masked_where(mask_patch == 0, mask_patch)
        plt.imshow(masked_arr, alpha=0.5)
    plt.axis("off")

def display_patches(
        patches: np.ndarray, output_path: str, mask_patches: Union[np.ndarray, List] = [],
        is_mask: bool = False, is_overlay: bool = False
    ):
    """
    Given a list of ROIs, display them
    """
    num_of_patches = patches.shape[0] * patches.shape[1]
    logging.info(f"display_patches - total number of patches: {num_of_patches}")
    plt.figure(figsize=(15, 10))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            plt.subplot(patches.shape[0], patches.shape[1], i*patches.shape[1] + j + 1)
            if is_mask:
                patch = np.ma.masked_where(patches[i, j] == 0, patches[i, j])
            # set vmin and vmax to 0 and 255 to display the mask
            if is_overlay:
                mask_patch = mask_patches[i, j]
                display_patch(patch, mask_patch=mask_patch, is_overlay=True)
            else:
                display_patch(patch)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"display_patches - saved to {output_path}")

def resize_roi_and_masks(roi_arr: np.ndarray, mask: np.ndarray, downsample_factor: int = 2):
    """
    Given a ROI and its mask, resize them by the downsample factor
    """
    roi_arr = np.array(roi_arr, dtype=np.uint8)
    mask = mask.astype(np.uint8)

    # downsample the roi_arr and mask, W and H are reversed for cv2.resize
    roi_arr = cv2.resize(roi_arr, (roi_arr.shape[1]//downsample_factor, roi_arr.shape[0]//downsample_factor))
    mask = cv2.resize(mask, (mask.shape[1]//downsample_factor, mask.shape[0]//downsample_factor))

    return roi_arr, mask

def save_patches(patches: np.ndarray, mask_patches: np.ndarray, save_path: str = ".", verbose: bool = False):
    """
    Save patches to a given path
    """
    num_of_patches = patches.shape[0] * patches.shape[1]
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            # save patch
            patch = patches[i, j, 0]
            patch = Image.fromarray(patch)
            os.makedirs(f"{save_path}/patch", exist_ok=True)
            patch.save(f"{save_path}/patch/{i}_{j}.png")

            # save mask
            mask = mask_patches[i, j]
            mask = Image.fromarray(np.stack((mask,)*3, axis=-1).astype(np.uint8))
            
            os.makedirs(f"{save_path}/mask", exist_ok=True)
            mask.save(f"{save_path}/mask/{i}_{j}.png")

    if verbose:
        logging.info(f"save_patches - saved {num_of_patches} patches")

def patchify_and_save(roi_arr: np.ndarray, mask: np.ndarray, save_path: str = ".", overlap: bool = True, verbose: bool = False):
    assert roi_arr.shape[0] == mask.shape[0] and roi_arr.shape[1] == mask.shape[1], "ROI and mask must have the same shape"
    min_dim = min(roi_arr.shape[0], roi_arr.shape[1], 512)
    step = min_dim//2 if overlap else min_dim
    roi_patches = patchify(roi_arr, (min_dim, min_dim, 3), step=step)
    mask_patches = patchify(mask, (min_dim, min_dim), step=step)
    save_patches(roi_patches, mask_patches, save_path=save_path, verbose=verbose)

def create_patches_dataset(annot_path: str, patch_folder: str, data_folder: str, set_type: str, patch_size: int, step_size: int):
    file_id = annot_path.split("/")[-1].split(".")[0]

    # create if not exists
    if not f"{patch_folder}/{set_type}/{file_id}" in glob.glob(f"{patch_folder}/{set_type}/{file_id}"):
        logging.info(f"create_patches_dataset - processing: {file_id}")
        wsi_file_paths = glob.glob(f"{data_folder}/{set_type}/**/{file_id}.svs", recursive=True)

        if len(wsi_file_paths) != 0:
            try:
                slide = openslide.OpenSlide(wsi_file_paths[0])
                annotations = geojson.load(open(annot_path))
                bbox_list = create_bbox_list(annotations)
                rois = get_annotated_rois(slide, bbox_list, patch_size, step_size)
                masks = create_masks(rois, annotations)

                for idx, (roi, mask) in enumerate(zip(rois, masks)):
                    roi_arr = np.array(roi["image"])
                    patchify_and_save(roi_arr, mask, save_path=f"{patch_folder}/{set_type}/{file_id}/{roi['class']}-{idx}-40x", overlap=False)

                    roi_arr, mask = resize_roi_and_masks(roi["image"], mask, downsample_factor=2)
                    patchify_and_save(roi_arr, mask, save_path=f"{patch_folder}/{set_type}/{file_id}/{roi['class']}-{idx}-20x", overlap=False)

            except Exception as e:
                logging.exception(f"create_patches_dataset - fail to process {file_id}:")
                raise e
            finally:
                slide.close()
            
        gc.collect()
    else:
        logging.info(f"create_patches_dataset - {patch_folder}/{set_type}/{file_id} already exists")

if __name__ == "__main__":
    setup_logging()

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--project_root", help="project root path to export the extracted patch data, e.g. -p /path/to/project/", type=str, default=".", required=True)
    argParser.add_argument("-r", "--raw_data_folder", help="raw data folder path, e.g. -r /path/to/raw", type=str, required=True)
    argParser.add_argument("-a", "--annot_folder", help="annotation folder path, e.g. -a /path/to/annot", type=str, required=True)
    argParser.add_argument("-t", "--tile_size", help="tile size, e.g. -t 512", type=int, default=512)
    argParser.add_argument("-s", "--step_size", help="step size, e.g. -s 512", type=int, default=512)
    args = argParser.parse_args()

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    RAW_DATA_FOLDER_PATH = args.raw_data_folder
    ANNOT_PATH = args.annot_folder
    PATCH_PATH = f"{PROJECT_ROOT}/data/roi_patches"
    OUTPUT_PATH = f"{PROJECT_ROOT}/output/"
    OUTPUT_PLOT_PATH = f"{OUTPUT_PATH}/plots/extract_patches"
    PATCH_SIZE = args.tile_size
    STEP_SIZE = args.step_size

    Path(PATCH_PATH).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)

    # get slide sample 1
    slide = openslide.OpenSlide(f"{RAW_DATA_FOLDER_PATH}/train/Group_AT/Type_ADH/BRACS_1486.svs")

    logging.info(f"main - levels: {slide.level_count}")
    logging.info(f"main - level dimensions {slide.level_dimensions}")
    logging.info(f"main - level downsamples {slide.level_downsamples}")

    # get annotations 1
    annotation_file = f"{ANNOT_PATH}/train/Group_AT/Type_ADH/BRACS_1486.geojson"
    annotations = geojson.load(open(annotation_file))
    bbox_list = create_bbox_list(annotations)
    rois = get_annotated_rois(slide, bbox_list, PATCH_SIZE, STEP_SIZE)
    masks = create_masks(rois, annotations)

    # display lowest resolution 1
    slide.get_thumbnail(slide.level_dimensions[-1]).save(f"{OUTPUT_PLOT_PATH}/BRACS_1486.png")
    logging.info(f"main - saved to {OUTPUT_PLOT_PATH}/BRACS_1486.png")

    # display ROIs
    display_rois(rois, output_path=f"{OUTPUT_PLOT_PATH}/BRACS_1486_rois.png")

    # patchify and save training sample patches 1
    for idx, (roi, mask) in enumerate(zip(rois, masks)):
        # sample 40x and 20x patches
        patchify_and_save(np.array(roi["image"]), mask, save_path=f"{PATCH_PATH}/sample/BRACS_1486/{roi['class']}-{idx}-40x", overlap=False)

        roi_arr, mask = resize_roi_and_masks(np.array(roi["image"]), mask, downsample_factor=2)
        patchify_and_save(roi_arr, mask, save_path=f"{PATCH_PATH}/sample/BRACS_1486/{roi['class']}-{idx}-20x", overlap=False)

    logging.info("main - saved BRACS_1486 patches")

    # plot sample patches 1
    for sample_patch_folder_path in sorted(glob.glob(f"{PATCH_PATH}/sample/BRACS_1486/*"))[0:2]:
        patch_paths = sorted(glob.glob(f"{sample_patch_folder_path}/patch/*"))
        mask_paths = sorted(glob.glob(f"{sample_patch_folder_path}/mask/*"))

        patches = []
        for patch_path in patch_paths:
            patch = cv2.imread(patch_path)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patches.append(patch)
        patches = np.array(patches)

        mask_patches = []
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0.5
            mask_patches.append(mask)
        mask_patches = np.array(mask_patches)

        # reshape patches and mask_patches to (vertical position, horizontal position, H, W, C)
        vertical_patch_idx, horizontal_patch_idx = patch_paths[-1].split("/")[-1].split(".")[0].split("_")
        vertical_patch_count, horizontal_patch_count = int(vertical_patch_idx) + 1, int(horizontal_patch_idx) + 1
        patches = patches.reshape((vertical_patch_count, horizontal_patch_count, patches.shape[1], patches.shape[2], patches.shape[3]))
        mask_patches = mask_patches.reshape((vertical_patch_count, horizontal_patch_count, mask_patches.shape[1], mask_patches.shape[2]))

        display_patches(
            patches,
            output_path=f"{OUTPUT_PLOT_PATH}/BRACS_1486/{sample_patch_folder_path.split('/')[-1]}_sample_overlay_patches.png",
            mask_patches=mask_patches,
            is_mask=True,
            is_overlay=True
        )

    # get slide sample 2
    slide = openslide.OpenSlide(f"{RAW_DATA_FOLDER_PATH}/train/Group_BT/Type_PB/BRACS_745.svs")

    logging.info(f"main - levels: {slide.level_count}")
    logging.info(f"main - level dimensions {slide.level_dimensions}")
    logging.info(f"main - level downsamples {slide.level_downsamples}")

    # get annotations 2
    annotation_file = f"{ANNOT_PATH}/train/Group_BT/Type_PB/BRACS_745.geojson"
    annotations = geojson.load(open(annotation_file))
    bbox_list = create_bbox_list(annotations)
    rois = get_annotated_rois(slide, bbox_list, PATCH_SIZE, STEP_SIZE)
    masks = create_masks(rois, annotations)

    # display lowest resolution 2
    slide.get_thumbnail(slide.level_dimensions[-1]).save(f"{OUTPUT_PLOT_PATH}/BRACS_745.png")
    logging.info("main - saved BRACS_745.png")

    display_rois(rois, output_path=f"{OUTPUT_PLOT_PATH}/BRACS_745_rois.png")

    # patchify sample patches 2
    roi_image_arr = np.array(rois[0]["image"])
    sample_patches = patchify(roi_image_arr, (PATCH_SIZE, PATCH_SIZE, 3), step=STEP_SIZE)
    sample_mask_patches = patchify(masks[0], (PATCH_SIZE, PATCH_SIZE), step=STEP_SIZE)

    # plot sample patches 2
    display_patches(sample_patches, output_path=f"{OUTPUT_PLOT_PATH}/BRACS_745_sample_patches.png")
    display_patches(sample_mask_patches, output_path=f"{OUTPUT_PLOT_PATH}/BRACS_745_sample_mask_patches.png", is_mask=True)
    display_patches(sample_patches, output_path=f"{OUTPUT_PLOT_PATH}/BRACS_745_sample_overlay_patches.png", mask_patches=sample_mask_patches, is_overlay=True)

    gc.collect()

    # get all annotations
    train_set = glob.glob(f"{ANNOT_PATH}/train/**/*.geojson", recursive=True)
    val_set = glob.glob(f"{ANNOT_PATH}/val/**/*.geojson", recursive=True)
    test_set = glob.glob(f"{ANNOT_PATH}/test/**/*.geojson", recursive=True)

    failed_file_path = Path(f"{OUTPUT_PATH}/failed_preprocess_files-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")

    logging.info("main - start processing train set")   
    for train_annot_path in train_set:
        try:
            create_patches_dataset(train_annot_path, PATCH_PATH, RAW_DATA_FOLDER_PATH, "train", PATCH_SIZE, STEP_SIZE)
        except Exception as e:
            file_id = train_annot_path.split('/')[-1].split('.')[0]
            with open(failed_file_path, 'a+') as f:
                f.write(f'{file_id}\n')

    logging.info("main - start processing validation set")
    for val_annot_path in val_set:
        try:
            create_patches_dataset(val_annot_path, PATCH_PATH, RAW_DATA_FOLDER_PATH, "val", PATCH_SIZE, STEP_SIZE)
        except Exception as e:
            file_id = val_annot_path.split('/')[-1].split('.')[0]
            with open(failed_file_path, 'a+') as f:
                f.write(f'{file_id}\n')
