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


from extract_patches import create_bbox_list
from coord_utils import pad_roi_coordinates
from log_utils import setup_logging

OPENSLIDE_PATH = r"C:/openslide/openslide-win64/bin"
if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def get_scaled_img_and_rois(
    slide: openslide.OpenSlide,
    bbox_list: List[Dict],
    level: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get scaled image and rois from slide and bbox list at given downsample level, default as max downsample.
    """
    roi_bboxes = np.array(list(map(lambda x: x["coordinates"], bbox_list)))
    slide_arr = np.array(slide.get_thumbnail(slide.level_dimensions[level]))

    slide_width, slide_height = slide.dimensions
    height, width = slide_arr.shape[:2]
    width_ratio = width / slide_width
    height_ratio = height / slide_height

    scaled_bboxes = roi_bboxes * np.array([width_ratio, height_ratio])
    return slide_arr, scaled_bboxes


def get_extract_area_coord(
    slide_arr: np.ndarray,
    scaled_roi_bboxes: np.ndarray,
    patch_size: int,
    step_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get the coordinates of the area to extract from the slide.
    """
    # get the min and max x and y coordinates among the ROIs
    min_bbox_x, min_bbox_y = scaled_roi_bboxes.reshape(
        -1, 2).min(axis=0).astype(int)
    max_bbox_x, max_bbox_y = scaled_roi_bboxes.reshape(
        -1, 2).max(axis=0).astype(int)

    # extend the min and max coordinates value to create the slide extraction area
    height, width = slide_arr.shape[:2]

    # first, add 30 pixel padding to the min and max coordinates
    # check if the the points are within the bounds of slide
    extract_min_x = max(0, min_bbox_x - 30)
    extract_min_y = max(0, min_bbox_y - 30)
    extract_max_x = min(width, max_bbox_x + 30)
    extract_max_y = min(height, max_bbox_y + 30)

    # then, pad the coordinates to make the width and height mod step size
    extract_min_coord, extract_max_coord = pad_roi_coordinates(
        (extract_min_x, extract_min_y),
        (extract_max_x, extract_max_y),
        (width, height),
        patch_size,
        step_size
    )

    return extract_min_coord, extract_max_coord


def patchify_area_and_rois(
    slide_arr: np.ndarray,
    extract_area_coord: Tuple[Tuple[int, int], Tuple[int, int]],
    patch_size: int, step_size: int, scaled_bboxes: np.ndarray
) -> List[List[Dict[str, Union[np.ndarray, List[np.ndarray]]]]]:
    """
    Patchify the slide image and retrieve the relative coordinates of the bboxes in the patchified images
    """
    (extract_min_x, extract_min_y) = extract_area_coord[0]
    (extract_max_x, extract_max_y) = extract_area_coord[1]
    extract_area_arr = slide_arr[
        extract_min_y:extract_max_y,
        extract_min_x:extract_max_x,
        :
    ]

    # patchify the slide image, each patch is 512x512, step is 256
    slide_patches = patchify(
        extract_area_arr, (patch_size, patch_size, 3), step=step_size)

    # create a list of dict including patches and the corresponding list of roi in relative coordinates
    area_and_rois = []
    for i in range(slide_patches.shape[0]):
        patch_annots_row = []
        for j in range(slide_patches.shape[1]):
            patch_annots_row.append({
                "patch": slide_patches[i, j, 0],
                "roi_bboxes": []
            })
        area_and_rois.append(patch_annots_row)

    assert (area_and_rois[0][0]["patch"] == slide_patches[0, 0, ::]).all()

    # retrieve relative coordinates of the bboxes in the patchified images
    for scaled_roi_bbox in scaled_bboxes:
        # get the bbox coordinates
        (min_x, min_y), (max_x, max_y) = scaled_roi_bbox

        # get the step index of the bbox
        min_x_step_idx = int((min_x - extract_min_x) // step_size)
        min_y_step_idx = int((min_y - extract_min_y) // step_size)
        max_x_step_idx = int((max_x - extract_min_x) // step_size)
        max_y_step_idx = int((max_y - extract_min_y) // step_size)

        # compile a list of patches that the bbox is in, indicated by the patch indices
        patch_list: List[Tuple[int]] = []
        for x in range(  # the range is computed by the step index of the bbox
            max(0, min_x_step_idx - patch_size//step_size + 1),
            min(slide_patches.shape[1], max_x_step_idx + 1)
        ):
            for y in range(  # the range is computed by the step index of the bbox
                max(0, min_y_step_idx - patch_size // step_size + 1),
                min(slide_patches.shape[0], max_y_step_idx + 1)
            ):
                patch_list.append((y, x))

        for patch_idx in patch_list:
            # get the relative coordinates of the bbox in the patch
            patch_min_x = max(
                0, min_x - extract_min_x - patch_idx[1] * STEP_SIZE)
            patch_min_y = max(
                0, min_y - extract_min_y - patch_idx[0] * STEP_SIZE)
            patch_max_x = min(
                PATCH_SIZE, max_x - extract_min_x - patch_idx[1] * STEP_SIZE)
            patch_max_y = min(
                PATCH_SIZE, max_y - extract_min_y - patch_idx[0] * STEP_SIZE)

            area_and_rois[patch_idx[0]][patch_idx[1]]["roi_bboxes"].append(
                np.array([
                    (patch_min_x, patch_min_y),
                    (patch_max_x, patch_max_y)], dtype=np.float32)
            )

    return area_and_rois


def save_obj_detect_patch_and_roi(
    area_and_rois: List[List[Dict[str, Union[np.ndarray, List[np.ndarray]]]]],
    save_path: str = ".",
    verbose: bool = False
):
    """
    Save the area and rois to a file
    """
    for row_idx, row in enumerate(area_and_rois):
        for col_idx, patch_rois in enumerate(row):
            patch = patch_rois["patch"]
            rois = patch_rois["roi_bboxes"]

            # Save the patch
            os.makedirs(f"{save_path}/patch", exist_ok=True)
            Image.fromarray(patch).save(
                f"{save_path}/patch/{row_idx}_{col_idx}.png"
            )

            # Save the rois
            os.makedirs(f"{save_path}/roi", exist_ok=True)
            np.save(f"{save_path}/roi/{row_idx}_{col_idx}.npy", rois)

    if verbose:
        logging.info(
            f"save_obj_detect_patch_and_roi - Saved patches image and bbox json to {save_path}"
        )


def create_rois_dataset(annot_path: str, slide_folder: str, data_folder: str, set_type: str, patch_size: int, step_size: int):
    file_id = annot_path.split("/")[-1].split(".")[0]
    # create if not exists
    if not f"{slide_folder}/{set_type}/{file_id}" in glob.glob(f"{slide_folder}/{set_type}/{file_id}"):
        logging.info(f"create_roi_dataset - processing: {file_id}")
        wsi_file_paths = glob.glob(
            f"{data_folder}/{set_type}/**/{file_id}.svs",
            recursive=True
        )

        if len(wsi_file_paths) != 0:
            try:
                slide = openslide.OpenSlide(wsi_file_paths[0])
                annotations = geojson.load(open(annot_path))
                bbox_list = create_bbox_list(annotations)

                slide_arr, scaled_bboxes = get_scaled_img_and_rois(
                    slide, bbox_list, level=-1
                )
                slide_arr = np.pad(
                    slide_arr,
                    (
                        (0, STEP_SIZE - slide_arr.shape[0] % STEP_SIZE),
                        (0, STEP_SIZE - slide_arr.shape[1] % STEP_SIZE),
                        (0, 0)
                    ),
                    constant_values=255
                )

                Path(f"{slide_folder}/{set_type}").mkdir(
                    parents=True,
                    exist_ok=True
                )
                patch_to_bboxes_list = patchify_area_and_rois(
                    slide_arr,
                    ((0, 0), slide_arr.shape[:2]),
                    PATCH_SIZE, STEP_SIZE, scaled_bboxes
                )
                save_obj_detect_patch_and_roi(
                    patch_to_bboxes_list,
                    save_path=f"{slide_folder}/{set_type}/{file_id}"
                )
            except Exception as e:
                logging.exception(
                    f"create_roi_dataset - failed to process {file_id}"
                )
                raise e
            finally:
                slide.close()

        gc.collect()
    else:
        logging.info(
            f"create_roi_dataset - {slide_folder}/train/{file_id} already exists"
        )


if __name__ == "__main__":
    setup_logging()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-p", "--project_root",
        help="project root path to export the extracted patch data, e.g. -p /path/to/project/", type=str, default=".", required=True)
    argParser.add_argument(
        "-r", "--raw_data_folder",
        help="raw data folder path, e.g. -r /path/to/raw", type=str, required=True)
    argParser.add_argument(
        "-a", "--annot_folder", help="annotation folder path, e.g. -a /path/to/annot", type=str, required=True)
    argParser.add_argument(
        "-t", "--tile_size", help="tile size, e.g. -t 512", type=int, default=512)
    argParser.add_argument(
        "-s", "--step_size", help="step size, e.g. -s 512", type=int, default=256)
    args = argParser.parse_args()

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    RAW_DATA_FOLDER_PATH = args.raw_data_folder
    ANNOT_PATH = args.annot_folder
    SLIDE_PATH = f"{PROJECT_ROOT}/data/slide_patches"
    OUTPUT_PATH = f"{PROJECT_ROOT}/output/"
    OUTPUT_PLOT_PATH = f"{OUTPUT_PATH}/plots/extract_slides"
    PATCH_SIZE = args.tile_size
    STEP_SIZE = args.step_size

    Path(SLIDE_PATH).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(
        f"{RAW_DATA_FOLDER_PATH}/train/Group_AT/Type_ADH/BRACS_1486.svs"
    )

    logging.info(f"main - levels: {slide.level_count}")
    logging.info(f"main - level dimensions {slide.level_dimensions}")
    logging.info(f"main - level downsamples {slide.level_downsamples}")

    # get annotations
    annotation_file = f"{ANNOT_PATH}/train/Group_AT/Type_ADH/BRACS_1486.geojson"
    annotations = geojson.load(open(annotation_file))
    bbox_list = create_bbox_list(annotations)

    # get slide and scaled rois
    slide_arr, scaled_bboxes = get_scaled_img_and_rois(
        slide, bbox_list, level=-1
    )
    slide_arr = np.pad(
        slide_arr,
        (
            (0, STEP_SIZE - slide_arr.shape[0] % STEP_SIZE),
            (0, STEP_SIZE - slide_arr.shape[1] % STEP_SIZE),
            (0, 0)
        ),
        constant_values=255
    )

    # draw rectangle on the slide
    slide_plot_arr = slide_arr.copy()

    for scaled_roi_bbox in scaled_bboxes:
        min_coord, max_coord = scaled_roi_bbox
        cv2.rectangle(
            slide_plot_arr,
            min_coord.astype(int),
            max_coord.astype(int),
            (0, 0, 0),
            2
        )

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(slide_plot_arr)
    plt.title("Sample Extract Area with ROIs")
    # draw lines at 256 pixel intervals vertically and horizontally to indictate the 256x256 patches
    for i in range(0, slide_plot_arr.shape[1], STEP_SIZE):
        plt.axvline(i, color="black")
    for i in range(0, slide_plot_arr.shape[0], STEP_SIZE):
        plt.axhline(i, color="black")

    plt.savefig(f"{OUTPUT_PLOT_PATH}/extract_slide_cropping_sample.png")
    plt.close()
    logging.info(
        f"main - saved extract slide cropping sample to {OUTPUT_PLOT_PATH}/extract_slide_cropping_sample.png"
    )

    # patchify the slide image and get the relative coordinates of the bboxes in the patchified images
    patch_to_bboxes_list = patchify_area_and_rois(
        slide_arr,
        ((0, 0), slide_arr.shape[:2]),
        PATCH_SIZE, STEP_SIZE, scaled_bboxes
    )

    # save the patches and rois
    save_obj_detect_patch_and_roi(
        patch_to_bboxes_list,
        save_path=f"{SLIDE_PATH}/sample"
    )
    logging.info(
        f"main - saved sample patches and roi bboxes to {SLIDE_PATH}/sample"
    )

    # plot a single patch and its rois
    sample_slide_patch_plot_arr = cv2.imread(
        f"{SLIDE_PATH}/sample/patch/0_0.png"
    )
    sample_slide_patch_plot_arr = cv2.cvtColor(
        sample_slide_patch_plot_arr,
        cv2.COLOR_BGR2RGB
    )

    # draw rectangle on the patch copy
    for annot in np.load(f"{SLIDE_PATH}/sample/roi/0_0.npy"):
        cv2.rectangle(
            sample_slide_patch_plot_arr,
            annot[0].astype(int),
            annot[1].astype(int),
            (0, 0, 0),
            2
        )

    plt.imshow(sample_slide_patch_plot_arr)
    plt.title("Sample Patch with ROIs")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/extract_slide_patch_sample.png")
    plt.close()
    logging.info(
        f"main - saved extract slide patch sample to {OUTPUT_PLOT_PATH}/extract_slide_patch_sample.png"
    )

    # get all annotations
    train_set = glob.glob(f"{ANNOT_PATH}/train/**/*.geojson", recursive=True)
    val_set = glob.glob(f"{ANNOT_PATH}/val/**/*.geojson", recursive=True)

    failed_file_path = Path(
        f"{OUTPUT_PATH}/failed_slide_files-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    )

    logging.info("main - start processing train set")
    for train_annot_path in train_set:
        try:
            create_rois_dataset(
                train_annot_path, SLIDE_PATH,
                RAW_DATA_FOLDER_PATH, "train", PATCH_SIZE, STEP_SIZE
            )
        except:
            file_id = train_annot_path.split("/")[-1].split(".")[0]
            with open(failed_file_path, "a+") as f:
                f.write(f"{file_id}\n")

    logging.info("main - start processing validation set")
    for val_annot_path in val_set:
        try:
            create_rois_dataset(
                val_annot_path, SLIDE_PATH,
                RAW_DATA_FOLDER_PATH, "val", PATCH_SIZE, STEP_SIZE
            )
        except:
            file_id = val_annot_path.split("/")[-1].split(".")[0]
            with open(failed_file_path, "a+") as f:
                f.write(f"{file_id}\n")
