import argparse
import glob
import logging
from pathlib import Path
import random

import cv2

import numpy as np
import matplotlib.pyplot as plt

import torchstain

from log_utils import setup_logging

if __name__ == "__main__":
    setup_logging()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-p", "--project_root",
        help="project root path to export the extracted patch data, e.g. -p /path/to/project/", type=str, default=".", required=True)
    args = argParser.parse_args()

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    PATCH_PATH = f"{PROJECT_ROOT}/data/roi_patches"
    NORM_PATH = f"{PROJECT_ROOT}/data/norms"
    OUTPUT_PATH = f"{PROJECT_ROOT}/output/"
    OUTPUT_PLOT_PATH = f"{OUTPUT_PATH}/plots/norm_values"

    Path(NORM_PATH).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)

    # load dataset
    train_patches_paths = sorted(glob.glob(f"{PATCH_PATH}/train/**/*-*"))

    train_img_path = []
    train_mask_path = []

    for roi in train_patches_paths:
        train_img_path.extend(glob.glob(roi + "/patch/*.png"))
        train_mask_path.extend(glob.glob(roi + "/mask/*.png"))

    train_img_path.sort()
    train_mask_path.sort()

    assert len(train_img_path) == len(
        train_mask_path), "Number of images and masks should be equal"

    logging.info(f"Number of train patch images: {len(train_img_path)}")

    norm_img_selected = False
    img_idx = len(train_img_path)//2
    while not norm_img_selected:
        train_norm_img_path = train_img_path[img_idx]
        bw_img = cv2.cvtColor(cv2.imread(
            train_norm_img_path), cv2.COLOR_BGR2GRAY)
        norm_img_selected = bw_img.mean() <= 200
        img_idx = (img_idx + 1) % len(train_img_path)

    norm_img_arr = cv2.cvtColor(cv2.imread(
        train_norm_img_path), cv2.COLOR_BGR2RGB)
    logging.info(f"main - train_norm_img_path: {train_norm_img_path}")
    plt.imshow(norm_img_arr)
    plt.title("Image selected for stain normalisation from training set")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/norm_img.png")

    np.save(f"{NORM_PATH}/stain_norm_img.npy", norm_img_arr)
    logging.info("Saved image selected for stain normalisation")

    stain_normaliser = torchstain.normalizers.MacenkoNormalizer(
        backend='numpy')
    stain_normaliser.fit(norm_img_arr)

    # calculate the mean and standard deviation of channels for reference
    num_img_sampled = 0
    random.seed(0)
    rand_img_path_list = random.sample(train_img_path, k=len(train_img_path))

    # list of sampled images of the colour map
    rgb_pixel_list = []
    cielab_pixel_list = []

    for img_path in rand_img_path_list:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        try:
            img, _, _ = stain_normaliser.normalize(img)
        except Exception as e:
            logging.error(
                f"Error in normalising image: {img_path}")
            logging.error(e)
            continue

        # add the image to the list of images of the colour map list
        rgb_pixel_list.append(img.reshape(-1, img.shape[-1]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        cielab_pixel_list.append(img.reshape(-1, img.shape[-1]))
        num_img_sampled += 1

        if num_img_sampled == 1000:
            logging.info("Sampled 1000 images for calculating mean and std")
            break

    flatten_rgb_pixel_array = np.concatenate(rgb_pixel_list, axis=0)
    flatten_cielab_pixel_array = np.concatenate(cielab_pixel_list, axis=0)

    rgb_mean = np.mean(flatten_rgb_pixel_array, axis=0) / 255.0
    rgb_std = np.std(flatten_rgb_pixel_array, axis=0) / 255.0
    logging.info(f"rgb_mean: {rgb_mean}")
    logging.info(f"rgb_std: {rgb_std}")

    cielab_mean = np.mean(flatten_cielab_pixel_array, axis=0) / 255.0
    cielab_std = np.std(flatten_cielab_pixel_array, axis=0) / 255.0
    logging.info(f"cielab_mean: {cielab_mean}")
    logging.info(f"cielab_std: {cielab_std}")

    np.save(f"{NORM_PATH}/rgb_mean.npy", rgb_mean)
    np.save(f"{NORM_PATH}/rgb_std.npy", rgb_std)
    np.save(f"{NORM_PATH}/cielab_mean.npy", cielab_mean)
    np.save(f"{NORM_PATH}/cielab_std.npy", cielab_std)
    logging.info("Saved mean and std values for RGB and CIELAB channels")
