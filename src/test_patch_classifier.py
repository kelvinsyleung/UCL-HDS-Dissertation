import argparse
import glob
from pathlib import Path
import logging

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

import torch
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm import tqdm

import torchstain
from patchify import patchify
import cv2

from log_utils import setup_logging

if __name__ == "__main__":
    setup_logging()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-p", "--project_root", help="project root path, e.g. -p /path/to/data", type=str, default=".", required=True)
    argParser.add_argument(
        "-c", "--colour_space", help="colour space: RGB, CIELAB e.g. -c RGB", type=str, default="RGB")
    argParser.add_argument(
        "-m", "--mag", help="magnification of patches for training: 20x or 40x, e.g. -m 20x", type=str, default="20x")
    args = argParser.parse_args()

    MAGNIFICATION = args.mag
    logging.info(
        f"main - MAGNIFICATION: {'mixed' if MAGNIFICATION == '*' else MAGNIFICATION}")

    # colour space
    COLOUR_SPACE = args.colour_space
    logging.info(f"main - COLOUR_SPACE: {COLOUR_SPACE}")

    if COLOUR_SPACE not in ["RGB", "CIELAB"]:
        raise ValueError("Invalid colour space")

    class_map = {
        "0": 0, "1": 0, "2": 0, "3": 1, "4": 1, "5": 2, "6": 2
    }

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    DATA_PATH = f"{PROJECT_ROOT}/data"
    ROI_TEST_PATH = f"{DATA_PATH}/roi_test_imgs"
    NORM_PATH = f"{DATA_PATH}/norms"

    # relative to script execution path
    OUTPUT_PLOT_PATH = f"{PROJECT_ROOT}/output/plots/test_patch_classifier"

    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"pytorch device using: {device}")

    # load dataset
    test_img_paths = sorted(glob.glob(f"{ROI_TEST_PATH}/*/*.png"))
    assert len(test_img_paths) > 0, "No test images found"

    test_class_gt = list(map(lambda path: class_map[Path(
        path).parent.stem.split("_")[0]], test_img_paths))

    raw_imgs_and_gt = []
    for img_path, class_gt in zip(test_img_paths, test_class_gt):
        img = cv2.imread(img_path)
        raw_imgs_and_gt.append((img, class_gt))

    if COLOUR_SPACE == "RGB":
        mean = np.load(f"{NORM_PATH}/rgb_mean.npy")
        std = np.load(f"{NORM_PATH}/rgb_std.npy")
        logging.info("main - RGB mean and std loaded")
    elif COLOUR_SPACE == "CIELAB":
        mean = np.load(f"{NORM_PATH}/cielab_mean.npy")
        std = np.load(f"{NORM_PATH}/cielab_std.npy")
        logging.info("main - CIELAB mean and std loaded")

    stain_norm_img = np.load(f"{NORM_PATH}/stain_norm_img.npy")

    stain_normaliser = torchstain.normalizers.MacenkoNormalizer("numpy")
    stain_normaliser.fit(stain_norm_img)

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    bag_of_imgs = []

    for img, gt in raw_imgs_and_gt:
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        raw_img_bag = patchify(
            img,
            (
                min(1024, h),
                min(1024, w),
                3
            ),
            1024
        ).squeeze(2)

        raw_img_bag = raw_img_bag.reshape(
            (raw_img_bag.shape[0]*raw_img_bag.shape[1]), *raw_img_bag.shape[2:])
        bag_of_imgs.append((raw_img_bag, gt))

    model = torchvision.models.resnext101_32x8d(num_classes=3)
    model.load_state_dict(
        torch.load(
            f"{PROJECT_ROOT}/models/train_patch_classifier/weighted_resnext_101_{COLOUR_SPACE}_{'mixed' if MAGNIFICATION == '*' else MAGNIFICATION}_best_model.pth")[
                "model_state_dict"
        ]
    )

    model.to(device)
    model.eval()

    preds = []
    preds_gt = []
    with torch.no_grad():
        for bag_of_img, gt in tqdm(bag_of_imgs):
            bag = torch.Tensor(bag_of_img.shape[0], 3, 256, 256)
            for i, img in enumerate(bag_of_img):
                try:
                    img, _, _ = stain_normaliser.normalize(img)
                except:
                    pass

                if COLOUR_SPACE == "CIELAB":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                bag[i] = transform(image=img)["image"]

            output = model(bag.to(device))
            preds.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())
            preds_gt.extend([gt]*len(bag_of_img))

    ConfusionMatrixDisplay.from_predictions(
        preds_gt,
        preds,
        xticks_rotation="vertical",
        cmap=plt.cm.Blues
    )
    plt.savefig(f"{OUTPUT_PLOT_PATH}/cm_{COLOUR_SPACE}_{MAGNIFICATION}.png")
    print(classification_report(
        preds_gt,
        preds,
        digits=4
    ))
