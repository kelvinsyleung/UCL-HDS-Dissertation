import argparse
import glob
from pathlib import Path
import logging
import random
import time

import cv2
import torchstain

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from log_utils import setup_logging
from patch_dataset import PatchDataset
from class_mapping import NAME2TYPELABELS_MAP, LABELS2TYPE_MAP
from train_utils import run_train_loop, seed_worker, plot_history


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
        "-c", "--color_space", help="color space: RGB, CIELAB, or BW e.g. -c RGB", type=str, default="RGB")
    argParser.add_argument(
        "-m", "--mag", help="magnification of patches for training: 20x or 40x, e.g. -m 20x", type=str, default="20x")
    argParser.add_argument(
        "-t", "--transfer_learning", help="transfer learning: True or False, e.g. -t True", type=bool, default=False)
    args = argParser.parse_args()

    MAGNIFICATION = args.mag
    logging.info(
        f"main - MAGNIFICATION: {'mixed' if MAGNIFICATION == '*' else MAGNIFICATION}")

    # color space
    COLOR_SPACE = args.color_space
    logging.info(f"main - COLOR_SPACE: {COLOR_SPACE}")

    # transfer learning
    TRANSFER_LEARNING = args.transfer_learning
    logging.info(f"main - TRANSFER_LEARNING: {TRANSFER_LEARNING}")

    if TRANSFER_LEARNING and COLOR_SPACE != "RGB":
        raise ValueError("Transfer learning only works with RGB color space")

    CLASS_MAP = NAME2TYPELABELS_MAP
    LABEL_MAP = LABELS2TYPE_MAP

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    DATA_PATH = f"{PROJECT_ROOT}/data"
    PATCH_PATH = f"{DATA_PATH}/roi_patches"

    # relative to script execution path
    OUTPUT_PLOT_PATH = f"{PROJECT_ROOT}/output/plots/train_patch_classifier"
    MODEL_SAVEPATH = f"{PROJECT_ROOT}/models/train_patch_classifier"
    PRETRAINED_MODEL_PATH = f"{PROJECT_ROOT}/models/pretrained"

    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVEPATH).mkdir(parents=True, exist_ok=True)

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"pytorch device using: {device}")

    # load dataset
    train_patches_paths = sorted(
        glob.glob(f"{PATCH_PATH}/train/**/*-{'*' if MAGNIFICATION == 'mixed' else MAGNIFICATION}"))

    train_img_path = []
    train_mask_path = []

    for roi in train_patches_paths:
        train_img_path.extend(glob.glob(roi + "/patch/*.png"))
        train_mask_path.extend(glob.glob(roi + "/mask/*.png"))

    train_img_path.sort()
    train_mask_path.sort()

    assert len(train_img_path) == len(
        train_mask_path), "Number of images and masks should be equal"

    logging.info(
        f"main - Number of {'mixed' if MAGNIFICATION == '*' else MAGNIFICATION} train images: {len(train_img_path)}")

    val_patches_paths = sorted(
        glob.glob(f"{PATCH_PATH}/val/**/*-{'*' if MAGNIFICATION == 'mixed' else MAGNIFICATION}"))

    val_img_path = []
    val_mask_path = []

    for roi in val_patches_paths:
        val_img_path.extend(glob.glob(roi + "/patch/*.png"))
        val_mask_path.extend(glob.glob(roi + "/mask/*.png"))

    val_img_path.sort()
    val_mask_path.sort()

    assert len(val_img_path) == len(
        val_mask_path), "Number of images and masks should be equal"

    logging.info(
        f"main - Number of {'mixed' if MAGNIFICATION == '*' else MAGNIFICATION} val images: {len(val_img_path)}")

    # albumentations transforms
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.Rotate([90, 90], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])

    # stain normalisation
    train_norm_img_path = train_img_path[len(train_img_path)//2]
    train_norm_img_arr = cv2.cvtColor(
        cv2.imread(train_norm_img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(train_norm_img_arr)
    plt.title("Image selected for normalisation in training set")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/train_norm_img.png")
    plt.close()
    logging.info(f"main - train_norm_img_path: {train_norm_img_path}")

    val_norm_img_path = val_img_path[len(val_img_path)//2]
    val_norm_img_arr = cv2.cvtColor(
        cv2.imread(val_norm_img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(val_norm_img_arr)
    plt.title("Image selected for normalisation in validation set")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/val_norm_img.png")
    plt.close()
    logging.info(f"main - val_norm_img_path: {val_norm_img_path}")

    train_stain_normaliser = torchstain.normalizers.MacenkoNormalizer(
        backend='numpy')
    train_stain_normaliser.fit(train_norm_img_arr)

    val_stain_normaliser = torchstain.normalizers.MacenkoNormalizer(
        backend='numpy')
    val_stain_normaliser.fit(val_norm_img_arr)

    logging.info("main - stain normalisation setup complete")

    # define the hyperparameters
    LEARNING_RATE = 1e-4
    BATCHSIZE = 32
    EPOCHS = 100
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 4
    WEIGHT_DECAY = 1e-4

    num_classes = len(LABEL_MAP) + 1

    # model
    if TRANSFER_LEARNING:
        model = torchvision.models.resnext50_32x4d()
        model.load_state_dict(torch.load(
            f"{PRETRAINED_MODEL_PATH}/resnext50_32x4d-1a0047aa.pth"))
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    else:
        model = torchvision.models.resnext50_32x4d(num_classes=num_classes)

    logging.info(f"main - model setup complete")
    logging.info(f"main - model: {model}")

    # datasets
    patch_train_dataset = PatchDataset(
        img_paths=train_img_path,
        mask_paths=train_mask_path,
        mode=COLOR_SPACE,
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        stain_normaliser=train_stain_normaliser,
        level="patch",
        transform=train_transform,
        seed=0
    )

    patch_val_dataset = PatchDataset(
        img_paths=val_img_path,
        mask_paths=val_mask_path,
        mode=COLOR_SPACE,
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        stain_normaliser=val_stain_normaliser,
        level="patch",
        transform=val_transform,
        seed=0
    )

    # dataloaders
    worker_g = torch.Generator()
    worker_g.manual_seed(0)

    train_batches = DataLoader(
        patch_train_dataset, batch_size=BATCHSIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    worker_g.manual_seed(0)
    valid_batches = DataLoader(
        patch_val_dataset, batch_size=BATCHSIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )

    # define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_time = time.time()

    set_name = f"resnext_{COLOR_SPACE}_{'mixed' if MAGNIFICATION == '*' else MAGNIFICATION}{'_transfer_learning' if TRANSFER_LEARNING else ''}"

    def eval_fn(output, targets, num_classes):
        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(
            output, dim=1).long(), targets, mode="multiclass", num_classes=num_classes)
        return smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

    # train the network
    history = run_train_loop(
        model, num_classes, device,
        train_batches, valid_batches,
        EPOCHS, criterion, optimizer,
        set_name, eval_fn, model_type="classification",
        save_interval=50, save_path=MODEL_SAVEPATH
    )
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        f"{MODEL_SAVEPATH}/{set_name}_model_final.pth"
    )

    # display the total time needed to perform the training
    end_time = time.time()
    logging.info(
        f"main - Total time taken to train the {set_name} model: {(end_time - start_time):.2f}s")

    # plot the loss and accuracy history
    plot_history(
        history, save_path=f"{OUTPUT_PLOT_PATH}/{set_name}_history.png", model_type="classification")
