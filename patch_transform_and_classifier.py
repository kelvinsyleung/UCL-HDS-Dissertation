from typing import Union
import random
import os
from pathlib import Path
import glob
import time
import gc
import logging
import sys

import numpy as np
from matplotlib import pyplot as plt
import cv2

import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader

from class_mapping import NAME2SUBTYPELABELS_MAP, LABELS2SUBTYPE_MAP, NAME2TYPELABELS_MAP, LABELS2TYPE_MAP
from patch_dataset import PatchDataset
from model import UNet
from train_utils import run_train_loop, seed_worker, plot_history

root = logging.getLogger()
root.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")

logging_handler_out = logging.StreamHandler(sys.stdout)
logging_handler_out.setLevel(logging.INFO)
logging_handler_out.setFormatter(formatter)
root.addHandler(logging_handler_out)

logging_handler_err = logging.StreamHandler(sys.stderr)
logging_handler_err.setLevel(logging.ERROR)
logging_handler_err.setFormatter(formatter)
root.addHandler(logging_handler_err)

OPENSLIDE_PATH  = r"C:/openslide/openslide-win64/bin"

import os
if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

HOME_PATH = os.path.expanduser("~")
PATCH_PATH = "./data/patches"
OUTPUT_PLOT_PATH = "./output/plots"
OUTPUT_MODEL_PATH = "./output/models"

Path(OUTPUT_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"pytorch device using: {device}")

def cielab_intensify_to_rgb(img: Union[np.ndarray, torch.Tensor], rate: float):
    cie_img = np.array(img)
    if isinstance(img, torch.Tensor):
        cie_img = cie_img * 255
    cie_img[:, :, 0][cie_img[:, :, 0] > 127] = np.clip(cie_img[:, :, 0][cie_img[:, :, 0] > 127] * (1 + rate), 0, 255)
    cie_img[:, :, 0][cie_img[:, :, 0] < 127] = np.clip(cie_img[:, :, 0][cie_img[:, :, 0] < 127] * (1 - rate), 0, 255)

    cie_img[:, :, 1][cie_img[:, :, 1] > 127] = np.clip(cie_img[:, :, 1][cie_img[:, :, 1] > 127] * (1 - rate), 0, 255)
    cie_img[:, :, 1][cie_img[:, :, 1] < 127] = np.clip(cie_img[:, :, 1][cie_img[:, :, 1] < 127] * (1 + rate), 0, 255)

    cie_img[:, :, 2][cie_img[:, :, 2] > 127] = np.clip(cie_img[:, :, 2][cie_img[:, :, 2] > 127] * (1 - rate), 0, 255)
    cie_img[:, :, 2][cie_img[:, :, 2] < 127] = np.clip(cie_img[:, :, 2][cie_img[:, :, 2] < 127] * (1 + rate), 0, 255)
    
    cie_img = cie_img.astype(np.uint8)
    rgb_img = cv2.cvtColor(cie_img, cv2.COLOR_LAB2RGB)
    return rgb_img


if __name__ == "__main__":
    train_sample_patches_paths = glob.glob(PATCH_PATH + "/train_sample/*")
    test_img_path = ""
    test_mask_path = ""
    for roi in glob.glob(train_sample_patches_paths[0] + "**/**"):
        img_paths = glob.glob(roi + "/patch/*.png")
        mask_paths = glob.glob(roi + "/mask/*.png")
        test_img_path = img_paths[0]
        test_mask_path = mask_paths[0]
        img = cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_paths[0], cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.savefig(f"{OUTPUT_PLOT_PATH}/read_patch_example.png")
        break

    test_img = cv2.imread(test_img_path, cv2.IMREAD_UNCHANGED)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGRA2RGB)
    cielab_test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2LAB)
    bw_test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))


    logging.info(f"main - RGB image shape: {test_img.shape}")
    logging.info(f"main - RGB example pixel values: {test_img[200, 100, :]}")

    logging.info(f"main - CIELAB image shape: {cielab_test_img.shape}")
    logging.info(f"main - CIELAB example pixel values:{cielab_test_img[200, 100, :]}")

    logging.info(f"main - Black and white image shape: {bw_test_img.shape}")
    logging.info(f"main - Black and white example pixel values: {bw_test_img[200, 100]}")

    ax[0].imshow(test_img)
    ax[1].imshow(cielab_test_img)
    ax[2].imshow(bw_test_img, cmap="gray")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/colorspace_example.png")
    logging.info(f"main - saved colorspace example plot to {OUTPUT_PLOT_PATH}/colorspace_example.png")

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))

    ax[0].imshow(test_img)
    ax[0].set_title("Original image")
    ax[1].imshow(cielab_intensify_to_rgb(cielab_test_img, 0.1))
    ax[1].set_title("Intensified image")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/cielab_intensify_example.png")
    logging.info(f"main - saved cielab intensify example plot to {OUTPUT_PLOT_PATH}/cielab_intensify_example.png")

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

    # load dataset
    train_20x_patches_paths = sorted(glob.glob(PATCH_PATH + "/train/**/*-20x"))

    set(map(lambda x: "-".join(os.path.split(x)[-1].split("-")[:-2]), train_20x_patches_paths))

    train_20x_img_path = []
    train_20x_mask_path = []

    for roi in train_20x_patches_paths:
        train_20x_img_path.extend(glob.glob(roi + "/patch/*.png"))
        train_20x_mask_path.extend(glob.glob(roi + "/mask/*.png"))

    train_20x_img_path.sort()
    train_20x_mask_path.sort()

    assert len(train_20x_img_path) == len(train_20x_mask_path), "Number of images and masks should be equal"

    logging.info(f"main - Number of 20x train images: {len(train_20x_img_path)}")

    with open('data/train_20x_img_paths.txt', 'w+') as f:
        f.write('\n'.join(train_20x_img_path))

    with open('data/train_20x_mask_paths.txt', 'w+') as f:
        f.write('\n'.join(train_20x_mask_path))

    val_20x_patches_paths = sorted(glob.glob(PATCH_PATH + "/val/**/*-20x"))

    val_20x_img_path = []
    val_20x_mask_path = []

    for roi in val_20x_patches_paths:
        val_20x_img_path.extend(glob.glob(roi + "/patch/*.png"))
        val_20x_mask_path.extend(glob.glob(roi + "/mask/*.png"))

    val_20x_img_path.sort()
    val_20x_mask_path.sort()

    assert len(val_20x_img_path) == len(val_20x_mask_path), "Number of images and masks should be equal"

    logging.info(f"main - Number of 20x val images: {len(val_20x_img_path)}")

    with open('data/val_20x_img_paths.txt', 'w+') as f:
        f.write('\n'.join(val_20x_img_path))

    with open('data/val_20x_mask_paths.txt', 'w+') as f:
        f.write('\n'.join(val_20x_mask_path))

    # test 20x datasets creation
    patch_rgb_20x_8cls_dataset = PatchDataset(
        imgPaths=train_20x_img_path,
        maskPaths=train_20x_mask_path,
        mode="RGB",
        name_to_class_mapping=NAME2SUBTYPELABELS_MAP,
        transform=train_transform,
        seed=0
    )

    patch_cielab_20x_8cls_dataset = PatchDataset(
        imgPaths=train_20x_img_path,
        maskPaths=train_20x_mask_path,
        mode="CIELAB",
        name_to_class_mapping=NAME2SUBTYPELABELS_MAP,
        transform=train_transform,
        seed=0
    )

    patch_bw_20x_8cls_dataset = PatchDataset(
        imgPaths=train_20x_img_path,
        maskPaths=train_20x_mask_path,
        mode="BW",
        name_to_class_mapping=NAME2SUBTYPELABELS_MAP,
        transform=train_transform,
        seed=0
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].title.set_text("RGB")
    rgb_img, rgb_mask = patch_rgb_20x_8cls_dataset[0]
    ax[0].imshow(rgb_img.permute(1, 2, 0))
    ax[0].imshow(rgb_mask > 0, alpha=0.3, cmap="gray")

    ax[1].title.set_text("CIELAB intensified image to RGB")
    cielab_img, cielab_mask = patch_cielab_20x_8cls_dataset[0]
    ax[1].imshow(cielab_intensify_to_rgb(cielab_img.permute(1, 2, 0), 0.1))
    ax[1].imshow(cielab_mask > 0, alpha=0.3, cmap="gray")

    ax[2].title.set_text("BW")
    bw_img, bw_mask = patch_bw_20x_8cls_dataset[0]
    ax[2].imshow(bw_img.permute(1, 2, 0), cmap="gray")
    ax[2].imshow(bw_mask > 0, alpha=0.3, cmap="gray")

    plt.savefig(f"{OUTPUT_PLOT_PATH}/data_augmentation_example.png")
    logging.info(f"main - Data augmentation example saved to {OUTPUT_PLOT_PATH}/data_augmentation_example.png")

    rgb_20x_loader = DataLoader(patch_rgb_20x_8cls_dataset, batch_size=8, shuffle=True)
    cielab_20x_loader = DataLoader(patch_cielab_20x_8cls_dataset, batch_size=8, shuffle=True)
    bw_20x_loader = DataLoader(patch_bw_20x_8cls_dataset, batch_size=8, shuffle=True)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    img, mask = next(iter(rgb_20x_loader))

    plt.title("RGB 20x Data loader first batch first image")
    plt.imshow(img[0].permute(1, 2, 0))
    plt.imshow(mask[0] > 0, alpha=0.3, cmap="gray")
    plt.savefig(f"{OUTPUT_PLOT_PATH}/rgb_dataloader_example.png")


    torch.cuda.empty_cache()
    gc.collect()

    # define the hyperparameters
    LEARNING_RATE = 1e-4
    BATCHSIZE = 32
    EPOCHS = 300
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 4

    # RGB 20x 3cls
    # model
    model = UNet(in_channels=3, out_channels=4)

    # datasets
    patch_rgb_20x_3cls_train_dataset = PatchDataset(
        imgPaths=train_20x_img_path,
        maskPaths=train_20x_mask_path,
        mode="RGB",
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        transform=train_transform,
        seed=0
    )

    patch_rgb_20x_3cls_val_dataset = PatchDataset(
        imgPaths=val_20x_img_path,
        maskPaths=val_20x_mask_path,
        mode="RGB",
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        transform=val_transform,
        seed=0
    )

    # dataloaders
    worker_g = torch.Generator()
    worker_g.manual_seed(0)

    train_batches = DataLoader(
        patch_rgb_20x_3cls_train_dataset, batch_size=BATCHSIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    worker_g.manual_seed(0)
    valid_batches = DataLoader(
        patch_rgb_20x_3cls_val_dataset, batch_size=BATCHSIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    
    # define the loss function and the optimizer
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    set_name = "unet_rgb_20x_8=3cls"
    # train the network
    history = run_train_loop(
        model, 8, device,
        train_batches, valid_batches, len(patch_rgb_20x_3cls_train_dataset), len(patch_rgb_20x_3cls_val_dataset),
        EPOCHS, criterion, optimizer, set_name, save_interval=50
    )
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        f"{OUTPUT_MODEL_PATH}/{set_name}_model_final.pth"
    )

    # display the total time needed to perform the training
    end_time = time.time()
    logging.info(f"main - Total time taken to train the {set_name} model: {(end_time - start_time):.2f}s")

    # plot the loss and accuracy history
    plot_history(history, save_path=f"{OUTPUT_PLOT_PATH}/{set_name}_history.png")

    # CIELAB 20x 3cls
    # model
    model = UNet(in_channels=3, out_channels=4)

    # datasets
    patch_cielab_20x_3cls_train_dataset = PatchDataset(
        imgPaths=train_20x_img_path,
        maskPaths=train_20x_mask_path,
        mode="CIELAB",
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        transform=train_transform,
        seed=0
    )

    patch_cielab_20x_3cls_val_dataset = PatchDataset(
        imgPaths=val_20x_img_path,
        maskPaths=val_20x_mask_path,
        mode="CIELAB",
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        transform=val_transform,
        seed=0
    )

    # dataloaders
    worker_g = torch.Generator()
    worker_g.manual_seed(0)

    train_batches = DataLoader(
        patch_cielab_20x_3cls_train_dataset, batch_size=BATCHSIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    worker_g.manual_seed(0)
    valid_batches = DataLoader(
        patch_cielab_20x_3cls_val_dataset, batch_size=BATCHSIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    
    # define the loss function and the optimizer
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    set_name = "unet_cielab_20x_3cls"
    # train the network
    history = run_train_loop(
        model, 8, device,
        train_batches, valid_batches, len(patch_cielab_20x_3cls_train_dataset), len(patch_cielab_20x_3cls_val_dataset),
        EPOCHS, criterion, optimizer, set_name, save_interval=50
    )
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        f"{OUTPUT_MODEL_PATH}/{set_name}_model_final.pth"
    )

    # display the total time needed to perform the training
    end_time = time.time()
    logging.info(f"main - Total time taken to train the {set_name} model: {(end_time - start_time):.2f}s")

    # plot the loss and accuracy history
    plot_history(history, save_path=f"{OUTPUT_PLOT_PATH}/{set_name}_history.png")


    # BW 20x 3cls
    # model
    model = UNet(in_channels=1, out_channels=4)

    # datasets
    patch_bw_20x_3cls_train_dataset = PatchDataset(
        imgPaths=train_20x_img_path,
        maskPaths=train_20x_mask_path,
        mode="BW",
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        transform=train_transform,
        seed=0
    )

    patch_bw_20x_3cls_val_dataset = PatchDataset(
        imgPaths=val_20x_img_path,
        maskPaths=val_20x_mask_path,
        mode="BW",
        name_to_class_mapping=NAME2TYPELABELS_MAP,
        transform=val_transform,
        seed=0
    )

    # dataloaders
    worker_g = torch.Generator()
    worker_g.manual_seed(0)

    train_batches = DataLoader(
        patch_bw_20x_3cls_train_dataset, batch_size=BATCHSIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    worker_g.manual_seed(0)
    valid_batches = DataLoader(
        patch_bw_20x_3cls_val_dataset, batch_size=BATCHSIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    
    # define the loss function and the optimizer
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    set_name = "unet_cielab_20x_3cls"
    # train the network
    history = run_train_loop(
        model, 8, device,
        train_batches, valid_batches, len(patch_bw_20x_3cls_train_dataset), len(patch_bw_20x_3cls_val_dataset),
        EPOCHS, criterion, optimizer, set_name, save_interval=50
    )
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        f"{OUTPUT_MODEL_PATH}/{set_name}_model_final.pth"
    )

    # display the total time needed to perform the training
    end_time = time.time()
    logging.info(f"main - Total time taken to train the {set_name} model: {(end_time - start_time):.2f}s")

    # plot the loss and accuracy history
    plot_history(history, save_path=f"{OUTPUT_PLOT_PATH}/{set_name}_history.png")
