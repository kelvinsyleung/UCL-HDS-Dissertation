import argparse
import glob
from pathlib import Path
import logging
import random
import time

import torchstain

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import albumentations as A
from albumentations.pytorch import ToTensorV2


from patch_dataset import SlideROIDataset
from train_utils import run_train_loop, seed_worker, plot_history
from log_utils import setup_logging


def collate_fn(batch):
    return tuple(zip(*batch))


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
        "-c", "--colour_space", help="colour space: RGB, CIELAB e.g. -c RGB", type=str, default="RGB")
    args = argParser.parse_args()

    # colour space
    COLOUR_SPACE = args.colour_space
    logging.info(f"main - COLOUR_SPACE: {COLOUR_SPACE}")

    if COLOUR_SPACE not in ["RGB", "CIELAB"]:
        raise ValueError("Invalid colour space")

    # absolute path for loading patches
    PROJECT_ROOT = args.project_root
    DATA_PATH = f"{PROJECT_ROOT}/data"
    SLIDE_PATH = f"{DATA_PATH}/slide_patches"
    NORM_PATH = f"{DATA_PATH}/norms"

    # relative to script execution path
    OUTPUT_PLOT_PATH = f"{PROJECT_ROOT}/output/plots/train_obj_detector"
    MODEL_SAVEPATH = f"{PROJECT_ROOT}/models/train_obj_detector"

    Path(OUTPUT_PLOT_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVEPATH).mkdir(parents=True, exist_ok=True)

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"pytorch device using: {device}")

    # load dataset
    train_slide_feat_paths = sorted(glob.glob(f"{SLIDE_PATH}/train/**"))

    train_img_path = []
    train_roi_path = []

    for slide_feat in train_slide_feat_paths:
        train_img_path.extend(glob.glob(slide_feat + "/patch/*.png"))
        train_roi_path.extend(glob.glob(slide_feat + "/roi/*.npy"))

    train_img_path.sort()
    train_roi_path.sort()

    assert len(train_img_path) == len(
        train_roi_path), "Number of images and rois should be equal"

    logging.info(f"Number of train images: {len(train_img_path)}")

    val_slide_feat_paths = sorted(glob.glob(f"{SLIDE_PATH}/val/**"))

    val_img_path = []
    val_roi_path = []

    for slide_feat in val_slide_feat_paths:
        val_img_path.extend(glob.glob(slide_feat + "/patch/*.png"))
        val_roi_path.extend(glob.glob(slide_feat + "/roi/*.npy"))

    val_img_path.sort()
    val_roi_path.sort()

    assert len(val_img_path) == len(
        val_roi_path), "Number of images and rois should be equal"

    logging.info(f"Number of val images: {len(val_img_path)}")

    if COLOUR_SPACE == "RGB":
        mean = np.load(f"{NORM_PATH}/rgb_mean.npy")
        std = np.load(f"{NORM_PATH}/rgb_std.npy")
        logging.info("main - RGB mean and std loaded")
    elif COLOUR_SPACE == "CIELAB":
        mean = np.load(f"{NORM_PATH}/cielab_mean.npy")
        std = np.load(f"{NORM_PATH}/cielab_std.npy")
        logging.info("main - CIELAB mean and std loaded")

    # albumentations transforms
    train_transform = A.Compose([
        A.Rotate([90, 90], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

    val_transform = A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

    # stain normalisation
    norm_img_arr = np.load(f"{NORM_PATH}/stain_norm_img.npy")
    stain_normaliser = torchstain.normalizers.MacenkoNormalizer(
        backend='numpy')
    stain_normaliser.fit(norm_img_arr)

    logging.info("main - stain normalisation setup complete")

    # define the hyperparameters
    LEARNING_RATE = 1e-4
    BATCHSIZE = 16
    EPOCHS = 100
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    WEIGHT_DECAY = 1e-4

    # model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        image_mean=mean, image_std=std, num_classes=2, pretrained_backbone=None)

    logging.info(f"main - model setup complete")
    logging.info(f"main - model: {model}")

    # datasets
    train_dataset = SlideROIDataset(
        img_paths=train_img_path,
        roi_paths=train_roi_path,
        mode=COLOUR_SPACE,
        stain_normaliser=stain_normaliser,
        transform=train_transform
    )

    val_dataset = SlideROIDataset(
        img_paths=val_img_path,
        roi_paths=val_roi_path,
        mode=COLOUR_SPACE,
        stain_normaliser=stain_normaliser,
        transform=val_transform
    )

    # dataloaders
    worker_g = torch.Generator()
    worker_g.manual_seed(0)

    train_batches = DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, collate_fn=collate_fn,
        pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )
    worker_g.manual_seed(0)
    valid_batches = DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, collate_fn=collate_fn,
        pin_memory=True, prefetch_factor=PREFETCH_FACTOR
    )

    # define the loss function and the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5)

    start_time = time.time()

    set_name = f"fasterrcnn_resnet50_fpn_{COLOUR_SPACE}"
    # train the network
    history = run_train_loop(
        model, 2, device,
        train_batches, valid_batches,
        EPOCHS, criterion=None, optimizer=optimizer, scheduler=scheduler,
        set_name=set_name, model_type="detection",
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
        history, save_path=f"{OUTPUT_PLOT_PATH}/{set_name}_history.png", model_type="detection")
