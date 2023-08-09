from typing import Dict
import random
import logging

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from tqdm import tqdm

def run_train_loop(
        model: nn.Module, num_classes:int, device: torch.device,
        train_batches: DataLoader, valid_batches: DataLoader, train_set_len: int, val_set_len: int,
        epochs: int, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
        set_name: str, save_interval: int=50
    ):
    """
    Train the model for the given number of epochs and save the model after every save_interval epochs.
    
    Parameters
    ----------
        model: nn.Module
            The model to train.
        num_classes: int
            The number of classes in the dataset.
        device: torch.device
            The device to use for training.
        train_batches: DataLoader
            The training data loader.
        valid_batches: DataLoader
            The validation data loader.
        train_set_len: int
            The length of the training set.
        val_set_len: int
            The length of the validation set.
        epochs: int
            The number of epochs to train the model for.
        criterion: torch.nn.Module
            The loss function to use.
        optimizer: torch.optim.Optimizer
            The optimizer to use.
        set_name: str
            The name of the dataset.
        save_interval: int
            The interval at which to save the model.
    """
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": []
    }

    for epoch in range(epochs):
        # set the model in training phase
        model.train()

        total_train_loss = 0
        total_val_loss = 0
        train_dice = 0
        val_dice = 0

        for data, targets in tqdm(train_batches):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = model(data)
            loss = criterion(output, targets)

            # backward
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            total_train_loss += loss.detach()*data.size(0)

            # evaluate dice coefficient on test set
            tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output, dim=1).long(), targets, mode="multiclass", num_classes=num_classes)
            train_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # set the model in evaluation phase
        with torch.no_grad():
            model.eval()

            for data, targets in valid_batches:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # forward
                output = model(data)
                loss = criterion(output, targets)

                total_val_loss += loss.detach()*data.size(0)
                
                # evaluate dice coefficient on test set
                tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output, dim=1).long(), targets, mode="multiclass", num_classes=num_classes)
                val_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # compute the average loss and accuracy
        avg_train_loss = total_train_loss.cpu() / train_set_len
        avg_val_loss = total_val_loss.cpu() / val_set_len
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        if (epoch+1) % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                f"{set_name}_model_epoch{epoch+1}.pth"
            )

        # print the loss and accuracy for the epoch
        logging.info(f"train_utils - Epoch {(epoch+1)}/{epochs} Train Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}, Train Dice Score: {train_dice:.4f} Validation Dice Score: {val_dice:.4f}")

    return history

def seed_worker(worker_id):
    """
    Seed a worker for reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def plot_history(history, save_path: str):
    """
    Plot the training history.

    Parameters
    ----------
        history: Dict
            The training history.
        save_path: str
            The path to save the plot.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.title("Loss vs epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_dice"], label="train dice score")
    plt.plot(history["val_dice"], label="validation dice score")
    plt.title("Dice score vs epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.savefig(save_path)
