from typing import Union
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
        model: nn.Module, num_classes: int, device: torch.device,
        train_batches: DataLoader, valid_batches: DataLoader,
        epochs: int, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
        eval_fn: callable,
        set_name: str,
        patience: int=10, save_interval: Union[int, None]=50, save_path: str="./models/"
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
        epochs: int
            The number of epochs to train the model for.
        criterion: torch.nn.Module
            The loss function to use.
        optimizer: torch.optim.Optimizer
            The optimizer to use.
        set_name: str
            The name of the dataset.
        patience: int
            The number of epochs to wait before stopping training if the validation loss does not improve.
        save_interval: Union[int, None]
            The interval at which to save extra models. None to save only the best model.
        save_path: str
            The path to save the models.
    """
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_score": [],
        "val_score": []
    }

    best_model_val_loss = np.inf
    best_model_val_score = 0
    best_model_epoch = 0

    for epoch in range(epochs):
        # set the model in training phase
        model.train()

        total_train_loss = 0
        total_val_loss = 0
        train_score = 0
        val_score = 0

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

            total_train_loss += loss.item()

            # evaluate score on test set
            train_score = eval_fn(output, targets, num_classes=num_classes)

        # set the model in evaluation phase
        with torch.no_grad():
            model.eval()

            for data, targets in valid_batches:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # forward
                output = model(data)
                loss = criterion(output, targets)

                total_val_loss += loss.item()
                
                # evaluate score on test set
                val_score = eval_fn(output, targets, num_classes=num_classes)

        # compute the average loss and accuracy
        avg_train_loss = total_train_loss / len(train_batches)
        avg_val_loss = total_val_loss / len(valid_batches)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_score"].append(train_score)
        history["val_score"].append(val_score)

        if save_interval and (epoch+1) % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                f"{save_path}/{set_name}_model_epoch{epoch+1}.pth"
            )

        if avg_val_loss < best_model_val_loss:
            best_model_val_score = val_score
            best_model_val_loss = avg_val_loss
            best_model_epoch = epoch
            torch.save(
                {
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                f"{save_path}/{set_name}_best_model.pth"
            )

        if epoch > best_model_epoch + patience:
            logging.info(f"train_utils - Early stopping at epoch {epoch+1}, best model at epoch {best_model_epoch+1}")
            logging.info(f"train_utils - Best model validation loss: {best_model_val_loss:.4f}, validation score: {best_model_val_score:.4f}")
            break

        # print the loss and accuracy for the epoch
        logging.info(f"train_utils - Epoch {(epoch+1)}/{epochs} Train Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}, Train Score: {train_score:.4f} Validation Score: {val_score:.4f}")

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
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train loss")
    plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="validation loss")
    plt.title("Loss vs epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history["train_score"]) + 1), history["train_score"], label="train score")
    plt.plot(range(1, len(history["val_score"]) + 1), history["val_score"], label="validation score")
    plt.title("score vs epochs")
    plt.xlabel("Epochs")
    plt.ylabel("score")
    plt.legend()
    plt.savefig(save_path)
