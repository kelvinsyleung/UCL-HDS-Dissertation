from typing import Literal, Union, Dict, List, Tuple
import random
import logging

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

def forward_step(
        model: nn.Module, num_classes: int, device: torch.device,
        criterion: torch.nn.Module, eval_fn: callable, model_type: str,
        metrics: Dict[str, float], forward_type: Literal["train", "val"],
        data: torch.Tensor, targets: Union[torch.Tensor, Tuple[Dict[str, torch.Tensor]]]
    ):
    if model_type != "detection":
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        data = [d.to(device, non_blocking=True) for d in data]
        targets = [
            {
                k: v.to(device, non_blocking=True) for k, v in t.items()
            } for t in targets
        ]

    loss = None
    if model_type != "detection":
        output = model(data)
        loss = criterion(output, targets)
    else:
        loss_dict = model(data, targets)
        loss = sum(loss for loss in loss_dict.values())

    metrics[f"total_{forward_type}_loss"] += loss.item()

    if model_type != "detection":
        metrics[f"total_{forward_type}_score"] += eval_fn(output, targets, num_classes=num_classes)
    return loss

def train_one_epoch(
        model: nn.Module, num_classes: int, device: torch.device, train_batches: DataLoader,
        criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
        eval_fn: callable, model_type: str, metrics: Dict[str, float]
    ):
    for data, targets in tqdm(train_batches):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        loss = forward_step(
            model, num_classes, device, criterion,
            eval_fn, model_type, metrics, "train", data, targets
        )
        
        # backward
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

def evaluate_one_epoch(
        model: nn.Module, num_classes: int, device: torch.device, valid_batches: DataLoader,
        criterion: torch.nn.Module,
        eval_fn: callable, model_type: str, metrics: Dict[str, float]
    ):
    with torch.no_grad():
        for data, targets in valid_batches:
            # forward
            forward_step(
                model, num_classes, device, criterion,
                eval_fn, model_type, metrics, "val", data, targets
            )

def save_model(
        model: nn.Module, optimizer: torch.optim.Optimizer,
        set_name: str, save_interval: Union[int, None], save_path: str,
        history: Dict[str, List], best_model_val_loss: float, best_model_val_score: float, best_model_epoch: int,
        avg_val_loss: float, avg_val_score: float, epoch: int
    ):
    # regular save
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

    # best model save
    if avg_val_loss < best_model_val_loss:
        best_model_val_score = avg_val_score
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
        
    return best_model_val_loss, best_model_val_score, best_model_epoch

def run_train_loop(
        model: nn.Module, num_classes: int, device: torch.device,
        train_batches: DataLoader, valid_batches: DataLoader,
        epochs: int, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
        set_name: str,
        eval_fn: Union[callable, None]=None,
        model_type: str="classification",
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
        model_type: str
            The type of model. e.g. "classification", "detection", or "segmentation".
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
    best_model_val_score = 0.
    best_model_epoch = 0

    for epoch in range(epochs):
        # set the model in training phase
        model.train()

        metrics = {
            "total_train_loss": 0,
            "total_val_loss": 0,
            "total_train_score": 0,
            "total_val_score": 0
        }

        train_one_epoch(
            model, num_classes, device, train_batches, criterion, optimizer, eval_fn,
            model_type, metrics
        )

        # set the model in evaluation phase
        if model_type != "detection":
            model.eval()
        evaluate_one_epoch(
            model, num_classes, device, valid_batches, criterion, eval_fn,
            model_type, metrics
        )

        # compute the average loss and accuracy
        avg_train_loss = metrics["total_train_loss"] / len(train_batches)
        avg_val_loss = metrics["total_val_loss"] / len(valid_batches)
        avg_train_score = metrics["total_train_score"] / len(train_batches)
        avg_val_score = metrics["total_val_score"] / len(valid_batches)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_score"].append(avg_train_score)
        history["val_score"].append(avg_val_score)

        best_model_val_loss, best_model_val_score, best_model_epoch = save_model(
            model, optimizer, set_name, save_interval, save_path,
            history, best_model_val_loss, best_model_val_score, best_model_epoch,
            avg_val_loss, avg_val_score, epoch
        )

        if epoch > best_model_epoch + patience:
            logging.info(f"train_utils - Early stopping at epoch {epoch+1}, best model at epoch {best_model_epoch+1}")
            logging.info(f"train_utils - Best model validation loss: {best_model_val_loss:.4f}, validation score: {best_model_val_score:.4f}")
            break

        # print the loss and accuracy for the epoch
        logging.info(f"train_utils - Epoch {(epoch+1)}/{epochs}")
        logging.info(f"Train Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")
        if model_type != "detection":
            logging.info(f"Train Score: {avg_train_score:.4f} Validation Score: {avg_val_score:.4f}")

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
