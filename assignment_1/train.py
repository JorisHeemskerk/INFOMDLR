"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import copy
import logging
import torch

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

METRICS = {
    "MAE": torch.nn.functional.l1_loss, 
    "MSE": torch.nn.functional.mse_loss
}


def train(
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    n_epochs: int,
    device: str,
    logger: logging.Logger
)-> tuple[
    list[float],
    dict[str, list[float]],
    list[float],
    dict[str, list[float]],
    nn.Module
]:
    """
    Train a model for `n_epochs` epochs.

    :param train_dataloader: Dataset to train with.
    :type train_dataloader: DataLoader
    :param val_dataloader: Dataset to validate with.
    :type val_dataloader: DataLoader
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Per epoch train losses and metrics and validation losses 
        and metrics. Along with the model checkpoint that achieved the 
        best validation loss.
    :rtype: tuple[
        list[float],
        dict[str, list[float]],
        list[float],
        dict[str, list[float]],
        nn.Module
    ]
    """
    best = None
    train_losses_per_epoch, train_metrics_per_epoch = [], [] 
    val_losses_per_epoch, val_metrics_per_epoch = [], []
    for i in tqdm(range(n_epochs), "\033[33mEpoch"):
        print("\033[37m", end="") # Reset colour.
        logger.info(f"-----===== Epoch {i} (training) =====-----")
        train_loss, train_metrics = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            logger
        )
        train_losses_per_epoch.append(train_loss)
        train_metrics_per_epoch.append(train_metrics)

        logger.info(f"-----===== Epoch {i} (validation) =====-----")
        val_loss, val_metrics = val_epoch(
            val_dataloader, 
            model, 
            loss_fn, 
            device,
            logger
        )

        if val_loss < min(
            val_losses_per_epoch
        ) if len(val_losses_per_epoch) > 0 else float("inf"):
            best = copy.deepcopy(model.state_dict())
        val_losses_per_epoch.append(val_loss)
        val_metrics_per_epoch.append(val_metrics)

    logger.info("Done training")
    model.load_state_dict(best)

    train_metrics = {
        k: [d[k] for d in train_metrics_per_epoch] for k in METRICS
    }
    val_metrics = {k: [d[k] for d in val_metrics_per_epoch] for k in METRICS}
    return \
        train_losses_per_epoch, \
        train_metrics, \
        val_losses_per_epoch, \
        val_metrics, \
        model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    logger: logging.Logger
)-> tuple[float, dict[str, float]]:
    """
    Train a model for 1 epoch.
 
    :param dataloader: Dataset to train with.
    :type dataloader: DataLoader
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average training loss and metrics over the epoch.
    :rtype: tuple[float, dict[str, float]]
    """
    total_loss = 0
    train_metrics = {metric: [] for metric in METRICS}

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.squeeze(-1).to(device)
        y_hat = model(X).squeeze(-1)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        total_loss += loss.item()
        for metric, method in METRICS.items():
            train_metrics[metric].append(method(y_hat, y).item())

        if batch % 20 == 0:
            current = batch * len(y) + len(X)
            metrics_string = ", ".join(
                f"(normalised) {metric}: {np.mean(train_metrics[metric]):>2f}"
                for metric in METRICS.keys()
            )
            logger.debug(
                f"\033[30mtrain loss: {loss.item():>7f} | "
                f"{metrics_string} | "
                f"[{current:>5d}/{len(dataloader.dataset):>5d}]\033[37m"
            )

    return \
        total_loss / len(dataloader), \
        {key: np.mean(value) for key, value in train_metrics.items()}

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    logger: logging.Logger
)-> tuple[float, dict[str, float]]:
    """
    Validate loss for a given dataset and model.
 
    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to validate.
    :type model: nn.Module
    :param loss_fn: Loss function to validate with.
    :type loss_fn: nn.Module
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average validation loss and metrics over the epoch.
    :rtype: tuple[float, dict[str, float]]
    """
    total_loss = 0
    val_metrics = {metric: [] for metric in METRICS}

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.squeeze(-1).to(device)
            y_hat = model(X).squeeze(-1)
            loss = loss_fn(y_hat, y)

            total_loss += loss.item()
            for metric, method in METRICS.items():
                val_metrics[metric].append(method(y_hat, y).item())
            
    val_loss = total_loss / len(dataloader)
    metrics_string = ", ".join(
        f"(normalised) {metric}: {np.mean(val_metrics[metric]):>2f}"
        for metric in METRICS.keys()
    )
    logger.debug(f"Avg loss: {val_loss:>8f} | {metrics_string} |\n\033[37m")

    return \
        val_loss, \
        {key: np.mean(value) for key, value in val_metrics.items()}
    

def evaluate(
    dataloader: DataLoader, 
    model: nn.Module,
    device: str,
    logger: logging.Logger,
    mean: float | None=None,
    std: float | None=None,
)-> tuple[float, float]:
    """
    Evaluate a model by calculating the MAE and MSE on a dataset.

    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to evaluate.
    :type model: nn.Module
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param mean: Mean to denormalise with.
    :type mean: float | None
    :param std: Standard deviation to denormalise with.
    :type std: float | None
    :return: MAE and MSE on the dataset
    :rtype: tuple[float, float]
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            pred = model(x).squeeze(-1)
            predictions.append(pred.cpu())
            targets.append(y.squeeze(-1).cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    # Denormalise if stats are provided.
    if mean is not None and std is not None:
        predictions = predictions * std + mean
        targets = targets * std + mean

    return tuple([f(predictions, targets).item() for f in METRICS.values()])
