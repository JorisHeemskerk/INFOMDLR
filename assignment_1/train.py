"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import copy
import logging
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    list[float],
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
    :return: Per epoch train losses and validation losses. Along with 
        the model checkpoint that achieved the best validation loss.
    :rtype: tuple[
        list[float],
        list[float],
        nn.Module
    ]
    """
    best = None
    train_losses_per_epoch, val_losses_per_epoch = [], []
    for i in tqdm(range(n_epochs), "\033[33mEpoch"):
        print("\033[37m", end="") # Reset colour.
        logger.info(f"-----===== Epoch {i} (training) =====-----")
        train_loss = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            logger
        )
        train_losses_per_epoch.append(train_loss)

        logger.info(f"-----===== Epoch {i} (validation) =====-----")
        val_loss = val_epoch(
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

    logger.info("Done training")
    model.load_state_dict(best)

    return train_losses_per_epoch, val_losses_per_epoch, model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    logger: logging.Logger
)-> float:
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
    :return: Average training loss over the epoch.
    :rtype: float
    """
    total_loss = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.squeeze(-1).to(device)
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        total_loss += loss.item()

        if batch % 10 == 0:
            current = batch * len(y) + len(X)
            logger.debug(
                f"\033[30mtrain loss: {loss.item():>7f}  "
                f"[{current:>5d}/{len(dataloader.dataset):>5d}]\033[37m"
            )

    return total_loss / len(dataloader)

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    logger: logging.Logger
)-> float:
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
    :return: Average validation loss.
    :rtype: float
    """
    total_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.squeeze(-1).to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            total_loss += loss.item()
            
    val_loss = total_loss / len(dataloader)
    logger.debug(f"Avg loss: {val_loss:>8f} \n\033[37m")

    return val_loss

def evaluate(
    dataloader: DataLoader, 
    model: nn.Module,
    device: str,
    logger: logging.Logger  
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
    :return: MAE and MSE on the dataset
    :rtype: tuple[float, float]
    """
    total_mae = 0
    total_mse = 0

    mae = nn.L1Loss()
    mse = nn.MSELoss()

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.squeeze(-1).to(device)
            y_hat = model(X)

            total_mae += mae(y_hat, y).item()
            total_mse += mse(y_hat, y).item()

    return total_mae / len(dataloader), total_mse / len(dataloader)