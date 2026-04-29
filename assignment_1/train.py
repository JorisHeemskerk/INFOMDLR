import copy
import logging
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
from tqdm import tqdm

import handle_output

from early_stopper import EarlyStopper
from mean_average_precision import calculate_map
from visualise import visualise_batch


def _pad_to_longest(arrays: list[list[float]]) -> np.ndarray:
    """
    Pad fold arrays to equal length with NaN so that folds stopped
    early do not corrupt the mean/std calculation.

    :param arrays: arrays of different sizes
    :type arrays: list[list[float]]
    :returns: padded arrays
    :rtype: np.ndarray 
    """
    max_len = max(len(a) for a in arrays)
    padded = [
        np.pad(
            np.array(a, dtype=float), 
            (0, max_len - len(a)), 
            constant_values=np.nan
        )
        for a in arrays
    ]
    return np.array(padded)

def train_cross_validation(
    full_train_dataset: Dataset,
    k_folds: int,
    dataset_to_dataloader_function: Callable,
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    early_stopper: EarlyStopper,
    n_epochs: int,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
    logger: logging.Logger
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    nn.Module
]:
    """
    Train a model for `n_epochs` epochs using k-fold cross validation.
 
    :param full_train_dataset: Dataset to train with.
    :type full_train_dataset: Dataset
    :param k_folds: The number of folds to use.
    :type k_folds: int
    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param scheduler: Scheduler to change how the learning rate adapts.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler | None
    :param early_stopper: Early stopper to halt training when validation
        loss stops improving.
    :type early_stopper: EarlyStopper
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :param grid_size: Size of the grid.
    :type grid_size: int
    :param iou_thresholds: IoU thresholds at which to compute mAP.
    :type iou_thresholds: list[float]
    :param conf_threshold: Confidence threshold for filtering 
        predictions when computing mAP.
    :type conf_threshold: float
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Per epoch train losses, train mAPs, validation losses, and
        validation mAPs, each as a dict mapping keys to arrays of shape
        (k_folds, n_epochs). Along with the model checkpoint that 
        achieved the best validation mAP across all folds.
    :rtype: tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        nn.Module
    ]
    """
    best = None
    best_val_mAP = -1

    train_losses_per_fold: list[dict[str, list[float]]] = []
    val_losses_per_fold: list[dict[str, list[float]]] = []
    train_mAPs_per_fold: list[dict[str, list[float]]] = []
    val_mAPs_per_fold: list[dict[str, list[float]]] = []

    # Save the initial states to reset training every fold.
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimiser_state = copy.deepcopy(optimiser.state_dict())
    initial_scheduler_state = copy.deepcopy(
        scheduler.state_dict()
    ) if scheduler is not None else None

    fold_size = len(full_train_dataset) // k_folds
    for k in range(k_folds):
        logger.info(f"-----=====##### Fold {k+1}/{k_folds} #####=====-----")

        model.load_state_dict(copy.deepcopy(initial_model_state))
        optimiser.load_state_dict(copy.deepcopy(initial_optimiser_state))
        if scheduler is not None:
            scheduler.load_state_dict(copy.deepcopy(initial_scheduler_state))
        # Reset early stopper.
        early_stopper.reset()

        # Generate all the indexes of the items from each fold.
        val_idx = list(range(k * fold_size, k * fold_size + fold_size))
        train_idx = list(range(0, k * fold_size)) + \
            list(range(k * fold_size + fold_size, len(full_train_dataset)))

        train_dataloader = dataset_to_dataloader_function(
            Subset(full_train_dataset, train_idx)
        )[0]
        val_dataloader = dataset_to_dataloader_function(
            Subset(full_train_dataset, val_idx)
        )[0]

        train_losses, train_mAPs, val_losses, val_mAPs, model = \
            train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimiser=optimiser,
                scheduler=scheduler,
                early_stopper=early_stopper,
                n_epochs=n_epochs,
                device=device,
                grid_size=grid_size,
                iou_thresholds=iou_thresholds,
                conf_threshold=conf_threshold,
                logger=logger
            )
        
        fold_best_val_mAP = max(val_mAPs[str(iou_thresholds[0])])
        if fold_best_val_mAP > best_val_mAP:
            best_val_mAP = fold_best_val_mAP
            best = copy.deepcopy(model.state_dict())

        train_losses_per_fold.append(train_losses)
        train_mAPs_per_fold.append(train_mAPs)
        val_losses_per_fold.append(val_losses)
        val_mAPs_per_fold.append(val_mAPs)
    
    # Translate list of dicts of lists to dict-of-arrays, 
    # shape: (k_folds, n_epochs).
    loss_keys = train_losses_per_fold[0].keys()
    mAP_keys  = train_mAPs_per_fold[0].keys()
    train_losses_stacked = {
        k: _pad_to_longest([fold[k] for fold in train_losses_per_fold]) 
        for k in loss_keys
    }
    val_losses_stacked = {
        k: _pad_to_longest([fold[k] for fold in val_losses_per_fold]) 
        for k in loss_keys
    }
    train_mAPs_stacked = {
        k: _pad_to_longest([fold[k] for fold in train_mAPs_per_fold])
        for k in mAP_keys
    }
    val_mAPs_stacked = {
        k: _pad_to_longest([fold[k] for fold in val_mAPs_per_fold])
        for k in mAP_keys
    }

    model.load_state_dict(best)
    return \
        train_losses_stacked, \
        train_mAPs_stacked, \
        val_losses_stacked, \
        val_mAPs_stacked, \
        model

def train(
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    early_stopper: EarlyStopper,
    n_epochs: int,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
    logger: logging.Logger
)-> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
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
    :param scheduler: Scheduler to change how the learning rate adapts.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler | None
    :param early_stopper: Early stopper to halt training when validation
        loss stops improving.
    :type early_stopper: EarlyStopper
    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param device: Device to move data to.
    :type device: str
    :param grid_size: Size of the grid.
    :type grid_size: int
    :param iou_thresholds: IoU thresholds at which to compute mAP.
    :type iou_thresholds: list[float]
    :param conf_threshold: Confidence threshold for filtering 
        predictions when computing mAP.
    :type conf_threshold: float
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Per epoch train losses, train mAPs, validation losses, and
        validation mAPs — each a dict mapping keys to a list of 
        per-epoch values. Along with the model checkpoint that achieved 
        the best validation mAP (at the first IoU threshold).
    :rtype: tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        nn.Module
    ]
    """
    best = None
    train_losses_per_epoch, train_mAPs_per_epoch = [], []
    val_losses_per_epoch,   val_mAPs_per_epoch   = [], []
    for i in tqdm(range(n_epochs), "\033[33mEpoch"):
        print("\033[37m", end="") # Reset colour.
        logger.info(f"-----===== Epoch {i} (training) =====-----")
        train_loss_dict, train_mAPs = train_epoch(
            train_dataloader, 
            model, 
            loss_fn, 
            optimiser,
            device,
            grid_size,
            iou_thresholds,
            conf_threshold,
            logger
        )
        train_losses_per_epoch.append(train_loss_dict)
        train_mAPs_per_epoch.append(train_mAPs)

        logger.info(f"-----===== Epoch {i} (validation) =====-----")
        val_loss_dict, val_mAPs = val_epoch(
            val_dataloader, 
            model, 
            loss_fn, 
            device,
            grid_size,
            iou_thresholds,
            conf_threshold,
            logger
        )

        # TODO: what if the first threshold is not the best for this?
        if val_mAPs[str(iou_thresholds[0])] > (
            max(
                [d[str(iou_thresholds[0])] for d in val_mAPs_per_epoch]
            ) if len(val_mAPs_per_epoch) > 0 else -1
        ):
            best = copy.deepcopy(model.state_dict())
        val_losses_per_epoch.append(val_loss_dict)
        val_mAPs_per_epoch.append(val_mAPs)

        if scheduler is not None:
            scheduler.step(val_loss_dict["total"])

        if early_stopper.should_stop(val_loss_dict["total"]):
            logger.warning(f"Decided to stop early, at epoch {i}")
            break

    logger.info("Done training")
    model.load_state_dict(best)
    
    # Translate list of dicts to dict of lists.
    l_keys = train_losses_per_epoch[0].keys()
    mAP_keys = train_mAPs_per_epoch[0].keys()
    train_losses = {k: [d[k] for d in train_losses_per_epoch] for k in l_keys}
    train_mAPs = {k: [d[k] for d in train_mAPs_per_epoch] for k in mAP_keys}
    val_losses = {k: [d[k] for d in val_losses_per_epoch] for k in l_keys}
    val_mAPs= {k: [d[k] for d in val_mAPs_per_epoch] for k in mAP_keys}
    return train_losses, train_mAPs, val_losses, val_mAPs, model

def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimiser: torch.optim.Optimizer,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
    logger: logging.Logger
)-> tuple[dict[str, float], dict[str, float]]:
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
    :param grid_size: Size of the grid the model used to divide the
        images.
    :type grid_size: int
    :param iou_thresholds: IoU thresholds at which to compute mAP.
    :type iou_thresholds: list[float]
    :param conf_threshold: Confidence threshold for filtering predictions
        when computing mAP.
    :type conf_threshold: float
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average training losses and mAPs over the epoch, each as a
        dict mapping loss/threshold keys to scalar values.
    :rtype: tuple[dict[str, float], dict[str, float]]
    """
    train_losses = {
        "total": 0, 
        "xy": 0, 
        "wh": 0, 
        "conf_obj": 0, 
        "conf_noobj": 0, 
        "cls": 0
    }
    train_mAPs = {str(iou_threshold): [] for iou_threshold in iou_thresholds}

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        y_hat = y_hat.view(-1, grid_size, grid_size, 7)
        loss, (loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls) = \
            loss_fn(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        train_losses["total"] += loss.item()
        train_losses["xy"] += loss_xy.item()
        train_losses["wh"] += loss_wh.item()
        train_losses["conf_obj"] += loss_conf_obj.item()
        train_losses["conf_noobj"] += loss_conf_noobj.item()
        train_losses["cls"] += loss_cls.item()
        
        for iou_threshold in train_mAPs.keys():
            train_mAPs[iou_threshold].append(
                calculate_map(
                    y_hat, 
                    y, 
                    float(iou_threshold), 
                    conf_threshold
                ).item()
            )

        if batch % 100 == 0:
            train_loss, current = loss.item(), batch * len(y) + len(X)
            mAP_string = ", ".join(
                f"mAP@{threshold}: {np.mean(train_mAPs[threshold]):>2f}"
                for threshold in train_mAPs.keys()
            )
            logger.debug(
                f"train loss: {train_loss:>7f} | {mAP_string} | xy loss: "
                f"{loss_xy:>2f}, wh loss: {loss_wh:>2f}, conf loss: "
                f"{loss_conf_obj:>2f}, noobj conf loss: {loss_conf_noobj:>2f},"
                f" class loss: {loss_cls:>2f} | [{current:>5d}/"
                f"{len(dataloader.dataset):>5d}]"
            )
    
    return \
        {key: value / len(dataloader) for key, value in train_losses.items()},\
        {key: np.mean(value) for key, value in train_mAPs.items()}

def val_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
    logger: logging.Logger
)-> tuple[dict[str, float], dict[str, float]]:
    """
    Validate the mAP and loss for a given dataset and model.
 
    :param dataloader: Dataset to validate with.
    :type dataloader: DataLoader
    :param model: Model to validate.
    :type model: nn.Module
    :param loss_fn: Loss function to validate with.
    :type loss_fn: nn.Module
    :param device: Device to move data to.
    :type device: str
    :param grid_size: Size of the grid the model used to divide the
        images.
    :type grid_size: int
    :param iou_thresholds: IoU thresholds at which to compute mAP.
    :type iou_thresholds: list[float]
    :param conf_threshold: Confidence threshold for filtering predictions
        when computing mAP.
    :type conf_threshold: float
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average validation losses and mAPs over the epoch, each as a
        dict mapping loss/threshold keys to scalar values.
    :rtype: tuple[dict[str, float], dict[str, float]]
    """
    model.eval()
    val_losses = {
            "total": 0, 
            "xy": 0, 
            "wh": 0, 
            "conf_obj": 0, 
            "conf_noobj": 0, 
            "cls": 0
        }
    val_mAPs = {str(iou_threshold): [] for iou_threshold in iou_thresholds}

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = y_hat.view(-1, grid_size, grid_size, 7)
            loss, (
                loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls
            ) = loss_fn(y_hat, y)

            val_losses["total"] += loss.item()
            val_losses["xy"] += loss_xy.item()
            val_losses["wh"] += loss_wh.item()
            val_losses["conf_obj"] += loss_conf_obj.item()
            val_losses["conf_noobj"] += loss_conf_noobj.item()
            val_losses["cls"] += loss_cls.item()
            
            for iou_threshold in val_mAPs.keys():
                val_mAPs[iou_threshold].append(
                    calculate_map(
                        y_hat, 
                        y, 
                        float(iou_threshold), 
                        conf_threshold
                    ).item()
                )
    avg_losses = {
        key: value / len(dataloader) for key, value in val_losses.items()
    }
    val_mAPs = {key: np.mean(value) for key, value in val_mAPs.items()}

    mAP_string = ", ".join(
        f"mAP@{threshold}: {np.mean(val_mAPs[threshold]):>2f}"
        for threshold in val_mAPs.keys()
    )
    logger.debug(
        f"Validation error | avg loss: {avg_losses["total"]:>7f} | "
        f"{mAP_string} | xy loss: {avg_losses["xy"]:>2f}, wh loss: "
        f"{avg_losses["wh"]:>2f}, conf loss: {avg_losses["conf_obj"]:>2f}, "
        F"noobj conf loss: {avg_losses["conf_noobj"]:>2f}, class loss: "
        f"{avg_losses["cls"]:>2f} |"
    )
    return avg_losses, val_mAPs

def predict_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
    plotting_conf_threshold: float,
    visualise_first_batch: bool,
    logger: logging.Logger
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute the mAP and loss for a given dataset and model.
 
    :param dataloader: Dataset to predict with.
    :type dataloader: DataLoader
    :param model: Model to use for predictions.
    :type model: nn.Module
    :param loss_fn: Loss function to predict with.
    :type loss_fn: nn.Module
    :param device: Device to move data to.
    :type device: str
    :param grid_size: Size of the grid the model used to divide the
        images.
    :type grid_size: int
    :param iou_thresholds: IoU thresholds at which to compute mAP.
    :type iou_thresholds: list[float]
    :param conf_threshold: Confidence threshold for filtering predictions
        when computing mAP.
    :type conf_threshold: float
    :param plotting_conf_threshold: Confidence threshold for plotting the
        first batch, only used if `visualise_first_batch` is True.
    :type plotting_conf_threshold: float
    :param visualise_first_batch: True if you want to visualise the
        first batch of predictions along with the ground truth.
    :type visualise_first_batch: bool
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :return: Average losses and mAPs over the dataset, each as a dict 
        mapping loss/threshold keys to scalar values.
    :rtype: tuple[dict[str, float], dict[str, float]]
    """
    model.eval()
    test_losses = {
        "total": 0, 
        "xy": 0, 
        "wh": 0, 
        "conf_obj": 0, 
        "conf_noobj": 0, 
        "cls": 0
    }
    test_mAPs = {str(iou_threshold): [] for iou_threshold in iou_thresholds}

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = y_hat.view(-1, grid_size, grid_size, 7)
            if i == 0 and visualise_first_batch == True:
                visualise_batch(
                    X, 
                    y, 
                    y_hat, 
                    plotting_conf_threshold,
                    dataloader.dataset.dataset.classes, 
                    f"{handle_output.OUTPUT_DIR}predict_batch_1.png"
                )

            loss, (
                loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls
            ) = loss_fn(y_hat, y)
            
            test_losses["total"] += loss.item()
            test_losses["xy"] += loss_xy.item()
            test_losses["wh"] += loss_wh.item()
            test_losses["conf_obj"] += loss_conf_obj.item()
            test_losses["conf_noobj"] += loss_conf_noobj.item()
            test_losses["cls"] += loss_cls.item()
            
            for iou_threshold in test_mAPs.keys():
                test_mAPs[iou_threshold].append(
                    calculate_map(
                        y_hat, 
                        y, 
                        float(iou_threshold), 
                        conf_threshold
                    ).item()
                )

    return \
        {key: value / len(dataloader) for key, value in test_losses.items()}, \
        {key: np.mean(value) for key, value in test_mAPs.items()}

def compute_epoch_map(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    grid_size: int,
    iou_thresholds: list[float],
    conf_threshold: float,
) -> dict[str, float]:
    """
    Compute mAP over an entire dataloader.

    :param model: The model to evaluate.
    :type model: nn.Module
    :param dataloader: Dataset to evaluate.
    :type dataloader: DataLoader
    :param device: Device to move data to.
    :type device: str
    :param grid_size: Size of the grid.
    :type grid_size: int
    :param iou_thresholds: List of IoU thresholds to evaluate at.
    :type iou_thresholds: list[float]
    :param conf_threshold: Confidence threshold for predictions.
    :type conf_threshold: float
    :returns: Dict mapping IoU threshold to mAP value.
    :rtype: dict[str, float]
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            all_preds.append(preds.view(-1, grid_size, grid_size, 7).cpu())
            all_targets.append(targets.cpu())

    y_hat_all = torch.cat(all_preds, dim=0)
    y_all = torch.cat(all_targets, dim=0)

    mAPs = {}
    for threshold in iou_thresholds:
        mAPs[str(threshold)] = calculate_map(
            y_hat=y_hat_all,
            y=y_all,
            iou_threshold=threshold,
            conf_threshold=conf_threshold,
        ).item()

    return mAPs
