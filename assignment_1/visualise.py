"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualise_training(
        train_loss: torch.Tensor, 
        train_metrics: dict[str, torch.Tensor], 
        val_loss: torch.Tensor, 
        val_metrics: dict[str, torch.Tensor],
        output_dir: str,
        train_loss_std: dict[str, torch.Tensor] | None = None,
        train_metrics_std: dict[str, torch.Tensor] | None = None,
        val_loss_std: dict[str, torch.Tensor] | None = None,
        val_metrics_std: dict[str, torch.Tensor] | None = None,
    )-> None:
    """
    Visualise both the loss and accuracy over the epochs, with optional
    shaded standard deviation bands.

    :param train_loss: Loss values during training.
    :type train_loss: torch.Tensor
    :param train_metrics: Accuracy values during training.
    :type train_metrics: dict[str, torch.Tensor]
    :param val_loss: Loss values during validation.
    :type val_loss: torch.Tensor
    :param val_metrics: Accuracy values during validation.
    :type val_metrics: dict[str, torch.Tensor]
    :param output_dir: Where to save the images to.
    :type output_dir: str
    :param train_loss_std: Std of loss values during training. 
        (DEFAULT=None)
    :type train_loss_std: torch.Tensor | None
    :param train_metrics_std: Std of accuracy values during training. 
        (DEFAULT=None)
    :type train_metrics_std: dict[str, torch.Tensor] | None
    :param val_loss_std: Std of loss values during validation. 
        (DEFAULT=None)
    :type val_loss_std: torch.Tensor | None
    :param val_metrics_std: Std of accuracy values during validation. 
        (DEFAULT=None)
    :type val_metrics_std: dict[str, torch.Tensor] | None
    """
    fig_metrics, ax_metrics = plt.subplots(nrows=1, ncols=len(train_metrics))
    if len(train_metrics) == 1:
        ax_metrics = [ax_metrics]
    epochs = range(len(train_loss))

    def plot_with_band(axis, values, std, label):
        line, = axis.plot(epochs, values, label=label)
        if std is not None:
            values, std = np.array(values), np.array(std)
            axis.fill_between(
                epochs, 
                values - std, 
                values + std, 
                alpha=0.2, 
                color=line.get_color()
            )
    
    # Plot metrics side by side.
    for i, metrics_description in enumerate(train_metrics.keys()):
        plot_with_band(
            ax_metrics[i], 
            train_metrics[metrics_description], 
            train_metrics_std[metrics_description] \
                if train_metrics_std is not None else None, 
            label=f"Train {metrics_description}"
        )
        plot_with_band(
            ax_metrics[i], 
            val_metrics[metrics_description], 
            val_metrics_std[metrics_description] \
                if val_metrics_std is not None else None, 
            label=f"Val {metrics_description}"
        )
        ax_metrics[i].set_title(f"{metrics_description} over epochs")
        ax_metrics[i].set_xlabel("Epochs")
        ax_metrics[i].set_ylabel(f"{metrics_description}")
        ax_metrics[i].legend()
    fig_metrics.suptitle(f"Mean Average Precisions during training.")
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_results.png")
    plt.close(fig_metrics)

    # Plot loss.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_with_band(
        ax, 
        train_loss,
        train_loss_std if train_loss_std is not None else None, 
        label=f"Train loss"
    )
    plot_with_band(
        ax, 
        val_loss, 
        val_loss_std if val_loss_std is not None else None, 
        label=f"Val loss"
    )
    ax.set_title(f"Loss over epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_loss.png")
    plt.close(fig)
