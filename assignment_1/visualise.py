"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualise_training(
        train_losses: dict[str, torch.Tensor], 
        train_mAPs: dict[str, torch.Tensor], 
        val_losses: dict[str, torch.Tensor], 
        val_mAPs: dict[str, torch.Tensor],
        output_dir: str,
        train_losses_std: dict[str, torch.Tensor] | None = None,
        train_mAPs_std: dict[str, torch.Tensor] | None = None,
        val_losses_std: dict[str, torch.Tensor] | None = None,
        val_mAPs_std: dict[str, torch.Tensor] | None = None,
    )-> None:
    """
    Visualise both the loss and accuracy over the epochs, with optional
    shaded standard deviation bands.

    :param train_losses: Loss values during training.
    :type train_losses: dict[str, torch.Tensor]
    :param train_mAPs: Accuracy values during training.
    :type train_mAPs: dict[str, torch.Tensor]
    :param val_losses: Loss values during validation.
    :type val_losses: dict[str, torch.Tensor]
    :param val_mAPs: Accuracy values during validation.
    :type val_mAPs: dict[str, torch.Tensor]
    :param output_dir: Where to save the images to.
    :type output_dir: str
    :param train_losses_std: Std of loss values during training. 
        (DEFAULT=None)
    :type train_losses_std: dict[str, torch.Tensor] | None
    :param train_mAPs_std: Std of accuracy values during training. 
        (DEFAULT=None)
    :type train_mAPs_std: dict[str, torch.Tensor] | None
    :param val_losses_std: Std of loss values during validation. 
        (DEFAULT=None)
    :type val_losses_std: dict[str, torch.Tensor] | None
    :param val_mAPs_std: Std of accuracy values during validation. 
        (DEFAULT=None)
    :type val_mAPs_std: dict[str, torch.Tensor] | None
    """
    fig_mAP, ax_mAP = plt.subplots(nrows=1, ncols=len(train_mAPs))
    if len(train_mAPs) == 1:
        ax_mAP = [ax_mAP]
    epochs = range(len(train_losses[list(train_losses.keys())[0]]))

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
    
    # Plot mAPs side by side.
    for i, mAP_threshold in enumerate(train_mAPs.keys()):
        plot_with_band(
            ax_mAP[i], 
            train_mAPs[mAP_threshold], 
            train_mAPs_std[mAP_threshold] \
                if train_mAPs_std is not None else None, 
            label=f"Train mAP@{mAP_threshold}"
        )
        plot_with_band(
            ax_mAP[i], 
            val_mAPs[mAP_threshold], 
            val_mAPs_std[mAP_threshold] \
                if val_mAPs_std is not None else None, 
            label=f"Val mAP@{mAP_threshold}"
        )
        ax_mAP[i].set_title(f"mAP@{mAP_threshold} over epochs")
        ax_mAP[i].set_xlabel("Epochs")
        ax_mAP[i].set_ylabel(f"mAP@{mAP_threshold}")
        ax_mAP[i].legend()
    fig_mAP.suptitle(f"Mean Average Precisions during training.")
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_mAPs.png")
    plt.close(fig_mAP)

    # Plot losses in separate figures.
    for i, loss_type in enumerate(train_losses.keys()):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plot_with_band(
            ax, 
            train_losses[loss_type], 
            train_losses_std[loss_type] \
                if train_losses_std is not None else None, 
            label=f"Train {loss_type} loss"
        )
        plot_with_band(
            ax, 
            val_losses[loss_type], 
            val_losses_std[loss_type] \
                if val_losses_std is not None else None, 
            label=f"Val {loss_type} loss"
        )
        ax.set_title(f"{loss_type} loss over epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(f"{loss_type} loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}training_{loss_type}_loss.png")
        plt.close(fig)

    # Plot all losses combined into a single figure.
    n_losses = len(train_losses)
    fig_all, axes_all = plt.subplots(
        nrows=n_losses // 2, 
        ncols=2, 
        figsize=(12, 16)
    )
    axes_all = axes_all.flatten()

    for i, loss_type in enumerate(train_losses.keys()):
        plot_with_band(
            axes_all[i],
            train_losses[loss_type],
            train_losses_std[loss_type] \
                if train_losses_std is not None else None,
            label=f"Train {loss_type} loss"
        )
        plot_with_band(
            axes_all[i],
            val_losses[loss_type],
            val_losses_std[loss_type] \
                if val_losses_std is not None else None,
            label=f"Val {loss_type} loss"
        )
        axes_all[i].set_title(f"{loss_type} loss over epochs")
        axes_all[i].set_xlabel("Epochs")
        axes_all[i].set_ylabel(f"{loss_type} loss")
        axes_all[i].legend()

    fig_all.suptitle("All losses during training")
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_all_losses.png")
    plt.close(fig_all)
