"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import Any


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
    fig_metrics.suptitle(f"Performance during training.")
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

def visualise_tuning(
    tune_param_name: str,
    tune_param_values: list[Any],
    tune_results: list[dict[str, tuple[float, float, float, float]]],
    metric_names: list[str],
    output_dir: str
)-> None:
    """
    Visualise results for the tuned parameters.

    :param tune_param_name: The name for the parameter that was tuned.
    :type tune_param_name: str
    :param tune_param_values: The range of values for the results.
    :type tune_param_values: str
    :param tune_results: Train and val metrics
    :type tune_results: list[dict[tuple[float, float, float, float]]]
    :param metric_names: Names of the metrics (in order)
    :type metric_names: list[str]
    :param output_dir: Where to save the images to.
    :type output_dir: str
    """
    # Transform to dict[str, list[tuple[...]]]
    results = dict(
        (k, [v for d in tune_results if k in d for v in [d[k]]])
        for k in {k for d in tune_results for k in d}
    )

    fig, ax = plt.subplots(nrows=1, ncols=len(metric_names))
    if len(metric_names) == 1:
        ax = [ax]
    for model in results.keys():
        for i, metric in enumerate(metric_names):
            ax[i].plot(
                tune_param_values, 
                [res[i] for res in results[model]],
                label=f"Train {model.title()}"
            )
            ax[i].plot(
                tune_param_values, 
                [res[i + 1] for res in results[model]],
                label=f"Val {model.title()}"
            )
            ax[i].scatter(
                tune_param_values, 
                [res[i] for res in results[model]],
            )
            ax[i].scatter(
                tune_param_values, 
                [res[i + 1] for res in results[model]],
            )

            ax[i].set_title(f"{metric} for differing {tune_param_name}")
            ax[i].set_xlabel(tune_param_name)
            ax[i].set_ylabel(metric)
            ax[i].legend()

    fig.suptitle("Tuning results")
    plt.tight_layout()
    models = list(results.keys())
    plt.savefig(
        f"{output_dir}tuning_res__{tune_param_name}"
        f"{('__' + models[0].title()) if len(models) == 1 else ''}.png"
    )
    plt.close(fig)
