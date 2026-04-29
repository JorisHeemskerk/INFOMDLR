import copy
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from plum import dispatch
from torch.utils.data import DataLoader
from typing import Any

from decode import decode_predictions


COLOUR_CAT = (255, 127, 14)
COLOUR_DOG = (31, 119, 180)
COLOUR_CAT_NORMALISED = tuple(np.array(COLOUR_CAT) / 255)
COLOUR_DOG_NORMALISED = tuple(np.array(COLOUR_DOG) / 255)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

def denormalise(image: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation on a single image tensor.

    :param image: Float tensor of shape (C, H, W), normalised.
    :type image: torch.Tensor
    :return: Float tensor of shape (C, H, W), values clipped to [0, 1].
    :rtype: torch.Tensor
    """
    mean = IMAGENET_MEAN.to(image.device)
    std  = IMAGENET_STD.to(image.device)
    return (image * std[:, None, None] + mean[:, None, None]).clamp(0, 1)

def draw_boxes(
    image: cv2.typing.MatLike, 
    prediction_data: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ],
    confidence_threshold: float,
    class_names: list[str],
    line_thickness: int=2,
    font_scale: float=.5,
    text_thickness: int=1
)-> cv2.typing.MatLike:
    """
    Display bounding boxes on top of image.
    
    :param image: Image to draw boxes onto.
    :type image: cv2.typing.MatLike
    :param prediction_data: Decoded output from model.
    :type prediction_data: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ]
    :param confidence_threshold: Only boxes above this threshold are 
        displayed.
    :type confidence_threshold: float
    :param class_names: Names of the classes.
    :type class_names: list[str]
    :param line_thickness: The thickness of the box lines.
    :type line_thickness: int
    :param font_scale: Font scale factor that is multiplied by the
        font-specific base size.
    :type font_scale: float
    :param text_thickness: Thickness of lines used to render the text.
    :type text_thickness: int
    :return: Image with bounding boxes drawn onto it.
    :rtype: cv2.typing.MatLike
    """
    image = image.copy()

    img_h, img_w = image.shape[:2]

    corrected_x, corrected_y, w, h, object_confidence, classes = \
        prediction_data
    predicted_class = torch.argmax(classes, dim=-1) 
    
    # Filter on indices that contain objects with high enough threshold.
    valid_cells = (object_confidence > confidence_threshold)

    corrected_x = corrected_x[valid_cells]
    corrected_y = corrected_y[valid_cells]
    w = w[valid_cells]
    h = h[valid_cells]
    object_confidence = object_confidence[valid_cells]
    predicted_class = predicted_class[valid_cells]

    # Convert from relative size to pixel size.
    pixel_relative_x = corrected_x * img_w
    pixel_relative_y = corrected_y * img_h
    pixel_relative_w = w * img_w
    pixel_relative_h = h * img_h

    # Convert from centers to corners.
    x1 = (pixel_relative_x - pixel_relative_w / 2).int()
    y1 = (pixel_relative_y - pixel_relative_h / 2).int()
    x2 = (pixel_relative_x + pixel_relative_w / 2).int()
    y2 = (pixel_relative_y + pixel_relative_h / 2).int()

    for i in range(len(object_confidence)):
        cls = predicted_class[i].item()
        # Colours are friendly for colourblind people.
        color = COLOUR_CAT_NORMALISED if cls == 0 else COLOUR_DOG_NORMALISED
        label = f"{class_names[cls]} {object_confidence[i]:.2f}"

        # Bounding box.
        cv2.rectangle(
            image, 
            (x1[i].item(), y1[i].item()), 
            (x2[i].item(), y2[i].item()), 
            color, 
            line_thickness
        )
        ################### Label background & text. ###################
        (label_w, label_h), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_thickness
        )
        label_x = x1[i].item()
        label_y = y1[i].item() - 5

        # If label goes above the image, draw it inside the box top instead.
        if label_y - label_h - baseline < 0:
            label_y = y1[i].item() + label_h + baseline

        # If label goes off the right edge, shift it left.
        if label_x + label_w > img_w:
            label_x = img_w - label_w

        # Clamp to left edge.
        label_x = max(0, label_x)

        # Do some funky math to make the label background opaque.
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y - label_h - baseline),
            (label_x + label_w, label_y + baseline),
            color,
            cv2.FILLED
        )
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(
            image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (1.0, 1.0, 1.0),
            text_thickness
        )

    return image

@dispatch
def visualise_batch(
    dataloader: DataLoader, 
    confidence_threshold: float,
    output_path: str
)-> None:
    """
    Visualise a single batch from a dataloader.

    :param dataloader: Dataloader to visualise.
    :type dataloader: Dataloader
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    images, targets = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
            
    targets = decode_predictions(targets, dataloader.dataset.dataset.grid_size)
    visualise_batch(
        images, 
        targets, 
        axes, 
        confidence_threshold,
        dataloader.dataset.dataset.classes,
        output_path
    )

@dispatch
def visualise_batch(
    X: torch.Tensor, 
    y: torch.Tensor, 
    confidence_threshold: float,
    class_names: list[str],
    output_path: str
)-> None:
    """
    Visualise a single batch from a X and y.

    :param X: Batch of images.
    :type X: torch.Tensor.
    :param y: Batch of targets (cubed).
    :type y: torch.Tensor.
    :param y_hat: Batch of prediction targets (cubed).
    :type y_hat: torch.Tensor.
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param class_names: Names of all the possible classes.
    :type class_names: list[str]
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    fig, axes = plt.subplots(1, len(X), figsize=(15, 5))
    if len(X) == 1:
        axes = [axes]
            
    targets = decode_predictions(y, y.shape[1])
    visualise_batch(
        X, 
        targets, 
        axes, 
        class_names, 
        confidence_threshold, 
        output_path
    )

@dispatch
def visualise_batch(
    X: torch.Tensor, 
    y: torch.Tensor, 
    y_hat: torch.Tensor, 
    confidence_threshold: float,
    class_names: list[str],
    output_path: str
)-> None:
    """
    Plots the true labels on top, and the predictions on the bottom.

    :param X: Batch of images.
    :type X: torch.Tensor.
    :param y: Batch of targets (cubed).
    :type y: torch.Tensor.
    :param y_hat: Batch of prediction targets (cubed).
    :type y_hat: torch.Tensor.
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param class_names: Names of all the possible classes.
    :type class_names: list[str]
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    fig, axes = plt.subplots(2, len(X), figsize=(15, 5))
    targets_y = decode_predictions(y, y.shape[1])
    targets_y_hat = decode_predictions(y_hat, y.shape[1])

    for i, img in enumerate(X):
        pic = copy.deepcopy(img).cpu()
        pic = denormalise(pic)
        pic = pic.permute(1, 2, 0).numpy()
        pic = draw_boxes(
            pic,
            tuple(t[i] for t in targets_y),
            confidence_threshold, 
            class_names
        ) 
        axes[0,i].imshow(pic)

        pic = copy.deepcopy(img).cpu()
        pic = denormalise(pic)
        pic = pic.permute(1, 2, 0).numpy()
        pic = draw_boxes(
            pic,
            tuple(t[i] for t in targets_y_hat),
            confidence_threshold, 
            class_names
        ) 
        axes[1, i].imshow(pic)

        axes[0, i].axis('off')
        axes[1, i].axis('off')

    fig.text(0.01, 0.75, 'Ground Truth', va='center', ha='left',
             fontsize=13, fontweight='bold', rotation=90)
    fig.text(0.01, 0.25, 'Predictions', va='center', ha='left',
             fontsize=13, fontweight='bold', rotation=90)

    # Leave left margin for labels.
    plt.tight_layout(rect=[0.03, 0, 1, 1])  
    plt.savefig(output_path, bbox_inches='tight')

@dispatch
def visualise_batch(
    images: torch.Tensor, 
    targets: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ],
    axes: Any,
    confidence_threshold: float,
    class_names: list[str],
    output_path: str
)-> None:
    """
    Visualise a single batch from a set of images and targets.

    CODE PARTIALLY PROVIDED IN ASSIGNMENT.

    :param images: Images to visualise.
    :type images: torch.Tensor
    :param targets: Target objects, unpacked cubes.
    :type targets: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    :param axes: axes to display onto.
    :type axes: Any
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param class_names: Names of all the possible classes.
    :type class_names: list[str]
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    for i, img in enumerate(images):
        pic = copy.deepcopy(img).cpu()
        pic = denormalise(pic)
        pic = pic.permute(1, 2, 0).numpy()
        pic = draw_boxes(
            pic,
            tuple(t[i] for t in targets),
            confidence_threshold, 
            class_names
        ) 
        axes[i].imshow(pic)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)

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
