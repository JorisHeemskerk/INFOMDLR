from yolov1_base import YOLOv1Base
from create_logger import create_logger
from cat_dog_dataset import CatDogDataset
from data import to_dataloaders
from train import compute_epoch_map

import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split


def main():
    logger = create_logger("confidence threshold")

    DATA_IMAGES_PATH = "assignment_4/data/images/"
    DATA_ANNOTATIONS_PATH = "assignment_4/data/annotations/"

    IMG_SIZE = 112
    GRID_SIZE = 7
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    TRAIN_VAL_TEST_SPLIT = [.72, .18, .1]
    IOU_THRESHOLDS = [.5]

    CONF_START = 0
    CONF_STOP = 1
    CONF_STEP  = 0.05

    CONF_THRESHOLDS = [
        threshold for threshold in \
            np.arange(CONF_START, CONF_STOP + CONF_STEP / 2, CONF_STEP)
    ]

    logger.info(f"using thresholds: {CONF_THRESHOLDS}")

    # Initialise Device.
    DEVICE = torch.accelerator.current_accelerator().type if \
        torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {DEVICE} device")

    # Initialise model.
    model = YOLOv1Base(logger)
    model = model.load("assignment_4/models/base_model.pth", logger)
    model = model.to(DEVICE)

    # Load the data.
    dataset = CatDogDataset(
        img_dir=DATA_IMAGES_PATH, 
        ann_dir=DATA_ANNOTATIONS_PATH, 
        input_img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        logger=logger,
        transform=transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # These numbers are from ImageNet, for normalisation.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   
                std=[0.229, 0.224, 0.225]
            )
        ]),
    )

    # Create the dataloaders.
    logger.debug(f"Splitting the dataset into {TRAIN_VAL_TEST_SPLIT}.")
    labels = dataset._labels
    indices = list(range(len(dataset)))
    
    # Split in a stratisfied manner.
    _, val_test_idx, _, val_test_labels = train_test_split(
        indices, 
        labels,
        test_size=TRAIN_VAL_TEST_SPLIT[1] + TRAIN_VAL_TEST_SPLIT[2],
        stratify=labels,
        random_state=42
    )
    _, test_idx = train_test_split(
        val_test_idx,
        test_size=TRAIN_VAL_TEST_SPLIT[2] / (
            TRAIN_VAL_TEST_SPLIT[1] + TRAIN_VAL_TEST_SPLIT[2]
        ),
        stratify=val_test_labels,
        random_state=42
    )
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    logger.debug(f"{len(test_dataset) = }")

    # Convert DataSet objects to DataLoader objects.
    test_dataloader = to_dataloaders(
        [test_dataset], 
        batch_sizes=[BATCH_SIZE], 
        shuffles=[False],
        logger=logger,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # Find mAP per threshold
    all_mAPs = {}
    for threshold in CONF_THRESHOLDS:
        test_mAP = compute_epoch_map(
            model,
            test_dataloader,
            DEVICE,
            GRID_SIZE,
            IOU_THRESHOLDS, 
            threshold
        )
        
        mAP_string = ", ".join(
            f"mAP@{iou}: {test_mAP[str(iou)]*100:<2f}%"
            for iou in IOU_THRESHOLDS
        )
        logger.info(f"{round(threshold, 4)} : {mAP_string}")

        all_mAPs[f"{round(threshold, 4)}"] = test_mAP[str(IOU_THRESHOLDS[0])]

    best_conf = max(all_mAPs, key=all_mAPs.__getitem__)
    logger.critical(
        f"best conf_threshold: {best_conf} : "
        f"mAP@{IOU_THRESHOLDS[0]} = {all_mAPs[best_conf]:<2f}"
    )


if __name__ == "__main__":
    main()
