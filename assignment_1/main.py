"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import argparse
import logging
import numpy as np
import os
import shutil
import torch
import traceback
import yaml

from jsonschema import validate, ValidationError
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import ConcatDataset
from typing import Any

import handle_output

from timeseries_dataset import TimeseriesDataset
from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from early_stopper import EarlyStopper
from train import train, predict_epoch, train_cross_validation, compute_epoch_map
from visualise import visualise_training
from yolov1_base import YOLOv1Base
from yolov1_resnet import YOLOv1ResNet


def _process_job(
    job: dict[str, Any], 
    job_id: int, 
    logger: logging.Logger
)-> None:
    """
    This function executes the jobs according to their description.

    :param job: Job description, pulled from config
    :type job: dict[str, Any]
    :param job_id: ID of the current job (for logging).
    :type job_id: int
    :param logger: Logger to log to.
    :type logger: logging.Logger
    """
    # ############ Change output dir to specific job folder. #############
    # handle_output.OUTPUT_DIR = f"{handle_output.OUTPUT_DIR}job_{job_id}/" if \
    #     job_id == 0 else "/".join(
    #         handle_output.OUTPUT_DIR.split("/")[:-2]
    #     ) + f"/job_{job_id}/"
    # os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)

    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = TimeseriesDataset(
        source="assignment_1/Xtrain.mat",
        window_size=5,
        horizon=1,
        stride=1,
    )

    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    logger.debug(f"Splitting the dataset into {job["train_val_test_split"]}.")
    indices = list(range(len(dataset)))
    
    ################## Split in a stratisfied manner. ##################
    train_idx, val_idx = train_test_split(
        indices, 
        test_size= \
            job["train_val_test_split"][1] + job["train_val_test_split"][2],
        random_state=42
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    logger.debug(
        f"{len(train_dataset)= }, {len(val_dataset)= }"
    )

    ########## Convert DataSet objects to DataLoader objects. ##########
    train_dataloader, val_dataloader = to_dataloaders(
        [train_dataset, val_dataset], 
        batch_sizes=[job["batch_size"]] * 2, 
        shuffles=[True, False],
        logger=logger,
        num_workers=CONFIG["general"]["num_data_workers"],
        pin_memory=True,
        persistent_workers=True
        # collate_fn=lambda x: tuple(zip(*x)) # TODO: why is this needed????????????
    )
    exit() # TODO: remove

    ####################################################################
    #                     Load the (correct) model.                    #
    ####################################################################
    logger.debug(f"Initialising the model ({job['model']})")
    models = {
        "lstm": (LSTM, {
            "input_size": job["input_size"],
            "hidden_size": job["hidden_size"][0],
            "num_layers": job["num_layers"],
            "logger": logger
        })
    }
    model = None
    for name, (cls, kwargs) in models.items():
        if job['model'].lower() in name:
            model = cls(**kwargs)
            break
    assert model is not None, "Provided model in config does not exist."

    logger.debug(f"Model:\n{model}")
    logger.debug("Total number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    exit()

    model = model.to(DEVICE)

    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    OPTIMISER = torch.optim.Adam(
        params=model.parameters(),
        lr=job["learning_rate"],
        weight_decay=1e-4
    )
    SCHEDULER = None
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
        OPTIMISER, 
        mode='min', 
        patience=10, 
        factor=0.5
    )
    LOSS_FN = None
    EARLY_STOPPER = EarlyStopper(15, 0.01)

    # Arguments used by both normal training and cross_validation
    arguments = {
        "model" : model,
        "loss_fn" : LOSS_FN,
        "optimiser": OPTIMISER,
        "scheduler" : SCHEDULER,
        "early_stopper" : EARLY_STOPPER,
        "n_epochs" : job["n_epochs"],
        "device" : DEVICE,
        "grid_size" : CONFIG["general"]["grid_size"],
        "iou_thresholds" : job["iou_thresholds"],
        "conf_threshold" : job["conf_threshold"],
        "logger" : logger
    }
    # Only perform cross validation on k >= 2.
    # Normal training, no folds.
    if job["k_folds"] <= 1:
        train_losses, train_mAPs, val_losses, val_mAPs, model = train(
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            **arguments
        )
        train_losses_std, train_mAPs_std = None, None
        val_losses_std, val_mAPs_std = None, None
    # Training with k-folds
    else:
        all_train_dataset = ConcatDataset([train_dataset, val_dataset])

        train_losses, train_mAPs, val_losses, val_mAPs, model = \
            train_cross_validation(
                full_train_dataset=all_train_dataset,
                k_folds=job["k_folds"],
                dataset_to_dataloader_function=lambda dataset: to_dataloaders(
                    [dataset],
                    batch_sizes=[job["batch_size"]],
                    shuffles=[False],
                    logger=logger,
                    num_workers=CONFIG["general"]["num_data_workers"],
                    pin_memory=True,
                    persistent_workers=True
                ),
                **arguments, 
            )
        # Combine all folds into 1, remembering the data distributions.
        loss_keys = train_losses.keys()
        mAP_keys = train_mAPs.keys()

        mean_func = lambda x, y: {k: np.nanmean(x[k], axis=0) for k in y}
        std_func = lambda x, y: {k: np.nanstd(x[k], axis=0) for k in y}
        
        train_losses_std = std_func(train_losses, loss_keys)
        train_losses = mean_func(train_losses, loss_keys)
        
        val_losses_std = std_func(val_losses, loss_keys)
        val_losses = mean_func(val_losses, loss_keys)
        
        train_mAPs_std = std_func(train_mAPs, mAP_keys)
        train_mAPs = mean_func(train_mAPs, mAP_keys)

        val_mAPs_std = std_func(val_mAPs, mAP_keys)
        val_mAPs = mean_func(val_mAPs, mAP_keys)
    
    # Save the best performing model (based on the validation set).
    model.save(handle_output.OUTPUT_DIR)
    ####################################################################
    #                         Show the results.                        #
    ####################################################################
    ########### Log the best training and validation scores. ###########
    # TODO: what if the first threshold is not the best for this?
    mAP_train_string = ", ".join(
        f"mAP@{threshold}: {np.max(train_mAPs[str(threshold)])*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    train_best_epoch = np.nanargmax(
        train_mAPs[str(job["iou_thresholds"][0])]
    ) + 1
    mAP_val_string = ", ".join(
        f"mAP@{threshold}: {np.max(val_mAPs[str(threshold)])*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    val_best_epoch = np.nanargmax(val_mAPs[str(job["iou_thresholds"][0])]) + 1
    logger.critical(
        f"Best training scores: {mAP_train_string} | "
        f"achieved during epoch {train_best_epoch}."
    )
    logger.critical(
        f"Best validation scores: {mAP_val_string} | "
        f"achieved during epoch {val_best_epoch}."
    )

    ########### Get results on training and validation sets. ###########
    train_epoch_map = compute_epoch_map(
        model,
        train_dataloader,
        DEVICE,
        CONFIG["general"]["grid_size"],
        job["iou_thresholds"], 
        job["conf_threshold"]
    )
    mAP_val_epoch_string = ", ".join(
        f"mAP@{threshold}: {train_epoch_map[str(threshold)]*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    logger.critical(f"Training set over all images: {mAP_val_epoch_string}")
    val_epoch_map = compute_epoch_map(
        model,
        val_dataloader,
        DEVICE,
        CONFIG["general"]["grid_size"],
        job["iou_thresholds"], 
        job["conf_threshold"]
    )
    mAP_val_epoch_string = ", ".join(
        f"mAP@{threshold}: {val_epoch_map[str(threshold)]*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    logger.critical(f"Validation set over all images: {mAP_val_epoch_string}")
    
    ############### Produce all the loss and mAP figures. ##############
    visualise_training(
        train_losses, 
        train_mAPs, 
        val_losses, 
        val_mAPs, 
        handle_output.OUTPUT_DIR,
        train_losses_std, 
        train_mAPs_std, 
        val_losses_std, 
        val_mAPs_std
    )

    ###### Predict on the validation set, then visualise batch 1. ######
    predict_epoch(
        dataloader=val_dataloader,
        model=model,
        loss_fn=LOSS_FN,
        device=DEVICE,
        grid_size=CONFIG["general"]["grid_size"],
        iou_thresholds=job["iou_thresholds"],
        conf_threshold=job["conf_threshold"],
        plotting_conf_threshold=job["plotting_conf_threshold"],
        visualise_first_batch=True,
        logger=logger
    )
    ####################################################################
    #                          Apply test set.                         #
    ####################################################################
    
    test_mAP = compute_epoch_map(
        model,
        test_dataloader,
        DEVICE,
        CONFIG["general"]["grid_size"],
        job["iou_thresholds"], 
        job["conf_threshold"]
    )
    mAP_test_epoch_string = ", ".join(
        f"mAP@{threshold}: {test_mAP[str(threshold)]*100:<2f}%"
        for threshold in job["iou_thresholds"]
    )
    logger.critical(f"Test set over all images: {mAP_test_epoch_string}")



def main()-> None:
    ####################################################################
    #                         Execute all jobs.                        #
    ####################################################################
    for i, job in enumerate(CONFIG['jobs'].values()):
        logger.info(
           f"----- Processing Job {i:3.0f}/"
           f"{len(CONFIG['jobs'].values())-1:3.0f} -----"
        )
        logger.info(f"Job description: {job}")
        # This try-except catches individual job errors and attempts the 
        # next job if one of them crashes.
        try:
            if job in list(CONFIG['jobs'].values())[:i]:
                logger.warning(
                    "A job matching this exact configuration has already " 
                    "been executed. You likely have duplicate job descriptions"
                    ". This job will be skipped."
                )
                continue
            _process_job(
                job=job,
                job_id=i, 
                logger=logger
            )
        except KeyboardInterrupt as e:
            logger.critical(
                "PROGRAM MANUALLY HALTED BY KEYBOARD INTERRUPT "
                "(inside job execution loop)."
            )
            raise KeyboardInterrupt(
                "Keyboard interupt detected, halting program."
            ) from e
        except Exception as e:
            trace = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            logger.error(
                f"Error during handling of job {i} ({job = })\n\tTraceback:\n"
                f"\t{trace}\n\t'''{type(e)}: {e}'''\n"
                "Skipping this job, attempting to execute next job."
            )

if __name__ == "__main__":
    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument(
        '-c',
        '--config', 
        dest='config_file_path', 
        type=str, 
        default="assignment_1/config/config.yaml", 
        help="Path to config file. (default: %(default)s)"
    )
    args = parser.parse_args()

    # Initialise Logger.
    os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)
    logger = create_logger(
        name="Deep Learning - Assignment 1", 
        output_log_file_name=f"{handle_output.OUTPUT_DIR}process.log"
    )
    logger.info(f"Provided commandline arguments: {args.__dict__}")

    # Seed PyTorch.
    torch.manual_seed(42)

    # Initialise Device.
    DEVICE = torch.accelerator.current_accelerator().type if \
        torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {DEVICE} device")

    # validate the provided config file.
    with open(args.config_file_path, 'r') as stream:
        CONFIG = yaml.safe_load(stream)
    try:
        validate(CONFIG, CONFIG_TEMPLATE)
    except ValidationError as e:
        raise ValidationError(
            "\x1b[31;1mA validation error occurred in the config file" \
            f": {e.message}\x1b[0m"
        ) from e
    shutil.copy(args.config_file_path, handle_output.OUTPUT_DIR + "config.yml")

    ## Execute main. ###################################################
    main()
