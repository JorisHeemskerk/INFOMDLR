"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import argparse
import logging
import os
import shutil
import torch
import traceback
import yaml

from jsonschema import validate, ValidationError
from sklearn.model_selection import train_test_split
from torch import nn
from typing import Any

import handle_output

from timeseries_dataset import TimeseriesDataset
from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from early_stopper import EarlyStopper
from train import train, evaluate
from lstm import LSTM


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
    ############ Change output dir to specific job folder. #############
    handle_output.OUTPUT_DIR = f"{handle_output.OUTPUT_DIR}job_{job_id}/" if \
        job_id == 0 else "/".join(
            handle_output.OUTPUT_DIR.split("/")[:-2]
        ) + f"/job_{job_id}/"
    os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)

    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = TimeseriesDataset(
        source="assignment_1/Xtrain.mat",
        window_size=job["window_size"],
        stride=job["stride"],
    )
    logger.debug(f"Dataset size: {len(dataset)}")

    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    logger.debug(f"Splitting the dataset into {job["train_val_split"]}.")
    indices = list(range(len(dataset)))
    
    ######################### Split the data. ##########################
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=job["train_val_split"][1],
        random_state=42
    )
    # Normalise based on only the train partition.
    dataset.fit_normalisation(train_idx)
    logger.debug(
        f"Normalisation fitted on training set: "
        f"mean={dataset.mean:.4f}, std={dataset.std:.4f}"
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
    )

    ####################################################################
    #                     Load the (correct) model.                    #
    ####################################################################
    logger.debug(f"Initialising the model ({job['model']})")
    models = {
        "lstm": (LSTM, {
            "input_size": 1,
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

    model = model.to(DEVICE)

    ####################################################################
    #                       Initialize optimiser.                      #
    ####################################################################
    logger.debug(f"Initialising the optimiser ({job['optimiser']})")
    optimisers = {
        "adam": (torch.optim.Adam, {
            "params": model.parameters(),
            "lr": job["learning_rate"],
            "weight_decay": 1e-4
        })
    }
    OPTIMISER = None
    for name, (cls, kwargs) in optimisers.items():
        if job['optimiser'].lower() in name:
            OPTIMISER = cls(**kwargs)
            break
    assert OPTIMISER is not None, \
        "Provided optimiser in config does not exist."

    ####################################################################
    #                         Train the model.                         #
    ####################################################################
    # SCHEDULER = None
    # SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     OPTIMISER, 
    #     mode='min', 
    #     patience=10, 
    #     factor=0.5
    # )
    LOSS_FN = nn.MSELoss()
    # EARLY_STOPPER = EarlyStopper(15, 0.01)

    # Arguments used by both normal training and cross_validation
    arguments = {
        "model" : model,
        "loss_fn" : LOSS_FN,
        "optimiser": OPTIMISER,
        # "scheduler" : SCHEDULER,
        # "early_stopper" : EARLY_STOPPER,
        "n_epochs" : job["n_epochs"],
        "device" : DEVICE,
        "logger" : logger
    }

    train_losses, val_losses, model = train(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        **arguments
    )

    # Save the best performing model (based on the validation set).
    model.save(handle_output.OUTPUT_DIR)
    ####################################################################
    #                         Show the results.                        #
    ####################################################################
    ########### Log the training and validation scores. ###########
    train_mae, train_mse = evaluate(
        train_dataloader,
        model,
        DEVICE,
        logger,
        mean=dataset.mean,
        std=dataset.std
    )

    logger.critical(
        f"Train results: \nMAE: {train_mae:<2f} | MSE: {train_mse:<2f}"
    )

    val_mae, val_mse = evaluate(
        val_dataloader,
        model,
        DEVICE,
        logger,
        mean=dataset.mean,
        std=dataset.std
    )
    
    logger.critical(
        f"Validation results: \nMAE: {val_mae:<2f} | MSE: {val_mse:<2f}"
    )
    
    ################# Plot the predicted and real values ###############
    # TODO: plot comparing the predicted and real values 
    # NOTE: DO NOT FORGET TO DENORMALISE!!!

    ####################################################################
    #                          Apply test set.                         #
    ####################################################################
    # TODO: add on friday!
    # NOTE: DO NOT FORGET TO NORMALISE/DENORMALISE!!!


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
