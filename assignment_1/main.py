"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import argparse
import copy
import logging
import os
import math
import scipy.io
import shutil
import torch
import traceback
import yaml

from jsonschema import validate, ValidationError
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from typing import Any

import handle_output

from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from early_stopper import EarlyStopper
from lstm import LSTM
from rnn import RNN
from transformer import Transformer
from timeseries_dataset import TimeseriesDataset
from train import train, evaluate
from visualise import visualise_training, visualise_tuning, visualise_future


def _process_job(
    job: dict[str, Any], 
    job_id: int, 
    logger: logging.Logger
)-> None:
    """
    This function executes the jobs according to their description.

    For each tunable parameter in job description, spin up several jobs,
    each with different values for that parameter. For all others, the
    first item in the list will be used as to prevent highly 
    computationally expensive grid searches. 

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
    job_output_dir = handle_output.OUTPUT_DIR

    tunable_job_keys = [
        'window_size',
        'hidden_size',
        'num_layers',
        'learning_rate',
        'batch_size',
        'stride',
        'weight_decay',
    ]

    tune_changes = False
    for key, values in job.items():
        if key in tunable_job_keys:
            if len(values) > 1:
                tune_changes = True
                tune_results = []
                for i, value in enumerate(values):
                    run_description = copy.deepcopy(job)
                    for tune_key in tunable_job_keys:
                        if len(job[tune_key]) > 1 and tune_key != key:
                            logger.warning(
                                "Multiple parameters provided for multiple tun"
                                f"able parameters. The values for {tune_key} ("
                                f"{job[tune_key]}) will be ignored and the fir"
                                "st value will be used ({job[tune_key][0]})."
                            )
                        run_description[tune_key] = job[tune_key][0]
                    run_description[key] = value
                    logger.info(
                        f"----- Processing Job {job_id}, Run {i:3.0f}/"
                        f"{len(values)-1:3.0f} -----"
                    )
                    logger.info(f"Run description: {run_description}")
                    results = _process_run(
                        run=run_description,
                        run_id=i, 
                        logger=logger
                    )
                    tune_results.append(results)
                visualise_tuning(
                    tune_param_name=key,
                    tune_param_values=values,
                    tune_results=tune_results,
                    output_dir=job_output_dir
                ) 
    # If there were no instances of multiple parameters, run as 1 job.
    if not tune_changes:
        run_description = copy.deepcopy(job)
        for tune_key in tunable_job_keys:
            run_description[tune_key] = job[tune_key][0]
        results = _process_run(
            run=run_description,
            run_id=None, 
            logger=logger
        )

def _process_run(
    run: dict[str, Any], 
    run_id: int | None, 
    logger: logging.Logger
)-> dict[str, tuple[float, float, float, float]]:
    """
    This function executes the run according to their description.

    :param run: Run description.
    :type run: dict[str, Any]
    :param run_id: ID of the current run (only provide if planning to 
        perform multiple runs).
    :type run_id: int | None
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :returns: In order, the training and validation MAEs, then the MSEs.
    :rtype: dict[str, tuple[float, float, float, float]]
    """
    ############ Change output dir to specific run folder. #############
    if run_id is not None:
        handle_output.OUTPUT_DIR = \
            f"{handle_output.OUTPUT_DIR}run_{run_id}/" if \
                run_id == 0 else "/".join(
                    handle_output.OUTPUT_DIR.split("/")[:-2]
                ) + f"/run_{run_id}/"
        os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)
    
    ######################### Save run config. #########################
    with open(f'{handle_output.OUTPUT_DIR}run_config.yml', 'w') as outfile:
        yaml.dump(run, outfile)

    ####################################################################
    #                          Load the data.                          #
    ####################################################################
    dataset = TimeseriesDataset(
        source=run["dataset"],
        window_size=run["window_size"],
        stride=run["stride"],
        n_signals=run["n_signals"],
    )
    logger.debug(f"Dataset size: {len(dataset)}")
    logger.debug(f"Shape of first data x element: {dataset[0][0].shape}")
    logger.debug(f"Shape of first data y element: {dataset[0][1].shape}")

    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    logger.debug(f"Splitting the dataset into {run["train_val_split"]}.")
    indices = list(range(len(dataset)))
    
    ######################### Split the data. ##########################
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=run["train_val_split"][1],
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
    logger.debug(f"{len(train_dataset) = }, {len(val_dataset) = }")

    ########## Convert DataSet objects to DataLoader objects. ##########
    train_dataloader, val_dataloader = to_dataloaders(
        [train_dataset, val_dataset], 
        batch_sizes=[run["batch_size"]] * 2, 
        shuffles=[True, False],
        logger=logger,
        num_workers=CONFIG["general"]["num_data_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    ############## Defer task to the individual model(s). ##############
    if run["model"].lower() == "all":
        MODELS = ["lstm", "rnn", "transformer"]
        all_model_results = {model: None for model in MODELS}
        for model_id, model in enumerate(MODELS):
            model_specific_run = copy.deepcopy(run)
            model_specific_run["model"] = model
            model_results = _process_model(
                model_specific_run, 
                model_id, 
                dataset,
                train_dataloader, 
                val_dataloader, 
                logger
            )
            all_model_results[model] = model_results
        # Remove model specific sub-directory before starting next run.
        handle_output.OUTPUT_DIR = "/".join(
            handle_output.OUTPUT_DIR.split("/")[:-2]
        ) + "/"
        return all_model_results
    else:
        return {run["model"]: _process_model(
            run, 
            None, 
            dataset,
            train_dataloader, 
            val_dataloader, 
            logger
        )}

def _process_model(
    run: dict[str, Any], 
    model_id: int | None, 
    dataset: TimeseriesDataset,
    train_dataloader: DataLoader[Any], 
    val_dataloader: DataLoader[Any],
    logger: logging.Logger
)-> tuple[float, float, float, float]:
    """
    Applies dataset to specific model. 
    
    This function makes it possible to do multiple models per job and 
    run.

    :param run: Run description.
    :type run: dict[str, Any]
    :param model_id: ID of the current model (only use if multiple 
        models are being trained for a run).
    :type model_id: int | None
    :param train_dataloader: Dataloader for training data.
    :type train_dataloader: DataLoader[Any]
    :param val_dataloader: Dataloader for validation data.
    :type val_dataloader: DataLoader[Any]
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :returns: In order, the training and validation MAEs, then the MSEs.
    :rtype: tuple[float, float, float, float]
    """
    ############ Change output dir to specific run folder. #############
    if model_id is not None:
        handle_output.OUTPUT_DIR = \
            f"{handle_output.OUTPUT_DIR}{run["model"]}/" if \
                model_id == 0 else "/".join(
                    handle_output.OUTPUT_DIR.split("/")[:-2]
                ) + f"/{run["model"]}/"
        os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)

    ####################################################################
    #                     Load the (correct) model.                    #
    ####################################################################
    logger.debug(f"Initialising the model ({run['model']})")
    models = {
        "lstm": (LSTM, {
            "input_size": run["n_signals"],
            "hidden_size": run["hidden_size"],
            "num_layers": run["num_layers"],
            "logger": logger
        }),
        "rnn": (RNN, {
            "input_size": run["n_signals"],
            "hidden_size": run["hidden_size"],
            "num_layers": run["num_layers"],
            "logger": logger
        }),
        "transformer": (Transformer, {
            "input_size": run["n_signals"],
            "hidden_size": run["hidden_size"],
            "num_layers": run["num_layers"],
            "logger": logger
        })
    }
    model = None
    for name, (cls, kwargs) in models.items():
        if run['model'].lower() in name:
            model = cls(**kwargs)
            break
    assert model is not None, \
        f"Provided model in config does not exist ({model})."

    logger.debug(f"Model:\n{model}")
    logger.debug("Total number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    model = model.to(DEVICE)

    ####################################################################
    #                       Initialize optimiser.                      #
    ####################################################################
    logger.debug(f"Initialising the optimiser ({run['optimiser']})")
    optimisers = {
        "adam": (torch.optim.Adam, {
            "params": model.parameters(),
            "lr": run["learning_rate"],
            "weight_decay": run["weight_decay"]
        })
    }
    OPTIMISER = None
    for name, (cls, kwargs) in optimisers.items():
        if run['optimiser'].lower() in name:
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
        "n_epochs" : run["n_epochs"],
        "device" : DEVICE,
        "logger" : logger
    }

    train_losses, train_metrics, val_losses, val_metrics, model = train(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        **arguments
    )
    # Denormalise if stats are provided.
    if dataset.std is not None:
        new_containers = [{}, {}]
        for i, normalised_metrics in enumerate([train_metrics, val_metrics]):
            for metric, values in normalised_metrics.items():
                new_containers[i][metric] = []
                for value in values:
                    if metric == "MAE":
                         new_containers[i][metric].append(value * dataset.std)
                    elif metric == "MSE":
                         new_containers[i][metric].append(
                            value * dataset.std**2
                        )
                    else:
                        logger.error(
                            "Metric provided cannot be denormalised: "
                            f"{metric = }"
                        )
        train_metrics = new_containers[0] 
        val_metrics = new_containers[1]

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
    visualise_training(
        train_losses, 
        train_metrics,
        val_losses, 
        val_metrics,
        handle_output.OUTPUT_DIR,
    )
    ####################################################################
    #                  Predict 200 future datapoints.                  #
    ####################################################################
    logger.info("Predicting 200 future elements.")
    n_skipped_predictions = len(dataset) % run['n_signals']
    window = dataset[-1][0].unsqueeze(0).to(DEVICE)
    predictions = []
    with torch.no_grad():
        for _ in range(math.ceil(200 / run['n_signals'])):
            pred = model(window)
            predictions.append(pred)
            window = torch.cat([window[:, 1:, :], pred.unsqueeze(1)], dim=1)

    future_predictions = torch.stack(predictions).reshape(-1)[
        n_skipped_predictions : 200 + n_skipped_predictions
    ]
    # denormalise
    if dataset.mean is not None and dataset.std is not None:
        future_predictions = future_predictions * dataset.std + dataset.mean

    all_data = torch.tensor(scipy.io.loadmat(run['dataset'])["Xtrain"]).cpu()

    visualise_future(
        past=all_data,
        future=future_predictions.cpu(),
        output_dir=handle_output.OUTPUT_DIR
    )
    
    ####################################################################
    #                          Apply test set.                         #
    ####################################################################
    # TODO: add on friday!
    # NOTE: DO NOT FORGET TO NORMALISE/DENORMALISE!!!

    return train_mae, val_mae, train_mse, val_mse

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
    parser.add_argument(
        '-d',
        '--device', 
        dest='device', 
        type=str, 
        default=None, 
        help=
            "Device to run the models on. If not provided, an optimal device "
            "will be determined and used. (default: %(default)s)"
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

    torch.set_num_threads(1)
    # Initialise Device.
    if args.device is None:
        DEVICE = torch.accelerator.current_accelerator().type if \
            torch.accelerator.is_available() else "cpu"
    else:
        DEVICE = args.device
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
