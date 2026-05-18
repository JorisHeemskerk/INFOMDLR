"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import argparse
import copy
import logging
import os
import shutil
import torch
import traceback
import yaml

from jsonschema import validate, ValidationError
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from typing import Any

import handle_output

from create_logger import create_logger
from config.config_validation_template import CONFIG_TEMPLATE
from data import to_dataloaders
from baseline import Baseline
from meg_dataset import MEGDataset, LABEL_MAP
from train import train_cross_validation, train, evaluate, METRICS
from visualise import visualise_training, visualise_tuning

DATASET_MAPPING = {
    "intra": {
        "train" : "assignment_2/data/Intra/train/",
        "test" : "assignment_2/data/Intra/test/"
    },
    "cross": {
        "train" : "assignment_2/data/Cross/train/",
        "test" : [
            "assignment_2/data/Cross/test1/", 
            "assignment_2/data/Cross/test2/", 
            "assignment_2/data/Cross/test3/", 
        ]
    },
}


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
        job_id == 0 else "/".join((
                handle_output.OUTPUT_DIR.split("/")[:-2] if 
                    "run" not in handle_output.OUTPUT_DIR else 
                    handle_output.OUTPUT_DIR.split("/")[:-3]
        )) + f"/job_{job_id}/"
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
                    metric_names=list(METRICS.keys()),
                    output_dir=job_output_dir
                ) 
    # If there were no instances of multiple parameters, run as 1 job.
    if not tune_changes:
        run_description = copy.deepcopy(job)
        for tune_key in tunable_job_keys:
            run_description[tune_key] = job[tune_key][0]
        _process_run(
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
    dataset = MEGDataset(
        data_dirs=DATASET_MAPPING[run["dataset"].lower()]["train"],
        window_size=run["window_size"],
        stride=run["stride"],
        downsample_factor=run["downsample_factor"],
        lazy=run["lazy"],
    )
    logger.debug(f"Dataset size: {len(dataset)}")
    logger.debug(f"Shape of first x element: {dataset[0][0].shape}")
    logger.debug(f"Shape of first y element: {dataset[0][1].shape}")
    test_data = MEGDataset(
        data_dirs=DATASET_MAPPING[run["dataset"].lower()]["test"],
        window_size=run["window_size"],
        stride=1,
        downsample_factor=run["downsample_factor"],
        lazy=run["lazy"],
    )
    logger.debug(f"Test dataset size: {len(test_data)}")

    ####################################################################
    #                      Create the DataLoaders.                     #
    ####################################################################
    # TODO: skip this part when kfolds > 1, but normalise on all data instead!!!!!!!!!!
    logger.debug(f"Splitting the dataset into {run["train_val_split"]}.")
    indices = list(range(len(dataset)))
    
    ######################### Split the data. ##########################
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=run["train_val_split"][1],
        random_state=42
    )
    # Normalise based on only the train partition.
    logger.debug("Fitting normalisation.")
    dataset.fit_normalisation(train_idx)
    logger.debug(
        f"Normalisation fitted on training set: "
        f"mean[:2]={dataset.mean[:2]}, std[:2]={dataset.std[:2]}"
    )
    test_data.mean = dataset.mean
    test_data.std = dataset.std

    logger.debug("Creating subsets.")
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    logger.debug(f"{len(train_dataset) = }, {len(val_dataset) = }")

    ########## Convert DataSet objects to DataLoader objects. ##########
    logger.debug("converting to dataloaders")
    train_dataloader, val_dataloader = to_dataloaders(
        [train_dataset, val_dataset], 
        batch_sizes=[run["batch_size"]] * 2, 
        shuffles=[True, False],
        logger=logger,
        num_workers=CONFIG["general"]["num_data_workers"],
        pin_memory=True, # TODO: check if this should be replaced with run["lazy"]
        persistent_workers=True
    )

    test_dataloader = to_dataloaders(
        [test_data], 
        batch_sizes=[run["batch_size"]], 
        shuffles=[False],
        logger=logger,
        num_workers=0,
        pin_memory=False,
    )[0]

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
                test_dataloader,
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
            test_dataloader, 
            logger
        )}

def _process_model(
    run: dict[str, Any], 
    model_id: int | None, 
    dataset: MEGDataset,
    train_dataloader: DataLoader[Any], 
    val_dataloader: DataLoader[Any],
    test_dataloader: DataLoader[Any],
    logger: logging.Logger
)-> tuple[float, float]:
    """
    Applies dataset to specific model. 
    
    This function makes it possible to do multiple models per job and 
    run.

    :param run: Run description.
    :type run: dict[str, Any]
    :param model_id: ID of the current model (only use if multiple 
        models are being trained for a run).
    :type model_id: int | None
    :param dataset: the dataset.
    :type dataset: MEGDataset
    :param train_dataloader: Dataloader for training data.
    :type train_dataloader: DataLoader[Any]
    :param val_dataloader: Dataloader for validation data.
    :type val_dataloader: DataLoader[Any]
    :param test_dataloader: Dataloader for test data.
    :type test_dataloader: DataLoader[Any]
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :returns: In order, the training and validation accuracies.
    :rtype: tuple[float, float]
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
        "baseline": (
            Baseline, {
                "network_shape": [
                    dataset.get_n_sensors() * run["window_size"], 
                    *([run["hidden_size"]] * run["num_layers"]),
                    len(LABEL_MAP),
                ],
                "logger": logger,
            }
        )
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

    # Should speed up model after epoch 1, but has not proven effective.
    # model = torch.compile(model, backend="aot_eager") 

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
    LOSS_FN = nn.CrossEntropyLoss()

    # Arguments used by both normal training and cross_validation
    arguments = {
        "model" : model,
        "loss_fn" : LOSS_FN,
        "optimiser": OPTIMISER,
        "n_epochs" : run["n_epochs"],
        "device" : DEVICE,
        "logger" : logger
    }

    ################ Don't use k-fold cross validation #################
    if run["k_folds"] == 1:
        # Train it the normal way.
        train_losses, train_metrics, val_losses, val_metrics, model = train(
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            **arguments
        )
        train_losses_std, train_metrics_std = None, None
        val_losses_std, val_metrics_std = None, None
    else:
    ################### Use k-fold cross validation ####################
        train_lossess, train_metricss, val_lossess, val_metricss, model=\
            train_cross_validation(
                full_train_dataset=dataset, 
                k_folds=run["k_folds"],
                dataset_to_dataloader_function=lambda dataset: to_dataloaders(
                    datasets=[dataset],
                    batch_sizes=[run["batch_size"]],
                    shuffles=[True],
                    logger=logger,
                    num_workers=CONFIG["general"]["num_data_workers"],
                    pin_memory=True,
                    persistent_workers=True,
                ),
                **arguments
            )
        
        train_losses = np.mean(train_lossess, axis=0)
        train_losses_std = np.std(train_lossess, axis=0)

        train_metrics = {
            k : np.mean(v, axis=0) for k, v in train_metricss.items()
        }
        train_metrics_std = {
            k : np.std(v, axis=0) for k, v in train_metricss.items()
        }

        val_losses = np.mean(val_lossess, axis=0)
        val_losses_std = np.std(val_lossess, axis=0)
        
        val_metrics = {
            k : np.mean(v, axis=0) for k, v in val_metricss.items()
        }
        val_metrics_std = {
            k : np.std(v, axis=0) for k, v in val_metricss.items()
        }

    # Save the best performing model (based on the validation set).
    model.save(handle_output.OUTPUT_DIR)

    ####################################################################
    #                         Show the results.                        #
    ####################################################################
    logger.critical(
        f"Best training accuracy: {max(train_metrics["accuracy"])}, achieved "
        f"during epoch {np.argmax(train_metrics["accuracy"]) + 1}.\nBest "
        f"validation accuracy: {max(val_metrics["accuracy"])}, achieved during"
        f" epoch {np.argmax(val_metrics["accuracy"]) + 1}."
    )

    ################# Plot the predicted and real values ###############
    visualise_training(
        train_losses, 
        train_metrics,
        val_losses, 
        val_metrics,
        handle_output.OUTPUT_DIR,
        train_losses_std,
        train_metrics_std,
        val_losses_std,
        val_metrics_std
    )

    ####################################################################
    #                         Test the model.                          #
    ####################################################################
    # logger.info("Evaluating on test set.")
    # logger.error("ONLY DO THIS AFTER HYPERPAREMTER TUNING!!")
    # test_accuracy = evaluate(
    #     dataloader=test_dataloader, 
    #     model=model,
    #     device=DEVICE,
    #     logger=logger,
    # )
    # logger.critical(f"Test accuracy: {test_accuracy}")

    return max(train_metrics["accuracy"]), max(val_metrics["accuracy"])

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
        default="assignment_2/config/config.yaml", 
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
        name="Deep Learning - Assignment 2", 
        output_log_file_name=f"{handle_output.OUTPUT_DIR}process.log"
    )
    logger.info(f"Provided commandline arguments: {args.__dict__}")

    # Seed PyTorch.
    torch.manual_seed(42)

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
