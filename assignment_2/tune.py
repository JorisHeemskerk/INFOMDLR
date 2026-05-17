"""
Optuna-based hyperparameter tuning with Bayesian optimisation (TPE)
and Hyperband pruning.

Usage: called from main.py when a job has `tune: true` set.
"""

import copy
import logging
import os
from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch
from torch import nn
from torch.utils.data import DataLoader

import handle_output
from train import METRICS


# ---------------------------------------------------------------------------
# Search-space helpers
# ---------------------------------------------------------------------------

def _build_search_space(
    trial: optuna.Trial,
    job: dict[str, Any],
) -> dict[str, Any]:
    """
    Derive an Optuna search space from the job config.

    Convention
    ----------
    * A parameter with **one** value  →  fixed (not tuned).
    * A parameter with **two** values →  treated as [low, high] and sampled
      over that range using a log-scale for lr / weight_decay and a linear
      int-scale for the rest.
    * A parameter with **three or more** values →  treated as an explicit
      categorical list.

    :param trial: Current Optuna trial.
    :param job: Job description from config.
    :returns: Flat dict of concrete hyperparameter values for this trial.
    """
    TUNABLE = {
        "learning_rate": ("float", True),   # (type, log_scale)
        "weight_decay":  ("float", True),
        "hidden_size":   ("int",   False),
        "num_layers":    ("int",   False),
        "batch_size":    ("int",   False),
        "window_size":   ("int",   False),
        "stride":        ("int",   False),
    }

    params: dict[str, Any] = {}
    for key, (dtype, log) in TUNABLE.items():
        values = job.get(key, [])
        n = len(values)

        if n == 0:
            raise ValueError(f"Job is missing required key '{key}'.")
        elif n == 1:
            # Fixed – not a search dimension.
            params[key] = values[0]
        elif n == 2:
            low, high = values[0], values[1]
            if dtype == "float":
                params[key] = trial.suggest_float(key, low, high, log=log)
            else:
                params[key] = trial.suggest_int(key, int(low), int(high),
                                                log=log)
        else:
            # Categorical list.
            params[key] = trial.suggest_categorical(key, values)
    
    return params


# ---------------------------------------------------------------------------
# Pruning callback for train.py
# ---------------------------------------------------------------------------

class OptunaPruningCallback:
    """
    Passed into the training loop so Hyperband can prune bad trials.

    After each epoch the callback reports the current validation accuracy
    to Optuna and raises ``TrialPruned`` when the trial should be stopped.

    :param trial: Current Optuna trial.
    :param monitor: Key in the val_metrics dict to track (default: 
        "accuracy").
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "accuracy") -> None:
        self.trial = trial
        self.monitor = monitor
        self._epoch = 0

    def __call__(self, val_metrics: dict[str, float]) -> None:
        value = val_metrics[self.monitor]
        self.trial.report(value, step=self._epoch)
        self._epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {self._epoch - 1} "
                f"(val_{self.monitor}={value:.4f})."
            )


# ---------------------------------------------------------------------------
# Core tuning function
# ---------------------------------------------------------------------------

def tune_job(
    job: dict[str, Any],
    job_id: int,
    build_run_fn: Callable[[dict[str, Any], int, logging.Logger,
                            optuna.Trial | None], float],
    logger: logging.Logger,
    n_trials: int = 30,
    n_startup_trials: int = 10,
    direction: str = "maximize",
    study_name: str | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """
    Run an Optuna study for one job.

    :param job: Job description from config.
    :param job_id: Used for directory naming / logging.
    :param build_run_fn: Callable with signature
        ``(run_dict, trial_number, logger, trial) -> float``.
        It must return the scalar metric Optuna should optimise (e.g. best
        validation accuracy).  Pass the Optuna trial object through so the
        function can attach the pruning callback.
    :param logger: Logger.
    :param n_trials: Total number of trials (default 30).
    :param n_startup_trials: Random warm-up trials before TPE kicks in
        (default 10).
    :param direction: "maximize" or "minimize" (default "maximize").
    :param study_name: Optional name for the Optuna study.
    :param storage: Optional Optuna storage URL for persistence
        (e.g. "sqlite:///study.db").
    :returns: Completed Optuna study.
    """
    study_name = study_name or f"job_{job_id}_tuning"

    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
        multivariate=True,   # joint modelling of hyperparameters
    )
    # min_resource  = earliest epoch Hyperband can prune
    # max_resource  = n_epochs (reduction factor η=3 by default)
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=job["n_epochs"],
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    tuning_output_dir = handle_output.OUTPUT_DIR  # snapshot before trials move it

    def objective(trial: optuna.Trial) -> float:
        # Reset output dir to the job-level directory for each trial.
        handle_output.OUTPUT_DIR = \
            f"{tuning_output_dir}trial_{trial.number}/"
        os.makedirs(handle_output.OUTPUT_DIR, exist_ok=True)

        # Build concrete hyperparameters for this trial.
        sampled = _build_search_space(trial, job)

        # Merge with fixed job keys (dataset, model, etc.).
        run = copy.deepcopy(job)
        run.update(sampled)
        logger.info(f"{run = }")
        logger.info(f"{sampled = }")

        logger.info(
            f"[Optuna] Trial {trial.number} | params: {sampled}"
        )

        try:
            score = build_run_fn(run, trial.number, logger, trial)
        except optuna.TrialPruned:
            raise  # let Optuna handle it
        except Exception as e:
            logger.error(
                f"[Optuna] Trial {trial.number} failed with {type(e).__name__}: "
                f"{e}"
            )
            raise optuna.exceptions.TrialPruned() from e

        logger.info(
            f"[Optuna] Trial {trial.number} finished | "
            f"{direction} metric = {score}"
        )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Restore output dir to job level after all trials.
    handle_output.OUTPUT_DIR = tuning_output_dir

    logger.info(
        f"[Optuna] Best trial: #{study.best_trial.number} | "
        f"value={study.best_trial.value:.4f} | "
        f"params={study.best_trial.params}"
    )

    _save_study_summary(study, tuning_output_dir, logger)
    return study


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def _save_study_summary(
    study: optuna.Study,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Write a human-readable CSV and a best-params YAML to *output_dir*.
    """
    import yaml

    # Full trials dataframe.
    df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, "optuna_trials.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"[Optuna] Trials saved to {csv_path}")

    # Best params.
    best_path = os.path.join(output_dir, "best_params.yml")
    with open(best_path, "w") as f:
        yaml.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_trial.value,
                "best_params": study.best_trial.params,
            },
            f,
        )
    logger.info(f"[Optuna] Best params saved to {best_path}")