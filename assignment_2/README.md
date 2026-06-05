# Hypertuning
> This chapter describes the steps we took/will take to produce our hypertuning results.

Below is a step by step tutorial for how to go from cloning the repo to reproducing our hyperparameter tuning results.

### Step 1: Setting up the config file
Below is the config file used for all of the hypertuning runs. The only thing that changes between the runs is the model used and the dataset partition. Copy this config over into the `config/config.yaml` file. If you are running multiple jobs, be sure to give each a unique job id.
```yml
    job0:
        ### Data ###
        dataset: cross # Can be from [intra, cross]
        window_size: [32, 1024]
        stride: 1024 # Replaced by window_size when tune == true
        downsample_factor: [1, 6]
        lazy: true
        train_val_split: [.8, .2] # Only used if k_folds > 1
        batch_size: [8, 128]
        ### Model ###
        model: "baseline" # Can be from: [baseline, eegnet, eegnettransformer, meggpt, meggcnet, all]
        model_params: # Contents depend on the model used
            dropout: [0.0, 0.75]
        optimiser: "adam"
        learning_rate: [0.000001, 0.001]
        weight_decay: [0.000001, 0.001]
        n_epochs: 30
        k_folds: 3  # use 1 to ignore, 3 to fold over people in the cross dataset, and 8 to use 1 file per task for val
        ## Tune ###
        tune: true
        n_trials: 40
        n_startup_trials: 7
        min_epochs: 10
        reduction_factor: 3
```

### Step 2: Selecting a model

There are 5 models you can choose from:
1. Baseline
2. EEGNet
3. EEGnet-SA
4. MEG-gpt
5. MEG-GCNet

Please select the relevant model and put its name in the `model` field (note that the spelling changes, see the comment in the yaml). 

### Step 2: Selecting a dataset partition

There are 2 dataset partitions
1. intra-person
2. cross-person

Please select the relevant partition and put its name in the `model` field (note that the spelling changes, see the comment in the yaml). 

### Step 3: Running experiments
Per model option, both datasets should be tuned, this results in 10 tunable jobs. That is a lot of jobs, so it would be preferable to run overnight or to divide the work across multiple computers. Jobs can be run independently. 

To run the experiment, execute the main.py script, it will automatically grab `assignment_2/config/config.yaml`, if you are using a different file, pass it to the executable with the flag `-c` or `--config`. The flag `-d` or `--device` can be used to set the pytorch device. Run the script with:
```cmd
python assignment_2/main.py
```

### Step 4: Waiting...
Each job may take quite a while to run, so be patient and let it do its thing. You can check the progress of an individual trial with the yellow progress bar and the over-all progress with the blue bar. Some runs may take multiple hours. Note that not all trials get fully executed, some may stop early, saving time. This also means the progress bar is not linear per se.

### Step 5: Validating the results
Now that your job is finished, the output folder should contain at least 1 job folder with multiple trials inside. Each trial can be explored individually, but the most interesting results are likely in the `best_model` folder. Here you will find your best model, along with the visualisations and result metrics.

**IMPORTANT:** save the path to this folder! You will need it later.

Visually inspect the results, if all the optimal runs are near the minimum or maximum values (as described in the config) for one or mulitple of the hyperparameters, this likely means the range should be extended. In this case, you will have to re-run the experiment with a different range. If there is no clear local optima with respect to the hyperparameters, perhaps you will need to give it more trials, increase the minimum epochs, extend the maximum number of epochs, or make other changes. If this is the case, please discuss with your team if this means needing to re-run multiple experiments.

If all looks good and you think the roughly-optimal hyperparameter values have been found, you can continue to the next step.

### Step 6: Testing the best model
You can use the `test_best_model.ipynb` notebook to test your model performance on the test set. At the top of the notebook, you will find a field to input the path to your best model folder. 

Execute the entire notebook. There should now be new files in your best_model folder.

### Step 7: Exporting the results to your report
Now that have all required results, you can place them in your report.

- Copy over the Parallel Coordinates plot to the appendix, make sure to label it correctly and give it a caption.
- Copy over the training and validation results from the hypertuning to the relevant table in the report.
- Copy over the training and testing results from the notebook to the relevant table in the report.
- Copy over the values of the best found hyperparameters to the relevant table in the report.

### Step 8: Informing your team
Thats it! You're done! Now just send a quick message to your team that they can check out your new results in the report.
