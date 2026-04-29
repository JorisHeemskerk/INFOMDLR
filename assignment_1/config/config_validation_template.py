"""
DISCLAIMER: 
This code was previously part of Joris Heemskerks Bachelors thesis, 
and is being re-used here. All rights are reserved to Joris Heemskerk, 
and Technolution BV, Gouda NL. Joris was granted the rights to use and 
modify this code, at the express notion that a disclaimer was put in.
"""

""" Config yaml template

    This template can be used to validate the composition of the configuration yaml file.
    Minimal viable yaml file looks like (_'s are placeholders for data):

    ```yaml
    general:
        data_images_path : _
        data_annotations_path : _
        grid_size: _
        num_data_workers: _
    jobs:
        job0:
            model: _
            start_from_checkpoint_path: _
            input_image_size: _
            train_val_test_split: _
            batch_size : _
            n_epochs: _
            k_folds: _
            learning_rate: _
            l1_coefficient: _
            lambda_coord: _
            lambda_noobj: _
            iou_thresholds: _
            conf_threshold: _
            plotting_conf_threshold: _
    ```
"""

CONFIG_TEMPLATE = {
    'type': 'object',
    'properties': {
        'general': {
            'type': 'object',
            'properties': {
                'data_images_path': {
                    'type': 'string', 
                },
                'data_annotations_path': {
                    'type': 'string', 
                },
                'grid_size': {
                    'type': 'number',
                    'minimum': 1
                },
                'num_data_workers': {
                    'type': 'number',
                    'minimum': 1
                },
            },
            'required': [
                'data_images_path', 
                'data_annotations_path',
                'grid_size',
                'num_data_workers',
            ],
            'additionalProperties' : False
        },
        'jobs': {
            'type': 'object',
            'patternProperties': {
                '^job\\d+$': {
                    'type': 'object',
                    'properties': {
                        'model': {
                            'type': 'string', 
                        },
                        'start_from_checkpoint_path': {
                            'type': 'string', 
                        },
                        'input_image_size': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'train_val_test_split': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 3,
                            'maxItems': 3
                        },
                        'batch_size': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'n_epochs': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'k_folds': {
                            'type': 'number', 
                            'minimum': 0,
                        },
                        'learning_rate': {
                            'type': 'number'
                        },
                        'lambda_coord': {
                            'type': 'number'
                        },
                        'lambda_noobj': {
                            'type': 'number'
                        },
                        'iou_thresholds': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1,
                        },
                        'conf_threshold': {
                            'type': 'number'
                        },
                        'plotting_conf_threshold': {
                            'type': 'number'
                        }
                    },
                    'required': [
                        'train_val_test_split',
                        'batch_size',
                        'n_epochs',
                        'k_folds',
                        'learning_rate',
                        'lambda_coord',
                        'lambda_noobj',
                        'iou_thresholds',
                        'conf_threshold',
                        'plotting_conf_threshold',
                    ],
                    'additionalProperties' : False
                }
            }
        },
    },
    'required': ['general', 'jobs'],
    'additionalProperties' : False
}
