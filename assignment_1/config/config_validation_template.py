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
            input_size: _
            hidden_size: _
            num_layers: _
            optimiser: _
            train_val_split: _
            batch_size: _
            n_epochs: _
            learning_rate: _
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
                        'input_size': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'hidden_size': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 1
                        },
                        'num_layers': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'optimiser': {
                            'type': 'string', 
                        },
                        'train_val_split': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 2,
                            'maxItems': 2
                        },
                        'batch_size': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'n_epochs': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'learning_rate': {
                            'type': 'number'
                        }
                    },
                    'required': [
                        'model',
                        'input_size',
                        'hidden_size',
                        'num_layers',
                        'optimiser',
                        'train_val_split',
                        'batch_size',
                        'n_epochs',
                        'learning_rate'
                    ],
                    'additionalProperties' : False
                }
            }
        },
    },
    'required': ['general', 'jobs'],
    'additionalProperties' : False
}
