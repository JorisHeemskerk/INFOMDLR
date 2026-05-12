import logging
import torch
from torch import nn

from base_model import BaseModel


class Baseline(BaseModel):
    """
    A minimal baseline for one-step-ahead time-series prediction.

        Input layer  : window_size * n_signals neurons
                       (the full flattened input window)
        Hidden layer : 3 neurons, ReLU activation
        Output layer : 1 neuron (predicts the next value)

    The ``hidden_size`` and ``num_layers`` config keys are accepted but
    intentionally ignored so that the job description in config.yaml can
    stay in the same format as the other models (RNN, LSTM, Transformer).
    A debug-level warning is logged if non-default values are supplied.
    """

    # Fixed constants to configure a baseline MLP.
    _HIDDEN_NEURONS: int = 3
    _OUTPUT_NEURONS: int = 7

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        logger: logging.Logger,
        window_size: int = 3,
    ) -> None:
        """
        Build the baseline model.

        :param input_size: Number of signals per timestep (``n_signals``
            in the config).  For this dataset this is 1.
        :type input_size: int
        :param hidden_size: Ignored. The hidden layer is always 3 neurons.
            Kept for API compatibility with the other model classes.
        :type hidden_size: int
        :param num_layers: Ignored. There is always exactly 1 hidden layer.
            Kept for API compatibility with the other model classes.
        :type num_layers: int
        :param logger: Logger to log to.
        :type logger: logging.Logger
        :param window_size: Number of past timesteps fed to the model.
            Defaults to 3 so that the flattened input has exactly 3
            features (matching the assignment spec: "3 inputs").
            When you set ``window_size: [3]`` in config.yaml this value
            is passed automatically via the constructor call in main.py.
        :type window_size: int
        """
        super().__init__(logger)

        if hidden_size != self._HIDDEN_NEURONS:
            logger.debug(
                f"Baseline: hidden_size={hidden_size} supplied in config "
                f"but is ignored. The hidden layer is fixed at "
                f"{self._HIDDEN_NEURONS} neurons per the assignment spec."
            )
        if num_layers != 1:
            logger.debug(
                f"Baseline: num_layers={num_layers} supplied in config "
                "but is ignored. There is always exactly 1 hidden layer."
            )

        # The flattened input size: window_size timesteps x n_signals features.
        # With window_size=3 and n_signals=1 this is exactly 3.
        self._flat_input_size: int = window_size * input_size

        self.backbone = nn.Sequential(
            # Hidden layer: flat_input -> 3 neurons, ReLU.
            nn.Linear(self._flat_input_size, self._HIDDEN_NEURONS),
            nn.ReLU(),
        )

        # Output layer: 3 neurons -> 1 prediction.
        # No activation on the output we are predicting a continuous value.
        self.head = nn.Linear(self._HIDDEN_NEURONS, self._OUTPUT_NEURONS)

        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the next value from an input window.

        :param x: Input tensor of shape ``(batch_size, window_size,
            input_size)``.
        :type x: torch.Tensor
        :return: Predicted next value of shape ``(batch_size, 1)``.
        :rtype: torch.Tensor
        """
        # Flatten (B, W, C) → (B, W*C) so a plain Linear layer can consume it.
        x = x.reshape(x.size(0), -1)
        return self.head(self.backbone(x))
