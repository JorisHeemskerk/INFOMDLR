import logging
import torch
from torch import nn

from base_model import BaseModel


class Baseline(BaseModel):
    """
    A minimal baseline for one-step-ahead time-series prediction.
    """

    def __init__(
        self,
        network_shape: list[int],
        dropout: float,
        logger: logging.Logger,
    ) -> None:
        """
        Build the baseline model.

        :param network_shape: list equal to number of layers, containing
            the number of neurons per layers (include input and output).
        :type network_shape: list[int]
        :param logger: Logger to log to.
        :type logger: logging.Logger
        """
        super().__init__(logger)

        self.model = []
        for i, layer_size in enumerate(network_shape[:-1]):
            self.model.append(
                nn.Linear(layer_size, network_shape[i+1])
            )
            if i < len(network_shape) - 2:
                self.model.append(
                    nn.ReLU(),
                )
                self.model.append(
                    nn.Dropout(p=dropout)
                )
        self.model = nn.Sequential(*self.model)

        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the next value from an input window.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Class prediction(s).
        :rtype: torch.Tensor
        """
        # Flatten (B, W, C) → (B, W*C) so a plain Linear layer can consume it.
        x = x.reshape(x.size(0), -1)
        return self.model(x)

