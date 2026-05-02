import logging
import torch
from torch import nn

from base_model import BaseModel


class RNN(BaseModel):
    """
    A RNN model for predicting time-series data.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        logger: logging.Logger
    ) -> None:
        """
        Define the layers of the model.

        :param input_size: Number of features per timestep.
        :type input_size: int
        :param hidden_size: The number of features in the hidden state.
        :type hidden_size: int
        :param num_layers: Number of RNN layers.
        :type num_layers: int
        :param logger: Logger to log to.
        :type logger: logging.Logger
        """

        super().__init__(logger)
 
        self.backbone = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.head = nn.Linear(hidden_size, 1)
 
        self._initialise_weights()

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Perform a forward pass on the network.

        :param x: Input tensor of shape (batch_size, window_size, 
            input_size).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, 1).
        :rtype: torch.Tensor
        """
        backbone, _ = self.backbone(x)
        return self.head(backbone[:, -1, :])
