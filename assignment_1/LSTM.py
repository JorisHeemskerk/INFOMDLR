import logging
import torch
from torch import nn


class LSTM(nn.Module):
    """
    A LSTM model for predicting time-series data.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        logger: logging.Logger
    )-> None:
        """
        Define the layers of the model.

        :param input_size: Number of features per timestep.
        :type input_size: int
        :param hidden_size: The number of features in the hidden state.
        :type hidden_size: int
        :param num_layers: Number of LSTM layers.
        :type num_layers: int
        :param logger: Logger to log to.
        :type logger: logging.Logger
        """
        super().__init__()

        self.logger = logger
        self.backbone = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_size, 1)

        self.__initialise_weights()

    def __initialise_weights(self)-> None:
        """
        Apply kaiming uniform initialization to all layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LSTM)):
                for name, param in module.named_parameters():
                    if "bias" in name:
                        nn.init.zeros_(param)
                    if "weight" in name:
                        nn.init.xavier_normal_(param)

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

    def save(self, destination: str)-> None:
        """
        Save internal state to file.

        :param destination: Directory/file to output model to.
        :type destination: str
        """
        filename = f"{destination}/best_{self.__class__.__name__}.pth"
        if ".pth" in destination:
            filename = destination
        self.logger.info(f"Saving model to {filename}...")
        torch.save(self, filename)

    @classmethod
    def load(cls, source: str, logger: logging.Logger)-> LSTM:
        """
        Load a model from a file.

        :param source: Directory or .pth file to load the model from.
        :type source: str
        :param logger: Logger to assign to the loaded model, as it would
            otherwise load the old logger.
        :type logger: logging.Logger
        :return: The loaded model instance.
        :rtype: LSTM
        """
        filename = f"{source}/best_{cls.__name__}.pth"
        if ".pth" in source:
            filename = source
        logger.info(f"Loading model from {filename}...")
        model = torch.load(filename, weights_only=False)
        model.logger = logger
        return model
