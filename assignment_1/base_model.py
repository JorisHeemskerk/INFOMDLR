import logging
import torch
from torch import nn


class BaseModel(nn.Module):
    """
    A base model class that can be used to build other specific models 
    on top of.
    """
    def __init__(self, logger: logging.Logger, *args, **kwargs)-> None:
        self.logger = logger
        super().__init__(*args, **kwargs)

    def _initialise_weights(self)-> None:
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
        raise NotImplementedError

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
    def load(cls, source: str, logger: logging.Logger)-> "BaseModel":
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
