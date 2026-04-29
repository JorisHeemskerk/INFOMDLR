import logging
import torch
from torch import nn


class YOLOv1Base(nn.Module):
    """
    A YOLOv1-inspired model architecture.
    """
    def __init__(self, logger: logging.Logger)-> None:
        """
        Define the convolutional, pooling and fully connected layers.
        
        :param logger: Logger to log to.
        :type logger: logging.Logger
        """
        super().__init__()
        self.logger = logger
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(.5), # NOTE: how much dropout do we need?

            # 112 -> 56 -> 28 -> 14 -> 7 | 7 * 7 * 32 = 1568
            nn.Linear(in_features=1568, out_features=512), 
            nn.LeakyReLU(negative_slope=0.1),

            # 7 * 7 grid * (1 object + 4 bbox + 2 classes) = 343
            nn.Linear(in_features=512, out_features=343),
            nn.Sigmoid(),
        )
        self.__initialise_weights()

    def __initialise_weights(self)-> None:
        """
        Apply kaiming uniform initialization to all layers.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Perform a forward pass on the network.

        :param x: Input tensor of shape 
            (batch_size, 3, img_size, img_size).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, 343)
        :rtype: torch.Tensor
        """
        return self.head(self.backbone(x))

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
    def load(cls, source: str, logger: logging.Logger)-> YOLOv1Base:
        """
        Load a model from a file.

        :param source: Directory or .pth file to load the model from.
            If a directory is given, loads 'best_YOLOv1Base.pth'.
        :type source: str
        :param logger: Logger to assign to the loaded model, as it would
            otherwise load the old logger.
        :type logger: logging.Logger
        :return: The loaded model instance.
        :rtype: YOLOv1Base
        """
        filename = f"{source}/best_{cls.__name__}.pth"
        if ".pth" in source:
            filename = source
        logger.info(f"Loading model from {filename}...")
        model = torch.load(filename, weights_only=False)
        model.logger = logger
        return model
