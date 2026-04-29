import logging
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from yolov1_base import YOLOv1Base


class YOLOv1ResNet(YOLOv1Base):
    """
    YOLOv1-style detection head on a pretrained ResNet-18 backbone.
    """

    def __init__(
        self, 
        logger: logging.Logger, 
        freeze_backbone: bool=False,
    ) -> None:
        """
        Define the convolutional, pooling and fully connected layers.

        Loads the ResNet18 backbone, all except the last 2 child 
        modules. Then loads the coco weights into these layers. The 
        head is custom and mimics YOLOv1Base, except with more inputs.
        
        :param logger: Logger to log to.
        :type logger: logging.Logger
        :param freeze_backbone: If True, freeze the backbone weights so
            they do not get trained.
        :type freeze_backbone: bool
        """
        super().__init__(logger)

        # The last two children are avgpool and fully connected layers.
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #  224 * 224 input -> 512 * 7 * 7 backbone output 
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            self.logger.debug("Freezing ResNet-18 backbone.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),

            # 7 * 7 * 512 = 25088
            nn.Linear(25_088, 512),
            nn.ReLU(),

            # 7 * 7 grid * (1 object + 4 bbox + 2 classes) = 343
            nn.Linear(512, 343),
            nn.Sigmoid(),
        )
        self.__initialise_head_weights()

    def __initialise_head_weights(self) -> None:
        """
        Apply kaiming uniform initialization to only the head layers.
        """
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
