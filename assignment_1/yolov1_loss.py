import torch
import torch.nn as nn

from decode import unpack_cube


class YOLOv1Loss(nn.Module):
    """
    Class for the YOLOv1 loss function.

    Loss consists of 5 main parts:
    1. Euclidean distance between the (x, y) coordinates.
    2. Relative difference between the width and the height (summed)
    3. The squared difference between the objectness scores when there 
        is an object.
    4. The squared difference between the objectness scores when there
        is an object.
    5. Sum of Squared Errors, over all classes.
    """

    def __init__(self, lambda_coord: float, lambda_noobj: float)-> None:
        """
        Initialiser for YOLOv1Loss class.

        :param lambda_coord: Scaling factor for losses part 1 and 2.
        :type lambda_coord: float
        :param lambda_noobj: Scaling factor for loss part 4.
        :type lambda_noobj: float
        """
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj  = lambda_noobj

    def forward(
        self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor,
    )-> tuple[
        torch.Tensor, 
        tuple[
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor
        ]
    ]:
        """
        Calculate the YOLO loss based on the prediction & target.

        Loss consists of 5 main parts:
        1. Euclidean distance between the (x, y) coordinates.
        2. Relative difference between the width and the height (summed)
        3. The squared difference between the objectness scores when 
            there is an object.
        4. The squared difference between the objectness scores when 
            there is no object.
        5. Sum of Squared Errors, over all classes.

        Parts 1 and 2 are weighed by `lambda_coord` and 4 is weighed by 
        `lambda_noobj`. The parts are then summed into the final loss.

        :param y_hat: Prediction tensor.
        :type y_hat: torch.Tensor
        :pram y: ground truth / target tensor.
        :type y: torch.Tensor
        :return: Summed loss, and a tuple with all the individual parts.
        :rtype: tuple(
            torch.Tensor, 
            tuple[
                torch.Tensor, 
                torch.Tensor, 
                torch.Tensor, 
                torch.Tensor, 
                torch.Tensor
            ]
        )
        """
        pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = unpack_cube(
            y_hat
        )
        true_x, true_y, true_w, true_h, true_conf, true_cls = unpack_cube(y)

        # Mask when there is or is not an object.
        obj_mask  = true_conf
        noobj_mask = 1.0 - obj_mask

        # Part 1.
        loss_xy = self.lambda_coord * (obj_mask * (
            (pred_x - true_x) ** 2 +
            (pred_y - true_y) ** 2
        )).sum()

        # Part 2.
        loss_wh = self.lambda_coord * (obj_mask * (
            (pred_w.sqrt() - true_w.sqrt()) ** 2 +
            (pred_h.sqrt() - true_h.sqrt()) ** 2
        )).sum()

        # Part 3.
        loss_conf_obj = (obj_mask * (pred_conf - true_conf) ** 2).sum()

        # Part 4.
        loss_conf_noobj = self.lambda_noobj * (
            noobj_mask * (pred_conf - true_conf) ** 2
        ).sum()

        # Part 5.
        loss_cls = (obj_mask.unsqueeze(-1) * (pred_cls - true_cls) ** 2).sum()

        total_loss = \
            loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_cls
        return \
            total_loss, \
            (loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_cls)
