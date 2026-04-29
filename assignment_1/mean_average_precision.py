import torch

from decode import decode_predictions


def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise IoU between two sets of boxes.

    :param boxes_a: Box parameters in (N, cx, cy, w, h) format.
    :type boxes_a: torch.Tensor
    :param boxes_b: Box parameters in (M, cx, cy, w, h) format.
    :type boxes_b: torch.Tensor
    :return: IoU matrix of shape (N, M).
    :rtype: torch.Tensor
    """
    cx_a, cy_a, w_a, h_a = boxes_a.unbind(-1)
    cx_b, cy_b, w_b, h_b = boxes_b.unbind(-1)

    half_w_a, half_h_a = w_a / 2, h_a / 2
    half_w_b, half_h_b = w_b / 2, h_b / 2

    # Find the width of the intersection by finding the minimal right sides of
    # the boxes when comparing boxes_a and boxes_b and the maximal left sides.
    # If there is a negative result set this to 0.
    inter_w = (
        torch.minimum(
            cx_a[:, None] + half_w_a[:, None], cx_b[None] + half_w_b[None]
        ) - torch.maximum(
            cx_a[:, None] - half_w_a[:, None], cx_b[None] - half_w_b[None]
        )
    ).clamp(min=0)
    
    # Find the height of the intersection by finding the minimal top sides of
    # the boxes when comparing boxes_a and boxes_b and the maximal bottom
    # sides. If there is a negative result set this to 0.
    inter_h = (
        torch.minimum(
            cy_a[:, None] + half_h_a[:, None], cy_b[None] + half_h_b[None]
        ) - torch.maximum(
            cy_a[:, None] - half_h_a[:, None], cy_b[None] - half_h_b[None]
        )
    ).clamp(min=0)

    inter_area = inter_w * inter_h
    union_area = (w_a * h_a)[:, None] + (w_b * h_b)[None] - inter_area

    # Clamp union to prevent zero division.
    return inter_area / union_area.clamp(min=1e-6)

def calculate_map(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    iou_threshold: float,
    conf_threshold: float
) -> torch.Tensor:
    """
    Calculate the mean average precision(mAP) over a batch of
    predictions and ground truths. Predictions are first filtered by a
    confidence threshold, then further filtered by a intersection over
    union (IoU) threshold. Each ground truth is only matched once,
    preferring the prediction with the highest confidence. The average
    precision is calculated per class before and are then averaged over
    all classes that appear in the batch.

    Inspiration for many functionalities came from:
    https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

    :param y_hat: Model predictions in cube format.
    :type y_hat: torch.Tensor
    :param y: Ground truths in cube format.
    :type y: torch.Tensor
    :param iou_threshold: IoU threshold above which a prediction is 
        considered a match with a ground truth prediction.
    :type iou_threshold: float
    :param conf_threshold: Objectness threshold above which predictions
        are kept.
    :type conf_threshold: float
    :returns: The mAP of the predictions.
    :rtype: torch.Tensor
    """
    pred_x, pred_y, pred_w, pred_h, pred_obj, pred_cls = \
        decode_predictions(y_hat)
    true_x, true_y, true_w, true_h, true_obj, true_cls = decode_predictions(y)

    batches, grid_size, _ = pred_x.shape
    n_classes = pred_cls.shape[-1]
    batch_grid_cells = batches * grid_size * grid_size
    device = y_hat.device

    # Keep track of which image each cell came from.
    batch_idx = (
        torch.arange(batches, device=device)
            .unsqueeze(1)
            .expand(batches, grid_size * grid_size)
            .reshape(batch_grid_cells, 1)
            .float()
    )

    # Stack box info (cx, cy, w, h) into shape (batch_grid_cells, 4).
    pred_boxes = torch.stack(
        [pred_x, pred_y, pred_w, pred_h],
        dim=-1
    ).reshape(batch_grid_cells, 4)

    true_boxes = torch.stack(
        [true_x, true_y, true_w, true_h],
        dim=-1
    ).reshape(batch_grid_cells, 4)

    # Flatten per-cell predictions so every cell can be used independently.
    pred_obj = pred_obj.reshape(batch_grid_cells, 1)
    true_obj = true_obj.reshape(batch_grid_cells)
    pred_cls = pred_cls.reshape(batch_grid_cells, n_classes)
    true_cls = true_cls.reshape(batch_grid_cells, n_classes)

    # Filter ground truth to only keep grid cells that contain an object.
    true_mask = true_obj > 0.5
    true_boxes = true_boxes[true_mask]
    true_cls = true_cls[true_mask]
    true_batch = batch_idx[true_mask]

    # Calculate per-class confidence by multiplying the objectness and the 
    # class prediction confidence.
    cls_scores = pred_obj * pred_cls

    # Filter predictions to only keep grid cells that contain an object with an
    # objectness above the confidence threshold.
    pred_mask = pred_obj.squeeze(-1) >= conf_threshold
    pred_boxes = pred_boxes[pred_mask]
    cls_scores = cls_scores[pred_mask]
    pred_batch = batch_idx[pred_mask]

    # If there are no predictions or ground truths left return mAP of zero.
    if pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
        return torch.zeros(n_classes, device=device).mean()

    # Sort all predictions by per-class confidence.
    sort_idx = cls_scores.T.argsort(descending=True, dim=-1)
    pred_batch_sorted = pred_batch[sort_idx]

    # Match all predictions against all ground truths and compute their IoU.
    iou_mat = pairwise_iou(pred_boxes, true_boxes)

    # Sort IoU scores per class by per-class confidence.
    iou_sorted = iou_mat[sort_idx]

    # Create a mask per class for when a prediction and ground truth belong to
    # the same image.
    batch_true = true_batch.T.unsqueeze(0)
    batch_mask = pred_batch_sorted == batch_true

    # Create a mask per class for when a ground truth image belongs to that
    # class.
    true_cls_mask = true_cls.T.unsqueeze(1)

    # Zero out iou scores that don't belong to the same image or the correct 
    # class.
    iou_sorted = iou_sorted * batch_mask * true_cls_mask

    # For each prediction, find the ground truth with the highest IoU score and
    # it's index, then create a mask for when the IoU is larger than the
    # threshold.
    best_iou, best_gt = iou_sorted.max(dim=-1)
    is_match = best_iou >= iou_threshold

    # Build a matrix where per class a 1 indicates that a prediction was
    # matched to a ground truth.
    match_matrix = torch.zeros(
        n_classes,
        pred_boxes.shape[0],
        true_boxes.shape[0],
        device=device
    )
    class_idx, pred_idx = is_match.nonzero(as_tuple=True)
    match_matrix[class_idx, pred_idx, best_gt[class_idx, pred_idx]] = 1.0

    # Make sure each ground truth is only matched once by taking the cumulative
    # sum and ensuring those that are duplicates aren't kept.
    cum_matches = match_matrix.cumsum(dim=1)
    valid_match = (cum_matches <= 1.0) & (match_matrix == 1.0)

    # Decide per class true positives. True positives are when it had a valid
    # match with it's ground truth. 
    tp = valid_match.any(dim=-1).float()

    # Compute per class the cumulative true positives and false positives.
    cum_tp = tp.cumsum(dim=-1)
    cum_fp = (1.0 - tp).cumsum(dim=-1)

    # Calculate per class the precision.
    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)

    # Calculate recall by dividing cumulative true positives by the total
    # number of ground truths per class.
    true_per_class = true_cls.sum(dim=0)
    recall = cum_tp / true_per_class.unsqueeze(-1).clamp(min=1e-6)

    # Ensure each precision-recall curve starts at recall = 0 and
    # precision = 1.
    ones = torch.ones(n_classes, 1, device=device)
    zeros = torch.zeros(n_classes, 1, device=device)
    precision = torch.cat([ones,  precision], dim=-1)
    recall = torch.cat([zeros, recall], dim=-1)

    # Calculate the area under the precision-recall curve per class.
    precision_env = precision.flip(-1).cummax(dim=-1).values.flip(-1)
    ap_per_class = torch.trapezoid(precision_env, recall, dim=-1)

    # Print the info for confusion matrix (comment this line if not in use)
    # print_confusion_matrix(cls_scores, true_cls, valid_match)

    # Only calculate the mean over classes that had at least one ground truth
    # box in this batch.
    classes_with_gt = true_per_class > 0
    return ap_per_class[classes_with_gt].mean()

def print_confusion_matrix(
    cls_scores: torch.Tensor,
    true_cls: torch.Tensor,
    valid_match: torch.Tensor
)-> None:
    """
    Print a confusion matrix for the predictions. For each class: print
    how many ground truths were matched to each class or were not 
    matched (None).

    :param cls_scores: Per-class confidence scores.
    :type cls_scores: torch.Tensor
    :param true_cls: Ground truth class labels.
    :type true_cls: torch.Tensor
    :param valid_match: Matrix containing the valid matches.
    :type valid_match: torch.Tensor
    """
    pred_class = cls_scores.argmax(dim=-1)
    true_class = true_cls.argmax(dim=-1)

    # For each GT, find which predictions were matched to it.
    pred_per_gt = valid_match.any(dim=0).T

    for actual_cls_idx, cls_name in enumerate(["cat", "dog"]):
        gt_indices = (true_class == actual_cls_idx).nonzero(as_tuple=True)[0]

        counts = {"cat": 0, "dog": 0, "None": 0}
        for gt_idx in gt_indices:
            matched_preds = pred_per_gt[gt_idx].nonzero(as_tuple=True)[0]
            if len(matched_preds) == 0:
                counts["None"] += 1
            else:
                pred_cls = pred_class[matched_preds[0]].item()
                counts[["cat", "dog"][pred_cls]] += 1

        print(
            f"actual {cls_name}: "
            f"predicted cat: {counts['cat']} times, "
            f"predicted dog: {counts['dog']} times, "
            f"predicted None: {counts['None']} times"
        )

    # Count ground truths that had no prediction.
    # Also prints unmatched predictions -> actual None.
    for pred_cls_idx, cls_name in enumerate(["cat", "dog"]):
        pred_indices = (pred_class == pred_cls_idx).nonzero(as_tuple=True)[0]
        unmatched = [
            pred for pred in pred_indices if not pred_per_gt[:, pred].any()
        ]
        print(f"predicted {cls_name}: actual None: {len(unmatched)} times")
