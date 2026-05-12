"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

class EarlyStopper:
    """
    Class used to indicate when to stop training early.
    """
    def __init__(self, patience: int, min_delta: float):
        """
        Initialiser for `EarlyStopper`.

        :param patience: How many epochs there need be minimal / no 
            change in loss in order to stop early.
        :type patience: int
        :param min_delta: Minimal difference in loss that would indicate
            changes in the loss.
        :type min_delta: float

        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        """
        Indicates if should stop early.

        :param val_loss: Loss to use for indicating if learning has
            stalled.
        :type val_loss: float
        :returns: True if should stop early, False otherwise.
        :rtype: bool
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def reset(self)-> None:
        """
        Reset to base values.
        """
        self.best_loss = float("inf")
        self.counter = 0
