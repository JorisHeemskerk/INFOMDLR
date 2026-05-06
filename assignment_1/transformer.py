import logging
import math

import torch
from torch import nn

from base_model import BaseModel


class Transformer(BaseModel):
    """
    A compact Transformer-based model for one-step-ahead time-series prediction.

    Input  (B, W, C) 
    Where:
        B = batch size
        W = window size / number of past timesteps
        C = number of input signals
        d_model = hidden_size
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        logger: logging.Logger,
        dropout: float = 0.1,
    ) -> None:
        """
        Define the Transformer model.

        :param input_size: Number of features/signals per timestep.
        :type input_size: int
        :param hidden_size: Transformer embedding dimension, also known as
            d_model.
        :type hidden_size: int
        :param num_layers: Number of Transformer encoder layers.
        :type num_layers: int
        :param logger: Logger to log to.
        :type logger: logging.Logger
        :param dropout: Dropout probability inside the Transformer layers.
        :type dropout: float
        """
        super().__init__(logger)

        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}.")

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.input_size = input_size
        self.d_model = hidden_size

        # The number of heads must divide hidden_size exactly
        nhead = self._select_num_heads(hidden_size)

        logger.debug(
            f"Transformer nhead automatically set to {nhead} "
            f"(d_model={hidden_size})."
        )

        # Project raw input features into Transformer embedding space
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Add fixed sinusoidal positional encodings so the Transformer knows
        # the order of the timesteps inside the input window
        self.pos_encoder = _PositionalEncoding(
            d_model=hidden_size,
            dropout=dropout,
        )

        # One standard Transformer encoder layer.
        #
        # batch_first=True means tensors have shape:
        # (batch_size, window_size, hidden_size)
        #
        # norm_first=True uses pre-layer normalisation, which is often more
        # stable for small datasets.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Stack multiple Transformer encoder layers.
        self.backbone = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Final normalisation before the prediction head.
        self.final_norm = nn.LayerNorm(hidden_size)

        # Map the last timestep representation back to the required output size.
        #
        # For this project:
        # (B, hidden_size) -> (B, 1)
        self.head = nn.Linear(hidden_size, input_size)

        self._initialise_weights()

    @staticmethod
    def _select_num_heads(hidden_size: int) -> int:
        """
        Select a valid number of attention heads for the given hidden size.

        The number of heads must divide hidden_size. For this small time-series
        task, we prefer 4 heads for hidden_size=32 and 8 heads for larger models.

        :param hidden_size: Transformer embedding dimension.
        :type hidden_size: int
        :return: Number of attention heads.
        :rtype: int
        """
        if hidden_size >= 64 and hidden_size % 8 == 0:
            return 8

        if hidden_size % 4 == 0:
            return 4

        if hidden_size % 2 == 0:
            return 2

        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next value given an input window of past values.
        :param x: Input tensor of shape
            (batch_size, window_size, input_size).
        :type x: torch.Tensor
        :return: Predicted next value of shape
            (batch_size, input_size).
        :rtype: torch.Tensor
        """
        # Raw signal values -> Transformer embeddings.
        #
        # (B, W, input_size) -> (B, W, hidden_size)
        x = self.input_projection(x)

        # Add positional information.
        x = self.pos_encoder(x)

        # Process the full input window using self-attention.
        encoded = self.backbone(x)

        # Stabilise the final encoded representations.
        encoded = self.final_norm(encoded)

        # Use the final timestep representation to predict the next value.
        return self.head(encoded[:, -1, :])


class _PositionalEncoding(nn.Module):
    """
    This implementation follows the classic sinusoidal encoding used in
    Transformer models.

    :param d_model: Transformer embedding dimension.
    :type d_model: int
    :param dropout: Dropout probability applied after adding positional encoding.
    :type dropout: float
    :param max_len: Maximum sequence length for which positional encodings are
        precomputed.
    :type max_len: int
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Position indices:
        # shape = (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # Frequency scaling terms for sine/cosine functions.
        # shape = approximately (d_model / 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # Positional encoding matrix:
        # shape = (1, max_len, d_model)
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)

        # Even dimensions use sine.
        pe[0, :, 0::2] = torch.sin(position * div_term)

        # Odd dimensions use cosine.
        #
        # The slicing on div_term makes this safe even if d_model is odd.
        pe[0, :, 1::2] = torch.cos(
            position * div_term[: pe[0, :, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        :param x: Tensor of shape (batch_size, seq_len, d_model).
        :type x: torch.Tensor
        :return: Tensor of the same shape with positional encoding added.
        :rtype: torch.Tensor
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)