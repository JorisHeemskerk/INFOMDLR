"""
EEGNetTransformer — Hybrid temporal-conv + spatial-attention model.

Motivation
EEGNet's spatial depth-wise convolution collapses all sensors into a
weighted sum whose weights are fixed by their sequential index. With 248
MEG sensors stored in arbitrary order this is a poor inductive bias.

This model keeps everything in EEGNet that works well (temporal conv,
temporal mixer) and replaces only the spatial depth-wise conv with a
Transformer encoder that treats each sensor as a token. A learnable CLS
token aggregates spatial information, so the model can learn which
sensors (and which combinations of sensors) are relevant, entirely from
data and without any geometric prior.
"""

import logging
import torch
from torch import nn

from base_model import BaseModel


class SpatialTransformerBlock(nn.Module):
    """
    Multi-head self-attention over the sensor (electrode/MEG channel) 
    dimension.

    Sensors are treated as sequence tokens. A single learnable CLS token
    is prepended; its representation after attention is returned as the 
    aggregated spatial feature vector, one per (batch, timestep) pair.
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.25,
    ):
        """
        Initialiser

        :param in_features: Feature dimension of each sensor token 
            coming in (equals F1, the number of temporal filters).
        :type in_features: int
        :param embed_dim: Internal attention dimensionality. If 
            different from in_features a learned linear projection is 
            applied first.
        :type embed_dim: int
        :param num_heads: Number of attention heads. Must evenly divide
            embed_dim.
        :type num_heads: int
        :param num_layers: Number of stacked TransformerEncoder layers.
        :type num_layers: int
        :param dropout: Dropout probability inside the transformer.
        :type dropout: float
        """
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads " \
            f"({num_heads})."

        self.input_proj = (
            nn.Linear(in_features, embed_dim, bias=False) 
                if in_features != embed_dim else nn.Identity()
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cls_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, F1, C, T) output of the temporal conv block.
        :type x: torch.Tensor
        :returns: (batch, embed_dim, 1, T) one spatial summary per 
            timestep, ready for the post-attention.
        :rtype: torch.Tensor
        """
        batch, F1, C, T = x.shape

        # Reshape so sensors are tokens.
        x = x.permute(0, 3, 2, 1).contiguous()

        x = x.reshape(batch * T, C, F1)
        x = self.input_proj(x)

        cls = self.cls_token.expand(batch * T, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        cls_out = self.cls_norm(x[:, 0, :])

        # Restore spatial dims.
        cls_out = cls_out.reshape(batch, T, -1)
        cls_out = cls_out.permute(0, 2, 1)
        cls_out = cls_out.unsqueeze(2)

        return cls_out

class EEGNetTransformer(BaseModel):
    """
    Hybrid temporal-conv / spatial-transformer model for MEG/EEG decoding.

    Drop-in replacement for EEGNet when sensor ordering carries no meaningful
    spatial information.  The spatial depthwise convolution is replaced by a
    Transformer encoder with a CLS token; everything else mirrors EEGNet so
    results are directly comparable.

    Args:
        logger (logging.Logger): Passed to BaseModel.
        chunk_size (int): Timepoints T per trial. (default: 151)
        num_electrodes (int): Number of sensors C. (default: 60)
        F1 (int): Temporal filter count. Block 1 output channels.
            (default: 8)
        F2 (int): Output channels of the temporal mixer: Block 3.
            (default: 16)
        D (int): Spatial embedding multiplier; embed_dim = F1 * D.
            Keeps the channel count entering Block 3 identical to 
            EEGNet. (default: 2)
        num_classes (int): Number of output classes. (default: 2)
        kernel_1 (int): Temporal conv kernel size in Block 1. 
            (default: 64)
        kernel_2 (int): Separable conv kernel size in Block 3. 
            (default: 16)
        num_heads (int): Attention heads in the spatial transformer.
            Must divide F1 * D. (default: 4)
        num_transformer_layers (int): Stacked encoder layers. 
            (default: 1)
        dropout (float): Dropout probability throughout. (default: 0.25)
    """

    def __init__(
        self,
        logger: logging.Logger,
        chunk_size: int = 151,
        num_electrodes: int = 60,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        num_classes: int = 2,
        kernel_1: int = 64,
        kernel_2: int = 16,
        num_heads: int = 4,
        num_transformer_layers: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__(logger)

        embed_dim = F1 * D

        # Block 1: Temporal Conv
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1, F1,
                kernel_size=(1, kernel_1),
                stride=1,
                padding=(0, kernel_1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            nn.AvgPool2d((1, 4), stride=4), # NOTE: moved here from spatial_post
        )

        # Block 2: Spatial Transformer
        self.spatial_transformer = SpatialTransformerBlock(
            in_features=F1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )
        self.spatial_post = nn.Sequential(
            nn.BatchNorm2d(embed_dim, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            # nn.AvgPool2d((1, 4), stride=4), # NOTE: moved this to temporal_conv
            nn.Dropout(p=dropout),
        )

        # Block 3: Temporal Mixer 
        self.temporal_mixer = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=(1, kernel_2),
                stride=1,
                padding=(0, kernel_2 // 2),
                bias=False,
                groups=embed_dim,
            ),
            nn.Conv2d(embed_dim, F2, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout),
        )

        self.lin = nn.Linear(
            self._feature_dim(chunk_size, num_electrodes),
            num_classes,
            bias=False,
        )

    def _feature_dim(self, chunk_size: int, num_electrodes: int) -> int:
        """
        Return the flattened feature size entering the linear classifier
        """
        with torch.no_grad():
            mock = torch.zeros(1, 1, num_electrodes, chunk_size)
            mock = self.temporal_conv(mock)
            mock = self.spatial_transformer(mock)
            mock = self.spatial_post(mock)
            mock = self.temporal_mixer(mock)
        return mock.flatten(start_dim=1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Shape (batch, num_electrodes, chunk_size).
        :type x: torch.Tensor
        :returns: Shape (batch, num_classes) with class logits.
        :rtype: torch.Tensor
        """
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_transformer(x)
        x = self.spatial_post(x)
        x = self.temporal_mixer(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        return x
