import logging
import torch
import torch.nn as nn

from base_model import BaseModel


class STBlock(nn.Module):
    """
    Spatio-temporal block (ST-block) from WeatherGCNet
    (Stańczyk & Mehrkanoon, ESANN 2021).

    Each block maintains its own learnable adjacency matrix which is
    normalised on every forward pass. A graph spatial convolution aggregates
    information across sensor nodes, a temporal convolution captures local
    dynamics along the time axis, and a residual connection is wrapped around
    both operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        temporal_kernel_size: int = 3,
        gamma_learnable: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialiser.

        :param in_channels: Number of feature channels entering this block.
        :type in_channels: int
        :param out_channels: Number of feature channels leaving this block.
        :type out_channels: int
        :param num_nodes: Number of graph nodes (MEG sensors).
        :type num_nodes: int
        :param temporal_kernel_size: Kernel length k for the temporal
            convolution. Must be an odd integer so that same-padding
            preserves the time dimension.
        :type temporal_kernel_size: int
        :param gamma_learnable: If True, the self-loop strength γ is a
            trainable scalar (learnt-γ variant from the paper). If False,
            γ is fixed to 1 (γ=1 variant from the paper).
        :type gamma_learnable: bool
        :param dropout: Dropout probability applied after each ReLU.
            Set to 0.0 to disable.
        :type dropout: float
        """
        super().__init__()

        assert temporal_kernel_size % 2 == 1, (
            f"temporal_kernel_size must be odd, got {temporal_kernel_size}."
        )

        self.num_nodes = num_nodes

        # --- Learnable adjacency matrix (one per block, shape V × V) ---
        self.A = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.A)

        # Self-loop strength γ: learnable scalar or fixed buffer
        if gamma_learnable:
            self.gamma = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("gamma", torch.ones(1))

        # --- Graph spatial convolution ---
        # After adjacency-matrix aggregation the channel dimension is
        # unchanged; a 1×1 Conv2d over the (T, V) grid projects to
        # out_channels (matching the paper's "1×1 convolution" step).
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_spatial = nn.BatchNorm2d(out_channels)

        # --- Temporal convolution ---
        # (k × 1) kernel: looks at k consecutive time-steps for each node
        # independently (no mixing across the node dimension here).
        padding = (temporal_kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=(padding, 0),
        )
        self.bn_temporal = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Residual projection: needed only when channel dimensions differ
        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def _normalised_adjacency(self) -> torch.Tensor:
        """
        Build the symmetrically normalised adjacency matrix.

        Follows the paper exactly:
          1. Add self-loop:   Â  = A + γI
          2. Min-max scale:   Â  = (Â − Â_min) / (Â_max − Â_min)
          3. Degree matrix:   D̂_ii = Σ_j Â_ij
          4. Symmetric norm:  D̂^{−½} Â D̂^{−½}

        :returns: Normalised adjacency matrix of shape (num_nodes, num_nodes).
        :rtype: torch.Tensor
        """
        # Step 1 — self-loop
        A_hat = self.A + self.gamma * torch.eye(
            self.num_nodes, dtype=self.A.dtype, device=self.A.device
        )

        # Step 2 — min-max normalisation
        A_hat = (A_hat - A_hat.min()) / (A_hat.max() - A_hat.min() + 1e-8)

        # Steps 3 & 4 — symmetric degree normalisation
        # Avoid materialising a dense (V × V) diagonal matrix by using
        # broadcasting: equivalent to D^{-½} Â D^{-½}.
        deg = A_hat.sum(dim=1)                        # (V,)
        d_inv_sqrt = torch.pow(deg + 1e-8, -0.5)     # (V,)
        A_norm = d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)

        return A_norm  # (V, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Feature tensor of shape (batch, in_channels, timesteps,
            num_nodes).
        :type x: torch.Tensor
        :returns: Feature tensor of shape (batch, out_channels, timesteps,
            num_nodes).
        :rtype: torch.Tensor
        """
        batch, C, T, V = x.shape
        residual = self.residual_proj(x)

        # --- Graph spatial convolution ---
        # Paper eq. (1): X_out = X_in · (D̂^{−½} Â D̂^{−½})
        # X_in is reshaped to (B·C·T, V) so that the right-multiply by
        # A_norm aggregates neighbourhood information per (batch, channel,
        # time) position across the node dimension.
        A_norm = self._normalised_adjacency()          # (V, V)
        x_flat = x.reshape(batch * C * T, V)           # (B·C·T, V)
        x_flat = x_flat @ A_norm                       # (B·C·T, V)
        x = x_flat.reshape(batch, C, T, V)             # (B, C, T, V)

        # 1×1 conv projects to out_channels and mixes features channel-wise
        x = self.relu(self.bn_spatial(self.spatial_conv(x)))  # (B, C_out, T, V)
        x = self.dropout(x)

        # --- Temporal convolution ---
        x = self.bn_temporal(self.temporal_conv(x))    # (B, C_out, T, V)

        # --- Residual + final activation ---
        x = self.relu(x + residual)
        x = self.dropout(x)

        return x


class MEGGCNet(BaseModel):
    """
    Spatio-temporal graph convolutional network for MEG-based brain state
    classification, adapted from WeatherGCNet (Stańczyk & Mehrkanoon,
    ESANN 2021).

    The 248 MEG sensors are treated as nodes in a graph whose connectivity
    is learned end-to-end via a per-block learnable adjacency matrix. Three
    stacked ST-blocks progressively extract cross-sensor (spatial) and
    within-sensor (temporal) features. A global average pool over the time
    and node dimensions followed by a linear classifier produces one
    prediction per input window.

    Accepted input shapes:
        (batch, 248, window_size)        3-D, channel dim will be added
        (batch, 1,   248, window_size)   4-D pipeline default

    Output shape:
        (batch, num_classes)             raw logits (no softmax applied)
    """

    def __init__(
        self,
        logger: logging.Logger,
        num_nodes: int = 248,
        in_channels: int = 1,
        num_classes: int = 4,
        temporal_kernel_size: int = 3,
        gamma_learnable: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialiser.

        :param num_nodes: Number of graph nodes, i.e. MEG sensors.
        :type num_nodes: int
        :param in_channels: Feature channels per node in the input tensor.
            For raw MEG amplitude time-series this is 1.
        :type in_channels: int
        :param num_classes: Number of target classes. 4 for this task
            (rest, task_motor, task_story_math, task_working_memory).
        :type num_classes: int
        :param temporal_kernel_size: Temporal convolution kernel length k
            passed to every ST-block. Must be odd.
        :type temporal_kernel_size: int
        :param gamma_learnable: Whether the self-loop strength γ inside
            each ST-block is learnable (True) or fixed to 1 (False).
        :type gamma_learnable: bool
        :param dropout: Dropout probability used inside every ST-block.
            Set to 0.0 to disable.
        :type dropout: float
        """
        super().__init__(logger)

        shared = dict(
            num_nodes=num_nodes,
            temporal_kernel_size=temporal_kernel_size,
            gamma_learnable=gamma_learnable,
            dropout=dropout,
        )

        # Three ST-blocks with channel progression: in_channels → 16 → 32 → 64
        self.st_block1 = STBlock(in_channels, 16, **shared)
        self.st_block2 = STBlock(16,          32, **shared)
        self.st_block3 = STBlock(32,          64, **shared)

        # 1×1 conv reduces 64 → 4 channels before pooling (paper Fig. 2b)
        self.conv_reduce = nn.Conv2d(64, 4, kernel_size=1)

        # Classification head: global average pool collapses (T, V) to a
        # scalar per channel, yielding a 4-dim vector; linear maps to logits.
        self.classifier = nn.Linear(4, num_classes)

        self.logger = logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Raw MEG input of shape (batch, 1, 248, window_size) or
            (batch, 248, window_size).
        :type x: torch.Tensor
        :returns: Raw class logits of shape (batch, num_classes).
        :rtype: torch.Tensor
        """
        # Accept both 3-D and 4-D inputs
        if x.dim() == 3:
            x = x.unsqueeze(1)                      # (B, 1, V, T)

        # Pipeline shape is (B, 1, V=248, T); WeatherGCNet expects (B, C, T, V)
        x = x.permute(0, 1, 3, 2).contiguous()     # (B, C=1, T, V=248)

        x = self.st_block1(x)                       # (B,  16, T, 248)
        x = self.st_block2(x)                       # (B,  32, T, 248)
        x = self.st_block3(x)                       # (B,  64, T, 248)

        x = self.conv_reduce(x)                     # (B,   4, T, 248)

        # Global average pool over time and node dimensions
        x = x.mean(dim=[2, 3])                      # (B, 4)

        return self.classifier(x)                   # (B, num_classes)
