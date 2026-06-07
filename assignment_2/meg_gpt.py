"""
MEG-GPT — a GPT-style (decoder-only) Transformer for raw MEG classification.

Inspired by:
    Huang et al. (2025) "MEG-GPT: A transformer-based foundation model for
    magnetoencephalography data"  arXiv:2510.18080
    https://github.com/OHBA-analysis/osl-foundation

The original MEG-GPT was built in TensorFlow for parcellated (38-region)
resting-state data and trained with a next-token-prediction objective.
This implementation re-uses the core design ideas in PyTorch and adapts them
for the supervised decoding setting used in this assignment:

  • Per-sensor linear projection  →  d_model tokens  (one token per sensor)
  • Learned positional embedding over the time axis is replaced by a
    per-timestep patch embedding (sensors are already spatial, not temporal)
  • Causal (lower-triangular) self-attention across the time dimension
  • Mean-pool over the resulting token sequence
  • Linear classification head

Input shape  : (B, C, T)  where C = n_sensors, T = window_size
Output shape : (B, num_classes)
"""

import logging
import math

import torch
from torch import nn

from base_model import BaseModel



class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    :param d_model: Embedding dimensionality.
    :type d_model: int
    :param max_len: Maximum sequence length supported.
    :type max_len: int
    :param dropout: Dropout applied after adding the encoding.
    :type dropout: float
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)           # (L, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()  # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (B, L, d_model).
        :type x: torch.Tensor
        :returns: Same shape with positional encoding added.
        :rtype: torch.Tensor
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)



class MEGGPT(BaseModel):
    """
    A GPT-style decoder-only Transformer for MEG brain-decoding.

    Architecture
    ~~~~~~~~~~~~
    1. **Sensor-axis patch embedding** — each sensor's time-series is
       projected from ``window_size`` scalar values to ``d_model``
       dimensions via a 1-D temporal convolution (patch size = ``patch_size``
       timesteps).  This gives a sequence of ``n_patches`` tokens per
       sensor.  All sensor tokens are concatenated, resulting in a
       sequence of length ``n_sensors × n_patches``.

    2. **Positional encoding** — sinusoidal encoding is added to every
       token so the model can distinguish both sensor position and
       temporal position within the sequence.

    3. **Causal Transformer decoder** — a stack of causal (masked)
       multi-head self-attention layers, each followed by a feed-forward
       sub-layer with GELU activations.  Causality ensures the model
       cannot look ahead in time, consistent with the autoregressive
       pre-training objective of the original MEG-GPT.

    4. **Classification head** — the token sequence is averaged (mean
       pooling) and passed through a linear layer that produces the
       final class logits.

    Parameters
    ~~~~~~~~~~
    :param num_electrodes: Number of MEG sensors (C).
    :type num_electrodes: int
    :param chunk_size: Window length in timesteps (T).
    :type chunk_size: int
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param d_model: Transformer embedding dimension (must be divisible by
        ``num_heads``).  Default: 64.
    :type d_model: int
    :param num_heads: Number of attention heads.  Default: 4.
    :type num_heads: int
    :param num_layers: Number of Transformer decoder layers.  Default: 2.
    :type num_layers: int
    :param patch_size: Number of consecutive timesteps grouped into one
        token (controls sequence length vs. resolution trade-off).
        Default: 32.
    :type patch_size: int
    :param ffn_dim: Hidden size of the feed-forward sublayer.  Defaults
        to ``2 × d_model``.
    :type ffn_dim: int | None
    :param dropout: Dropout probability used throughout.  Default: 0.1.
    :type dropout: float
    :param pretrained: If True, attempt to load transformer core weights
        from the original MEG-GPT TF checkpoint (requires d_model=400,
        tensorflow, and huggingface_hub).  Default: True.
    :type pretrained: bool
    """

    # HuggingFace repo id for the original MEG-GPT checkpoint (TF format).
    _PRETRAINED_REPO = "OHBA-analysis/MEG-GPT"
    # Expected d_model of the pretrained checkpoint.
    _PRETRAINED_D_MODEL = 400

    def __init__(
        self,
        num_electrodes: int,
        chunk_size: int,
        num_classes: int,
        logger: logging.Logger,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        patch_size: int = 32,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__(logger)

        if d_model % num_heads != 0:
            # Auto-adjust num_heads to the largest divisor of d_model that
            # is ≤ the requested value.
            adjusted = max(
                h for h in range(1, num_heads + 1) if d_model % h == 0
            )
            logger.warning(
                f"MEGGPT: d_model={d_model} is not divisible by "
                f"num_heads={num_heads}; adjusting num_heads to {adjusted}."
            )
            num_heads = adjusted

        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.patch_size = patch_size

        # ── 1. Per-sensor patch embedding ────────────────────────────────
        # Conv1d operates on (B*C, 1, T) → (B*C, d_model, n_patches)
        # We use a strided convolution with kernel == stride so patches
        # are non-overlapping, exactly as in vision transformers.
        self.patch_embed = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
            nn.LayerNorm(d_model),   # applied after transpose inside forward
        )

        # Compute resulting sequence length: each sensor contributes n_patches
        # tokens, and we have num_electrodes sensors → total_seq_len tokens.
        n_patches = chunk_size // patch_size
        self.n_patches = n_patches
        total_seq_len = num_electrodes * n_patches

        logger.debug(
            f"MEGGPT: n_patches={n_patches}, total_seq_len={total_seq_len}, "
            f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}"
        )

        # ── 2. Positional encoding ───────────────────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=total_seq_len + 1,
            dropout=dropout,
        )

        # ── 3. Causal Transformer decoder ────────────────────────────────
        # ffn_dim reduced from 4× to 2× d_model — halves FFN parameter count.
        ffn_dim = ffn_dim or d_model * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,    # expects (B, L, d_model)
            norm_first=True,     # Pre-LN (more stable, used in GPT-2+)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Causal (lower-triangular) attention mask — registered as buffer so
        # it moves to the correct device with the model.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            total_seq_len
        )  # (L, L) with -inf above diagonal
        self.register_buffer("causal_mask", causal_mask)

        # ── 4. Classification head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._initialise_weights()

        if pretrained:
            self._load_pretrained_weights()

    # ------------------------------------------------------------------
    def _load_pretrained_weights(self) -> None:
        """
        Download the original MEG-GPT TensorFlow checkpoint from HuggingFace
        and load the transformer core weights (attention + FFN + layer norms)
        into this model.

        Requires ``tensorflow`` and ``huggingface_hub`` to be installed.
        Falls back to random (Xavier) initialisation with a warning when
        either package is missing or when d_model does not match the
        pretrained checkpoint's d_model (400).

        Layers loaded   : transformer self-attention projections (Q, K, V,
                          output), feed-forward dense layers, layer norms.
        Layers skipped  : ``patch_embed`` (input dim mismatch: 52 parcels vs
                          this model's sensors), ``head`` (different number
                          of output classes), positional encoding.
        """
        if self.d_model != self._PRETRAINED_D_MODEL:
            self.logger.warning(
                f"MEGGPT pretrained: d_model={self.d_model} does not match "
                f"the checkpoint's d_model={self._PRETRAINED_D_MODEL}. "
                "Transformer weights cannot be transferred — falling back to "
                "random initialisation. Set d_model=400 (hidden_size: 400 in "
                "config) to enable pretrained loading."
            )
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            self.logger.warning(
                "MEGGPT pretrained: 'huggingface_hub' is not installed. "
                "Run 'pip install huggingface_hub' to enable pretrained "
                "weight loading. Falling back to random initialisation."
            )
            return

        try:
            import tensorflow as tf
        except ImportError:
            self.logger.warning(
                "MEGGPT pretrained: 'tensorflow' is not installed. "
                "Run 'pip install tensorflow' to enable pretrained weight "
                "loading. Falling back to random initialisation."
            )
            return

        self.logger.info(
            "MEGGPT pretrained: downloading checkpoint from HuggingFace "
            f"({self._PRETRAINED_REPO})..."
        )
        try:
            ckpt_index = hf_hub_download(
                repo_id=self._PRETRAINED_REPO,
                filename="meg-gpt/checkpoints/ckpt-60.index",
            )
            hf_hub_download(
                repo_id=self._PRETRAINED_REPO,
                filename="meg-gpt/checkpoints/ckpt-60.data-00000-of-00001",
            )
        except Exception as e:
            self.logger.warning(
                f"MEGGPT pretrained: failed to download checkpoint ({e}). "
                "Falling back to random initialisation."
            )
            return

        ckpt_path = ckpt_index.replace(".index", "")
        ckpt_reader = tf.train.load_checkpoint(ckpt_path)
        tf_vars = ckpt_reader.get_variable_to_shape_map()

        self.logger.debug(
            "MEGGPT pretrained: TF checkpoint variables:\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(tf_vars.items()))
        )

        # Build a mapping from PyTorch state dict keys to TF variable names.
        # The original decoder uses within-channel temporal self-attention
        # across n_layers=4 stacked TransformerEncoderLayer-equivalent blocks.
        # TF Keras names follow the pattern set by MultiHeadPASSTALayer and
        # the surrounding dense/norm layers — inspect the debug log above to
        # verify these names against the actual checkpoint.
        def _tf(name: str) -> str | None:
            """Return the TF variable array or None if the name is absent."""
            return ckpt_reader.get_tensor(name) if name in tf_vars else None

        loaded, skipped = [], []
        pt_state = self.state_dict()
        new_state: dict[str, torch.Tensor] = {}

        for layer_idx in range(len(self.transformer.layers)):
            prefix_pt = f"transformer.layers.{layer_idx}"
            # TF Keras indexing: first layer has no suffix, subsequent ones
            # are indexed starting at 1 (Keras default for list-of-layers).
            tf_suffix = "" if layer_idx == 0 else f"_{layer_idx}"
            prefix_tf = (
                f"meg_gpt/decoder/attention_layers{tf_suffix}"
            )
            ff_tf = (
                f"meg_gpt/decoder/feed_forward_layers{tf_suffix}"
            )
            norm1_tf = (
                f"meg_gpt/decoder/normalization_layers_1{tf_suffix}"
            )
            norm2_tf = (
                f"meg_gpt/decoder/normalization_layers_2{tf_suffix}"
            )

            # ── Self-attention ──────────────────────────────────────────
            # TF stores separate Q, K, V kernels; PyTorch packs them into
            # a single in_proj_weight of shape (3*d_model, d_model).
            q_w = _tf(f"{prefix_tf}/query/kernel")
            k_w = _tf(f"{prefix_tf}/key/kernel")
            v_w = _tf(f"{prefix_tf}/value/kernel")
            q_b = _tf(f"{prefix_tf}/query/bias")
            k_b = _tf(f"{prefix_tf}/key/bias")
            v_b = _tf(f"{prefix_tf}/value/bias")
            out_w = _tf(f"{prefix_tf}/output/kernel")
            out_b = _tf(f"{prefix_tf}/output/bias")

            if all(x is not None for x in [q_w, k_w, v_w, out_w]):
                # TF kernels are (in, out); PyTorch expects (out, in).
                qkv_w = torch.from_numpy(
                    __import__("numpy").concatenate(
                        [q_w.T, k_w.T, v_w.T], axis=0
                    )
                ).float()
                new_state[f"{prefix_pt}.self_attn.in_proj_weight"] = qkv_w
                loaded.append(f"{prefix_pt}.self_attn.in_proj_weight")

                if all(x is not None for x in [q_b, k_b, v_b]):
                    qkv_b = torch.from_numpy(
                        __import__("numpy").concatenate(
                            [q_b, k_b, v_b], axis=0
                        )
                    ).float()
                    new_state[f"{prefix_pt}.self_attn.in_proj_bias"] = qkv_b
                    loaded.append(f"{prefix_pt}.self_attn.in_proj_bias")

                new_state[f"{prefix_pt}.self_attn.out_proj.weight"] = (
                    torch.from_numpy(out_w.T).float()
                )
                loaded.append(f"{prefix_pt}.self_attn.out_proj.weight")

                if out_b is not None:
                    new_state[f"{prefix_pt}.self_attn.out_proj.bias"] = (
                        torch.from_numpy(out_b).float()
                    )
                    loaded.append(f"{prefix_pt}.self_attn.out_proj.bias")
            else:
                skipped.append(f"{prefix_pt}.self_attn (weights not found)")

            # ── Feed-forward ────────────────────────────────────────────
            ff1_w = _tf(f"{ff_tf}/dense/kernel")
            ff1_b = _tf(f"{ff_tf}/dense/bias")
            ff2_w = _tf(f"{ff_tf}/dense_1/kernel")
            ff2_b = _tf(f"{ff_tf}/dense_1/bias")

            if ff1_w is not None:
                new_state[f"{prefix_pt}.linear1.weight"] = (
                    torch.from_numpy(ff1_w.T).float()
                )
                loaded.append(f"{prefix_pt}.linear1.weight")
            if ff1_b is not None:
                new_state[f"{prefix_pt}.linear1.bias"] = (
                    torch.from_numpy(ff1_b).float()
                )
                loaded.append(f"{prefix_pt}.linear1.bias")
            if ff2_w is not None:
                new_state[f"{prefix_pt}.linear2.weight"] = (
                    torch.from_numpy(ff2_w.T).float()
                )
                loaded.append(f"{prefix_pt}.linear2.weight")
            if ff2_b is not None:
                new_state[f"{prefix_pt}.linear2.bias"] = (
                    torch.from_numpy(ff2_b).float()
                )
                loaded.append(f"{prefix_pt}.linear2.bias")

            # ── Layer norms ──────────────────────────────────────────────
            for pt_norm, tf_norm in [
                (f"{prefix_pt}.norm1", norm1_tf),
                (f"{prefix_pt}.norm2", norm2_tf),
            ]:
                g = _tf(f"{tf_norm}/gamma")
                b = _tf(f"{tf_norm}/beta")
                if g is not None:
                    new_state[f"{pt_norm}.weight"] = (
                        torch.from_numpy(g).float()
                    )
                    loaded.append(f"{pt_norm}.weight")
                if b is not None:
                    new_state[f"{pt_norm}.bias"] = (
                        torch.from_numpy(b).float()
                    )
                    loaded.append(f"{pt_norm}.bias")

        # Merge: keep existing values for keys not in new_state.
        pt_state.update(new_state)
        self.load_state_dict(pt_state, strict=False)

        self.logger.info(
            f"MEGGPT pretrained: loaded {len(loaded)} parameter tensors "
            f"from checkpoint. Skipped (random init): {skipped or ['none']}."
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: MEG window of shape (B, C, T) where B = batch size,
            C = ``num_electrodes``, T = ``chunk_size``.
        :type x: torch.Tensor
        :returns: Class logits of shape (B, num_classes).
        :rtype: torch.Tensor
        """
        B, C, T = x.shape

        # ── Patch embedding ──────────────────────────────────────────────
        # Reshape to treat every sensor independently:
        #   (B, C, T) → (B*C, 1, T)
        x = x.reshape(B * C, 1, T)

        # Conv1d: (B*C, 1, T) → (B*C, d_model, n_patches)
        x = self.patch_embed[0](x)                    # Conv1d
        x = x.permute(0, 2, 1)                        # → (B*C, n_patches, d_model)
        x = self.patch_embed[1](x)                    # LayerNorm

        # Re-assemble into (B, C * n_patches, d_model) so the transformer
        # attends across both sensors and time jointly.
        x = x.reshape(B, C * self.n_patches, self.d_model)

        # ── Positional encoding ──────────────────────────────────────────
        x = self.pos_enc(x)

        # ── Causal Transformer ───────────────────────────────────────────
        x = self.transformer(x, mask=self.causal_mask, is_causal=True)

        # ── Classification head ──────────────────────────────────────────
        # Mean-pool over the token sequence → (B, d_model)
        x = x.mean(dim=1)
        logits = self.head(x)

        return logits