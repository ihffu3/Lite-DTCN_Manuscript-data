"""Variable-token iTransformer adaptation for normalized complex CSI in RI form."""

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


_CONFIGS = {
    "legacy_adaptation": {
        "id": "legacy_adaptation",
        "d_model": 128,
        "heads": 4,
        "layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "optimizer": "Adam",
        "lr": 1e-4,
        "batch_size": 32,
    },
    "legacy_adaptation_lr3e4": {
        "id": "legacy_adaptation_lr3e4",
        "d_model": 128,
        "heads": 4,
        "layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "optimizer": "Adam",
        "lr": 3e-4,
        "batch_size": 32,
    },
}


def _resolve_config(config):
    if config is None:
        key = "legacy_adaptation"
    elif isinstance(config, str):
        key = config
    elif isinstance(config, Mapping):
        key = config.get("id", "legacy_adaptation")
        if key not in _CONFIGS:
            raise ValueError(f"unknown iTransformer configuration: {key!r}")
        expected = _CONFIGS[key]
        unknown = set(config) - set(expected)
        mismatched = {
            name: (config[name], expected[name])
            for name in set(config) & set(expected)
            if config[name] != expected[name]
        }
        if unknown or mismatched:
            raise ValueError(
                f"iTransformer config must match the frozen specification; "
                f"unknown={sorted(unknown)}, mismatched={mismatched}"
            )
    else:
        raise TypeError("config must be None, a frozen config id, or a mapping")
    try:
        return deepcopy(_CONFIGS[key])
    except KeyError as exc:
        raise ValueError(f"unknown iTransformer configuration: {key!r}") from exc


class InvertedEmbedding(nn.Module):
    """Embed each coordinate's observed L-frame history as one token."""

    def __init__(self, history_length, d_model, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(history_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history_by_token):
        return self.dropout(self.value_embedding(history_by_token))


class _VariableAttention(nn.Module):
    """Full multihead attention over coordinate tokens, never over batch rows."""

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by the number of heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.dropout = dropout
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, value):
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens):
        query = self._split_heads(self.query_projection(tokens))
        key = self._split_heads(self.key_projection(tokens))
        value = self._split_heads(self.value_projection(tokens))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.head_dim**-0.5,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(tokens.shape)
        return self.out_projection(attended)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attention = _VariableAttention(d_model, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, tokens):
        tokens = self.norm1(tokens + self.dropout(self.attention(tokens)))
        residual = tokens
        feed_forward = self.conv1(tokens.transpose(1, 2))
        feed_forward = self.dropout(self.activation(feed_forward))
        feed_forward = self.dropout(self.conv2(feed_forward).transpose(1, 2))
        return self.norm2(residual + feed_forward)


class CSIiTransformer(nn.Module):
    """iTransformer-style direct snapshot adapter with window-local RevIN."""

    def __init__(self, history_length=10, n_tx=8, n_freq=32, config=None):
        super().__init__()
        self.config = _resolve_config(config)
        self.history_length = int(history_length)
        self.n_tx = int(n_tx)
        self.n_freq = int(n_freq)
        if self.history_length < 1 or self.n_tx < 1 or self.n_freq < 1:
            raise ValueError("history_length, n_tx and n_freq must be positive")
        self.n_variates = self.n_tx * self.n_freq * 2
        self.d_model = int(self.config["d_model"])
        self.heads = int(self.config["heads"])
        self.d_ff = int(self.config["d_ff"])
        self.dropout_rate = float(self.config["dropout"])
        self.embedding = InvertedEmbedding(
            self.history_length, self.d_model, self.dropout_rate
        )
        self.encoder = nn.ModuleList(
            [
                _EncoderLayer(
                    self.d_model,
                    self.heads,
                    self.d_ff,
                    self.dropout_rate,
                )
                for _ in range(self.config["layers"])
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.d_model)
        self.projector = nn.Linear(self.d_model, 1)
        self.optimizer_config = {
            "name": self.config["optimizer"],
            "lr": self.config["lr"],
            "batch_size": self.config["batch_size"],
            "weight_decay": 0.0,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience": 10,
                "min_lr": 1e-6,
            },
            "max_epochs": 200,
            "early_stopping_patience": 30,
        }

    def forward(self, history, conditions=None):
        del conditions  # Dedicated baseline: known speed/horizon are deliberately unused.
        if not isinstance(history, torch.Tensor) or history.ndim != 5:
            raise ValueError("history must be a tensor with shape [B,L,Nt,Nf,2]")
        expected = (self.history_length, self.n_tx, self.n_freq, 2)
        if tuple(history.shape[1:]) != expected:
            raise ValueError(f"history trailing shape must be {expected}")

        batch = history.shape[0]
        series = history.reshape(batch, self.history_length, self.n_variates)
        # Per-sample, per-coordinate history-only RevIN; mean detach is legacy behavior.
        means = series.mean(dim=1, keepdim=True).detach()
        centered = series - means
        stdev = torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / stdev

        tokens = self.embedding(normalized.transpose(1, 2))
        for layer in self.encoder:
            tokens = layer(tokens)
        tokens = self.encoder_norm(tokens)
        forecast = self.projector(tokens).transpose(1, 2)
        restored = forecast * stdev + means
        return restored[:, 0].reshape(batch, self.n_tx, self.n_freq, 2)


def build_itransformer(config=None, *, history_length=10, n_tx=8, n_freq=32):
    """Construct one of the two predeclared iTransformer training configurations."""
    return CSIiTransformer(
        history_length=history_length,
        n_tx=n_tx,
        n_freq=n_freq,
        config=config,
    )


def itransformer_parameter_count(
    history_length=10,
    n_tx=8,
    n_freq=32,
    d_model=128,
    heads=4,
    layers=2,
    d_ff=256,
):
    """Analytical trainable-parameter count for the frozen variable-token model."""
    del heads  # Attention projections have d_model-wide outputs independent of heads.
    variates = int(n_tx) * int(n_freq) * 2
    length = int(history_length)
    width = int(d_model)
    feed_forward = int(d_ff)

    def linear(inputs, outputs):
        return inputs * outputs + outputs

    embedding = linear(length, width)
    attention = 4 * linear(width, width)
    feed_forward_parameters = linear(width, feed_forward) + linear(feed_forward, width)
    two_layer_norms = 2 * (2 * width)
    encoder = int(layers) * (
        attention + feed_forward_parameters + two_layer_norms
    )
    final_norm = 2 * width
    output = linear(width, 1)
    # One Linear(128,1) is shared over token positions by PyTorch.
    return int(embedding + encoder + final_norm + output)


def complexity(model):
    """Exact parameter count and token/matrix dimensions for the adapted model."""
    actual = sum(parameter.numel() for parameter in model.parameters())
    analytical = itransformer_parameter_count(
        history_length=model.history_length,
        n_tx=model.n_tx,
        n_freq=model.n_freq,
        d_model=model.d_model,
        heads=model.heads,
        layers=len(model.encoder),
        d_ff=model.d_ff,
    )
    return {
        "parameters": actual,
        "analytical_parameters": analytical,
        "history_length": model.history_length,
        "n_variates": model.n_variates,
        "attention_tokens": model.n_variates,
        "attention_heads": model.heads,
        "encoder_layers": len(model.encoder),
        "attention_score_shape_per_head": (model.n_variates, model.n_variates),
        "measured_latency": False,
    }
