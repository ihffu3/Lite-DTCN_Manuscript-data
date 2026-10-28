"""Predeclared Lite-DTCN candidates for development-only training.

The module has no import-time file access, training, or device allocation.
Inputs and outputs are normalized real/imaginary CSI tensors; all windows must
be assembled by the runner so that different history lengths share target
origins.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


RI_COMPONENTS = 2
DEFAULT_N_TX = 8
DEFAULT_N_FREQ = 32
DEFAULT_NODE_DIM = 16
DEFAULT_TEMPORAL_HIDDEN = 32
DEFAULT_HISTORY_LENGTH = 10
CONDITION_HIDDEN = 16


@dataclass(frozen=True)
class CandidateConfig:
    """One immutable entry from the T05 development candidate list."""

    id: str
    mode: str
    length: int
    residual: bool
    lr: float


_FROZEN_CONFIGS = {
    config.id: config
    for config in (
        CandidateConfig("full10_direct_lr1e3", "full_window", 10, False, 0.001),
        CandidateConfig("full10_residual_lr1e3", "full_window", 10, True, 0.001),
        CandidateConfig("two_direct_lr1e3", "exact_two", 2, False, 0.001),
        CandidateConfig("causal3_direct_lr1e3", "causal_k3", 10, False, 0.001),
        CandidateConfig("dilated_direct_lr1e3", "causal_dilated_1_2_4", 10, False, 0.001),
        CandidateConfig("full10_direct_lr1e2", "full_window", 10, False, 0.01),
        CandidateConfig("full10_residual_lr1e2", "full_window", 10, True, 0.01),
        CandidateConfig("two_residual_lr1e3", "exact_two", 2, True, 0.001),
    )
}


def available_candidate_configs():
    """Return IDs in the frozen protocol order; no exploratory configs exist."""
    return tuple(_FROZEN_CONFIGS)


def resolve_candidate_config(config):
    """Resolve an ID/config mapping and reject changes to frozen fields."""
    if isinstance(config, CandidateConfig):
        config_id = config.id
        supplied = {
            "mode": config.mode,
            "length": config.length,
            "residual": config.residual,
            "lr": config.lr,
        }
    elif isinstance(config, str):
        config_id = config
        supplied = {}
    elif isinstance(config, Mapping):
        config_id = config.get("id", config.get("config_id"))
        if not isinstance(config_id, str):
            raise ValueError("candidate config mapping must include a known id")
        allowed = {"id", "config_id", "mode", "length", "residual", "lr"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown candidate config fields: {sorted(unknown)}")
        supplied = {key: value for key, value in config.items() if key not in ("id", "config_id")}
    else:
        raise ValueError("config must be a frozen candidate ID, CandidateConfig, or mapping")

    if config_id not in _FROZEN_CONFIGS:
        raise ValueError(f"candidate config is not in the frozen Lite family: {config_id!r}")
    frozen = _FROZEN_CONFIGS[config_id]
    for key, value in supplied.items():
        if value != getattr(frozen, key):
            raise ValueError(f"candidate config {config_id!r} changes frozen field {key!r}")
    return frozen


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_history(history, *, length, n_tx, n_freq, expected_dtype):
    if not isinstance(history, torch.Tensor):
        raise ValueError("history must be a torch.Tensor")
    if history.ndim != 5 or history.shape[0] < 1:
        raise ValueError("history must have shape [B,L,Nt,Nf,2] with B>=1")
    expected = (length, n_tx, n_freq, RI_COMPONENTS)
    if tuple(history.shape[1:]) != expected:
        raise ValueError(
            "history shape must be [B," + ",".join(map(str, expected)) + "]"
        )
    if history.dtype not in (torch.float32, torch.float64):
        raise ValueError("history must use float32 or float64 normalized RI values")
    if history.dtype != expected_dtype:
        raise ValueError("history dtype must match model parameter dtype")


class _LiteTemporalBackbone(nn.Module):
    """Shared per-frame encoder and endpoint temporal readout."""

    def __init__(
        self,
        *,
        mode,
        history_length,
        n_tx,
        n_freq,
        node_dim,
        temporal_hidden,
    ):
        super().__init__()
        self.mode = mode
        self.history_length = _positive_int(history_length, "history_length")
        self.n_tx = _positive_int(n_tx, "n_tx")
        self.n_freq = _positive_int(n_freq, "n_freq")
        self.node_dim = _positive_int(node_dim, "node_dim")
        self.temporal_hidden = _positive_int(temporal_hidden, "temporal_hidden")
        if mode not in ("full_window", "causal_k3", "causal_dilated_1_2_4"):
            raise ValueError(f"unsupported temporal mode: {mode!r}")
        if mode == "causal_k3" and self.history_length < 3:
            raise ValueError("causal_k3 requires at least three observed frames")

        self.temporal_channels = self.n_tx * self.node_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.n_freq * RI_COMPONENTS, self.node_dim),
            nn.ReLU(),
        )
        if mode == "full_window":
            self.temporal_layers = nn.ModuleList(
                [
                    nn.Conv1d(
                        self.temporal_channels,
                        self.temporal_channels,
                        kernel_size=self.history_length,
                        padding=0,
                        groups=self.temporal_channels,
                    )
                ]
            )
            self.dilations = (1,)
        elif mode == "causal_k3":
            self.temporal_layers = nn.ModuleList(
                [
                    nn.Conv1d(
                        self.temporal_channels,
                        self.temporal_channels,
                        kernel_size=3,
                        padding=0,
                        groups=self.temporal_channels,
                    )
                ]
            )
            self.dilations = (1,)
        else:
            self.dilations = (1, 2, 4)
            self.temporal_layers = nn.ModuleList(
                [
                    nn.Conv1d(
                        self.temporal_channels,
                        self.temporal_channels,
                        kernel_size=3,
                        padding=0,
                        dilation=dilation,
                        groups=self.temporal_channels,
                    )
                    for dilation in self.dilations
                ]
            )
        self.pointwise_temporal = nn.Conv1d(
            self.temporal_channels, self.temporal_hidden, kernel_size=1
        )

    @property
    def receptive_field(self):
        if self.mode == "full_window":
            return self.history_length
        if self.mode == "causal_k3":
            return 3
        return 1 + 2 * sum(self.dilations)

    def _validate(self, history):
        _validate_history(
            history,
            length=self.history_length,
            n_tx=self.n_tx,
            n_freq=self.n_freq,
            expected_dtype=self.feature_encoder[0].weight.dtype,
        )

    def encode_endpoint(self, history):
        self._validate(history)
        batch, length = history.shape[:2]
        encoded = self.feature_encoder(
            history.reshape(batch * length * self.n_tx, self.n_freq * RI_COMPONENTS)
        ).reshape(batch, length, self.temporal_channels)
        temporal = encoded.transpose(1, 2)
        if self.mode == "full_window":
            temporal = self.temporal_layers[0](temporal)
        elif self.mode == "causal_k3":
            temporal = F.pad(temporal, (2, 0))
            temporal = self.temporal_layers[0](temporal)
        else:
            for layer, dilation in zip(self.temporal_layers, self.dilations):
                temporal = F.pad(temporal, (2 * dilation, 0))
                temporal = torch.relu(layer(temporal))
        temporal = torch.relu(self.pointwise_temporal(temporal))[:, :, -1]
        return temporal


class _LiteCandidateBase(nn.Module):
    def __init__(
        self,
        *,
        mode,
        history_length,
        n_tx=DEFAULT_N_TX,
        n_freq=DEFAULT_N_FREQ,
        node_dim=DEFAULT_NODE_DIM,
        temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
        residual=False,
        candidate_id=None,
        learning_rate=None,
    ):
        super().__init__()
        if not isinstance(residual, bool):
            raise ValueError("residual must be a bool")
        self.backbone = _LiteTemporalBackbone(
            mode=mode,
            history_length=history_length,
            n_tx=n_tx,
            n_freq=n_freq,
            node_dim=node_dim,
            temporal_hidden=temporal_hidden,
        )
        self.history_length = self.backbone.history_length
        self.n_tx = self.backbone.n_tx
        self.n_freq = self.backbone.n_freq
        self.node_dim = self.backbone.node_dim
        self.temporal_hidden = self.backbone.temporal_hidden
        self.mode = mode
        self.residual = residual
        self.candidate_id = candidate_id
        self.learning_rate = learning_rate
        self.prediction_head = nn.Linear(
            self.temporal_hidden,
            self.n_tx * self.n_freq * RI_COMPONENTS,
        )

    def forward(self, history, conditions=None):
        """Map observed normalized RI history to a normalized direct target.

        Dedicated candidates accept the interface's optional conditions field
        for runner compatibility but do not use it.
        """
        latent = self.backbone.encode_endpoint(history)
        batch = history.shape[0]
        prediction = self.prediction_head(latent).reshape(
            batch, self.n_tx, self.n_freq, RI_COMPONENTS
        )
        if self.residual:
            prediction = prediction + history[:, -1]
        return prediction


class FullWindowLiteDTCN(_LiteCandidateBase):
    """Endpoint full-window depthwise kernel; supports L=2,3,5,10,20."""

    def __init__(
        self,
        history_length=DEFAULT_HISTORY_LENGTH,
        n_tx=DEFAULT_N_TX,
        n_freq=DEFAULT_N_FREQ,
        node_dim=DEFAULT_NODE_DIM,
        temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
        residual=False,
    ):
        super().__init__(
            mode="full_window",
            history_length=history_length,
            n_tx=n_tx,
            n_freq=n_freq,
            node_dim=node_dim,
            temporal_hidden=temporal_hidden,
            residual=residual,
        )


class ExactTwoFrameLiteDTCN(FullWindowLiteDTCN):
    """Independently trainable L=2 full-window candidate."""

    def __init__(
        self,
        n_tx=DEFAULT_N_TX,
        n_freq=DEFAULT_N_FREQ,
        node_dim=DEFAULT_NODE_DIM,
        temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
        residual=False,
    ):
        super().__init__(
            history_length=2,
            n_tx=n_tx,
            n_freq=n_freq,
            node_dim=node_dim,
            temporal_hidden=temporal_hidden,
            residual=residual,
        )
        self.candidate_id = "independent_exact_two_frame"


class CausalK3LiteDTCN(_LiteCandidateBase):
    """Causal kernel-3 readout with exactly three-frame endpoint support."""

    def __init__(
        self,
        history_length=DEFAULT_HISTORY_LENGTH,
        n_tx=DEFAULT_N_TX,
        n_freq=DEFAULT_N_FREQ,
        node_dim=DEFAULT_NODE_DIM,
        temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
        residual=False,
    ):
        super().__init__(
            mode="causal_k3",
            history_length=history_length,
            n_tx=n_tx,
            n_freq=n_freq,
            node_dim=node_dim,
            temporal_hidden=temporal_hidden,
            residual=residual,
        )


class DilatedCausalLiteDTCN(_LiteCandidateBase):
    """Three causal depthwise blocks with dilation 1, 2, and 4."""

    def __init__(
        self,
        history_length=DEFAULT_HISTORY_LENGTH,
        n_tx=DEFAULT_N_TX,
        n_freq=DEFAULT_N_FREQ,
        node_dim=DEFAULT_NODE_DIM,
        temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
        residual=False,
    ):
        super().__init__(
            mode="causal_dilated_1_2_4",
            history_length=history_length,
            n_tx=n_tx,
            n_freq=n_freq,
            node_dim=node_dim,
            temporal_hidden=temporal_hidden,
            residual=residual,
        )


def create_candidate(
    config,
    *,
    n_tx=DEFAULT_N_TX,
    n_freq=DEFAULT_N_FREQ,
    node_dim=DEFAULT_NODE_DIM,
    temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
):
    """Construct exactly one of the eight frozen Lite-family candidates."""
    spec = resolve_candidate_config(config)
    common = {
        "n_tx": n_tx,
        "n_freq": n_freq,
        "node_dim": node_dim,
        "temporal_hidden": temporal_hidden,
        "residual": spec.residual,
    }
    if spec.mode == "full_window":
        model = FullWindowLiteDTCN(history_length=spec.length, **common)
    elif spec.mode == "exact_two":
        model = ExactTwoFrameLiteDTCN(**common)
    elif spec.mode == "causal_k3":
        model = CausalK3LiteDTCN(history_length=spec.length, **common)
    elif spec.mode == "causal_dilated_1_2_4":
        model = DilatedCausalLiteDTCN(history_length=spec.length, **common)
    else:  # Defensive: frozen mappings are statically declared above.
        raise ValueError(f"unsupported frozen candidate mode: {spec.mode!r}")
    model.candidate_id = spec.id
    model.learning_rate = spec.lr
    return model


def candidate_complexity(
    mode,
    *,
    history_length=DEFAULT_HISTORY_LENGTH,
    n_tx=DEFAULT_N_TX,
    n_freq=DEFAULT_N_FREQ,
    node_dim=DEFAULT_NODE_DIM,
    temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
    residual=False,
):
    """Return analytical parameters and MACs for the declared layer shapes.

    The count includes all Linear/Conv1d multiply-accumulates and their layer
    shapes, including left-padded kernel slots. It excludes activation, bias,
    and residual additions and is not a latency or memory measurement.
    """
    history_length = _positive_int(history_length, "history_length")
    n_tx = _positive_int(n_tx, "n_tx")
    n_freq = _positive_int(n_freq, "n_freq")
    node_dim = _positive_int(node_dim, "node_dim")
    temporal_hidden = _positive_int(temporal_hidden, "temporal_hidden")
    if not isinstance(residual, bool):
        raise ValueError("residual must be a bool")
    if mode == "exact_two":
        if history_length != 2:
            raise ValueError("exact_two complexity requires history_length=2")
        mode = "full_window"
    if mode not in ("full_window", "causal_k3", "causal_dilated_1_2_4"):
        raise ValueError(f"unsupported complexity mode: {mode!r}")
    if mode == "causal_k3" and history_length < 3:
        raise ValueError("causal_k3 complexity requires at least three frames")

    channels = n_tx * node_dim
    output_features = n_tx * n_freq * RI_COMPONENTS
    encoder_parameters = n_freq * RI_COMPONENTS * node_dim + node_dim
    pointwise_parameters = temporal_hidden * channels + temporal_hidden
    head_parameters = temporal_hidden * output_features + output_features
    encoder_macs = history_length * n_tx * n_freq * RI_COMPONENTS * node_dim
    pointwise_positions = 1 if mode == "full_window" else history_length
    pointwise_macs = pointwise_positions * temporal_hidden * channels
    head_macs = temporal_hidden * output_features

    if mode == "full_window":
        kernels = (history_length,)
        temporal_positions = 1
        receptive_field = history_length
    elif mode == "causal_k3":
        kernels = (3,)
        temporal_positions = history_length
        receptive_field = 3
    else:
        kernels = (3, 3, 3)
        temporal_positions = history_length
        receptive_field = 15
    depthwise_parameters = sum(channels * kernel + channels for kernel in kernels)
    depthwise_macs = temporal_positions * channels * sum(kernels)
    total_parameters = (
        encoder_parameters
        + depthwise_parameters
        + pointwise_parameters
        + head_parameters
    )
    total_macs = encoder_macs + depthwise_macs + pointwise_macs + head_macs
    return {
        "mode": mode,
        "history_length": history_length,
        "receptive_field": receptive_field,
        "parameters": total_parameters,
        "macs": total_macs,
        "residual_additions": output_features if residual else 0,
        "components": {
            "encoder_parameters": encoder_parameters,
            "depthwise_parameters": depthwise_parameters,
            "pointwise_parameters": pointwise_parameters,
            "head_parameters": head_parameters,
            "encoder_macs": encoder_macs,
            "depthwise_macs": depthwise_macs,
            "pointwise_macs": pointwise_macs,
            "head_macs": head_macs,
        },
    }

