"""Single-checkpoint speed/horizon-conditioned Lite-DTCN models."""
import torch
from torch import nn

from .candidates import (
    CONDITION_HIDDEN,
    DEFAULT_N_FREQ,
    DEFAULT_N_TX,
    DEFAULT_NODE_DIM,
    DEFAULT_TEMPORAL_HIDDEN,
    RI_COMPONENTS,
    _LiteTemporalBackbone,
    _positive_int,
    candidate_complexity,
    resolve_candidate_config,
)


CONDITION_SCALE = (300.0, 4.0)
CONDITION_VERSIONS = ("tau_only", "speed_tau")


def _validate_conditions(conditions, *, batch_size, expected_dtype, expected_device):
    if not isinstance(conditions, torch.Tensor):
        raise ValueError("conditions must be a torch.Tensor with shape [B,2]")
    if conditions.ndim != 2 or tuple(conditions.shape) != (batch_size, 2):
        raise ValueError("conditions must have shape [B,2] for [speed_kmh,horizon_ms]")
    if conditions.dtype not in (torch.float32, torch.float64):
        raise ValueError("conditions must use float32 or float64 values")
    if conditions.dtype != expected_dtype:
        raise ValueError("conditions dtype must match model parameter dtype")
    if conditions.device != expected_device:
        raise ValueError("conditions and history must be on the same device")


class ConditionedLiteDTCN(nn.Module):
    """One shared temporal backbone and one condition embedding per model.

    ``tau_only`` slices the horizon column before any speed operation, so a
    missing/NaN speed estimate cannot alter its prediction. ``speed_tau`` uses
    the current speed and horizon in raw km/h and ms, scaled internally by
    300 km/h and 4 ms. Neither variant uses future CSI or per-condition weight
    banks. Residual variants add the final observed normalized RI frame.
    """

    def __init__(
        self,
        backbone_config,
        *,
        version,
        n_tx=DEFAULT_N_TX,
        n_freq=DEFAULT_N_FREQ,
        node_dim=DEFAULT_NODE_DIM,
        temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
        condition_hidden=CONDITION_HIDDEN,
    ):
        super().__init__()
        if version not in CONDITION_VERSIONS:
            raise ValueError(f"version must be one of {CONDITION_VERSIONS}")
        if condition_hidden != CONDITION_HIDDEN:
            raise ValueError(f"condition_hidden is frozen at {CONDITION_HIDDEN}")
        spec = resolve_candidate_config(backbone_config)
        self.backbone_config = spec
        self.candidate_id = spec.id
        self.version = version
        self.condition_hidden = _positive_int(condition_hidden, "condition_hidden")
        self.condition_scale = CONDITION_SCALE
        self.history_length = spec.length
        self.n_tx = _positive_int(n_tx, "n_tx")
        self.n_freq = _positive_int(n_freq, "n_freq")
        self.temporal_hidden = _positive_int(temporal_hidden, "temporal_hidden")
        self.residual = spec.residual
        if spec.mode == "exact_two":
            backbone_mode = "full_window"
        else:
            backbone_mode = spec.mode
        self.backbone = _LiteTemporalBackbone(
            mode=backbone_mode,
            history_length=spec.length,
            n_tx=self.n_tx,
            n_freq=self.n_freq,
            node_dim=node_dim,
            temporal_hidden=self.temporal_hidden,
        )
        self.node_dim = self.backbone.node_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(1 if version == "tau_only" else 2, self.condition_hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.temporal_hidden + self.condition_hidden, self.temporal_hidden),
            nn.ReLU(),
        )
        self.prediction_head = nn.Linear(
            self.temporal_hidden,
            self.n_tx * self.n_freq * RI_COMPONENTS,
        )

    def forward(self, history, conditions=None):
        # Let the backbone enforce tensor shape and history dtype before the
        # condition encoder is evaluated. This performs only metadata checks.
        if not isinstance(history, torch.Tensor) or history.ndim < 1:
            raise ValueError("history must be a torch.Tensor with shape [B,L,Nt,Nf,2]")
        history_latent = self.backbone.encode_endpoint(history)
        _validate_conditions(
            conditions,
            batch_size=history.shape[0],
            expected_dtype=self.backbone.feature_encoder[0].weight.dtype,
            expected_device=history.device,
        )
        if self.version == "tau_only":
            condition_input = conditions[:, 1:2] / CONDITION_SCALE[1]
        else:
            # Stack only the declared current-condition scalars. No target or
            # future trajectory field is available to this model interface.
            condition_input = torch.stack(
                (
                    conditions[:, 0] / CONDITION_SCALE[0],
                    conditions[:, 1] / CONDITION_SCALE[1],
                ),
                dim=1,
            )
        condition_latent = self.condition_encoder(condition_input)
        fused = self.fusion(torch.cat((history_latent, condition_latent), dim=1))
        prediction = self.prediction_head(fused).reshape(
            history.shape[0], self.n_tx, self.n_freq, RI_COMPONENTS
        )
        if self.residual:
            prediction = prediction + history[:, -1]
        return prediction


def create_conditioned_candidate(
    backbone_config,
    *,
    version,
    n_tx=DEFAULT_N_TX,
    n_freq=DEFAULT_N_FREQ,
    node_dim=DEFAULT_NODE_DIM,
    temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
):
    """Factory for a single shared checkpoint over all supplied conditions."""
    return ConditionedLiteDTCN(
        backbone_config,
        version=version,
        n_tx=n_tx,
        n_freq=n_freq,
        node_dim=node_dim,
        temporal_hidden=temporal_hidden,
        condition_hidden=CONDITION_HIDDEN,
    )


def conditioned_complexity(
    backbone_config,
    *,
    version,
    n_tx=DEFAULT_N_TX,
    n_freq=DEFAULT_N_FREQ,
    node_dim=DEFAULT_NODE_DIM,
    temporal_hidden=DEFAULT_TEMPORAL_HIDDEN,
):
    """Analytical parameter/MAC counts, replacing the dedicated head."""
    if version not in CONDITION_VERSIONS:
        raise ValueError(f"version must be one of {CONDITION_VERSIONS}")
    spec = resolve_candidate_config(backbone_config)
    base = candidate_complexity(
        spec.mode,
        history_length=spec.length,
        n_tx=n_tx,
        n_freq=n_freq,
        node_dim=node_dim,
        temporal_hidden=temporal_hidden,
        residual=spec.residual,
    )
    output_features = n_tx * n_freq * RI_COMPONENTS
    condition_inputs = 1 if version == "tau_only" else 2
    condition_parameters = condition_inputs * CONDITION_HIDDEN + CONDITION_HIDDEN
    fusion_parameters = (
        (temporal_hidden + CONDITION_HIDDEN) * temporal_hidden + temporal_hidden
    )
    head_parameters = temporal_hidden * output_features + output_features
    condition_macs = condition_inputs * CONDITION_HIDDEN
    fusion_macs = (temporal_hidden + CONDITION_HIDDEN) * temporal_hidden
    head_macs = temporal_hidden * output_features
    base_head_parameters = base["components"]["head_parameters"]
    base_head_macs = base["components"]["head_macs"]
    parameters = (
        base["parameters"] - base_head_parameters
        + condition_parameters + fusion_parameters + head_parameters
    )
    macs = base["macs"] - base_head_macs + condition_macs + fusion_macs + head_macs
    return {
        "mode": base["mode"],
        "version": version,
        "history_length": spec.length,
        "receptive_field": base["receptive_field"],
        "parameters": parameters,
        "macs": macs,
        "residual_additions": base["residual_additions"],
        "components": {
            **{
                key: value
                for key, value in base["components"].items()
                if key not in ("head_parameters", "head_macs")
            },
            "condition_parameters": condition_parameters,
            "fusion_parameters": fusion_parameters,
            "head_parameters": head_parameters,
            "condition_macs": condition_macs,
            "fusion_macs": fusion_macs,
            "head_macs": head_macs,
        },
    }

