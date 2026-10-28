"""Side-effect-free CPU/GPU modules for auditing the historical Lite-DTCN.

The historical eight-node axis represents eight receive antennas in D0. It is
kept under an explicitly legacy name here and is not reinterpreted as a new
BS transmit array.
"""
import torch
from torch import nn


LEGACY_RX_ANTENNAS = 8
N_SUBCARRIERS = 32
RI_COMPONENTS = 2
RAW_NODE_FEATURES = N_SUBCARRIERS * RI_COMPONENTS
NODE_DIM = 16
TEMPORAL_CHANNELS = LEGACY_RX_ANTENNAS * NODE_DIM
TEMPORAL_HIDDEN = 32
TEMPORAL_KERNEL = 3


class _LegacyLiteDTCNBase(nn.Module):
    """Shared historical layer layout; no imports from training scripts."""

    def __init__(self):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(RAW_NODE_FEATURES, NODE_DIM),
            nn.ReLU(),
        )
        self.depthwise_temporal = nn.Conv1d(
            TEMPORAL_CHANNELS,
            TEMPORAL_CHANNELS,
            kernel_size=TEMPORAL_KERNEL,
            padding=1,
            groups=TEMPORAL_CHANNELS,
        )
        self.pointwise_temporal = nn.Conv1d(
            TEMPORAL_CHANNELS,
            TEMPORAL_HIDDEN,
            kernel_size=1,
        )
        self.prediction_head = nn.Linear(
            TEMPORAL_HIDDEN,
            LEGACY_RX_ANTENNAS * RAW_NODE_FEATURES,
        )

    def _validate_history(self, history, minimum_frames):
        if not isinstance(history, torch.Tensor):
            raise ValueError("history must be a torch.Tensor")
        if (
            history.ndim != 5
            or history.shape[0] < 1
            or history.shape[1] < minimum_frames
            or tuple(history.shape[2:])
            != (LEGACY_RX_ANTENNAS, N_SUBCARRIERS, RI_COMPONENTS)
        ):
            raise ValueError(
                "history must have shape [B,L,8 legacy-RX,32 subcarriers,2 RI] "
                f"with L>={minimum_frames}"
            )
        if history.dtype not in (torch.float32, torch.float64):
            raise ValueError("history must use float32 or float64 RI values")
        if history.dtype != self.feature_encoder[0].weight.dtype:
            raise ValueError("history dtype must match the model parameter dtype")
        if not torch.isfinite(history).all().item():
            raise ValueError("history must be finite")

    def _forward_sequence(self, history):
        batch_size, sequence_length = history.shape[:2]
        encoded = self.feature_encoder(
            history.reshape(
                batch_size * sequence_length,
                LEGACY_RX_ANTENNAS,
                RAW_NODE_FEATURES,
            )
        )
        encoded = encoded.reshape(
            batch_size,
            sequence_length,
            TEMPORAL_CHANNELS,
        )
        temporal = self.depthwise_temporal(encoded.transpose(1, 2))
        temporal = torch.relu(self.pointwise_temporal(temporal))[:, :, -1]
        prediction = self.prediction_head(temporal)
        return prediction.reshape(
            batch_size,
            LEGACY_RX_ANTENNAS,
            N_SUBCARRIERS,
            RI_COMPONENTS,
        )


class LegacyLiteDTCN(_LegacyLiteDTCNBase):
    """Full supplied Lite-DTCN forward path for historical replay/audit.

    Input is [B,L,8,32,2] real RI. In this legacy D0 bridge, the size-eight
    axis means receive antennas; the standardized tensor shape does not change
    that physical meaning. ``conditions`` is accepted for interface parity but
    intentionally unused, as the old network has no condition embedding.
    """

    def forward(self, history, conditions=None):
        self._validate_history(history, minimum_frames=1)
        return self._forward_sequence(history)


class TwoFrameEquivalentLiteDTCN(_LegacyLiteDTCNBase):
    """Same-weight endpoint-equivalent crop that processes only the last 2 frames.

    With the historical kernel-3, padding-1 convolution, the last output uses
    the last two observed inputs and a zero right-padding slot. Slicing before
    the same layers preserves that endpoint calculation and all parameter
    names/order; this is not a separately trained two-frame model.
    """

    def forward(self, history, conditions=None):
        self._validate_history(history, minimum_frames=2)
        return self._forward_sequence(history[:, -2:])


def legacy_complexity(history_frames):
    """Return historical parameter/MAC counts for an executed sequence length.

    MACs count configured Linear/Conv1d multiply-accumulates, including the
    padded kernel slots, and exclude activations and bias additions. For the
    two-frame class, pass ``history_frames=2``; no latency measurement is made.
    """
    if (
        isinstance(history_frames, bool)
        or not isinstance(history_frames, int)
        or history_frames < 1
    ):
        raise ValueError("history_frames must be a positive integer")
    encoder_params = RAW_NODE_FEATURES * NODE_DIM + NODE_DIM
    depthwise_params = TEMPORAL_CHANNELS * TEMPORAL_KERNEL + TEMPORAL_CHANNELS
    pointwise_params = TEMPORAL_HIDDEN * TEMPORAL_CHANNELS + TEMPORAL_HIDDEN
    head_outputs = LEGACY_RX_ANTENNAS * RAW_NODE_FEATURES
    head_params = TEMPORAL_HIDDEN * head_outputs + head_outputs
    encoder_macs = (
        history_frames * LEGACY_RX_ANTENNAS * RAW_NODE_FEATURES * NODE_DIM
    )
    depthwise_macs = history_frames * TEMPORAL_CHANNELS * TEMPORAL_KERNEL
    pointwise_macs = history_frames * TEMPORAL_HIDDEN * TEMPORAL_CHANNELS
    head_macs = TEMPORAL_HIDDEN * head_outputs
    return {
        "history_frames": history_frames,
        "parameters": encoder_params + depthwise_params + pointwise_params + head_params,
        "macs": encoder_macs + depthwise_macs + pointwise_macs + head_macs,
        "components": {
            "encoder_params": encoder_params,
            "depthwise_params": depthwise_params,
            "pointwise_params": pointwise_params,
            "head_params": head_params,
            "encoder_macs": encoder_macs,
            "depthwise_macs": depthwise_macs,
            "pointwise_macs": pointwise_macs,
            "head_macs": head_macs,
        },
    }
