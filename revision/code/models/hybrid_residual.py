"""Single-input Lite residual model over a frozen per-coefficient Wiener head."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from revision.code.models.candidates import create_candidate


def _normalizer(value, name):
    tensor = torch.as_tensor(value, dtype=torch.float64).detach().clone()
    if tuple(tensor.shape) != (2,) or not torch.isfinite(tensor).all().item():
        raise ValueError(name + " must be a finite two-element RI vector")
    if name == "std" and not (tensor > 0).all().item():
        raise ValueError("std must be positive")
    return tensor


def _linear_head(linear_head):
    if not isinstance(linear_head, Mapping) or set(linear_head) != {"coefficients", "intercept"}:
        raise ValueError("linear_head must contain coefficients and intercept")
    coefficients = np.asarray(linear_head["coefficients"])
    intercept = np.asarray(linear_head["intercept"])
    if (coefficients.dtype != np.dtype(np.complex128)
            or coefficients.shape != (256, 10)
            or not np.isfinite(coefficients).all()):
        raise ValueError("coefficients must be finite complex128 [256,10]")
    if (intercept.dtype != np.dtype(np.complex128)
            or intercept.shape != (256,)
            or not np.isfinite(intercept).all()):
        raise ValueError("intercept must be finite complex128 [256]")
    return torch.from_numpy(coefficients.copy()), torch.from_numpy(intercept.copy())


class PhysicalHistoryWienerResidual(nn.Module):
    """Predict from one physical history; normalization is internal and train-only."""

    def __init__(self, mean, std, linear_head, *, node_dim=16, temporal_hidden=32):
        super().__init__()
        mean64 = _normalizer(mean, "mean")
        std64 = _normalizer(std, "std")
        coefficients, intercept = _linear_head(linear_head)
        candidate = create_candidate(
            "full10_direct_lr1e3", node_dim=node_dim, temporal_hidden=temporal_hidden
        )
        self.backbone = candidate.backbone
        self.residual_head = nn.Linear(temporal_hidden + 8 * 32 * 2, 8 * 32 * 2)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.register_buffer("normalizer_mean", mean64)
        self.register_buffer("normalizer_std", std64)
        self.register_buffer("wiener_coefficients", coefficients)
        self.register_buffer("wiener_intercept", intercept)

    def wiener(self, physical_history):
        """Return the train-only per-coefficient complex128 linear forecast."""
        complex_history = torch.complex(physical_history[..., 0], physical_history[..., 1])
        features = complex_history.permute(0, 2, 3, 1).reshape(
            physical_history.shape[0], 256, 10
        )
        forecast = torch.einsum(
            "bdp,dp->bd", features, self.wiener_coefficients
        ) + self.wiener_intercept.unsqueeze(0)
        return torch.view_as_real(forecast.reshape(-1, 8, 32))

    def forward(self, physical_history, *, return_components=False):
        if (not isinstance(physical_history, torch.Tensor)
                or physical_history.ndim != 5
                or tuple(physical_history.shape[1:]) != (10, 8, 32, 2)
                or physical_history.shape[0] < 1
                or physical_history.dtype != torch.float64
                or not torch.isfinite(physical_history).all().item()):
            raise ValueError("physical_history must be finite float64 [B,10,8,32,2]")
        if (physical_history.device != self.normalizer_mean.device
                or any(p.device != physical_history.device for p in self.parameters())):
            raise ValueError("physical history, normalizer, and model must share a device")

        mean5 = self.normalizer_mean.reshape(1, 1, 1, 1, 2)
        std5 = self.normalizer_std.reshape(1, 1, 1, 1, 2)
        mean4 = self.normalizer_mean.reshape(1, 1, 1, 2)
        std4 = self.normalizer_std.reshape(1, 1, 1, 2)
        history_norm32 = ((physical_history - mean5) / std5).to(torch.float32)
        baseline = self.wiener(physical_history)
        baseline_norm32 = ((baseline - mean4) / std4).to(torch.float32)
        latent = self.backbone.encode_endpoint(history_norm32)
        fused = torch.cat((latent, baseline_norm32.reshape(len(latent), -1)), dim=1)
        residual_norm32 = self.residual_head(fused).reshape(-1, 8, 32, 2)
        prediction = baseline + residual_norm32.to(torch.float64) * std4
        if not torch.isfinite(prediction).all().item():
            raise ValueError("hybrid prediction is nonfinite")
        if return_components:
            prediction_norm32 = ((prediction - mean4) / std4).to(torch.float32)
            return {
                "prediction_physical": prediction,
                "prediction_norm": prediction_norm32,
                "wiener_physical": baseline,
                "residual_norm": residual_norm32,
            }
        return prediction
