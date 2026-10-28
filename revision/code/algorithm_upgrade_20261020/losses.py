"""Recipe-specific normalized and physical-domain training losses."""
from numbers import Real

import torch


_RECIPES = frozenset(("C0", "C1", "C2", "C3"))
_TARGET_SHAPE = (8, 32, 2)
_N_COMPLEX_ELEMENTS = 8 * 32


def _statistics(mean, std, *, device):
    try:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float64, device=device).detach()
        std_tensor = torch.as_tensor(std, dtype=torch.float64, device=device).detach()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("mean and std must be numeric two-component arrays") from exc
    if tuple(mean_tensor.shape) != (2,) or tuple(std_tensor.shape) != (2,):
        raise ValueError("mean and std must have shape [RI=2]")
    if not torch.isfinite(mean_tensor).all().item():
        raise ValueError("mean must be finite")
    if not torch.isfinite(std_tensor).all().item() or not (std_tensor > 0).all().item():
        raise ValueError("std must be finite and positive")
    return mean_tensor, std_tensor


def _epsilon_value(epsilon, *, device):
    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite positive scalar")
    if isinstance(epsilon, Real):
        value = float(epsilon)
    elif isinstance(epsilon, torch.Tensor) and epsilon.ndim == 0:
        value = float(epsilon.detach().to(device=device, dtype=torch.float64).item())
    else:
        raise ValueError("epsilon must be a finite positive scalar")
    if not torch.isfinite(torch.tensor(value, dtype=torch.float64)).item() or value <= 0:
        raise ValueError("epsilon must be a finite positive scalar")
    return value


def _environment_energies(values, batch, *, device):
    try:
        energy = torch.as_tensor(values, dtype=torch.float64, device=device).detach()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("environment_energy must provide one energy per row") from exc
    if energy.ndim == 0 and batch == 1:
        energy = energy.reshape(1)
    if tuple(energy.shape) != (batch,):
        raise ValueError("environment_energy must provide one energy per row")
    if not torch.isfinite(energy).all().item():
        raise ValueError("environment_energy must be finite and nonnegative")
    if (energy < 0).any().item():
        raise ValueError("environment_energy must be nonnegative")
    return energy


def training_loss(
    recipe,
    prediction_norm,
    target_physical_ri,
    mean,
    std,
    environment_energy,
    epsilon,
):
    """Compute the frozen C0 normalized MSE or C1-C3 physical weighted loss.

    The target is always supplied in physical float64 RI units. Energies for
    C1-C3 are per sampled row and correspond to that row's training environment.
    """
    if not isinstance(recipe, str) or recipe not in _RECIPES:
        raise ValueError("recipe must be one of C0, C1, C2, or C3")
    if not isinstance(prediction_norm, torch.Tensor):
        raise ValueError("prediction_norm must be a torch.Tensor")
    if not isinstance(target_physical_ri, torch.Tensor):
        raise ValueError("target_physical_ri must be a torch.Tensor")
    if (
        prediction_norm.ndim != 4
        or prediction_norm.shape[0] < 1
        or tuple(prediction_norm.shape[1:]) != _TARGET_SHAPE
    ):
        raise ValueError("prediction_norm must have shape [B,8,32,2]")
    if tuple(target_physical_ri.shape) != tuple(prediction_norm.shape):
        raise ValueError("target_physical_ri shape must match prediction_norm [B,8,32,2]")
    expected_dtype = torch.float64 if recipe == "C3" else torch.float32
    if prediction_norm.dtype != expected_dtype:
        raise ValueError(recipe + " prediction_norm has the wrong precision")
    if target_physical_ri.dtype != torch.float64:
        raise ValueError("target_physical_ri must use physical float64")
    if prediction_norm.device != target_physical_ri.device:
        raise ValueError("prediction and physical target must share a device")
    if not torch.isfinite(prediction_norm).all().item():
        raise ValueError("prediction_norm must be finite")
    if not torch.isfinite(target_physical_ri).all().item():
        raise ValueError("target_physical_ri must be finite")

    mean_tensor, std_tensor = _statistics(
        mean, std, device=prediction_norm.device
    )
    if recipe == "C0":
        target_norm32 = (
            target_physical_ri - mean_tensor
        ) / std_tensor
        loss = torch.mean((prediction_norm - target_norm32.to(torch.float32)) ** 2)
    else:
        energies = _environment_energies(
            environment_energy,
            prediction_norm.shape[0],
            device=prediction_norm.device,
        )
        epsilon_value = _epsilon_value(epsilon, device=prediction_norm.device)
        prediction_physical = (
            prediction_norm.to(torch.float64) * std_tensor + mean_tensor
        )
        physical_error = prediction_physical - target_physical_ri
        row_error_energy = torch.sum(
            physical_error * physical_error, dim=(1, 2, 3), dtype=torch.float64
        )
        denominator = _N_COMPLEX_ELEMENTS * torch.clamp_min(energies, epsilon_value)
        loss = torch.mean(row_error_energy / denominator)

    if not torch.isfinite(loss).item():
        raise ValueError(recipe + " training loss is nonfinite")
    return loss
