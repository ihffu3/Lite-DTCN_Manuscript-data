"""Prediction-only wrapper for recursively deploying a fitted direct h=1 head.

The wrapper deliberately does not fit, load, or inspect training records. It
accepts a pre-fitted Ridge or LMMSE predictor whose only fitted horizon is 1.
"""

from collections import Counter
import copy
import hashlib
from pathlib import Path

import numpy as np


RECURSIVE_VARIANT = "recursive_from_direct_h1"
_SUPPORTED_METHODS = frozenset(("ridge", "lmmse"))
_HISTORY_LENGTH = 10
_SOURCE_RELATIVE_PATH = (
    "revision/code/baselines/classical/recursive_deployment.py"
)


def _positive_horizon(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("horizon_steps must be an explicit positive integer")
    return value


def source_identity():
    """Return the wrapper source path and content hash for a run registry."""
    source = Path(__file__).resolve()
    try:
        relative = source.relative_to(source.parents[4]).as_posix()
    except (IndexError, ValueError):
        relative = _SOURCE_RELATIVE_PATH
    return {
        "path": relative,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


class RecursiveH1Predictor:
    """Roll a pre-fitted Ridge/LMMSE h=1 head forward using its own outputs."""

    model_variant = RECURSIVE_VARIANT

    def __init__(self, predictor):
        method = getattr(predictor, "method", None)
        if method not in _SUPPORTED_METHODS:
            raise ValueError("recursive h=1 deployment supports Ridge or LMMSE only")
        if getattr(predictor, "_fitted", False) is not True:
            raise ValueError("recursive h=1 deployment requires a pre-fitted predictor")
        config = getattr(predictor, "config", None)
        if not isinstance(config, dict):
            raise ValueError("pre-fitted predictor config is unavailable")
        if config.get("history_length") != _HISTORY_LENGTH:
            raise ValueError("recursive deployment requires history_length=10")
        horizons = config.get("horizon_steps")
        try:
            normalized_horizons = tuple(horizons)
        except TypeError as exc:
            raise ValueError("horizon_steps must contain exactly the fitted h=1 head") from exc
        if (
            len(normalized_horizons) != 1
            or isinstance(normalized_horizons[0], bool)
            or not isinstance(normalized_horizons[0], int)
            or normalized_horizons[0] != 1
        ):
            raise ValueError("horizon_steps must contain exactly the fitted h=1 head")
        self.predictor = predictor
        self.method = method
        self.route_id = f"{method}_challenge_{RECURSIVE_VARIANT}"

    def registry_metadata(self):
        """Metadata that keeps this deployment route distinct from direct D1."""
        return {
            "route_id": self.route_id,
            "method": self.method,
            "model_variant": self.model_variant,
            "fit_horizons": [1],
            "feedback_policy": "self_prediction_only",
            "deployment_source": source_identity(),
        }

    def predict(self, history, conditions=None, horizon_steps=None):
        """Predict a total horizon by repeatedly invoking the fitted h=1 head.

        Each row's non-ok status is sticky: the row may continue from the
        predictor's finite fallback output, but all failing steps remain
        visible in the returned status and diagnostics.
        """
        horizon = _positive_horizon(horizon_steps)
        history_array = np.asarray(history)
        if (
            history_array.ndim != 4
            or not history_array.size
            or history_array.shape[1] != _HISTORY_LENGTH
            or not np.iscomplexobj(history_array)
            or not np.isfinite(history_array).all()
        ):
            raise ValueError("history must be finite complex [B,10,Nt,Nf]")

        rollout = history_array.astype(np.complex128, copy=True)
        batch, _, n_tx, n_sc = rollout.shape
        step_statuses = []
        row_failures = [[] for _ in range(batch)]
        failure_counts = Counter()
        n_non_ok_step_rows = 0
        final_prediction = None

        for step in range(1, horizon + 1):
            # The predictor receives a copy, so even a misbehaving implementation
            # cannot mutate the history used to construct the next rollout step.
            output = self.predictor.predict(
                rollout.copy(),
                conditions=conditions,
                horizon_steps=1,
            )
            if not isinstance(output, dict):
                raise ValueError("h=1 predictor must return an output dictionary")
            prediction = np.asarray(output.get("prediction"))
            if (
                prediction.shape != (batch, n_tx, n_sc)
                or not np.iscomplexobj(prediction)
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("h=1 predictor returned invalid complex prediction shape or values")
            prediction = prediction.astype(np.complex128, copy=False)

            statuses = output.get("status")
            if not isinstance(statuses, (list, tuple)) or len(statuses) != batch:
                raise ValueError("h=1 predictor status count must match batch rows")
            statuses = list(statuses)
            if any(not isinstance(status, str) or not status for status in statuses):
                raise ValueError("h=1 predictor statuses must be nonempty strings")

            base_diagnostics = output.get("diagnostics", {})
            if not isinstance(base_diagnostics, dict):
                raise ValueError("h=1 predictor diagnostics must be a dictionary")
            step_statuses.append({
                "step": step,
                "row_statuses": statuses,
                "n_non_ok_rows": sum(status != "ok" for status in statuses),
                "predictor_diagnostics": copy.deepcopy(base_diagnostics),
            })
            for row, status in enumerate(statuses):
                if status != "ok":
                    row_failures[row].append({"step": step, "status": status})
                    failure_counts[status] += 1
                    n_non_ok_step_rows += 1

            final_prediction = prediction
            rollout = np.concatenate(
                (rollout[:, 1:], prediction[:, None, :, :]),
                axis=1,
            )

        final_statuses = []
        for events in row_failures:
            if not events:
                final_statuses.append("ok")
                continue
            detail = ";".join(
                f"step={event['step']}:{event['status']}" for event in events
            )
            final_statuses.append("recursive_non_ok:" + detail)

        diagnostics = {
            "method": self.method,
            "model_variant": self.model_variant,
            "route_id": self.route_id,
            "horizon_steps": horizon,
            "rollout_steps": horizon,
            "step_statuses": step_statuses,
            "row_failures": [
                {"row": row, "events": events}
                for row, events in enumerate(row_failures)
                if events
            ],
            "failure_counts": dict(sorted(failure_counts.items())),
            "n_failed_rows": sum(bool(events) for events in row_failures),
            "n_non_ok_step_rows": n_non_ok_step_rows,
        }
        return {
            "prediction": final_prediction.astype(np.complex128, copy=False),
            "status": final_statuses,
            "diagnostics": diagnostics,
        }
