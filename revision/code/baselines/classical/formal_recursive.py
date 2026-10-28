"""Formal prediction-only wrapper for a frozen direct h=1 classical state.

The wrapper has no training-record, target, or filesystem interface. Every
step consumes only the original history and prior model predictions; any
non-ok row status remains visible in the final status and diagnostics.
"""
from collections import Counter
import copy
import hashlib
from pathlib import Path

import numpy as np


RECURSIVE_VARIANT = "recursive_from_direct_h1"
_SUPPORTED_METHODS = frozenset(("ridge", "lmmse"))
_HISTORY_LENGTH = 10


def _positive_horizon(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("horizon_steps must be an explicit positive integer")
    return value


class FormalRecursiveH1Predictor:
    """Roll a fitted Ridge/LMMSE h=1 state forward with sticky failures."""

    model_variant = RECURSIVE_VARIANT

    def __init__(self, predictor):
        method = getattr(predictor, "method", None)
        if method not in _SUPPORTED_METHODS:
            raise ValueError("Formal recursive deployment supports Ridge/LMMSE only")
        if getattr(predictor, "_fitted", False) is not True:
            raise ValueError("Formal recursive deployment requires a fitted predictor")
        config = getattr(predictor, "config", None)
        if not isinstance(config, dict) or config.get("history_length") != _HISTORY_LENGTH:
            raise ValueError("Formal recursive deployment requires history_length=10")
        horizons = config.get("horizon_steps")
        try:
            horizons = tuple(horizons)
        except TypeError as exc:
            raise ValueError("The fitted predictor must contain only a direct h=1 head") from exc
        if len(horizons) != 1 or isinstance(horizons[0], bool) or horizons[0] != 1:
            raise ValueError("The fitted predictor must contain only a direct h=1 head")
        self.predictor = predictor
        self.method = method
        self.route_id = f"{method}_challenge_{RECURSIVE_VARIANT}"

    def registry_metadata(self):
        source = Path(__file__).resolve()
        return {
            "route_id": self.route_id,
            "method": self.method,
            "model_variant": self.model_variant,
            "fit_horizons": [1],
            "feedback_policy": "self_prediction_only",
            "deployment_source": {
                "path": "revision/code/baselines/classical/formal_recursive.py",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        }

    def predict(self, history, conditions=None, horizon_steps=None):
        horizon = _positive_horizon(horizon_steps)
        initial = np.asarray(history)
        if (initial.ndim != 4 or not initial.size or initial.shape[1] != _HISTORY_LENGTH
                or not np.iscomplexobj(initial) or not np.isfinite(initial).all()):
            raise ValueError("history must be finite complex [B,10,Nt,Nf]")
        rollout = initial.astype(np.complex128, copy=True)
        batch, _, n_tx, n_sc = rollout.shape
        step_statuses = []
        row_failures = [[] for _ in range(batch)]
        failure_counts = Counter()
        non_ok_step_rows = 0
        final_prediction = None

        for step in range(1, horizon + 1):
            # The target is intentionally absent from this API. The inner model
            # receives a private copy so it cannot mutate future rollout input.
            output = self.predictor.predict(
                rollout.copy(), conditions=conditions, horizon_steps=1,
            )
            if not isinstance(output, dict):
                raise ValueError("h=1 predictor must return an output dictionary")
            prediction = np.asarray(output.get("prediction"))
            if (prediction.shape != (batch, n_tx, n_sc)
                    or not np.iscomplexobj(prediction) or not np.isfinite(prediction).all()):
                raise ValueError("h=1 predictor returned invalid complex prediction shape or values")
            statuses = output.get("status")
            if (not isinstance(statuses, (list, tuple)) or len(statuses) != batch
                    or any(not isinstance(value, str) or not value for value in statuses)):
                raise ValueError("h=1 predictor statuses must be nonempty strings for every row")
            diagnostics = output.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                raise ValueError("h=1 predictor diagnostics must be a dictionary")
            statuses = list(statuses)
            step_statuses.append({
                "step": step,
                "row_statuses": statuses,
                "n_non_ok_rows": sum(value != "ok" for value in statuses),
                "predictor_diagnostics": copy.deepcopy(diagnostics),
            })
            for row, status in enumerate(statuses):
                if status != "ok":
                    row_failures[row].append({"step": step, "status": status})
                    failure_counts[status] += 1
                    non_ok_step_rows += 1
            final_prediction = prediction.astype(np.complex128, copy=False)
            rollout = np.concatenate((rollout[:, 1:], final_prediction[:, None]), axis=1)

        final_statuses = []
        for events in row_failures:
            if not events:
                final_statuses.append("ok")
            else:
                detail = ";".join(
                    f"step={event['step']}:{event['status']}" for event in events
                )
                final_statuses.append("recursive_non_ok:" + detail)
        return {
            "prediction": final_prediction,
            "status": final_statuses,
            "diagnostics": {
                "method": self.method,
                "model_variant": self.model_variant,
                "route_id": self.route_id,
                "horizon_steps": horizon,
                "rollout_steps": horizon,
                "step_statuses": step_statuses,
                "row_failures": [
                    {"row": row, "events": events}
                    for row, events in enumerate(row_failures) if events
                ],
                "failure_counts": dict(sorted(failure_counts.items())),
                "n_failed_rows": sum(bool(events) for events in row_failures),
                "n_non_ok_step_rows": non_ok_step_rows,
            },
        }
