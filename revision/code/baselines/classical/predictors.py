"""CPU-only direct CSI prediction baselines with train-only fitting.

The estimators are empirical adaptations, not full vector covariance/oracle methods.
All arrays use physical complex CSI and all training windows stay within one
record and user. No file access occurs during import or fitting.
"""
from collections import Counter
import copy
import time

import numpy as np

from revision.code.core.arrays import record_parts


_METHODS = {
    "persistence", "zero", "train_mean", "ar", "ridge", "lmmse", "kalman", "esprit"
}
_DEFAULT_ORIGIN = 19
_CHUNK_ORIGINS = 128
_KALMAN_RELATIVE_LOADING = 1e-9
_STATE_SCHEMA = "1.0"


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _nonnegative_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a finite nonnegative real number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative real number")
    return value


def _horizons(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        raw = [value]
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raise ValueError("horizon_steps must be a positive integer or a sequence of them")
    out = tuple(_positive_integer(x, "horizon_steps entry") for x in raw)
    if not out or len(set(out)) != len(out):
        raise ValueError("horizon_steps must be nonempty and unique")
    return out


def _copy_tree(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(v) for v in value)
    return copy.deepcopy(value)


def _environment_channel_means(records):
    sums = {}
    counts = Counter()
    for record in records:
        h, _, meta = record_parts(record)
        env = meta["environment_id"]
        h = h.astype(np.complex128, copy=False)
        if env not in sums:
            sums[env] = np.zeros(h.shape[2:], dtype=np.complex128)
        sums[env] += h.sum(axis=(0, 1), dtype=np.complex128)
        counts[env] += h.shape[0] * h.shape[1]
    if not sums:
        raise ValueError("At least one training environment is required")
    return np.mean(
        np.stack([sums[e] / counts[e] for e in sorted(sums)], axis=0),
        axis=0,
    )


def _environment_signal_power(records, mean):
    powers = {}
    counts = Counter()
    for record in records:
        h, _, meta = record_parts(record)
        env = meta["environment_id"]
        x = h.astype(np.complex128, copy=False) - mean[None, None, ...]
        if env not in powers:
            powers[env] = np.zeros(mean.shape, dtype=np.float64)
        powers[env] += np.sum(np.abs(x) ** 2, axis=(0, 1), dtype=np.float64)
        counts[env] += h.shape[0] * h.shape[1]
    return np.mean(
        np.stack([powers[e] / counts[e] for e in sorted(powers)], axis=0),
        axis=0,
    )


class _Moments:
    """Streaming raw moments for one environment, one feature, and temporal lags."""

    def __init__(self, features, width, dtype):
        self.count = 0
        self.sum_x = np.zeros((features, width), dtype=dtype)
        self.sum_y = np.zeros((features,), dtype=dtype)
        self.gram = np.zeros((features, width, width), dtype=dtype)
        self.cross = np.zeros((features, width), dtype=dtype)

    def add(self, x, y):
        if not len(x):
            return
        self.count += int(x.shape[0])
        self.sum_x += np.sum(x, axis=0)
        self.sum_y += np.sum(y, axis=0)
        self.gram += np.einsum("ndp,ndq->dpq", np.conjugate(x), x, optimize=True)
        self.cross += np.einsum("ndp,nd->dp", np.conjugate(x), y, optimize=True)


def _iter_record_windows(records, history_length, horizon, origin_start, chunk_origins=_CHUNK_ORIGINS):
    """Yield per-record/user windows; never concatenate across realization boundaries."""
    for record in records:
        h, _, meta = record_parts(record)
        h = h.astype(np.complex128, copy=False)
        n_time, n_user, n_tx, n_sc = h.shape
        stop = n_time - horizon
        if origin_start >= stop:
            continue
        for first in range(origin_start, stop, chunk_origins):
            origins = np.arange(first, min(first + chunk_origins, stop), dtype=np.int64)
            offsets = np.arange(-history_length + 1, 1, dtype=np.int64)
            indices = origins[:, None] + offsets[None, :]
            hist = h[indices]  # [origin, L, user, Nt, Nf]
            hist = np.transpose(hist, (0, 2, 3, 4, 1)).reshape(
                len(origins) * n_user, n_tx * n_sc, history_length
            )
            target = h[origins + horizon].reshape(len(origins) * n_user, n_tx * n_sc)
            yield meta["environment_id"], hist, target


def _as_real_features(x, y):
    # Complex feature order is (Nt,Nf); RI-expanded order is (Nt,Nf,RI).
    xri = np.stack((x.real, x.imag), axis=2).reshape(x.shape[0], x.shape[1] * 2, x.shape[2])
    yri = np.stack((y.real, y.imag), axis=2).reshape(y.shape[0], y.shape[1] * 2)
    return xri.astype(np.float64, copy=False), yri.astype(np.float64, copy=False)


def _solve_regularized(moments, relative_loading):
    gram = np.asarray(moments["gram"])
    cross = np.asarray(moments["cross"])
    p = gram.shape[-1]
    trace = np.real(np.trace(gram, axis1=-2, axis2=-1))
    trace = np.maximum(trace, 0.0)
    loading = relative_loading * (trace / p)
    identity = np.eye(p, dtype=gram.dtype)
    coefficients = np.zeros_like(cross)
    conditions = np.empty(len(trace), dtype=np.float64)
    fallback_features = 0
    for feature in range(len(trace)):
        try:
            conditions[feature] = np.linalg.cond(gram[feature])
        except np.linalg.LinAlgError:
            conditions[feature] = np.inf
        matrix = gram[feature] + loading[feature] * identity
        try:
            coefficients[feature] = np.linalg.solve(matrix, cross[feature])
        except np.linalg.LinAlgError:
            coefficients[feature] = np.linalg.lstsq(matrix, cross[feature], rcond=None)[0]
            fallback_features += 1
        if not np.isfinite(coefficients[feature]).all():
            coefficients[feature] = np.linalg.lstsq(matrix, cross[feature], rcond=None)[0]
            fallback_features += 1
        if not np.isfinite(coefficients[feature]).all():
            raise ValueError("Could not obtain finite coefficients for a training feature")
    return coefficients, loading, conditions, fallback_features


def _condition_summary(conditions):
    conditions = np.asarray(conditions, dtype=np.float64)
    finite = conditions[np.isfinite(conditions)]
    return {
        "max_condition_number": float(np.max(finite)) if len(finite) else float("inf"),
        "median_condition_number": float(np.median(finite)) if len(finite) else float("inf"),
        "ill_conditioned_features_gt_1e12": int(np.sum(~np.isfinite(conditions) | (conditions > 1e12))),
    }


def _ri_history_features(history):
    # [B,L,Nt,Nf] -> [B,Nt*Nf*2,L], with RI fastest.
    b, length, n_tx, n_sc = history.shape
    x = np.stack((history.real, history.imag), axis=-1)
    return np.transpose(x.reshape(b, length, n_tx * n_sc * 2), (0, 2, 1)).astype(np.float64)


def _complex_history_features(history):
    b, length, n_tx, n_sc = history.shape
    return np.transpose(history.reshape(b, length, n_tx * n_sc), (0, 2, 1)).astype(np.complex128)


def _ri_prediction(values, n_tx, n_sc):
    values = np.asarray(values, dtype=np.float64).reshape(-1, n_tx, n_sc, 2)
    return values[..., 0] + 1j * values[..., 1]


def _resolve_predict_horizon(horizon_steps, config):
    if horizon_steps is None:
        return config["horizon_steps"][0]
    return _positive_integer(horizon_steps, "horizon_steps")


def _training_records(training_records):
    records = list(training_records)
    if not records:
        raise ValueError("At least one train record is required")
    parsed = []
    envs = set()
    shape = None
    for record in records:
        h, _, meta = record_parts(record)
        if meta["split"] != "train":
            raise ValueError("Classical predictors may only be fitted on train records")
        current_shape = tuple(h.shape[2:])
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError("All train records must share Nt and Nf")
        parsed.append(record)
        envs.add(meta["environment_id"])
    return parsed, tuple(sorted(envs)), shape


def _normalize_config(config, method):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    cfg = copy.deepcopy(config)
    if "history_length" not in cfg:
        raise ValueError("config.history_length is required")
    cfg["history_length"] = _positive_integer(cfg["history_length"], "history_length")
    cfg["horizon_steps"] = _horizons(cfg.get("horizon_steps", (1, 2, 4, 8)))
    cfg["origin_start"] = _positive_integer(
        cfg.get("origin_start", _DEFAULT_ORIGIN), "origin_start"
    )
    if cfg["origin_start"] < cfg["history_length"] - 1:
        raise ValueError("origin_start must leave the configured history available")
    if method in {"ar", "kalman"}:
        cfg["order"] = _positive_integer(cfg.get("order"), "order")
    if method in {"ar", "ridge", "lmmse"}:
        cfg["relative_loading"] = _nonnegative_real(
            cfg.get("relative_loading"), "relative_loading"
        )
    if method == "kalman":
        cfg["process_noise_scale"] = float(cfg.get("process_noise_scale", 1.0))
        if not np.isfinite(cfg["process_noise_scale"]) or cfg["process_noise_scale"] <= 0:
            raise ValueError("process_noise_scale must be finite and positive")
        cfg["relative_loading"] = _nonnegative_real(
            cfg.get("relative_loading", _KALMAN_RELATIVE_LOADING), "relative_loading"
        )
    if method == "esprit":
        cfg["rank"] = _positive_integer(cfg.get("rank"), "rank")
        cfg["hankel_rows"] = _positive_integer(cfg.get("hankel_rows", 5), "hankel_rows")
        cfg["gain_fit_ridge"] = _nonnegative_real(
            cfg.get("gain_fit_ridge", 1e-8), "gain_fit_ridge"
        )
        columns = cfg["history_length"] - cfg["hankel_rows"]
        if cfg["rank"] > min(cfg["hankel_rows"] - 1, columns):
            raise ValueError("ESPRIT rank exceeds the identifiable shifted-subspace dimension")
    return cfg


class _MomentAccumulator:
    def __init__(self, features, width, dtype):
        self.features = features
        self.width = width
        self.dtype = dtype
        self.env = {}
        self.total_rows = 0

    def add(self, env, x, y):
        if env not in self.env:
            self.env[env] = _Moments(self.features, self.width, self.dtype)
        self.env[env].add(x, y)
        self.total_rows += len(x)

    def finish(self, fit_intercept=True):
        if not self.env:
            raise ValueError("No usable train windows")
        env_ids = sorted(self.env)
        mean_x = np.mean(np.stack([self.env[e].sum_x / self.env[e].count for e in env_ids]), axis=0)
        mean_y = np.mean(np.stack([self.env[e].sum_y / self.env[e].count for e in env_ids]), axis=0)
        raw_gram = np.mean(np.stack([self.env[e].gram / self.env[e].count for e in env_ids]), axis=0)
        raw_cross = np.mean(np.stack([self.env[e].cross / self.env[e].count for e in env_ids]), axis=0)
        if fit_intercept:
            gram = raw_gram - np.einsum("dp,dq->dpq", np.conjugate(mean_x), mean_x)
            gram = (gram + np.swapaxes(np.conjugate(gram), -1, -2)) * 0.5
            cross = raw_cross - np.conjugate(mean_x) * mean_y[:, None]
        else:
            gram, cross = raw_gram, raw_cross
        return {
            "mean_x": mean_x,
            "mean_y": mean_y,
            "gram": gram,
            "cross": cross,
            "n_training_rows": int(self.total_rows),
            "n_environments": len(env_ids),
        }


def _fit_moments(records, width, horizon, origin_start, *, real_ri=False,
                 center_offset=None, fit_intercept=True):
    features = None
    dtype = None
    accumulator = None
    for env, x, y in _iter_record_windows(records, width, horizon, origin_start):
        if real_ri:
            x, y = _as_real_features(x, y)
        else:
            x = x.astype(np.complex128, copy=False)
            y = y.astype(np.complex128, copy=False)
        if center_offset is not None:
            x = x - np.asarray(center_offset)[None, :, None]
            y = y - np.asarray(center_offset)[None, :]
        if features is None:
            features = x.shape[1]
            dtype = x.dtype
            accumulator = _MomentAccumulator(features, width, dtype)
        if x.shape[1] != features:
            raise ValueError("Inconsistent train feature dimensions")
        accumulator.add(env, x, y)
    if accumulator is None:
        raise ValueError(
            "No usable train windows: check duration, origin_start, history_length and horizon"
        )
    return accumulator.finish(fit_intercept=fit_intercept)


def _channel_mean_features(records, real_ri):
    mean = _environment_channel_means(records)
    if real_ri:
        return np.stack((mean.real, mean.imag), axis=-1).reshape(-1).astype(np.float64)
    return mean.reshape(-1).astype(np.complex128)


def _complex_residual_power(records, order, origin_start, mean, coefficients):
    by_env_sum = {}
    by_env_count = Counter()
    for env, x, y in _iter_record_windows(records, order, 1, origin_start):
        x = x - mean[None, :, None]
        y = y - mean[None, :]
        residual = y - np.einsum("ndp,dp->nd", x, coefficients, optimize=True)
        if env not in by_env_sum:
            by_env_sum[env] = np.zeros(y.shape[1], dtype=np.float64)
        by_env_sum[env] += np.sum(np.abs(residual) ** 2, axis=0, dtype=np.float64)
        by_env_count[env] += len(residual)
    if not by_env_sum:
        raise ValueError("No train innovations available for Kalman process-noise estimate")
    return np.mean(
        np.stack([by_env_sum[e] / by_env_count[e] for e in sorted(by_env_sum)]),
        axis=0,
    )


def _forecast_autoregression(history, coefficients, mean, steps):
    batch, features, length = history.shape
    order = coefficients.shape[-1]
    if length < order:
        raise ValueError("History is shorter than autoregressive order")
    work = history.copy()
    for _ in range(steps):
        centered = work[:, :, -order:] - mean[None, :, None]
        next_value = np.einsum("bdp,dp->bd", centered, coefficients, optimize=True) + mean[None, :]
        work = np.concatenate((work, next_value[..., None]), axis=2)
    return work[:, :, -1]


def _measurement_snr(conditions, batch):
    if not isinstance(conditions, dict) or "history_noise_snr_db" not in conditions:
        return None
    snr = np.asarray(conditions["history_noise_snr_db"], dtype=np.float64)
    if snr.ndim == 0:
        snr = np.full(batch, float(snr), dtype=np.float64)
    if snr.shape != (batch,) or not np.isfinite(snr).all():
        raise ValueError("history_noise_snr_db must be finite scalar or shape [B]")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        ratio = np.power(10.0, -snr / 10.0)
    if not np.isfinite(ratio).all():
        raise ValueError("history_noise_snr_db produced nonfinite measurement variance")
    return ratio


def _esprit_forecast(sequence, rank, hankel_rows, horizon, gain_fit_ridge):
    length = len(sequence)
    columns = length - hankel_rows
    if columns < rank or hankel_rows < 2 or rank > hankel_rows - 1:
        return None, "short_or_unidentifiable"
    y0 = np.stack([sequence[row:row + columns] for row in range(hankel_rows)], axis=0)
    y1 = np.stack([sequence[row + 1:row + 1 + columns] for row in range(hankel_rows)], axis=0)
    try:
        u, singular, _ = np.linalg.svd(y0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, "svd_failure"
    if not len(singular) or singular[0] <= np.finfo(np.float64).tiny:
        return None, "rank_deficient"
    tolerance = max(singular[0] * 1e-10, np.finfo(np.float64).tiny * 100)
    if rank > len(singular) or singular[rank - 1] <= tolerance:
        return None, "rank_deficient"
    basis = u[:, :rank]
    left, right = basis[:-1], basis[1:]
    try:
        # `basis` is unit-norm: rank thresholds for this shifted subspace must
        # not inherit the scale of the original Hankel matrix.
        if np.linalg.matrix_rank(left) < rank:
            return None, "rank_deficient"
        shift = np.linalg.lstsq(left, right, rcond=None)[0]
        poles = np.linalg.eigvals(shift)
    except np.linalg.LinAlgError:
        return None, "shift_solve_failure"
    pole_tolerance = np.finfo(np.float64).eps * 100
    if len(poles) != rank or not np.isfinite(poles).all() or np.any(np.abs(poles) <= pole_tolerance):
        return None, "invalid_poles"
    poles = poles / np.abs(poles)
    time_index = np.arange(length, dtype=np.int64)
    design = poles[None, :] ** time_index[:, None]
    gram = design.conj().T @ design
    rhs = design.conj().T @ sequence
    matrix = gram + gain_fit_ridge * np.eye(rank, dtype=np.complex128)
    try:
        gains = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        gains = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    prediction = np.sum(gains * poles ** (length - 1 + horizon))
    if not np.isfinite(prediction):
        return None, "nonfinite_forecast"
    return prediction, None


class ClassicalPredictor:
    """Factory-created predictor implementing fit(train_records, config)/predict(...)."""

    def __init__(self, method):
        if method not in _METHODS:
            raise ValueError("Unknown classical predictor: " + str(method))
        self.method = method
        self.config = None
        self._shape = None
        self._training_environment_ids = ()
        self._parameters = {}
        self._fit_diagnostics = {}
        self._fitted = False

    def fit(self, training_records, config):
        started = time.perf_counter()
        records, env_ids, shape = _training_records(training_records)
        cfg = _normalize_config(config, self.method)
        self.config = cfg
        self._shape = tuple(shape)
        self._training_environment_ids = env_ids
        self._parameters = {}
        self._fit_diagnostics = {
            "n_training_records": len(records),
            "n_training_environments": len(env_ids),
            "training_environment_ids": list(env_ids),
            "precision": "complex128/float64",
            "origin_start": cfg["origin_start"],
        }

        if self.method == "train_mean":
            self._parameters["mean"] = _environment_channel_means(records)
        elif self.method in {"ar", "kalman"}:
            self._fit_ar(records)
        elif self.method in {"ridge", "lmmse"}:
            self._fit_direct(records)
        elif self.method == "esprit":
            rows = cfg["hankel_rows"]
            if cfg["history_length"] <= rows:
                raise ValueError("ESPRIT history_length must exceed hankel_rows")
            self._parameters = {
                "rank": cfg["rank"],
                "hankel_rows": rows,
                "gain_fit_ridge": cfg["gain_fit_ridge"],
            }
            self._fit_diagnostics["training_data_used_for_coefficients"] = False
        self._fit_diagnostics["fit_elapsed_seconds"] = float(time.perf_counter() - started)
        self._fitted = True
        return self

    def _fit_ar(self, records):
        order = self.config["order"]
        mean = _channel_mean_features(records, real_ri=self.method == "ar")
        moments = _fit_moments(
            records, order, 1, self.config["origin_start"],
            real_ri=self.method == "ar", center_offset=mean, fit_intercept=False,
        )
        loading = self.config["relative_loading"]
        coefficients, loadings, conditions, fallback = _solve_regularized(moments, loading)
        self._parameters.update({
            "coefficients": coefficients,
            "mean": mean,
            "loadings": loadings,
        })
        diag = _condition_summary(conditions)
        diag.update({
            "order": order,
            "relative_loading": loading,
            "loading_rule": "alpha * trace(Gram_scaled_by_training_rows) / order",
            "solver_fallback_features": int(fallback),
            "n_training_windows": moments["n_training_rows"],
            "environment_equal_weighting": True,
        })
        self._fit_diagnostics["normal_equation"] = diag
        if self.method == "kalman":
            process_power = _complex_residual_power(
                records, order, self.config["origin_start"], mean, coefficients
            )
            channel_mean = _environment_channel_means(records)
            signal_power = _environment_signal_power(records, channel_mean).reshape(-1)
            process_scale = self.config["process_noise_scale"]
            self._parameters.update({
                "process_noise_variance": process_power * process_scale,
                "process_noise_unscaled": process_power,
                "signal_power": signal_power,
                "kalman_relative_loading": loading,
                "process_noise_scale": process_scale,
            })
            self._fit_diagnostics["kalman"] = {
                "process_noise_scale": process_scale,
                "process_noise_estimator": "equal-environment one-step innovation mean square",
                "measurement_model": "per-complex-coefficient observation of current CSI",
                "relative_loading_fixed_default": loading,
                "normal_equation": diag,
            }

    def _fit_direct(self, records):
        width = self.config["history_length"]
        relative_loading = self.config["relative_loading"]
        heads = {}
        summaries = {}
        for horizon in self.config["horizon_steps"]:
            moments = _fit_moments(
                records, width, horizon, self.config["origin_start"],
                real_ri=(self.method == "ridge"), fit_intercept=True,
            )
            coef, loadings, conditions, fallback = _solve_regularized(moments, relative_loading)
            intercept = moments["mean_y"] - np.sum(moments["mean_x"] * coef, axis=-1)
            heads[horizon] = {
                "coefficients": coef,
                "intercept": intercept,
                "loadings": loadings,
            }
            summary = _condition_summary(conditions)
            summary.update({
                "relative_loading": relative_loading,
                "loading_rule": "alpha * trace(centered Gram scaled by training rows) / width",
                "solver_fallback_features": int(fallback),
                "n_training_windows": moments["n_training_rows"],
                "environment_equal_weighting": True,
            })
            summaries[str(horizon)] = summary
        self._parameters["heads"] = heads
        self._fit_diagnostics["direct_heads"] = summaries
        self._fit_diagnostics["fit_form"] = (
            "per RI feature real affine temporal regression"
            if self.method == "ridge"
            else "per complex CSI coefficient affine empirical Wiener predictor"
        )

    def _validate_history(self, history):
        self._require_fitted()
        x = np.asarray(history)
        if (
            x.ndim != 4 or not x.size or not np.iscomplexobj(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("history must be finite complex [B,L,Nt,Nf]")
        if x.shape[1] != self.config["history_length"]:
            raise ValueError("history length does not match the fitted configuration")
        if tuple(x.shape[2:]) != self._shape:
            raise ValueError("history Nt,Nf does not match fitted training records")
        return x.astype(np.complex128, copy=False)

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit(training_records, config) before predict")

    def predict(self, history, conditions=None, horizon_steps=None):
        x = self._validate_history(history)
        horizon = _resolve_predict_horizon(horizon_steps, self.config)
        batch, length, n_tx, n_sc = x.shape
        started = time.perf_counter()
        statuses = ["ok"] * batch
        failure_counts = Counter()

        if self.method == "persistence":
            prediction = x[:, -1].copy()
        elif self.method == "zero":
            prediction = np.zeros((batch, n_tx, n_sc), dtype=np.complex128)
        elif self.method == "train_mean":
            prediction = np.broadcast_to(
                self._parameters["mean"], (batch, n_tx, n_sc)
            ).copy()
        elif self.method == "ar":
            features = _ri_history_features(x)
            predicted = _forecast_autoregression(
                features, self._parameters["coefficients"],
                self._parameters["mean"], horizon,
            )
            prediction = _ri_prediction(predicted, n_tx, n_sc)
        elif self.method == "ridge":
            if horizon not in self._parameters["heads"]:
                raise ValueError("Ridge has no direct head fitted for this horizon")
            head = self._parameters["heads"][horizon]
            features = _ri_history_features(x)
            predicted = np.einsum(
                "bdp,dp->bd", features, head["coefficients"], optimize=True
            ) + head["intercept"][None, :]
            prediction = _ri_prediction(predicted, n_tx, n_sc)
        elif self.method == "lmmse":
            if horizon not in self._parameters["heads"]:
                raise ValueError("LMMSE has no direct head fitted for this horizon")
            head = self._parameters["heads"][horizon]
            features = _complex_history_features(x)
            predicted = np.einsum(
                "bdp,dp->bd", features, head["coefficients"], optimize=True
            ) + head["intercept"][None, :]
            prediction = predicted.reshape(batch, n_tx, n_sc)
        elif self.method == "kalman":
            prediction, statuses, failure_counts, noise_mode = self._predict_kalman(
                x, conditions, horizon
            )
        elif self.method == "esprit":
            prediction, statuses, failure_counts = self._predict_esprit(x, horizon)
        else:
            raise RuntimeError("Unreachable predictor method")

        if prediction.shape != (batch, n_tx, n_sc):
            raise ValueError("Predictor produced malformed output shape")
        kalman_component_failures = (
            int(sum(failure_counts.values())) if self.method == "kalman" else None
        )
        finite_rows = np.isfinite(prediction).all(axis=(1, 2))
        for row in np.flatnonzero(~finite_rows):
            prediction[row] = x[row, -1]
            statuses[row] = "persistence_fallback:nonfinite_prediction"
            failure_counts["nonfinite_prediction"] += 1
        if not np.isfinite(prediction).all():
            raise ValueError("Persistence fallback could not produce finite output")
        diagnostics = {
            "method": self.method,
            "horizon_steps": horizon,
            "n_samples": batch,
            "failure_counts": dict(failure_counts),
            "n_failed_rows": int(sum(status != "ok" for status in statuses)),
            "fit_elapsed_seconds": self._fit_diagnostics.get("fit_elapsed_seconds"),
            "predict_elapsed_seconds": float(time.perf_counter() - started),
            "state_nbytes": self._state_nbytes(),
            "training_environment_ids": list(self._training_environment_ids),
        }
        if self.method == "kalman":
            diagnostics["measurement_noise_mode"] = noise_mode
            diagnostics["n_failed_components"] = kalman_component_failures
        if self.method == "esprit":
            diagnostics["rank"] = self._parameters["rank"]
        return {
            "prediction": prediction.astype(np.complex128, copy=False),
            "status": statuses,
            "diagnostics": diagnostics,
        }

    def _predict_kalman(self, history, conditions, horizon):
        batch, length, n_tx, n_sc = history.shape
        features = _complex_history_features(history)
        _, n_features, _ = features.shape
        order = self.config["order"]
        if length < order:
            raise ValueError("History is shorter than Kalman state order")
        mean = self._parameters["mean"]
        coefficients = self._parameters["coefficients"]
        process = self._parameters["process_noise_variance"]
        signal_power = self._parameters["signal_power"]
        snr_ratio = _measurement_snr(conditions, batch)
        if snr_ratio is None:
            measurement = np.zeros((batch, n_features), dtype=np.float64)
        else:
            measurement = snr_ratio[:, None] * signal_power[None, :]
        mode = "clean" if not np.any(measurement > 0) else "noisy"

        prediction = np.empty((batch, n_features), dtype=np.complex128)
        statuses = ["ok"] * batch
        failures = Counter()
        identity = np.eye(order, dtype=np.complex128)
        for b in range(batch):
            for d in range(n_features):
                series = features[b, d] - mean[d]
                if measurement[b, d] == 0:
                    forecast = _forecast_autoregression(
                        series[None, None, :],
                        coefficients[d:d + 1],
                        np.zeros(1, dtype=np.complex128),
                        horizon,
                    )[0, 0]
                    prediction[b, d] = forecast + mean[d]
                    continue

                transition = np.zeros((order, order), dtype=np.complex128)
                transition[0, :] = coefficients[d, ::-1]
                if order > 1:
                    transition[1:, :-1] = np.eye(order - 1, dtype=np.complex128)
                q = max(float(process[d]), 0.0)
                process_cov = np.zeros((order, order), dtype=np.complex128)
                process_cov[0, 0] = q
                r = float(measurement[b, d])
                state = np.asarray(series[:order][::-1], dtype=np.complex128).copy()
                covariance = r * identity
                failed = False
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for k in range(order, length):
                        state = transition @ state
                        covariance = transition @ covariance @ transition.conj().T + process_cov
                        covariance = (covariance + covariance.conj().T) * 0.5
                        innovation_variance = float(np.real(covariance[0, 0]) + r)
                        if (
                            not np.isfinite(innovation_variance)
                            or innovation_variance <= 0
                            or not np.isfinite(state).all()
                            or not np.isfinite(covariance).all()
                        ):
                            failed = True
                            break
                        gain = covariance[:, 0] / innovation_variance
                        innovation = series[k] - state[0]
                        state = state + gain * innovation
                        correction = identity - np.outer(gain, np.eye(order, dtype=np.complex128)[0])
                        covariance = (
                            correction @ covariance @ correction.conj().T
                            + r * np.outer(gain, gain.conj())
                        )
                        covariance = (covariance + covariance.conj().T) * 0.5
                        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
                            failed = True
                            break
                    if not failed:
                        for _ in range(horizon):
                            state = transition @ state
                            covariance = transition @ covariance @ transition.conj().T + process_cov
                            covariance = (covariance + covariance.conj().T) * 0.5
                            if not np.isfinite(state).all() or not np.isfinite(covariance).all():
                                failed = True
                                break
                if failed:
                    prediction[b, d] = features[b, d, -1]
                    statuses[b] = "component_persistence_fallback:singular_innovation"
                    failures["singular_innovation"] += 1
                    continue
                prediction[b, d] = state[0] + mean[d]
        return prediction.reshape(batch, n_tx, n_sc), statuses, failures, mode

    def _predict_esprit(self, history, horizon):
        batch, length, n_tx, n_sc = history.shape
        features = _complex_history_features(history)
        _, n_features, _ = features.shape
        rank = self._parameters["rank"]
        rows = self._parameters["hankel_rows"]
        ridge = self._parameters["gain_fit_ridge"]
        prediction = np.empty((batch, n_features), dtype=np.complex128)
        statuses = ["ok"] * batch
        failures = Counter()
        for b in range(batch):
            row_failure = None
            row_prediction = np.empty(n_features, dtype=np.complex128)
            for d in range(n_features):
                value, reason = _esprit_forecast(
                    features[b, d], rank, rows, horizon, ridge
                )
                if reason is not None:
                    row_failure = reason
                    break
                row_prediction[d] = value
            if row_failure is not None:
                prediction[b] = features[b, :, -1]
                statuses[b] = "persistence_fallback:" + row_failure
                failures[row_failure] += 1
            else:
                prediction[b] = row_prediction
        return prediction.reshape(batch, n_tx, n_sc), statuses, failures

    def _state_nbytes(self):
        def count(value):
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, dict):
                return sum(count(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count(v) for v in value)
            return 0
        return int(count(self._parameters))

    def state_dict(self):
        self._require_fitted()
        return {
            "schema_version": _STATE_SCHEMA,
            "method": self.method,
            "config": _copy_tree(self.config),
            "shape": tuple(self._shape),
            "training_environment_ids": tuple(self._training_environment_ids),
            "parameters": _copy_tree(self._parameters),
            "fit_diagnostics": _copy_tree(self._fit_diagnostics),
        }

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict) or state.get("schema_version") != _STATE_SCHEMA:
            raise ValueError("Unsupported classical predictor state schema")
        obj = cls(state.get("method"))
        shape = tuple(state.get("shape", ()))
        if len(shape) != 2 or any(_positive_integer(v, "state shape") < 1 for v in shape):
            raise ValueError("Invalid predictor state shape")
        if not isinstance(state.get("parameters"), dict) or not isinstance(state.get("config"), dict):
            raise ValueError("Malformed predictor state")
        obj.config = _copy_tree(state["config"])
        obj._shape = shape
        obj._training_environment_ids = tuple(state.get("training_environment_ids", ()))
        obj._parameters = _copy_tree(state["parameters"])
        obj._fit_diagnostics = _copy_tree(state.get("fit_diagnostics", {}))
        obj._fitted = True
        return obj


def build_classical_predictor(method):
    """Return a predictor implementing the frozen classical fit/predict interface."""
    return ClassicalPredictor(method)


def load_classical_predictor(state_dict):
    """Restore a fitted predictor from its pickle-friendly state dictionary."""
    return ClassicalPredictor.from_state_dict(state_dict)
