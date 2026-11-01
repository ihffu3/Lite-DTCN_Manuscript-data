#!/usr/bin/env python
"""Strict, read-only postprocessor for a complete A1 v2 replay.

This module never fits predictors, opens MAT files, or reads partial result rows.
The formal result directory is inspected only after its final completion marker
and checksum-bound run identity are present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
RUNNER_PATH = HERE.parent / "runner_v2.py"
PROTOCOL_PATH = HERE.parent / "protocol.json"
FIGURE_BUILDER_PATH = (
    PROJECT_ROOT / "revision/paper/figure_redraw_20261029/a1_noisy_link_figure_v1/"
    "build_a1_noisy_link_figure.py"
)
DEFAULT_RESULTS = HERE.parent / "results_v2"
METHODS = ("Lite", "LMMSE", "Ridge", "AR", "Kalman", "Persistence")
REFERENCE_METHODS = ("Perfect", "Outdated")
SPEEDS = (30, 300)
SNRS = (10, 20, 30)
N_ENVIRONMENTS = 30
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED_BASE = 20_261_030
FIGURE_FIELDS = (
    "metric", "method", "speed_kmh", "nominal_snr_db", "mean",
    "ci95_low", "ci95_high", "n_environments",
)
MODEL_ENV_FIELDS = (
    "speed_kmh", "nominal_snr_db", "method", "environment_id",
    "mat_sha256", "job_sha256", "qa_receipt_sha256",
    "expected_rows_per_method", "n_prediction_rows", "error_energy",
    "target_energy", "prediction_energy", "nmse", "nmse_db",
    "rzf_sum_se_bps_hz", "n_link_origins",
    "total_precoder_subcarriers", "precoder_fallback_subcarriers",
    "total_user_subcarriers", "su_fallback_user_subcarriers",
    "invalid_true_user_subcarriers", "receipt_path", "receipt_sha256",
    "cell_complete_sha256",
)
REFERENCE_ENV_FIELDS = (
    "speed_kmh", "nominal_snr_db", "method", "environment_id",
    "mat_sha256", "expected_rows_per_method", "n_prediction_rows",
    "error_energy", "target_energy", "prediction_energy", "nmse",
    "nmse_db", "rzf_sum_se_bps_hz", "n_link_origins",
    "total_precoder_subcarriers", "precoder_fallback_subcarriers",
    "total_user_subcarriers", "su_fallback_user_subcarriers",
    "invalid_true_user_subcarriers", "receipt_path", "receipt_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import reviewed helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_complete_marker(results_dir: Path) -> tuple[dict[str, Any], str]:
    """Fail before touching any cell/environment receipt unless final exists."""
    complete_path = Path(results_dir) / "complete.json"
    if not complete_path.is_file():
        raise ValueError(f"A1 run is partial: final marker is absent: {complete_path}")
    complete = read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError(f"A1 run is partial: final marker status is {complete.get('status')!r}")
    return complete, sha256(complete_path)


def complete_cell_map(complete_cells: Any) -> dict[tuple[int, int], dict[str, Any]]:
    expected = {(speed, snr) for speed in SPEEDS for snr in SNRS}
    if not isinstance(complete_cells, list):
        raise ValueError("Final marker lacks the complete cell ledger")
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for item in complete_cells:
        if not isinstance(item, dict):
            raise ValueError("Malformed A1 complete-cell ledger entry")
        try:
            key = (int(item["speed_kmh"]), int(item["history_noise_snr_db"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A1 complete-cell entry lacks speed/SNR identity") from exc
        if key in found:
            raise ValueError(f"Duplicate A1 complete cell: {key}")
        found[key] = item
    if set(found) != expected or len(found) != 6:
        raise ValueError("A1 final marker must bind exactly the 2-speed x 3-SNR grid")
    return found


def assert_reference_pointer(actual: Any, expected: dict[str, Any]) -> None:
    if not isinstance(actual, dict) or actual != expected:
        raise ValueError("Clean-reference reuse pointer/path/SHA does not match the SNR-10 source")


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def validate_rzf_metrics(link: Any, expected_rows: int) -> dict[str, Any]:
    """Validate persisted RZF denominators, finite values, ranges and fallback counts."""
    if not isinstance(link, dict):
        raise ValueError("RZF metrics are missing")
    if type(expected_rows) is not int or expected_rows < 4 or expected_rows % 4:
        raise ValueError("Expected prediction rows must cover complete four-user origins")
    n_origins = expected_rows // 4
    expected = {
        "receiver_snr_db": 10,
        "precoder": "RZF",
        "n_link_origins": n_origins,
        "total_precoder_subcarriers": n_origins * 32,
        "total_user_subcarriers": expected_rows * 32,
        "invalid_true_user_subcarriers": 0,
    }
    for field, value in expected.items():
        if link.get(field) != value:
            raise ValueError(f"RZF denominator/protocol mismatch for {field}: {link.get(field)!r}")
    se = _finite(link.get("mean_sum_se_bps_hz"), "mean_sum_se_bps_hz")
    if se < 0:
        raise ValueError("RZF sum spectral efficiency must be non-negative")
    pre_fallback = _integer(
        link.get("precoder_fallback_subcarriers"), "precoder_fallback_subcarriers",
    )
    su_fallback = _integer(
        link.get("su_fallback_user_subcarriers"), "su_fallback_user_subcarriers",
    )
    if pre_fallback > expected["total_precoder_subcarriers"]:
        raise ValueError("Precoder fallback count exceeds its subcarrier denominator")
    if su_fallback > expected["total_user_subcarriers"]:
        raise ValueError("SU fallback count exceeds its user-subcarrier denominator")

    efficiency = link.get("mean_beam_efficiency")
    loss_db = link.get("mean_beam_loss_db")
    if (efficiency is None) != (loss_db is None):
        raise ValueError("Beam-efficiency and beam-loss summaries must share valid-value support")
    if efficiency is not None:
        efficiency = _finite(efficiency, "mean_beam_efficiency")
        loss_db = _finite(loss_db, "mean_beam_loss_db")
        if not 0.0 <= efficiency <= 1.0 or loss_db < 0.0:
            raise ValueError("Beam efficiency/loss lies outside the physical range")
    return {
        "rzf_sum_se_bps_hz": se,
        "n_link_origins": n_origins,
        "total_precoder_subcarriers": expected["total_precoder_subcarriers"],
        "precoder_fallback_subcarriers": pre_fallback,
        "total_user_subcarriers": expected["total_user_subcarriers"],
        "su_fallback_user_subcarriers": su_fallback,
        "invalid_true_user_subcarriers": 0,
    }


def validate_metric_row(row: dict[str, Any], *, method: str, expected_rows: int) -> dict[str, Any]:
    if row.get("method") != method or row.get("status") != "complete":
        raise ValueError(f"Incomplete or mislabelled metric row for {method}")
    if _integer(row.get("n_prediction_rows"), "n_prediction_rows") != expected_rows:
        raise ValueError(f"No-drop prediction row count differs for {method}")
    if _integer(row.get("n_non_ok_rows"), "n_non_ok_rows") != 0:
        raise ValueError(f"Predictor fallback/status rows are not zero for {method}")
    expected_status = "ok" if method in METHODS else f"{method.lower()}_reference"
    if row.get("prediction_status_counts") != {expected_status: expected_rows}:
        raise ValueError(f"Unexpected prediction status counts for {method}")
    error = _finite(row.get("error_energy"), "error_energy")
    target = _finite(row.get("target_energy"), "target_energy")
    prediction = _finite(row.get("prediction_energy"), "prediction_energy")
    nmse = _finite(row.get("nmse"), "nmse")
    if error < 0 or target <= 0 or prediction < 0 or nmse < 0:
        raise ValueError(f"Energy/NMSE range invalid for {method}")
    if not math.isclose(nmse, error / target, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"NMSE does not equal error_energy/target_energy for {method}")
    nmse_db = row.get("nmse_db")
    if nmse == 0:
        if nmse_db is not None:
            raise ValueError(f"Zero NMSE must retain null dB value for {method}")
    else:
        expected_db = 10.0 * math.log10(nmse)
        if not math.isclose(_finite(nmse_db, "nmse_db"), expected_db,
                            rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"NMSE-dB does not match linear NMSE for {method}")
    if method == "Perfect" and (error != 0.0 or nmse != 0.0):
        raise ValueError("Perfect reference must have zero prediction error")
    link_summary = validate_rzf_metrics(row.get("rzf_10db"), expected_rows)
    return {
        "error_energy": error,
        "target_energy": target,
        "prediction_energy": prediction,
        "nmse": nmse,
        "nmse_db": None if nmse_db is None else float(nmse_db),
        **link_summary,
    }


def bootstrap_rate_summaries(
    rate_by_condition: dict[tuple[int, int, str], dict[str, float]],
    environment_ids_by_speed: dict[int, list[str]],
    *, draws: int = BOOTSTRAP_DRAWS,
    seed_base: int = BOOTSTRAP_SEED_BASE,
) -> tuple[dict[tuple[int, int, str], dict[str, float]], dict[tuple[int, int], dict[str, float]]]:
    """Share one PCG64 environment-resample matrix across each speed's full panel."""
    import numpy as np

    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draw count must be a positive integer")
    summaries: dict[tuple[int, int, str], dict[str, float]] = {}
    paired: dict[tuple[int, int], dict[str, float]] = {}
    for speed in SPEEDS:
        env_ids = environment_ids_by_speed.get(speed)
        if not isinstance(env_ids, list) or len(env_ids) != N_ENVIRONMENTS:
            raise ValueError(f"Expected exactly 30 sorted environment IDs at speed {speed}")
        if env_ids != sorted(env_ids) or len(set(env_ids)) != N_ENVIRONMENTS:
            raise ValueError(f"Environment IDs must be unique and sorted at speed {speed}")
        rng = np.random.Generator(np.random.PCG64(seed_base + speed))
        indices = rng.integers(0, N_ENVIRONMENTS, size=(draws, N_ENVIRONMENTS), dtype=np.int64)
        arrays: dict[tuple[int, int, str], Any] = {}
        for snr in SNRS:
            for method in METHODS:
                key = (speed, snr, method)
                values_by_env = rate_by_condition.get(key)
                if not isinstance(values_by_env, dict) or set(values_by_env) != set(env_ids):
                    raise ValueError(f"RZF environment grid is incomplete for {key}")
                values = np.asarray([_finite(values_by_env[env], f"RZF value {key}/{env}")
                                     for env in env_ids], dtype=np.float64)
                if np.any(values < 0):
                    raise ValueError(f"RZF sum spectral efficiency is negative for {key}")
                arrays[key] = values
                draws_mean = values[indices].mean(axis=1, dtype=np.float64)
                low, high = np.quantile(draws_mean, [0.025, 0.975], method="linear")
                summaries[key] = {
                    "mean": float(values.mean(dtype=np.float64)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "n_environments": N_ENVIRONMENTS,
                    "bootstrap_seed": seed_base + speed,
                    "bootstrap_replicates": draws,
                }
        for snr in SNRS:
            delta = arrays[(speed, snr, "Lite")] - arrays[(speed, snr, "LMMSE")]
            draws_mean = delta[indices].mean(axis=1, dtype=np.float64)
            low, high = np.quantile(draws_mean, [0.025, 0.975], method="linear")
            paired[(speed, snr)] = {
                "mean": float(delta.mean(dtype=np.float64)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "n_environments": N_ENVIRONMENTS,
                "bootstrap_seed": seed_base + speed,
                "bootstrap_replicates": draws,
            }
    return summaries, paired


def _manifest_ids(manifest: dict[str, Any], speed: int, split: str) -> list[str]:
    groups = manifest.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("Formal manifest has no group mapping")
    ids = sorted(
        env_id for env_id, group in groups.items()
        if group.get("family") == "D1_main" and group.get("split") == split
        and int(round(float(group.get("speed_kmh", -1)))) == speed
    )
    if len(ids) != N_ENVIRONMENTS:
        raise ValueError(f"Expected 30 D1_main/{split} identities at {speed} km/h")
    return ids


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_fit_identity(root: Path, manifest: dict[str, Any], speed: int) -> tuple[dict[str, Any], str]:
    path = root / f"S{speed}_fit_identity.json"
    if not path.is_file():
        raise ValueError(f"A1 fit identity is missing: {path}")
    fit = read_json(path)
    train_ids = _manifest_ids(manifest, speed, "train")
    test_ids = _manifest_ids(manifest, speed, "test")
    if (fit.get("speed_kmh") != speed
            or fit.get("train_environment_ids") != train_ids
            or fit.get("test_environment_ids") != test_ids):
        raise ValueError(f"A1 fit/test environment identity mismatch at {speed} km/h")
    expected_train_sha = {
        env_id: manifest["groups"][env_id]["output_sha256"] for env_id in train_ids
    }
    if fit.get("train_mat_sha256") != expected_train_sha:
        raise ValueError(f"A1 training MAT hashes differ from the formal manifest at {speed} km/h")
    for name in ("checkpoint_sha256", "run_manifest_sha256", "normalizer_sha256"):
        _require_sha(fit.get(name), f"fit {name}")
    scale = _finite(fit.get("channel_scale"), "fit channel_scale")
    if scale <= 0:
        raise ValueError("A1 fit channel_scale must be positive")
    return fit, sha256(path)


def _validate_cell_identity(
    identity: dict[str, Any], *, speed: int, snr: int, fit: dict[str, Any],
    source_hashes: dict[str, str], protocol_sha: str, runner_sha: str,
) -> None:
    expected = {
        "schema_version": "1.0",
        "speed_kmh": speed,
        "history_noise_snr_db": snr,
        "horizon_steps": 4,
        "horizon_ms": 2.0,
        "history_length": 10,
        "noise_seed": 2_602_236,
        "channel_scale": fit["channel_scale"],
        "protocol_sha256": protocol_sha,
        "runner_sha256": runner_sha,
        "source_sha256": source_hashes,
        "training_environment_ids": fit["train_environment_ids"],
        "training_mat_sha256": fit["train_mat_sha256"],
        "lite_checkpoint_sha256": fit["checkpoint_sha256"],
        "lite_run_manifest_sha256": fit["run_manifest_sha256"],
        "normalizer_sha256": fit["normalizer_sha256"],
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(f"A1 cell identity differs from reviewed fit/source binding: {field}")
    fingerprints = identity.get("fit_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(METHODS):
        raise ValueError("A1 cell does not bind exactly the six fixed predictor fingerprints")
    if fingerprints.get("Lite") != fit["checkpoint_sha256"]:
        raise ValueError("Lite fingerprint differs from the fixed checkpoint SHA")
    for method, fingerprint in fingerprints.items():
        _require_sha(fingerprint, f"{method} fit fingerprint")


def _validate_run_header(root: Path, results_dir: Path, complete: dict[str, Any],
                         complete_sha: str, runner, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], str]:
    identity_path = results_dir / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError("A1 final marker has no run identity")
    run_identity = read_json(identity_path)
    if run_identity.get("status") != "complete" or run_identity.get("complete_sha256") != complete_sha:
        raise ValueError("A1 run identity does not checksum-bind the final marker")
    protocol_sha = runner.sha256(PROTOCOL_PATH)
    runner_sha = sha256(RUNNER_PATH)
    source_hashes = runner._validate_static_sources(root, protocol)
    expected = {
        "schema_version": "1.0",
        "stage": "A1_noisy_input_link_replay",
        "protocol_sha256": protocol_sha,
        "runner_sha256": runner_sha,
        "formal_manifest_sha256": protocol["formal_manifest_sha256"],
        "source_sha256": source_hashes,
        "prior_a1_rows_sha256": protocol["prior_a1_binding"]["environment_rows_sha256"],
        "noise_seed": protocol["noise"]["seed"],
        "history_noise_snr_db": list(SNRS),
        "receiver_snr_db": 10,
        "speeds_kmh": list(SPEEDS),
        "horizon_steps": 4,
        "history_length": 10,
        "device": "cpu",
    }
    for field, value in expected.items():
        if run_identity.get(field) != value or complete.get(field) != value:
            raise ValueError(f"A1 completed run identity mismatch for {field}")
    if (complete.get("status") != "complete"
            or complete.get("prior_a1_nmse_matched") is not True
            or complete.get("test_rows_dropped") != 0
            or complete.get("n_model_link_environment_records") != 1080
            or complete.get("n_reference_link_environment_records") != 120):
        raise ValueError("A1 final summary is incomplete or reports a dropped/mismatched scope")
    return run_identity, source_hashes, runner_sha


def _validate_cell_complete(cell: dict[str, Any], speed: int, snr: int,
                            identity: dict[str, Any]) -> None:
    expected = {
        "status": "complete",
        "cell_identity": identity,
        "n_environments": N_ENVIRONMENTS,
        "n_model_methods_per_environment": len(METHODS),
        "n_reference_methods_evaluated_here": len(REFERENCE_METHODS) if snr == 10 else 0,
        "n_methods_per_environment": len(METHODS) + (len(REFERENCE_METHODS) if snr == 10 else 0),
        "n_reference_receipts_reused": 0 if snr == 10 else N_ENVIRONMENTS,
        "receiver_snr_db": 10,
        "n_model_link_environment_records": N_ENVIRONMENTS * len(METHODS),
        "n_reference_link_environment_records_evaluated_here": (
            N_ENVIRONMENTS * len(REFERENCE_METHODS) if snr == 10 else 0
        ),
    }
    for field, value in expected.items():
        if cell.get(field) != value:
            raise ValueError(f"A1 S{speed}/SNR{snr} completion field mismatch: {field}")


def analyze_completed_run(root: Path, results_dir: Path) -> dict[str, Any]:
    """Read only a final A1 run; never call the producer or open MAT/checkpoint data."""
    root = Path(root).resolve(strict=True)
    results_dir = Path(results_dir).resolve(strict=True)
    complete, complete_sha = read_complete_marker(results_dir)
    runner = _load_module(RUNNER_PATH, "a1_runner_v2_analysis_import")
    figure_builder = _load_module(FIGURE_BUILDER_PATH, "a1_figure_builder_analysis_import")
    protocol = read_json(PROTOCOL_PATH)
    run_identity, source_hashes, runner_sha = _validate_run_header(
        root, results_dir, complete, complete_sha, runner, protocol,
    )
    manifest_path = root / "revision/data/formal/manifest.json"
    manifest = read_json(manifest_path)  # JSON identity metadata only; no MAT access.
    prior_rows = runner._load_old_a1_rows(root)
    frozen_nmse = figure_builder.read_frozen_nmse()
    completion_cells = complete_cell_map(complete.get("complete_cells"))

    model_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    receipt_manifest: list[dict[str, Any]] = []
    reuse_manifest: list[dict[str, Any]] = []
    fit_bindings: dict[int, dict[str, Any]] = {}
    fit_sha_by_speed: dict[int, str] = {}
    invariant_fingerprints_by_speed: dict[int, dict[str, str]] = {}
    target_energy_by_env: dict[tuple[int, str], float] = {}
    expected_rows_by_env: dict[tuple[int, str], int] = {}
    ref_identity_by_speed: dict[int, dict[str, Any]] = {}
    condition_keys = {(speed, snr) for speed in SPEEDS for snr in SNRS}
    if set(completion_cells) != condition_keys:
        raise ValueError("A1 complete cell grid differs from the frozen scope")

    for speed in SPEEDS:
        fit, fit_sha = _validate_fit_identity(results_dir, manifest, speed)
        fit_bindings[speed] = fit
        fit_sha_by_speed[speed] = fit_sha
        if speed in ref_identity_by_speed:
            raise AssertionError("Unexpected duplicate speed identity")
        expected_test_ids = fit["test_environment_ids"]

        for snr in SNRS:
            cell_dir = results_dir / "environments" / f"S{speed}_SNR{snr}"
            identity_path = cell_dir / "cell_identity.json"
            cell_path = cell_dir / "cell_complete.json"
            if not identity_path.is_file() or not cell_path.is_file():
                raise ValueError(f"A1 condition is incomplete: {cell_dir}")
            identity = read_json(identity_path)
            _validate_cell_identity(
                identity, speed=speed, snr=snr, fit=fit, source_hashes=source_hashes,
                protocol_sha=run_identity["protocol_sha256"], runner_sha=runner_sha,
            )
            stable = {method: identity["fit_fingerprints"][method]
                      for method in ("Lite", "AR", "Persistence")}
            prior_stable = invariant_fingerprints_by_speed.setdefault(speed, stable)
            if stable != prior_stable:
                raise ValueError(f"A1 clean-fit/Lite fingerprints changed across SNR cells at {speed} km/h")
            cell = read_json(cell_path)
            _validate_cell_complete(cell, speed, snr, identity)
            ledger_entry = completion_cells[(speed, snr)]
            if (ledger_entry.get("n_environments") != N_ENVIRONMENTS
                    or ledger_entry.get("cell_complete_sha256") != sha256(cell_path)):
                raise ValueError(f"Top-level final marker does not bind S{speed}/SNR{snr}")
            env_entries = cell.get("environment_receipts")
            if not isinstance(env_entries, list) or len(env_entries) != N_ENVIRONMENTS:
                raise ValueError(f"A1 S{speed}/SNR{snr} environment ledger is incomplete")
            env_entry_by_id: dict[str, dict[str, Any]] = {}
            for entry in env_entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("environment_id"), str):
                    raise ValueError("Malformed A1 environment receipt entry")
                env_id = entry["environment_id"]
                if env_id in env_entry_by_id:
                    raise ValueError(f"Duplicate A1 environment receipt: {env_id}")
                env_entry_by_id[env_id] = entry
            if set(env_entry_by_id) != set(expected_test_ids):
                raise ValueError(f"A1 S{speed}/SNR{snr} test environment grid differs")
            expected_files = {"cell_identity.json", "cell_complete.json"} | {
                f"{env_id}.json" for env_id in expected_test_ids
            }
            if {path.name for path in cell_dir.glob("*.json")} != expected_files:
                raise ValueError(f"A1 S{speed}/SNR{snr} contains unbound or missing JSON receipts")

            ref_identity = ref_identity_by_speed.get(speed)
            if snr == 10:
                if ref_identity is not None:
                    raise ValueError(f"Duplicate SNR-10 reference cell at {speed} km/h")
                ref_identity = identity
                ref_identity_by_speed[speed] = identity
            elif ref_identity is None:
                raise ValueError(f"Missing SNR-10 reference cell before SNR {snr}")

            for env_id in expected_test_ids:
                group = manifest["groups"][env_id]
                expected_environment = runner._environment_identity(group, env_id)
                env_path = cell_dir / f"{env_id}.json"
                entry = env_entry_by_id[env_id]
                if env_path.is_symlink() or not env_path.is_file():
                    raise ValueError(f"A1 environment receipt is missing or is a symlink: {env_path}")
                env_sha = sha256(env_path)
                if entry.get("sha256") != env_sha:
                    raise ValueError(f"A1 completion ledger SHA mismatch: {env_path}")
                if Path(str(entry.get("path", ""))).resolve() != env_path.resolve():
                    raise ValueError(f"A1 completion ledger path mismatch: {env_path}")
                raw = read_json(env_path)
                expected_rows = _integer(
                    raw.get("expected_rows_per_method"), "expected_rows_per_method", minimum=4,
                )
                if expected_rows % 4:
                    raise ValueError(f"A1 row count does not form four-user origins: {env_id}")
                env_key = (speed, env_id)
                prior_expected_rows = expected_rows_by_env.setdefault(env_key, expected_rows)
                if prior_expected_rows != expected_rows:
                    raise ValueError(f"A1 row geometry changes across history-noise cells: {env_id}")

                reference_pointer = None
                if snr != 10:
                    reference_pointer = runner._reference_link_reuse(
                        output_dir=results_dir, speed=speed,
                        environment=expected_environment, expected_rows=expected_rows,
                        prior_rows=prior_rows, reference_cell_identity=ref_identity,
                    )
                    assert_reference_pointer(raw.get("reference_link_reuse"), reference_pointer)
                    reuse_manifest.append({
                        "speed_kmh": speed,
                        "history_noise_snr_db": snr,
                        "environment_id": env_id,
                        **reference_pointer,
                    })
                checked = runner._validate_saved_env(
                    env_path, identity, expected_environment, expected_rows,
                    prior_rows, speed, snr, reference_link_reuse=reference_pointer,
                )
                receipt_manifest.append({
                    "speed_kmh": speed,
                    "history_noise_snr_db": snr,
                    "environment_id": env_id,
                    "path": env_path.relative_to(results_dir).as_posix(),
                    "sha256": env_sha,
                })
                method_rows = checked["method_rows"]
                method_by_name = {row["method"]: row for row in method_rows}
                expected_methods = set(METHODS) | (set(REFERENCE_METHODS) if snr == 10 else set())
                if len(method_rows) != len(expected_methods) or set(method_by_name) != expected_methods:
                    raise ValueError(f"A1 method grid is incomplete for {env_id}/SNR{snr}")

                for method in sorted(expected_methods):
                    parsed = validate_metric_row(
                        method_by_name[method], method=method, expected_rows=expected_rows,
                    )
                    prior_target = target_energy_by_env.setdefault(env_key, parsed["target_energy"])
                    if not math.isclose(parsed["target_energy"], prior_target,
                                        rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"Clean target energy differs across methods/SNRs: {env_id}")
                    base = {
                        "speed_kmh": speed,
                        "nominal_snr_db": snr,
                        "method": method,
                        "environment_id": env_id,
                        "mat_sha256": expected_environment["mat_sha256"],
                        "expected_rows_per_method": expected_rows,
                        "n_prediction_rows": expected_rows,
                        "error_energy": parsed["error_energy"],
                        "target_energy": parsed["target_energy"],
                        "prediction_energy": parsed["prediction_energy"],
                        "nmse": parsed["nmse"],
                        "nmse_db": parsed["nmse_db"],
                        "rzf_sum_se_bps_hz": parsed["rzf_sum_se_bps_hz"],
                        "n_link_origins": parsed["n_link_origins"],
                        "total_precoder_subcarriers": parsed["total_precoder_subcarriers"],
                        "precoder_fallback_subcarriers": parsed["precoder_fallback_subcarriers"],
                        "total_user_subcarriers": parsed["total_user_subcarriers"],
                        "su_fallback_user_subcarriers": parsed["su_fallback_user_subcarriers"],
                        "invalid_true_user_subcarriers": parsed["invalid_true_user_subcarriers"],
                        "receipt_path": env_path.relative_to(results_dir).as_posix(),
                        "receipt_sha256": env_sha,
                    }
                    if method in METHODS:
                        base.update({
                            "job_sha256": expected_environment["job_sha256"],
                            "qa_receipt_sha256": expected_environment["qa_receipt_sha256"],
                            "cell_complete_sha256": sha256(cell_path),
                        })
                        model_rows.append(base)
                    else:
                        reference_rows.append(base)

    expected_model_keys = {
        (speed, snr, method, env_id)
        for speed in SPEEDS for snr in SNRS for method in METHODS
        for env_id in fit_bindings[speed]["test_environment_ids"]
    }
    model_keys = [(r["speed_kmh"], r["nominal_snr_db"], r["method"], r["environment_id"])
                  for r in model_rows]
    if len(model_rows) != 1080 or len(set(model_keys)) != 1080 or set(model_keys) != expected_model_keys:
        raise ValueError("A1 model rows must cover exactly 2 x 3 x 6 x 30 = 1,080 unique records")
    expected_reference_keys = {
        (speed, method, env_id) for speed in SPEEDS for method in REFERENCE_METHODS
        for env_id in fit_bindings[speed]["test_environment_ids"]
    }
    reference_keys = [(r["speed_kmh"], r["method"], r["environment_id"]) for r in reference_rows]
    if (len(reference_rows) != 120 or len(set(reference_keys)) != 120
            or set(reference_keys) != expected_reference_keys):
        raise ValueError("A1 SNR-10 references must cover exactly 2 x 30 x 2 = 120 records")
    if len(receipt_manifest) != 180 or len(reuse_manifest) != 120:
        raise ValueError("A1 receipt or reference reuse-pointer coverage is incomplete")

    rate_by_condition: dict[tuple[int, int, str], dict[str, float]] = {}
    raw_nmse_db: dict[tuple[int, int, str], list[float]] = {}
    for row in model_rows:
        key = (row["speed_kmh"], row["nominal_snr_db"], row["method"])
        rate_by_condition.setdefault(key, {})[row["environment_id"]] = row["rzf_sum_se_bps_hz"]
        if row["nmse_db"] is None:
            raise ValueError(f"Model NMSE-dB cannot be null: {key}/{row['environment_id']}")
        raw_nmse_db.setdefault(key, []).append(row["nmse_db"])
    for key, values in raw_nmse_db.items():
        prior_mean = frozen_nmse[key][0]
        replay_mean = math.fsum(values) / N_ENVIRONMENTS
        if not math.isclose(replay_mean, prior_mean, rel_tol=2e-9, abs_tol=1e-12):
            raise ValueError(f"Replay equal-environment NMSE mean differs from frozen A1: {key}")
    env_ids_by_speed = {speed: fit_bindings[speed]["test_environment_ids"] for speed in SPEEDS}
    rate_summaries, paired_summaries = bootstrap_rate_summaries(
        rate_by_condition, env_ids_by_speed,
    )
    return {
        "root": root,
        "results_dir": results_dir,
        "complete": complete,
        "complete_sha256": complete_sha,
        "run_identity": run_identity,
        "source_hashes": source_hashes,
        "runner_sha256": runner_sha,
        "protocol_sha256": runner.sha256(PROTOCOL_PATH),
        "protocol": protocol,
        "model_rows": model_rows,
        "reference_rows": reference_rows,
        "receipt_manifest": receipt_manifest,
        "reuse_manifest": reuse_manifest,
        "fit_sha_by_speed": fit_sha_by_speed,
        "fit_bindings": fit_bindings,
        "frozen_nmse": frozen_nmse,
        "frozen_nmse_sha256": sha256(figure_builder.FROZEN_NMSE_CSV),
        "rate_summaries": rate_summaries,
        "paired_summaries": paired_summaries,
        "figure_builder": figure_builder,
    }


def build_output_rows(analysis: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = analysis["frozen_nmse"]
    rate = analysis["rate_summaries"]
    fallback = {}
    for row in analysis["model_rows"]:
        key = (row["speed_kmh"], row["nominal_snr_db"], row["method"])
        state = fallback.setdefault(key, {"n_precoder_fallback_env": 0,
                                          "sum_precoder_fallback_subcarriers": 0,
                                          "n_su_fallback_env": 0,
                                          "sum_su_fallback_user_subcarriers": 0})
        pre = row["precoder_fallback_subcarriers"]
        su = row["su_fallback_user_subcarriers"]
        state["n_precoder_fallback_env"] += int(pre > 0)
        state["sum_precoder_fallback_subcarriers"] += pre
        state["n_su_fallback_env"] += int(su > 0)
        state["sum_su_fallback_user_subcarriers"] += su

    summary_rows = []
    for speed in SPEEDS:
        for snr in SNRS:
            for method in METHODS:
                key = (speed, snr, method)
                nmse_mean, nmse_low, nmse_high = frozen[key]
                link = rate[key]
                summary_rows.append({
                    "speed_kmh": speed,
                    "nominal_snr_db": snr,
                    "method": method,
                    "n_environments": N_ENVIRONMENTS,
                    "mean_nmse_db": nmse_mean,
                    "nmse_ci95_low_frozen": nmse_low,
                    "nmse_ci95_high_frozen": nmse_high,
                    "mean_rzf_sum_se_bps_hz": link["mean"],
                    "rzf_ci95_low": link["ci95_low"],
                    "rzf_ci95_high": link["ci95_high"],
                    "bootstrap_seed": link["bootstrap_seed"],
                    "bootstrap_replicates": link["bootstrap_replicates"],
                    **fallback[key],
                })
    paired_rows = []
    for speed in SPEEDS:
        for snr in SNRS:
            item = analysis["paired_summaries"][(speed, snr)]
            paired_rows.append({
                "speed_kmh": speed,
                "nominal_snr_db": snr,
                "contrast": "Lite-PC-Wiener (LMMSE)",
                "mean_lite_minus_lmmse_rzf_sum_se": item["mean"],
                "ci95_low": item["ci95_low"],
                "ci95_high": item["ci95_high"],
                "n_environments": item["n_environments"],
                "bootstrap_seed": item["bootstrap_seed"],
                "bootstrap_replicates": item["bootstrap_replicates"],
            })
    figure_rows: list[dict[str, Any]] = []
    for metric in ("nmse_db", "rzf_sum_se"):
        for speed in SPEEDS:
            for snr in SNRS:
                for method in METHODS:
                    key = (speed, snr, method)
                    if metric == "nmse_db":
                        mean, low, high = frozen[key]
                    else:
                        item = rate[key]
                        mean, low, high = item["mean"], item["ci95_low"], item["ci95_high"]
                    figure_rows.append({
                        "metric": metric,
                        "method": method,
                        "speed_kmh": speed,
                        "nominal_snr_db": snr,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_environments": N_ENVIRONMENTS,
                    })
    parsed = analysis["figure_builder"].validate_source_rows(
        [{field: str(row[field]) for field in FIGURE_FIELDS} for row in figure_rows], frozen,
    )
    if len(parsed) != 72:
        raise AssertionError("Figure source validation did not retain all 72 keys")
    return summary_rows, paired_rows, figure_rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)
    return sha256(path)


def write_analysis_outputs(analysis: dict[str, Any], output_dir: Path,
                           *, root_verified_complete: bool) -> dict[str, Any]:
    if not root_verified_complete:
        raise ValueError("Pass --root-verified-complete only after root review of this completed replay")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, paired_rows, figure_rows = build_output_rows(analysis)
    model_sha = _write_csv(output_dir / "environment_rows.csv", MODEL_ENV_FIELDS,
                           analysis["model_rows"])
    ref_sha = _write_csv(output_dir / "reference_rows.csv", REFERENCE_ENV_FIELDS,
                         analysis["reference_rows"])
    summary_fields = tuple(summary_rows[0])
    summary_sha = _write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    paired_sha = _write_csv(output_dir / "paired_rate.csv", tuple(paired_rows[0]), paired_rows)
    figure_sha = _write_csv(output_dir / "figure_source.csv", FIGURE_FIELDS, figure_rows)
    np_version = __import__("numpy").__version__
    report = {
        "schema": "a1-noisy-link-analysis-v2",
        "status": "root_verified_complete",
        "source_sha256": figure_sha,
        "source_csv": "figure_source.csv",
        "scope": {
            "family": "D1_main",
            "speeds_kmh": list(SPEEDS),
            "history_noise_snr_db": list(SNRS),
            "predictor_methods": list(METHODS),
            "formal_test_environments_per_speed": N_ENVIRONMENTS,
            "model_environment_condition_rows": 1080,
            "SNR10_reference_rows": 120,
            "SNR20_30_reference_reuse_pointers": 120,
            "target_policy": "same clean target energy across all methods and history-noise cells per speed/environment",
            "unit_of_resampling": "independent formal test environment",
            "test_access": analysis["protocol"]["formal_test_gate"]["test_access_disclosure"],
        },
        "bindings": {
            "results_directory": str(analysis["results_dir"]),
            "complete_json_sha256": analysis["complete_sha256"],
            "run_identity_json_sha256": sha256(analysis["results_dir"] / "run_identity.json"),
            "runner_v2_sha256": analysis["runner_sha256"],
            "protocol_sha256": analysis["protocol_sha256"],
            "formal_manifest_sha256": analysis["protocol"]["formal_manifest_sha256"],
            "source_files_sha256": analysis["source_hashes"],
            "prior_a1_environment_rows_sha256": analysis["protocol"]["prior_a1_binding"]["environment_rows_sha256"],
            "prior_a1_result_sha256": analysis["protocol"]["prior_a1_binding"]["result_json_sha256"],
            "prior_a1_postprocess_sha256": analysis["protocol"]["prior_a1_binding"]["postprocess_report_sha256"],
            "frozen_nmse_summary_sha256": analysis["frozen_nmse_sha256"],
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "fit_identity_sha256_by_speed": {
                str(speed): analysis["fit_sha_by_speed"][speed] for speed in SPEEDS
            },
            "environment_receipt_manifest": analysis["receipt_manifest"],
            "reference_reuse_pointer_manifest": analysis["reuse_manifest"],
        },
        "outputs": {
            "environment_rows_csv": {"path": "environment_rows.csv", "sha256": model_sha, "rows": 1080},
            "reference_rows_csv": {"path": "reference_rows.csv", "sha256": ref_sha, "rows": 120},
            "summary_csv": {"path": "summary.csv", "sha256": summary_sha, "rows": 36},
            "paired_rate_csv": {"path": "paired_rate.csv", "sha256": paired_sha, "rows": 6},
            "figure_source_csv": {"path": "figure_source.csv", "sha256": figure_sha, "rows": 72},
        },
        "bootstrap": {
            "generator": "NumPy PCG64",
            "numpy_version": np_version,
            "draws": BOOTSTRAP_DRAWS,
            "seed_base": BOOTSTRAP_SEED_BASE,
            "seed_by_speed": {str(speed): BOOTSTRAP_SEED_BASE + speed for speed in SPEEDS},
            "resampling": "one shared 30-environment index matrix per speed across all six predictors and three history-noise conditions",
            "interval": "pointwise 2.5th/97.5th percentile of bootstrap means, linear quantile interpolation",
            "nmse_intervals": "copied exactly from the SHA-pinned existing 36-point A1 summary; not recomputed",
            "paired_contrasts": "Lite minus LMMSE RZF sum-SE within each environment; same speed-level resamples",
            "tests_or_new_inferential_families": "none",
        },
        "row_audit": {
            "model_rows": len(analysis["model_rows"]),
            "reference_rows": len(analysis["reference_rows"]),
            "environment_receipts": len(analysis["receipt_manifest"]),
            "reference_reuse_pointers": len(analysis["reuse_manifest"]),
            "all_1080_model_nmse_rows_match_frozen_prior": True,
            "all_model_and_reference_prediction_fallback_rows": 0,
            "clean_target_energy_parity": True,
            "fallback_subcarrier_counts_retained_in_environment_and_summary_csv": True,
            "beam_valid_sample_denominator_persisted_by_producer": False,
        },
        "limitations": [
            "The frozen producer does not persist the valid origin-user denominator used for mean beam efficiency/loss. Those means are checked for joint nullness, finiteness, and range; the saved link denominators, fallback counts, and all plotted sum-SE values are checked exactly.",
            "Intervals describe variation across the 30 fixed environments conditional on the recorded model and noise realization; they do not estimate training-seed uncertainty or imply causality.",
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    report_path = output_dir / "figure_source.verification.json"
    temp = report_path.with_suffix(report_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--root-verified-complete", action="store_true",
        help="Root attestation gate; only use after complete-run review.",
    )
    args = parser.parse_args(argv)
    if not args.root_verified_complete:
        parser.error("--root-verified-complete is required after root review; no outputs were written")
    analysis = analyze_completed_run(args.root, args.results_dir)
    report = write_analysis_outputs(
        analysis, args.output_dir, root_verified_complete=True,
    )
    print(json.dumps({
        "status": report["status"],
        "figure_source_sha256": report["source_sha256"],
        "verification_path": str(args.output_dir.resolve() / "figure_source.verification.json"),
        "model_rows": report["row_audit"]["model_rows"],
        "reference_rows": report["row_audit"]["reference_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
