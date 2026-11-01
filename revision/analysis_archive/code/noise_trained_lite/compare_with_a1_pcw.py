#!/usr/bin/env python
"""Prepare a strict paired comparison of fresh noise-Lite fits with A1 PC-Wiener."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROTOCOL_PATH = HERE / "protocol.json"
PROTOCOL_SHA256 = "6d552ed681b7cd6e03746ede4f9a01b5b6b0ab8b1a0c3380e4928e412e773f08"
SPEEDS = (30, 300)
HISTORY_SNRS = (10, 20, 30)
NOISE_CONDITIONS = ("clean_infinite_snr", "10", "20", "30")
RECIPES = ("clean", "noiseaug")
METRICS = ("nmse_db", "rzf_sum_se")
A1_METHODS = ("Lite", "LMMSE", "Ridge", "AR", "Kalman", "Persistence")
OUTPUT_FIELDS = (
    "metric", "training_recipe", "speed_kmh", "history_snr_db", "contrast",
    "mean_difference", "ci95_low", "ci95_high", "n_environments",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(data)


def load_protocol() -> dict[str, Any]:
    raw = PROTOCOL_PATH.read_bytes()
    if sha256_bytes(raw) != PROTOCOL_SHA256:
        raise ValueError("Cross-comparison protocol bytes differ from the reviewed pin")
    protocol = json.loads(raw.decode("utf-8"))
    if (not isinstance(protocol, dict)
            or protocol.get("schema") != "noise-lite-vs-a1-pcw-paired-analysis-v1"
            or protocol.get("status") != "source_prepared_waiting_for_root_review_and_complete_a1_v2"):
        raise ValueError("Unexpected cross-comparison protocol identity/status")
    if protocol.get("grid", {}).get("paired_contrasts") != 24:
        raise ValueError("The reviewed source must retain the complete 24-contrast grid")
    return protocol


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import strict completed-result reader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_pinned_sources(root: Path, protocol: dict[str, Any]) -> None:
    inputs = protocol["inputs"]
    bindings = (
        (inputs["noise_lite"]["analysis_reader_path"], inputs["noise_lite"]["analysis_reader_sha256"]),
        (inputs["noise_lite"]["analysis_protocol_path"], inputs["noise_lite"]["analysis_protocol_sha256"]),
        (inputs["noise_lite"]["evaluation_runner_path"], inputs["noise_lite"]["evaluation_runner_sha256"]),
        (inputs["noise_lite"]["evaluation_protocol_path"], inputs["noise_lite"]["evaluation_protocol_sha256"]),
        (inputs["a1_pcw"]["analysis_reader_path"], inputs["a1_pcw"]["analysis_reader_sha256"]),
        (inputs["a1_pcw"]["runner_path"], inputs["a1_pcw"]["runner_sha256"]),
        (inputs["a1_pcw"]["protocol_path"], inputs["a1_pcw"]["protocol_sha256"]),
        (inputs["shared_noise_sources"]["formal_noise_path"], inputs["shared_noise_sources"]["formal_noise_sha256"]),
        (inputs["shared_noise_sources"]["formal_observation_noise_path"], inputs["shared_noise_sources"]["formal_observation_noise_sha256"]),
        ("revision/code/runner/formal_evaluate.py", inputs["shared_noise_sources"]["formal_evaluate_sha256"]),
        ("revision/code/runner/formal_classical_evaluate.py", inputs["shared_noise_sources"]["formal_classical_evaluate_sha256"]),
    )
    for relative, expected in bindings:
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Reviewed source SHA changed: {relative}")


def read_a1_only_after_complete(reader, root: Path, results_dir: Path):
    """The complete marker gate runs before the strict reader can open any row."""
    _marker, before_sha = reader.read_complete_marker(results_dir)
    analysis = reader.analyze_completed_run(root, results_dir)
    _marker_after, after_sha = reader.read_complete_marker(results_dir)
    if (before_sha != after_sha
            or analysis.get("complete_sha256") != before_sha):
        raise ValueError("A1 complete marker changed while its strict reader was running")
    return analysis, before_sha


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; rows are never filtered")
    return result


def validate_scientific_contract(
    noise_protocol: dict[str, Any], noise_snapshot: dict[str, Any],
    a1_protocol: dict[str, Any], a1_run_identity: dict[str, Any],
    fit_bindings: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Check geometry, seeds, receiver SNR, global g and normalizer before pairing."""
    data = noise_protocol.get("formal_data", {})
    evaluation = noise_protocol.get("evaluation", {})
    noise = a1_protocol.get("noise", {})
    panel = a1_protocol.get("panel", {})
    expected = {
        "speeds_kmh": [30, 300], "environments_per_speed": 30,
        "expected_time_samples": 4001, "users": 4, "history_length": 10,
        "origin_start": 19, "horizon_steps": 4, "horizon_ms": 2.0,
        "stride": 1, "expected_rows_per_environment": 15912,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError("Noise-Lite replay geometry differs from the paired comparison protocol")
    stream = evaluation.get("streaming", {})
    if (stream.get("expected_rows") != 480 or stream.get("row_count_per_receipt") != 15912
            or evaluation.get("conditions") != ["clean_infinite_snr", 10, 20, 30]
            or evaluation.get("history_noise", {}).get("seed") != 2602236
            or evaluation.get("link", {}).get("receiver_snr_db") != 10):
        raise ValueError("Noise-Lite complete-reader protocol differs from the fixed test grid")
    a1_expected = {
        "speeds_kmh": [30, 300], "test_environments_per_speed": 30,
        "history_length": 10, "origin_start": 19, "horizon_steps": 4,
        "horizon_ms": 2.0, "stride": 1, "users": 4,
    }
    if any(panel.get(key) != value for key, value in a1_expected.items()):
        raise ValueError("A1 completed-reader panel geometry differs from the paired comparison")
    if (noise.get("seed") != 2602236 or noise.get("history_snr_db") != [10, 20, 30]
            or a1_run_identity.get("noise_seed") != 2602236
            or a1_run_identity.get("history_noise_snr_db") != [10, 20, 30]
            or a1_run_identity.get("receiver_snr_db") != 10):
        raise ValueError("A1 seed/history/receiver-SNR identity differs from the paired grid")
    a1_link_text = str(a1_protocol.get("link", {}).get("evaluation", ""))
    if "receiver nominal SNR 10 dB" not in a1_link_text:
        raise ValueError("A1 protocol does not state receiver SNR 10 dB")
    formal_sha = noise_snapshot.get("formal_manifest_sha256")
    if (not _is_sha256(formal_sha)
            or a1_protocol.get("formal_manifest_sha256") != formal_sha
            or a1_run_identity.get("formal_manifest_sha256") != formal_sha):
        raise ValueError("A1 and noise-Lite do not bind the same formal manifest SHA")
    normalizer_sha = noise_snapshot.get("normalizer_sha256")
    global_scale = noise_snapshot.get("channel_scale")
    if not _is_sha256(normalizer_sha) or not math.isfinite(float(global_scale)) or float(global_scale) <= 0:
        raise ValueError("Noise-Lite global channel scale or normalizer binding is malformed")
    scale_by_speed: dict[str, float] = {}
    for speed in SPEEDS:
        fit = fit_bindings.get(speed)
        if (not isinstance(fit, dict)
                or fit.get("channel_scale") != global_scale
                or fit.get("normalizer_sha256") != normalizer_sha):
            raise ValueError(f"A1 channel_scale/normalizer differs from noise-Lite at {speed} km/h")
        scale_by_speed[str(speed)] = float(global_scale)
    return {
        "formal_manifest_sha256": formal_sha,
        "normalizer_sha256": normalizer_sha,
        "global_channel_scale_by_speed": scale_by_speed,
        "history_noise_seed": 2602236,
        "receiver_snr_db": 10,
        "expected_rows_per_environment": 15912,
        "same_noisy_history_parameters_verified": True,
    }


def posthoc_disclosures(noise_analysis_protocol, noise_protocol, a1_protocol):
    noise_is_posthoc = noise_analysis_protocol.get("access_boundary", {}).get(
        "replay_results_are_posthoc",
    )
    noise_text = noise_protocol.get("evaluation", {}).get("claim_boundary", {}).get(
        "access_disclosure",
    )
    a1_text = a1_protocol.get("formal_test_gate", {}).get("test_access_disclosure")
    if (noise_is_posthoc is not True or not isinstance(noise_text, str) or not noise_text.strip()
            or not isinstance(a1_text, str) or not a1_text.strip()):
        raise ValueError("Both fixed replay protocols must retain their posthoc access disclosures")
    return {
        "noise_lite_replay_is_posthoc": True,
        "noise_lite_access_disclosure": noise_text,
        "a1_disclosure": a1_text,
    }


def validate_input_paths(root: Path, noise_manifest_path: Path, a1_results_dir: Path,
                         protocol: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(root).resolve(strict=True)

    def resolved(value):
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve(strict=True)

    expected_manifest = resolved(protocol["inputs"]["noise_lite"]["pretest_manifest_path"])
    expected_a1_results = resolved(protocol["inputs"]["a1_pcw"]["results_directory"])
    actual_manifest = resolved(noise_manifest_path)
    actual_a1_results = resolved(a1_results_dir)
    if actual_manifest != expected_manifest:
        raise ValueError("Noise-Lite manifest path must match the protocol-pinned path")
    if actual_a1_results != expected_a1_results:
        raise ValueError("A1 results directory must match the protocol-pinned path")
    return actual_manifest, actual_a1_results


def _environment_group_map(groups_by_speed: dict[str, Any], environments: dict[int, list[str]]):
    expected_speeds = {str(speed) for speed in SPEEDS}
    if not isinstance(groups_by_speed, dict) or set(groups_by_speed) != expected_speeds:
        raise ValueError("Noise-Lite pretest manifest must bind exactly the two speeds")
    result = {}
    for speed in SPEEDS:
        rows = groups_by_speed[str(speed)]
        if not isinstance(rows, list) or len(rows) != 30:
            raise ValueError(f"Noise-Lite test environment list is incomplete at {speed} km/h")
        seen = set()
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError("Malformed noise-Lite environment identity")
            env = item.get("environment_id")
            if env in seen or env not in environments[speed]:
                raise ValueError(f"Unexpected or duplicate noise-Lite environment: {env}")
            seen.add(env)
            identity = {key: item.get(key) for key in ("mat_sha256", "job_sha256", "qa_receipt_sha256")}
            if not all(_is_sha256(value) for value in identity.values()):
                raise ValueError(f"Noise-Lite environment {env} lacks full MAT/job/QA SHA identity")
            result[(speed, env)] = identity
        if seen != set(environments[speed]):
            raise ValueError(f"Noise-Lite environment IDs differ from strict reader at {speed} km/h")
    return result


def _validate_a1_rows(a1_rows, group_by_env, environments, *, expected_rows):
    expected_keys = {
        (speed, snr, method, env)
        for speed in SPEEDS for snr in HISTORY_SNRS for method in A1_METHODS
        for env in environments[speed]
    }
    if not isinstance(a1_rows, list) or len(a1_rows) != 1080:
        raise ValueError("A1 strict reader must return exactly 1080 model rows")
    by_key = {}
    for row in a1_rows:
        try:
            key = (int(row["speed_kmh"]), int(row["nominal_snr_db"]),
                   row["method"], row["environment_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A1 model row lacks fixed comparison identity") from exc
        if key not in expected_keys or key in by_key:
            raise ValueError(f"A1 model-row grid has an unexpected or duplicate key: {key}")
        speed, _snr, _method, env = key
        group = group_by_env[(speed, env)]
        for field, value in group.items():
            if row.get(field) != value:
                raise ValueError(f"A1 {field} differs from noise-Lite for {(speed, env)}")
        if (row.get("expected_rows_per_method") != expected_rows
                or row.get("n_prediction_rows") != expected_rows):
            raise ValueError(f"A1 full-row count differs for {key}")
        by_key[key] = row
    if set(by_key) != expected_keys:
        raise ValueError(f"A1 strict-reader model grid is missing {len(expected_keys - set(by_key))} rows")
    return by_key


def _validate_noise_rows(noise_records, group_by_env, environments):
    expected_keys = {
        (speed, env, recipe, condition)
        for speed in SPEEDS for env in environments[speed]
        for recipe in RECIPES for condition in NOISE_CONDITIONS
    }
    if not isinstance(noise_records, list) or len(noise_records) != 480:
        raise ValueError("Noise-Lite strict reader must return exactly 480 complete rows")
    by_key = {}
    targets_by_env = {}
    for row in noise_records:
        try:
            key = (int(row["speed_kmh"]), row["environment_id"],
                   row["training_recipe"], str(row["history_condition"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Noise-Lite row lacks fixed comparison identity") from exc
        if key not in expected_keys or key in by_key:
            raise ValueError(f"Noise-Lite row grid has an unexpected or duplicate key: {key}")
        speed, env, _recipe, _condition = key
        group = group_by_env[(speed, env)]
        if row.get("mat_sha256") != group["mat_sha256"]:
            raise ValueError(f"Noise-Lite MAT identity differs from its manifest: {(speed, env)}")
        target = _finite(row.get("target_energy"), f"target_energy {key}")
        if target <= 0:
            raise ValueError(f"Noise-Lite target energy must be positive: {key}")
        values = {metric: _finite(row.get(metric), f"{metric} {key}") for metric in METRICS}
        if values["rzf_sum_se"] < 0:
            raise ValueError(f"Noise-Lite RZF sum-SE must be nonnegative: {key}")
        target_key = (speed, env)
        prior = targets_by_env.setdefault(target_key, target)
        if target != prior:
            raise ValueError(f"Noise-Lite clean target energy differs within {(speed, env)}")
        by_key[key] = {"target_energy": target, **values}
    if set(by_key) != expected_keys:
        raise ValueError(f"Noise-Lite strict-reader grid is missing {len(expected_keys - set(by_key))} rows")
    return by_key


def _shared_bootstrap_indices(speed: int, n_environments: int = 30):
    import numpy as np

    seed = 20261030 + speed
    indices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, n_environments, size=(10000, n_environments), dtype=np.int64,
    )
    return indices, {
        "seed": seed,
        "draws": 10000,
        "n_environments": n_environments,
        "resample_index_sha256": sha256_bytes(indices.tobytes()),
    }


def build_paired_contrasts(
    noise_records, noise_groups_by_speed, a1_model_rows, environments, *, expected_rows: int,
):
    """Build all prespecified per-environment Lite-minus-PCW contrasts."""
    import numpy as np

    if set(environments) != set(SPEEDS) or any(
            len(environments[speed]) != 30 or environments[speed] != sorted(set(environments[speed]))
            for speed in SPEEDS):
        raise ValueError("Expected 30 sorted distinct test environments at each speed")
    group_by_env = _environment_group_map(noise_groups_by_speed, environments)
    a1_by_key = _validate_a1_rows(a1_model_rows, group_by_env, environments,
                                  expected_rows=expected_rows)
    noise_by_key = _validate_noise_rows(noise_records, group_by_env, environments)
    pcw_by_key = {
        (speed, snr, env): row
        for (speed, snr, method, env), row in a1_by_key.items()
        if method == "LMMSE"
    }
    if len(pcw_by_key) != 180:
        raise ValueError("A1 PC-Wiener (LMMSE) rows must cover exactly 180 environment-condition keys")

    for speed in SPEEDS:
        for env in environments[speed]:
            group = group_by_env[(speed, env)]
            noise_targets = [noise_by_key[(speed, env, recipe, condition)]["target_energy"]
                             for recipe in RECIPES for condition in NOISE_CONDITIONS]
            for snr in HISTORY_SNRS:
                pcw = pcw_by_key[(speed, snr, env)]
                pcw_target = _finite(pcw.get("target_energy"), f"PC-Wiener target_energy {(speed, snr, env)}")
                if (not math.isclose(pcw_target, noise_targets[0], rel_tol=1e-12, abs_tol=1e-18)
                        or any(target != noise_targets[0] for target in noise_targets[1:])):
                    raise ValueError(f"Target energy differs across paired rows: {(speed, env, snr)}")
                if pcw.get("n_prediction_rows") != expected_rows:
                    raise ValueError(f"PC-Wiener full-row count differs: {(speed, env, snr)}")
                if pcw.get("mat_sha256") != group["mat_sha256"]:
                    raise ValueError(f"PC-Wiener MAT SHA differs from paired noise-Lite: {(speed, env, snr)}")

    rows = []
    bootstrap = {}
    for speed in SPEEDS:
        envs = environments[speed]
        indices, bootstrap[str(speed)] = _shared_bootstrap_indices(speed, len(envs))
        for metric in METRICS:
            a1_metric = "nmse_db" if metric == "nmse_db" else "rzf_sum_se_bps_hz"
            for recipe in RECIPES:
                for snr in HISTORY_SNRS:
                    differences = np.asarray([
                        noise_by_key[(speed, env, recipe, str(snr))][metric]
                        - _finite(pcw_by_key[(speed, snr, env)].get(a1_metric),
                                  f"PC-Wiener {a1_metric} {(speed, snr, env)}")
                        for env in envs
                    ], dtype=np.float64)
                    if not np.isfinite(differences).all():
                        raise ValueError("Paired difference is nonfinite; no rows may be dropped")
                    draw_means = differences[indices].mean(axis=1, dtype=np.float64)
                    low, high = np.quantile(draw_means, [0.025, 0.975], method="linear")
                    rows.append({
                        "metric": metric,
                        "training_recipe": recipe,
                        "speed_kmh": speed,
                        "history_snr_db": snr,
                        "contrast": "fresh_lite_minus_a1_pcw",
                        "mean_difference": float(differences.mean(dtype=np.float64)),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "n_environments": len(envs),
                    })
    expected_keys = {
        (metric, recipe, speed, snr)
        for metric in METRICS for recipe in RECIPES
        for speed in SPEEDS for snr in HISTORY_SNRS
    }
    keys = {(row["metric"], row["training_recipe"], row["speed_kmh"], row["history_snr_db"])
            for row in rows}
    if len(rows) != 24 or keys != expected_keys:
        raise AssertionError("Paired output must retain the full prespecified 24-contrast grid")
    return rows, {
        "noise_lite_rows": len(noise_by_key),
        "a1_model_rows": len(a1_by_key),
        "a1_pcw_rows": len(pcw_by_key),
        "paired_contrasts": len(rows),
        "environment_pairs": len(rows) * 30,
        "bootstrap_by_speed": bootstrap,
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _atomic_bundle(output_parent: Path, run_name: str, rows, verification):
    output_dir = output_parent / run_name
    staging = output_parent / f".{run_name}.staging"
    if output_dir.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite a comparison output/staging directory: {output_dir}")
    staging.mkdir(parents=True, exist_ok=False)
    csv_data = _csv_bytes(rows)
    csv_path = staging / "paired_contrasts.csv"
    csv_path.write_bytes(csv_data)
    verification["outputs"] = {
        "paired_contrasts_csv": {
            "path": "paired_contrasts.csv", "sha256": sha256_bytes(csv_data), "rows": len(rows),
        },
    }
    verification_data = (json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    (staging / "verification.json").write_bytes(verification_data)
    os.replace(staging, output_dir)
    return output_dir


def run_comparison(
    root: Path, *, noise_manifest_path: Path, a1_results_dir: Path,
    root_verified_complete: bool,
) -> Path:
    if not root_verified_complete:
        raise PermissionError("Pass --root-verified-complete only after root reviews both complete source chains")
    root = Path(root).resolve(strict=True)
    protocol = load_protocol()
    manifest_path, a1_results_dir = validate_input_paths(
        root, noise_manifest_path, a1_results_dir, protocol,
    )
    validate_pinned_sources(root, protocol)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    a1_config = protocol["inputs"]["a1_pcw"]
    a1_reader = _load_module(root / a1_config["analysis_reader_path"], "a1_complete_comparison_reader")
    a1_analysis, a1_complete_sha = read_a1_only_after_complete(a1_reader, root, a1_results_dir)

    noise_config = protocol["inputs"]["noise_lite"]
    noise_reader = _load_module(root / noise_config["analysis_reader_path"], "noise_lite_complete_comparison_reader")
    noise_analysis_protocol = noise_reader._load_analysis_protocol()
    evaluator, noise_protocol = noise_reader._load_evaluator(root, noise_analysis_protocol)
    manifest, manifest_sha = evaluator.read_pretest_manifest(root, noise_protocol, manifest_path)
    noise_records, environments, noise_provenance = noise_reader.load_validated_replay(
        root, manifest_path, noise_protocol, evaluator,
    )
    if manifest_sha != noise_provenance.get("pretest_manifest_sha256"):
        raise ValueError("Noise-Lite validated reader and manifest lookup returned different identities")
    a1_complete_after, a1_complete_sha_after = a1_reader.read_complete_marker(a1_results_dir)
    if a1_complete_sha_after != a1_complete_sha or a1_analysis.get("complete") != a1_complete_after:
        raise ValueError("A1 complete marker changed during cross-result validation")

    a1_protocol = a1_analysis["protocol"]
    scientific_contract = validate_scientific_contract(
        noise_protocol, manifest["snapshot"], a1_protocol,
        a1_analysis["run_identity"], a1_analysis["fit_bindings"],
    )
    disclosures = posthoc_disclosures(noise_analysis_protocol, noise_protocol, a1_protocol)
    groups = manifest["snapshot"].get("test_groups_by_speed")
    rows, pairing_audit = build_paired_contrasts(
        noise_records, groups, a1_analysis["model_rows"], environments,
        expected_rows=scientific_contract["expected_rows_per_environment"],
    )

    run_identity_path = a1_results_dir / "run_identity.json"
    verification = {
        "schema": "noise-lite-vs-a1-pcw-paired-verification-v1",
        "status": "root_verified_complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "root_attestation": "--root-verified-complete",
        "noise_lite": {
            "analysis_reader_sha256": sha256_file(root / noise_config["analysis_reader_path"]),
            "evaluation_protocol_sha256": noise_provenance["evaluation_protocol_sha256"],
            "evaluation_runner_sha256": noise_provenance["evaluation_runner_sha256"],
            "pretest_manifest_sha256": noise_provenance["pretest_manifest_sha256"],
            "replay_identity_sha256": noise_provenance["replay_identity_sha256"],
            "replay_summary_sha256": noise_provenance["replay_summary_sha256"],
            "validated_receipt_table_sha256": noise_provenance["validated_receipt_table_sha256"],
            "rows": noise_provenance["replay_summary_rows"],
            "root_approval": noise_provenance["root_approval"],
        },
        "a1_pcw": {
            "analysis_reader_sha256": sha256_file(root / a1_config["analysis_reader_path"]),
            "runner_sha256": a1_analysis["runner_sha256"],
            "protocol_sha256": a1_analysis["protocol_sha256"],
            "complete_json_sha256": a1_complete_sha,
            "run_identity_sha256": sha256_file(run_identity_path),
            "receipt_manifest_sha256": canonical_sha256(a1_analysis["receipt_manifest"]),
            "model_rows": len(a1_analysis["model_rows"]),
            "method": "LMMSE",
            "method_label": "PC-Wiener",
        },
        "pairing": {
            **scientific_contract,
            **pairing_audit,
            "required_identity_fields": protocol["pairing_contract"]["alignment_fields"],
            "target_energy_match": protocol["pairing_contract"]["target_energy_match"],
            "same_noisy_histories": True,
            "same_noisy_histories_basis": protocol["noise_identity_audit"]["same_observation_condition"],
        },
        "posthoc_disclosure": {
            **disclosures,
            "not_blind": True,
            "test_based_selection": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
        "outputs": {},
    }
    output_parent = HERE / "results"
    output_parent.mkdir(parents=True, exist_ok=True)
    output_name = f"{manifest_sha[:16]}_{a1_complete_sha[:16]}"
    return _atomic_bundle(output_parent, output_name, rows, verification)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--noise-manifest", type=Path,
                        default=ROOT / "revision/paper/remaining_fixes_20261029/noise_trained_lite_v1/evaluation_v1/manifests/PRETEST_noise_trained_lite_c8e2a156fb51a7ac.json")
    parser.add_argument("--a1-results-dir", type=Path,
                        default=ROOT / "revision/paper/remaining_fixes_20261029/a1_noisy_link_v1/results_v2")
    parser.add_argument("--root-verified-complete", action="store_true",
                        help="Root attestation after source review and both complete-result reviews")
    args = parser.parse_args(argv)
    output = run_comparison(
        args.root, noise_manifest_path=args.noise_manifest,
        a1_results_dir=args.a1_results_dir,
        root_verified_complete=args.root_verified_complete,
    )
    print(json.dumps({"status": "root_verified_complete", "output_directory": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
