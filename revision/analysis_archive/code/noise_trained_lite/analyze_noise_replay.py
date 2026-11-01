#!/usr/bin/env python
"""Summarize only the complete, approved noise-Lite replay receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ANALYSIS_PROTOCOL = HERE / "protocol.json"
METRICS = ("nmse_db", "rzf_sum_se")
RECIPES = ("clean", "noiseaug")
SPEEDS = (30, 300)
CONDITIONS = ("10", "20", "30", "clean_infinite_snr")
FIGURE_FIELDS = (
    "metric", "training_recipe", "speed_kmh", "history_condition",
    "mean", "ci95_low", "ci95_high", "n_environments",
)
PAIRED_FIELDS = (
    "metric", "contrast", "speed_kmh", "history_condition",
    "mean_difference", "ci95_low", "ci95_high", "n_environments",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def aggregate_records(records: list[dict[str, Any]], expected_environments: dict[int, list[str]]):
    """Build equal-environment means and paired bootstrap CIs from validated rows."""
    import numpy as np

    expected_keys = {
        (speed, env, recipe, condition)
        for speed in SPEEDS for env in expected_environments.get(speed, [])
        for recipe in RECIPES for condition in CONDITIONS
    }
    if (set(expected_environments) != set(SPEEDS)
            or any(len(envs) != 30 or len(set(envs)) != 30 for envs in expected_environments.values())):
        raise ValueError("Expected exactly 30 distinct environments at each fixed speed")
    by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in records:
        try:
            key = (row["speed_kmh"], row["environment_id"],
                   row["training_recipe"], row["history_condition"])
            mat_sha = row["mat_sha256"]
            target_energy = row["target_energy"]
            values = {metric: row[metric] for metric in METRICS}
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Validated replay row is missing an analysis field: {exc}") from exc
        if key not in expected_keys:
            raise ValueError(f"Unexpected replay row key: {key}")
        if key in by_key:
            raise ValueError(f"Duplicate replay row key: {key}")
        if not isinstance(mat_sha, str) or not mat_sha or not _finite_number(target_energy) or float(target_energy) <= 0:
            raise ValueError(f"Missing or invalid physical pairing identity for {key}")
        if not all(_finite_number(value) for value in values.values()):
            raise ValueError(f"Nonfinite saved metric; refusing to filter or floor {key}")
        by_key[key] = {"mat_sha256": mat_sha, "target_energy": float(target_energy), **values}
    if len(by_key) != 480 or set(by_key) != expected_keys:
        raise ValueError(f"Expected exactly 480 complete replay rows; found {len(by_key)}")

    for speed in SPEEDS:
        for env in expected_environments[speed]:
            rows = [by_key[(speed, env, recipe, condition)]
                    for recipe in RECIPES for condition in CONDITIONS]
            if len({row["mat_sha256"] for row in rows}) != 1:
                raise ValueError(f"MAT identity differs within paired environment {(speed, env)}")
            if len({row["target_energy"] for row in rows}) != 1:
                raise ValueError(f"Saved target energy differs within paired environment {(speed, env)}")

    figure_rows, paired_rows, bootstrap = [], [], {}
    for speed in SPEEDS:
        envs = expected_environments[speed]
        seed = 20261030 + speed
        rng = np.random.Generator(np.random.PCG64(seed))
        indices = rng.integers(0, len(envs), size=(10000, len(envs)))
        bootstrap[str(speed)] = {
            "seed": seed,
            "resample_index_sha256": sha256_bytes(indices.tobytes()),
            "replicates": 10000,
            "n_environments": len(envs),
        }
        cells: dict[tuple[str, str, str], np.ndarray] = {}
        for metric in METRICS:
            for recipe in RECIPES:
                for condition in CONDITIONS:
                    values = np.asarray([
                        by_key[(speed, env, recipe, condition)][metric] for env in envs
                    ], dtype=np.float64)
                    cells[(metric, recipe, condition)] = values
                    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975], method="linear")
                    figure_rows.append({
                        "metric": metric, "training_recipe": recipe, "speed_kmh": speed,
                        "history_condition": condition, "mean": float(values.mean()),
                        "ci95_low": float(low), "ci95_high": float(high), "n_environments": 30,
                    })
            for condition in CONDITIONS:
                differences = cells[(metric, "noiseaug", condition)] - cells[(metric, "clean", condition)]
                low, high = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975], method="linear")
                paired_rows.append({
                    "metric": metric, "contrast": "noiseaug_minus_clean", "speed_kmh": speed,
                    "history_condition": condition, "mean_difference": float(differences.mean()),
                    "ci95_low": float(low), "ci95_high": float(high), "n_environments": 30,
                })
    if len(figure_rows) != 32 or len(paired_rows) != 16:
        raise AssertionError("Internal output grid is not 32 summary rows and 16 paired contrasts")
    return figure_rows, paired_rows, bootstrap


def load_validated_replay(root: Path, manifest_path: Path, protocol: dict[str, Any], evaluator):
    """Read result JSON only after the frozen-manifest, smoke, approval and 480-row gates."""
    root = Path(root).resolve(strict=True)
    manifest, manifest_sha = evaluator.read_pretest_manifest(root, protocol, manifest_path)
    approval = evaluator.require_root_approval(root, protocol, manifest, manifest_sha)
    snapshot = manifest["snapshot"]
    bundles = snapshot["runs"]
    environments = {speed: sorted(group["environment_id"]
                                   for group in snapshot["test_groups_by_speed"][str(speed)])
                    for speed in SPEEDS}

    output_root = evaluator._resolve(root, protocol["paths"]["result_directory"]) / manifest_sha
    identity = {
        "schema": "noise-trained-lite-fixed-replay-run-v1",
        "family": "noise_trained_lite", "pretest_manifest_sha256": manifest_sha,
        "evaluation_protocol_sha256": snapshot["evaluation_protocol_sha256"],
        "evaluation_runner_sha256": snapshot["evaluation_runner_sha256"],
        "training_protocol_sha256": snapshot["training_protocol_sha256"],
        "root_approval_sha256": approval["sha256"],
        "channel_scale": snapshot["channel_scale"],
        "receiver_snr_db": 10, "test_based_selection": False,
    }
    identity_path = output_root / "run_identity.json"
    identity_bytes = identity_path.read_bytes()
    if json.loads(identity_bytes.decode("utf-8")) != identity:
        raise ValueError("Replay run_identity.json does not bind this exact approved replay")
    summary_path = output_root / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes.decode("utf-8"))
    evaluator._validate_completed_summary(
        root, summary, identity, snapshot, protocol, output_root, approval=approval,
    )
    bundle_by_run = {bundle["run_id"]: bundle for bundle in bundles}
    records = []
    for item in summary["environment_receipts"]:
        if item["run_id"] not in bundle_by_run:
            raise ValueError("Validated summary contains an unexpected run identity")
        path = evaluator._resolve(root, item["path"])
        path.relative_to(output_root.resolve())
        row_bytes = path.read_bytes()
        if sha256_bytes(row_bytes) != item.get("sha256"):
            raise ValueError(f"Saved row changed after the 480-row validator: {path}")
        saved = json.loads(row_bytes.decode("utf-8"))
        payload = saved.get("payload")
        if saved.get("status") != "complete" or not isinstance(payload, dict):
            raise ValueError(f"Saved row changed after the 480-row validator: {path}")
        metrics = payload["metrics"]
        bundle = bundle_by_run[item["run_id"]]
        condition = str(item["history_condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected history condition in completed row: {condition}")
        link = metrics["rzf_10db"]
        records.append({
            "speed_kmh": bundle["speed_kmh"], "environment_id": item["environment_id"],
            "training_recipe": bundle["history_recipe"], "history_condition": condition,
            "mat_sha256": payload["identity"]["mat_sha256"],
            "target_energy": metrics["target_energy"], "nmse_db": metrics["nmse_db"],
            "rzf_sum_se": link["mean_sum_se_bps_hz"],
        })
    if len(records) != 480:
        raise ValueError("The complete replay row set changed after strict validation")
    frozen_path = Path(manifest_path)
    if not frozen_path.is_absolute():
        frozen_path = root / frozen_path
    provenance = {
        "analysis_completion_gate": "passed_full_480_row_root_approval_and_receipt_validation",
        "pretest_manifest_path": frozen_path.resolve().relative_to(root).as_posix(),
        "pretest_manifest_sha256": manifest_sha,
        "pretest_snapshot_sha256": evaluator.canonical_sha256(snapshot),
        "root_approval": approval,
        "replay_identity_path": identity_path.relative_to(root).as_posix(),
        "replay_identity_sha256": sha256_bytes(identity_bytes),
        "replay_summary_path": summary_path.relative_to(root).as_posix(),
        "replay_summary_sha256": sha256_bytes(summary_bytes),
        "validated_receipt_table_sha256": evaluator.canonical_sha256(summary["environment_receipts"]),
        "replay_summary_rows": len(records),
        "fit_bundles": bundles,
        "pretest_source_sha256": snapshot["source_sha256"],
        "formal_manifest_sha256": snapshot["formal_manifest_sha256"],
        "training_statistics_sha256": snapshot["training_statistics_sha256"],
        "training_protocol_sha256": snapshot["training_protocol_sha256"],
        "training_runner_sha256": snapshot["training_runner_sha256"],
        "evaluation_protocol_sha256": snapshot["evaluation_protocol_sha256"],
        "evaluation_runner_sha256": snapshot["evaluation_runner_sha256"],
        "normalizer_sha256": snapshot["normalizer_sha256"],
    }
    return records, environments, provenance


def _csv_bytes(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (format(value, ".17g") if isinstance(value, float) else value)
                         for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def write_bundle(output_parent: Path, figure_rows: list[dict[str, Any]],
                 paired_rows: list[dict[str, Any]], info: dict[str, Any]) -> Path:
    approval = info.get("root_approval", {})
    if (info.get("analysis_completion_gate") != "passed_full_480_row_root_approval_and_receipt_validation"
            or info.get("replay_summary_rows") != 480
            or not isinstance(info.get("fit_bundles"), list) or len(info["fit_bundles"]) != 4
            or approval.get("status") != "approved_for_formal_test_replay"
            or len(figure_rows) != 32 or len(paired_rows) != 16):
        raise PermissionError("Output requires the complete, root-approved four-fit/480-row validation path")
    output_parent = Path(output_parent)
    name = str(info.get("pretest_manifest_sha256", ""))[:16]
    if len(name) != 16:
        raise ValueError("Output bundle requires a frozen pretest-manifest SHA-256")
    destination = output_parent / name
    output_parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing analysis bundle: {destination}")
    figure_bytes = _csv_bytes(figure_rows, FIGURE_FIELDS)
    paired_bytes = _csv_bytes(paired_rows, PAIRED_FIELDS)
    verification = {
        **info,
        "schema": "noise-trained-lite-analysis-verification-v1",
        "status": "root_verified_complete",
        "source_file": "figure_source.csv",
        "source_sha256": sha256_bytes(figure_bytes),
        "figure_source_rows": len(figure_rows),
        "paired_contrasts_file": "paired_contrasts.csv",
        "paired_contrasts_sha256": sha256_bytes(paired_bytes),
        "paired_contrast_rows": len(paired_rows),
        "replay_summary_rows": 480,
        "artifact_sha256s": {
            "figure_source.csv": sha256_bytes(figure_bytes),
            "paired_contrasts.csv": sha256_bytes(paired_bytes),
        },
        "environment_unit": "independent formal test environment",
        "fit_seed": 42,
        "history_noise_test_seed": 2602236,
        "test_mat_bytes_opened_by_analysis": False,
        "test_based_selection_or_tuning": False,
    }
    stage = Path(tempfile.mkdtemp(prefix=".analysis-", dir=output_parent))
    (stage / "figure_source.csv").write_bytes(figure_bytes)
    (stage / "paired_contrasts.csv").write_bytes(paired_bytes)
    (stage / "verification.json").write_bytes(json_bytes(verification))
    os.replace(stage, destination)
    return destination


def _load_analysis_protocol() -> dict[str, Any]:
    if sha256(ANALYSIS_PROTOCOL) != "5f9ccda32fbe8bae65bb027777b330525d6a31ca8bcc0a34a0e29831ec83a7ae":
        raise ValueError("Pinned analysis protocol bytes changed")
    value = json.loads(ANALYSIS_PROTOCOL.read_text(encoding="utf-8"))
    if (value.get("schema") != "noise-trained-lite-complete-analysis-v1"
            or value.get("status") != "source_prepared_waiting_for_complete_approved_replay"):
        raise ValueError("Analysis protocol is not the expected source-only preparation")
    return value


def _load_evaluator(root: Path, config: dict[str, Any]):
    source = root / config["evaluation_source"]["runner_path"]
    protocol_path = root / config["evaluation_source"]["protocol_path"]
    if (sha256(source) != config["evaluation_source"]["runner_sha256"]
            or sha256(protocol_path) != config["evaluation_source"]["protocol_sha256"]):
        raise ValueError("Pinned fixed-replay evaluator source or protocol SHA changed")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from revision.paper.remaining_fixes_20261029.noise_trained_lite_v1.evaluation_v1 import runner as module
    if Path(module.__file__).resolve() != source.resolve():
        raise RuntimeError("Imported evaluator path differs from the pinned source")
    protocol = module.json_read(module.PROTOCOL_PATH)
    module.validate_protocol(protocol)
    return module, protocol


def run_analysis(root: Path, manifest_path: Path) -> Path:
    import numpy as np

    root = Path(root).resolve(strict=True)
    config = _load_analysis_protocol()
    evaluator, protocol = _load_evaluator(root, config)
    records, environments, provenance = load_validated_replay(
        root, manifest_path, protocol, evaluator,
    )
    figure_rows, paired_rows, bootstrap = aggregate_records(records, environments)
    provenance.update({
        "analysis_protocol_sha256": sha256(ANALYSIS_PROTOCOL),
        "analysis_runner_sha256": sha256(Path(__file__)),
        "analysis_protocol_id": config["schema"],
        "bootstrap": bootstrap,
        "bootstrap_generator": "numpy.random.PCG64",
        "bootstrap_replicates": 10000,
        "bootstrap_base_seed": 20261030,
        "bootstrap_seed_rule": "base_seed + speed_kmh",
        "bootstrap_quantile_method": "linear",
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "ci_method": "pointwise percentile 95% intervals over equal-environment means",
        "paired_contrast": "noiseaug minus clean, paired by speed/environment/history condition",
        "replay_is_posthoc": True,
    })
    return write_bundle(HERE / "results", figure_rows, paired_rows, provenance)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Exact frozen pretest manifest path; never auto-select a manifest")
    parser.add_argument("--execute-approved-analysis", action="store_true",
                        help="Read saved approved replay rows after every completion gate passes")
    args = parser.parse_args(argv)
    if not args.execute_approved_analysis:
        parser.error("--execute-approved-analysis is required; this entry point reads saved formal replay results")
    print(run_analysis(args.root, args.manifest))


if __name__ == "__main__":
    main()
