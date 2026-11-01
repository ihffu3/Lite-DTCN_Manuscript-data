"""Read-only strict audit and descriptive aggregation for the frozen iTransformer replay."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr


FAMILY = "itransformer"
REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "revision/paper/remaining_fixes_20261029"
RUNNER_PATH = BASE / "fixed_checkpoint_replay_v1/runner.py"
MANIFEST_PATH = BASE / "fixed_checkpoint_replay_v1/manifests/PRETEST_itransformer_b651f4140b24bc5f.json"
OUTPUT_DIR = BASE / "fixed_checkpoint_replay_v1/results/itransformer/6cef454656a827cebac900cbfb1628b428cc676ab5b66d8c393b58b95a1f1acd"
DEST = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, content: str) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(path)


def close(a: float, b: float, *, rel: float = 1e-10, abs_: float = 1e-14) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_fixed_checkpoint_replay_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the pinned replay runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = runner.ROOT
    if Path(root).resolve() != REPO.resolve():
        raise AssertionError(f"Frozen runner root differs from repository: {root}")

    protocol = runner.read_json(runner.PROTOCOL_PATH)
    protocol_sha = runner.sha256(runner.PROTOCOL_PATH)
    runner_sha = runner.sha256(RUNNER_PATH)
    manifest, manifest_sha = runner._read_pretest_manifest(root, protocol, FAMILY, MANIFEST_PATH)
    if manifest_sha != OUTPUT_DIR.name:
        raise AssertionError("Result directory key does not equal the supplied pretest manifest SHA")
    snapshot = manifest["snapshot"]
    if snapshot["source_sha256"]["replay_runner"]["sha256"] != runner_sha:
        raise AssertionError("Current replay runner is not the frozen source bound by the manifest")
    approval = runner.require_root_approval(root, protocol, FAMILY, manifest_sha)
    identity = runner.make_run_identity(FAMILY, manifest_sha, snapshot)
    if runner.read_json(OUTPUT_DIR / "run_identity.json") != identity:
        raise AssertionError("Stored run identity differs from the fixed manifest")

    training_protocol_path = runner._resolve(root, protocol["families"][FAMILY]["training_protocol"])
    training_protocol = runner.read_json(training_protocol_path)
    run_specs = runner._training_run_specs(FAMILY, training_protocol)
    groups_by_speed = {int(k): v for k, v in snapshot["test_groups_by_speed"].items()}
    baseline_maps, baseline_evidence = runner._load_baseline_maps(root, FAMILY, groups_by_speed)

    suppressed_reader_output = io.StringIO()
    with redirect_stdout(suppressed_reader_output), redirect_stderr(suppressed_reader_output):
        results, existing_keys = runner._read_resumable_rows(
            OUTPUT_DIR, FAMILY, run_specs, snapshot, manifest_sha, baseline_maps
        )
    complete = runner._validate_completed_run(
        OUTPUT_DIR, identity, snapshot, FAMILY, manifest_sha, results, run_specs,
        protocol, approval,
    )
    if complete is not True:
        raise AssertionError("Frozen runner did not validate the stored run as complete")

    expected_keys = {
        (run["run_id"], env["environment_id"])
        for run in run_specs
        for env in groups_by_speed[run["speed_kmh"]]
    }
    actual_keys = {(row["run_id"], row["environment_id"]) for row in results}
    if len(results) != 120 or len(existing_keys) != 120 or actual_keys != expected_keys:
        raise AssertionError("Formal result does not contain exactly the fixed 4 x 30 rows")

    # Compare the serialized CSV at both whole-file and field-by-field levels.
    expected_csv_text = runner._build_csv_text(results, FAMILY)
    csv_path = OUTPUT_DIR / "per_environment.csv"
    actual_csv_bytes = csv_path.read_bytes()
    if actual_csv_bytes != expected_csv_text.encode("utf-8"):
        raise AssertionError("per_environment.csv differs from all validated JSON rows")
    actual_csv_rows = list(csv.DictReader(io.StringIO(actual_csv_bytes.decode("utf-8"))))
    expected_csv_rows = list(csv.DictReader(io.StringIO(expected_csv_text)))
    if len(actual_csv_rows) != 120 or len(expected_csv_rows) != 120:
        raise AssertionError("CSV row count is not 120")
    actual_by_key = {(row["run_id"], row["environment_id"]): row for row in actual_csv_rows}
    expected_by_key = {(row["run_id"], row["environment_id"]): row for row in expected_csv_rows}
    if len(actual_by_key) != 120 or actual_by_key != expected_by_key:
        raise AssertionError("CSV field values differ from their environment JSON rows")

    contract = snapshot["evaluation_contract"]
    expected_origins = len(range(
        contract["origin_start"],
        contract["expected_time_samples"] - contract["horizon_steps"],
        contract["stride"],
    ))
    if expected_origins != 3978:
        raise AssertionError(f"Unexpected fixed origin count: {expected_origins}")
    rows_by_key = {(row["run_id"], row["environment_id"]): row for row in results}
    for row in results:
        metrics = row["metrics"]
        if (metrics["n_origins"] != expected_origins
                or metrics["n_rows"] != expected_origins * contract["n_users"]
                or metrics["horizon_steps"] != 4
                or metrics["stride"] != 1
                or not close(metrics["horizon_ms"], 2.0, rel=0, abs_=1e-12)):
            raise AssertionError(f"Fixed test geometry mismatch: {row['run_id']}/{row['environment_id']}")
        if not close(metrics["nmse"], metrics["error_energy"] / metrics["target_energy"]):
            raise AssertionError(f"Energy/NMSE ratio mismatch: {row['run_id']}/{row['environment_id']}")
        if not close(metrics["nmse_db"], 10.0 * math.log10(metrics["nmse"]), abs_=1e-10):
            raise AssertionError(f"NMSE-dB mismatch: {row['run_id']}/{row['environment_id']}")
        if metrics["link"]["scheme"] != "RZF" or not close(metrics["link"]["snr_db"], 10.0, rel=0, abs_=1e-12):
            raise AssertionError(f"Link metric contract mismatch: {row['run_id']}/{row['environment_id']}")

    specs_by_speed_condition = {(s["speed_kmh"], s["condition"]): s for s in run_specs}
    paired_target_checks = 0
    for speed in sorted(groups_by_speed):
        c0 = specs_by_speed_condition[(speed, "C0")]
        c1 = specs_by_speed_condition[(speed, "C1")]
        for env in groups_by_speed[speed]:
            env_id = env["environment_id"]
            a = rows_by_key[(c0["run_id"], env_id)]
            b = rows_by_key[(c1["run_id"], env_id)]
            if a["metrics"]["target_energy"] != b["metrics"]["target_energy"]:
                raise AssertionError(f"C0/C1 target energy differs for {speed}/{env_id}")
            if a["identity"] != b["identity"]:
                raise AssertionError(f"C0/C1 test identity differs for {speed}/{env_id}")
            paired_target_checks += 1

    # Validate pinned historical raw rows and their exact identity/target alignment.
    baseline_join_checks = 0
    baseline_target_energy_comparisons = 0
    baseline_target_energy_exact_matches = 0
    baseline_target_energy_max_abs_delta = 0.0
    baseline_target_energy_max_relative_delta = 0.0
    baseline_raw_rows = {}
    for speed_text, refs in protocol["families"][FAMILY]["baseline_replays"].items():
        speed = int(speed_text)
        expected_ids_at_speed = {item["environment_id"] for item in groups_by_speed[speed]}
        for ref in refs:
            path = runner._resolve(root, ref["path"])
            if runner.sha256(path) != ref["sha256"]:
                raise AssertionError(f"Pinned baseline SHA mismatch: {ref['label']} at {speed}")
            baseline_data = runner.read_json(path)
            runner.validate_historical_baseline_metadata(baseline_data, speed, ref["label"])
            by_env = {row["environment_id"]: row for row in baseline_data["environment_metrics"]}
            if len(by_env) != 30 or set(by_env) != expected_ids_at_speed:
                raise AssertionError(f"Baseline ID coverage mismatch: {ref['label']} at {speed}")
            for env in groups_by_speed[speed]:
                env_id = env["environment_id"]
                base_row = by_env[env_id]
                runner.validate_historical_baseline_row(base_row, env, contract)
                if not close(base_row["nmse"], base_row["error_energy"] / base_row["target_energy"]):
                    raise AssertionError(f"Baseline energy/NMSE mismatch: {ref['label']}/{env_id}")
                if not close(base_row["nmse_db"], 10.0 * math.log10(base_row["nmse"]), abs_=1e-10):
                    raise AssertionError(f"Baseline NMSE-dB mismatch: {ref['label']}/{env_id}")
                for condition in ("C0", "C1"):
                    run = specs_by_speed_condition[(speed, condition)]
                    model_row = rows_by_key[(run["run_id"], env_id)]
                    ident = model_row["identity"]
                    target_model = model_row["metrics"]["target_energy"]
                    target_baseline = base_row["target_energy"]
                    target_abs_delta = abs(target_baseline - target_model)
                    target_rel_delta = target_abs_delta / max(abs(target_baseline), abs(target_model), 1e-300)
                    if (base_row["output_sha256"] != ident["mat_sha256"]
                            or base_row["job_sha256"] != ident["job_sha256"]
                            or base_row["qa_receipt_sha256"] != ident["qa_receipt_sha256"]
                            or not math.isclose(target_baseline, target_model, rel_tol=1e-12, abs_tol=1e-24)):
                        raise AssertionError(f"Baseline test identity/target mismatch: {ref['label']}/{condition}/{env_id}")
                    baseline_target_energy_comparisons += 1
                    baseline_target_energy_exact_matches += target_abs_delta == 0.0
                    baseline_target_energy_max_abs_delta = max(baseline_target_energy_max_abs_delta, target_abs_delta)
                    baseline_target_energy_max_relative_delta = max(baseline_target_energy_max_relative_delta, target_rel_delta)
                    baseline_join_checks += 1
            baseline_raw_rows[(speed, ref["label"])] = by_env

    # Explicitly bind and compare same-speed C0/C1 training initialization receipts.
    snapshot_runs = {run["run_id"]: run for run in snapshot["runs"]}
    receipt_evidence = {}
    init_pair_checks = []
    for run in run_specs:
        bundle = snapshot_runs[run["run_id"]]
        receipt_path = runner._resolve(root, bundle["receipt_path"])
        mirror_receipt_path = runner._resolve(root, bundle["mirror_receipt_path"])
        if runner.sha256(receipt_path) != bundle["receipt_sha256"]:
            raise AssertionError(f"Training receipt hash mismatch: {run['run_id']}")
        receipt = runner.read_json(receipt_path)
        mirror_receipt = runner.read_json(mirror_receipt_path)
        if receipt != mirror_receipt or runner.sha256(mirror_receipt_path) != bundle["mirror_receipt_sha256"]:
            raise AssertionError(f"Main/mirror receipt mismatch: {run['run_id']}")
        if (receipt.get("run_id") != run["run_id"] or receipt.get("status") != "complete"
                or receipt.get("test_files_opened") != []):
            raise AssertionError(f"Training receipt is incomplete or records test access: {run['run_id']}")
        receipt_evidence[run["run_id"]] = {
            "receipt_sha256": runner.sha256(receipt_path),
            "initial_parameter_sha256": receipt["initial_parameter_sha256"],
            "first_batch_sha256": receipt["first_batch_sha256"],
            "completed_epochs": receipt["completed_epochs"],
            "best_epoch": receipt["best_epoch"],
        }
    for speed in sorted(groups_by_speed):
        c0 = specs_by_speed_condition[(speed, "C0")]["run_id"]
        c1 = specs_by_speed_condition[(speed, "C1")]["run_id"]
        for field in ("initial_parameter_sha256", "first_batch_sha256"):
            if receipt_evidence[c0][field] != receipt_evidence[c1][field]:
                raise AssertionError(f"C0/C1 same-speed training {field} differs at {speed}")
        init_pair_checks.append({
            "speed_kmh": speed,
            "run_ids": [c0, c1],
            "initial_parameter_sha256": receipt_evidence[c0]["initial_parameter_sha256"],
            "first_batch_sha256": receipt_evidence[c0]["first_batch_sha256"],
            "matches": True,
        })

    summary_json = runner.read_json(OUTPUT_DIR / "summary.json")
    computed_models, computed_contrasts = runner._result_summary(results, FAMILY, run_specs)
    if (summary_json.get("models") != computed_models
            or summary_json.get("paired_descriptive_contrasts") != computed_contrasts
            or summary_json.get("status") != "complete"
            or summary_json.get("test_based_selection") is not False
            or summary_json.get("test_based_tuning") is not False):
        raise AssertionError("Summary differs from the strict row-derived recomputation")

    summary_fields = [
        "speed_kmh", "condition", "run_id", "n_environments",
        "mean_environment_linear_nmse", "mean_environment_nmse_db", "mean_environment_rzf_sum_se_10db",
        "PC-Wiener_mean_environment_nmse_db", "PC-Wiener_mean_environment_rzf_sum_se_10db",
        "Original-F26-Lite_mean_environment_nmse_db", "Original-F26-Lite_mean_environment_rzf_sum_se_10db",
        "model_minus_PC-Wiener_mean_environment_nmse_db", "model_minus_PC-Wiener_mean_environment_rzf_sum_se_10db",
        "model_minus_Original-F26-Lite_mean_environment_nmse_db", "model_minus_Original-F26-Lite_mean_environment_rzf_sum_se_10db",
    ]
    summary_buffer = io.StringIO(newline="")
    summary_writer = csv.DictWriter(summary_buffer, fieldnames=summary_fields, lineterminator="\n")
    summary_writer.writeheader()
    model_rows = []
    for run in sorted(run_specs, key=lambda x: (x["speed_kmh"], x["condition"])):
        item = computed_models[run["run_id"]]
        row = {
            "speed_kmh": run["speed_kmh"], "condition": run["condition"], "run_id": run["run_id"],
            "n_environments": item["n_environments"],
            "mean_environment_linear_nmse": item["mean_environment_linear_nmse"],
            "mean_environment_nmse_db": item["mean_environment_nmse_db"],
            "mean_environment_rzf_sum_se_10db": item["mean_environment_rzf_sum_se_10db"],
        }
        for label in ("PC-Wiener", "Original-F26-Lite"):
            baseline = item["historical_baseline_comparison"][label]
            row[f"{label}_mean_environment_nmse_db"] = baseline["mean_environment_nmse_db"]
            row[f"{label}_mean_environment_rzf_sum_se_10db"] = baseline["mean_environment_rzf_sum_se_10db"]
            row[f"model_minus_{label}_mean_environment_nmse_db"] = baseline["model_minus_baseline_mean_nmse_db"]
            row[f"model_minus_{label}_mean_environment_rzf_sum_se_10db"] = baseline["model_minus_baseline_mean_rzf_sum_se_10db"]
        summary_writer.writerow(row)
        model_rows.append(row)

    paired_fields = [
        "speed_kmh", "n_paired_environments", "mean_delta_nmse_db_C1_minus_C0",
        "mean_delta_rzf_sum_se_10db_C1_minus_C0", "C0_mean_environment_nmse_db",
        "C1_mean_environment_nmse_db", "C0_mean_environment_rzf_sum_se_10db",
        "C1_mean_environment_rzf_sum_se_10db", "exact_environment_pairing", "target_energy_equal_per_environment",
    ]
    paired_buffer = io.StringIO(newline="")
    paired_writer = csv.DictWriter(paired_buffer, fieldnames=paired_fields, lineterminator="\n")
    paired_writer.writeheader()
    paired_rows = []
    for speed in sorted(groups_by_speed):
        contrast = next(row for row in computed_contrasts if row["speed_kmh"] == speed)
        c0 = computed_models[specs_by_speed_condition[(speed, "C0")]["run_id"]]
        c1 = computed_models[specs_by_speed_condition[(speed, "C1")]["run_id"]]
        row = {
            "speed_kmh": speed,
            "n_paired_environments": contrast["n_paired_environments"],
            "mean_delta_nmse_db_C1_minus_C0": contrast["mean_delta_nmse_db"],
            "mean_delta_rzf_sum_se_10db_C1_minus_C0": contrast["mean_delta_rzf_sum_se_10db"],
            "C0_mean_environment_nmse_db": c0["mean_environment_nmse_db"],
            "C1_mean_environment_nmse_db": c1["mean_environment_nmse_db"],
            "C0_mean_environment_rzf_sum_se_10db": c0["mean_environment_rzf_sum_se_10db"],
            "C1_mean_environment_rzf_sum_se_10db": c1["mean_environment_rzf_sum_se_10db"],
            "exact_environment_pairing": True,
            "target_energy_equal_per_environment": True,
        }
        paired_writer.writerow(row)
        paired_rows.append(row)

    interpretation = """# iTransformer C0/C1 控制对照（描述性汇总）

这份分析基于固定 pretest manifest 下完成的正式 D1 test 回放。冻结 replay runner 对 manifest、root approval、checkpoint/source 身份、完整结果与基线身份均通过严格校验；此后对已保存的环境 JSON/CSV 作汇总。本后处理没有重新打开 MAT 文件，也没有计算 p 值或置信区间。

每个速度各有 30 个完全配对环境，每个条件只训练了一个 seed。30 km/h 下，C1 相对 C0 的环境 NMSE-dB 均值差为 −1.229 dB（C1 −24.941 dB，C0 −23.712 dB），RZF 10 dB 平均 sum-SE 差为 +0.011 bit/s/Hz（6.453 对 6.442）。300 km/h 下，相应差值为 −0.387 dB（−1.304 对 −0.917 dB）和 +0.125 bit/s/Hz（1.897 对 1.771）。这些是对 30 个环境逐一配对后再取平均的描述性差值；单 seed 不能支持训练随机性层面的推断。

在 300 km/h，C1 的平均 sum-SE 仍低于 PC-Wiener 的 2.251 bit/s/Hz，且 C1 的平均 NMSE-dB（−1.304 dB）也高于 PC-Wiener（−1.734 dB）。在 30 km/h，C1 平均 sum-SE 为 6.453 bit/s/Hz，低于 PC-Wiener 的 6.499 bit/s/Hz，NMSE-dB 也明显高于 PC-Wiener 的 −36.482 dB。结果只描述本次固定 checkpoint 与 test 环境的差异，不解释为因果效应或跨 seed 的稳定优势。

结果来自事后固定 checkpoint test 回放；测试集已用于这项正式评估。测试集未用于 checkpoint/超参数选择，但这份控制对照并非独立预注册的 test 分析。
"""

    summary_path = DEST / "summary.csv"
    paired_path = DEST / "paired.csv"
    interpretation_path = DEST / "INTERPRETATION.md"
    summary_sha = write_new(summary_path, summary_buffer.getvalue())
    paired_sha = write_new(paired_path, paired_buffer.getvalue())
    interpretation_sha = write_new(interpretation_path, interpretation)

    run_manifest_path = OUTPUT_DIR / "run_manifest.json"
    verification = {
        "schema": "itransformer-control-analysis-verification-v1",
        "status": "passed",
        "family": FAMILY,
        "result_directory": OUTPUT_DIR.relative_to(REPO).as_posix(),
        "pretest_manifest": MANIFEST_PATH.relative_to(REPO).as_posix(),
        "pretest_manifest_sha256": manifest_sha,
        "protocol_sha256": protocol_sha,
        "frozen_runner_sha256": runner_sha,
        "root_approval_file_sha256": approval["approval_file_sha256"],
        "run_identity_sha256": runner.sha256(OUTPUT_DIR / "run_identity.json"),
        "run_manifest_sha256": runner.sha256(run_manifest_path),
        "summary_json_sha256": runner.sha256(OUTPUT_DIR / "summary.json"),
        "per_environment_csv_sha256": runner.sha256(csv_path),
        "output_sha256": {
            "summary.csv": summary_sha,
            "paired.csv": paired_sha,
            "INTERPRETATION.md": interpretation_sha,
            Path(__file__).name: sha256(Path(__file__)),
        },
        "checks": {
            "strict_pretest_manifest_revalidated": True,
            "exact_root_approval_revalidated": True,
            "frozen_runner_complete_run_gate": True,
            "exact_run_identity": True,
            "complete_unique_model_environment_rows": len(results),
            "expected_rows": 120,
            "rows_per_run": {s["run_id"]: 30 for s in run_specs},
            "exact_csv_bytes_and_all_fields_match_environment_json": True,
            "geometry_and_energy_nmse_db_algebra_rows_checked": len(results),
            "expected_geometry": {
                "speeds_kmh": sorted(groups_by_speed), "horizon_steps": 4,
                "horizon_ms": 2.0, "stride": 1, "origins_per_environment": expected_origins,
                "users": contract["n_users"], "rows_per_environment": expected_origins * contract["n_users"],
            },
            "paired_c0_c1_target_energy_and_identity_pairs_checked": paired_target_checks,
            "historical_baseline_model_identity_target_join_checks": baseline_join_checks,
            "historical_baseline_target_energy_comparisons": baseline_target_energy_comparisons,
            "historical_baseline_target_energy_exact_matches": baseline_target_energy_exact_matches,
            "historical_baseline_target_energy_max_abs_roundoff": baseline_target_energy_max_abs_delta,
            "historical_baseline_target_energy_max_relative_roundoff": baseline_target_energy_max_relative_delta,
            "historical_baseline_target_energy_tolerance": {"rel_tol": 1e-12, "abs_tol": 1e-24},
            "training_receipt_sha_and_main_mirror_identity_checks": len(receipt_evidence),
            "same_speed_c0_c1_initial_parameter_and_first_batch_pairs": init_pair_checks,
            "summary_and_pair_contrasts_recomputed_from_rows": True,
            "test_based_selection": False,
            "test_based_tuning": False,
            "test_mat_opened_by_this_analysis": False,
            "new_p_values_or_confidence_intervals": False,
        },
        "baseline_inputs": baseline_evidence,
        "training_receipts": receipt_evidence,
        "computed_models": computed_models,
        "computed_paired_descriptive_contrasts": computed_contrasts,
        "notes": [
            "Experimental unit is the environment; C0/C1 differences pair the exact same 30 environment IDs within speed.",
            "One seed per condition; results are descriptive, with no p-values or confidence intervals.",
            "The test outcomes were used by the frozen formal replay after pretest fixation; this post-hoc control analysis is not independently preregistered.",
        ],
    }
    verification_path = DEST / "verification.json"
    write_new(verification_path, json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "passed",
        "rows": len(results),
        "paired_contrasts": len(computed_contrasts),
        "destination": DEST.relative_to(REPO).as_posix(),
        "manifest_sha256": manifest_sha,
        "output_sha256": {
            "summary.csv": summary_sha,
            "paired.csv": paired_sha,
            "INTERPRETATION.md": interpretation_sha,
            "verification.json": sha256(verification_path),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
