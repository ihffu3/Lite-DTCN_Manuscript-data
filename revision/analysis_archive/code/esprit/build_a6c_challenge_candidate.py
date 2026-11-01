#!/usr/bin/env python
"""Format an independently audited A6c result as a review-only paper candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
AUDIT_DIR = PROJECT_ROOT / "revision/paper/remaining_fixes_20261029/classical_analysis/a6c_heldout_v3"
AUDIT_JSON = AUDIT_DIR / "verification.json"
AUDIT_CSV = AUDIT_DIR / "source_data.csv"
DEFAULT_OUTPUT = HERE / "candidate_outputs"
METHOD_LABEL = "ESPRIT (D1-validation-selected rank 4, six Hankel rows)"
TASKS = (
    ("D2_static_1p5ms", "D2_unseen_speed", 1.5, False),
    ("D2_static_3ms", "D2_unseen_speed", 3.0, False),
    ("D2_dynamic_1p5_3ms", "D2_unseen_speed", None, True),
    ("D3_variable_speed_2ms", "variable_speed", 2.0, False),
    ("D3_variable_speed_4ms", "variable_speed", 4.0, False),
    ("D3_smooth_curve_turn_2ms", "smooth_curve_turn", 2.0, False),
    ("D3_smooth_curve_turn_4ms", "smooth_curve_turn", 4.0, False),
    ("D3_cluster_transition_2ms", "cluster_transition", 2.0, False),
    ("D3_cluster_transition_4ms", "cluster_transition", 4.0, False),
)
REQUIRED_CSV_FIELDS = {
    "task_id", "environment_id", "family", "dynamic_horizon", "horizon_ms", "horizon_step_counts",
    "nmse_db", "status", "n_fallback_rows", "oracle_current_nmse_db", "paired_delta_db",
    "run_fingerprint",
}
FACT_FIELDS = (
    "fact_id", "quantity", "method", "speed_kmh", "horizon_ms", "snr_db", "value", "unit",
    "aggregation", "source_file", "source_selector",
)
MAP_FIELDS = ("asset", "row_selector", "source_csv", "fact_ids")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _csv_int(value: Any, label: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", text):
        raise ValueError(f"{label} must be a non-negative integer")
    return int(text)


def _expected_task_specs() -> dict[str, tuple[str, float | None, bool]]:
    return {task_id: (family, horizon_ms, dynamic) for task_id, family, horizon_ms, dynamic in TASKS}


def validate_audit_package(
    report: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    source_csv_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Consume only the strict auditor's completed, hashed 180-cell output."""
    if report.get("schema") != "a6c-esprit-heldout-strict-audit-v1":
        raise ValueError("Unexpected A6c strict-audit schema")
    if report.get("status") not in {"verified_complete", "verified_complete_with_fallbacks"}:
        raise ValueError("A6c audit is not integration-eligible")
    if report.get("integration_eligible") is not True:
        raise ValueError("A6c strict audit did not authorize candidate formatting")
    source_record = report.get("source_data_csv", {})
    if source_record.get("sha256") != source_csv_sha256 or source_record.get("rows") != 180:
        raise ValueError("A6c source_data.csv hash or row count differs from its audit receipt")

    coverage = report.get("coverage", {})
    if (coverage.get("task_count") != 9 or coverage.get("environment_count") != 80
            or coverage.get("expected_cells") != 180 or coverage.get("observed_cells") != 180
            or coverage.get("cells_per_task") != 20 or coverage.get("exact_task_environment_grid") is not True
            or coverage.get("unavailable_cells_retained") != 0):
        raise ValueError("A6c audit does not cover all 180 scored task/environment cells")
    statuses = coverage.get("cell_status_counts", {})
    if (statuses.get("unavailable") != 0
            or statuses.get("complete", 0) + statuses.get("complete_with_fallbacks", 0) != 180):
        raise ValueError("Unavailable cells or incomplete status accounting block table preparation")
    joins = report.get("same_mat_target_join", {})
    if (joins.get("historical_a6_raw_rows") != 900
            or joins.get("historical_a6_corrected_rows") != 900
            or joins.get("oracle_current_rows") != 180
            or joins.get("a6c_cells_joined") != 180
            or joins.get("all_900_historical_classical_rows_joined_by_task_env_mat_and_target") is not True
            or joins.get("all_180_oracle_current_rows_joined_by_task_env_mat_and_target") is not True):
        raise ValueError("A6c audit did not confirm the historical and oracle-current joins")
    dynamic = report.get("dynamic_horizon_audit", {})
    if (dynamic.get("dynamic_two_bin_geometry_preserved") is not True
            or dynamic.get("all_task_horizon_counts_match_frozen_grid_and_oracle_anchor") is not True):
        raise ValueError("A6c dynamic 3/6-step horizon audit is incomplete")
    fallback_policy = report.get("fallback_and_unavailable_policy", {})
    if (fallback_policy.get("fallback_rows_remain_scored_in_nmse") is not True
            or fallback_policy.get("unavailable_cells_are_retained_and_never_scored_or_dropped") is not True):
        raise ValueError("A6c fallback/unavailable accounting is not verified")

    plan = report.get("coverage", {}).get("challenge_plan")
    expected = _expected_task_specs()
    if not isinstance(plan, list) or len(plan) != 9:
        raise ValueError("A6c plan must have nine tasks")
    actual_specs = {}
    for item in plan:
        task_id = item.get("task_id")
        if task_id in actual_specs:
            raise ValueError(f"Duplicate A6c task in audit plan: {task_id}")
        actual_specs[task_id] = (item.get("family"), item.get("horizon_ms"), item.get("dynamic"))
    if actual_specs != expected:
        raise ValueError("A6c task IDs, families, or horizon definitions differ from the fixed plan")

    fixed_panel = report.get("coverage", {}).get("fixed_panel")
    if fixed_panel is not None:
        if not isinstance(fixed_panel, dict):
            raise ValueError("A6c fixed_panel is malformed")
        family_envs = fixed_panel
    else:
        # The strict auditor currently records the plan and coverage but not the full panel map.
        # Environment IDs are therefore checked for exact per-task counts and uniqueness below.
        family_envs = None

    summaries = report.get("metric_audit", {}).get("paired_summaries")
    if report.get("metric_audit", {}).get("paired_delta_definition") != \
            "ESPRIT NMSE dB minus oracle_current NMSE dB; positive is worse":
        raise ValueError("A6c paired-difference direction is missing or changed")
    if not isinstance(summaries, list) or len(summaries) != 9:
        raise ValueError("A6c audit must include all nine recomputed paired summaries")
    summary_by_task: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        task_id = summary.get("task_id")
        if task_id in summary_by_task or task_id not in expected:
            raise ValueError(f"Unexpected or duplicate A6c paired summary: {task_id}")
        if (summary.get("method") != "esprit" or summary.get("n_expected_environments") != 20
                or summary.get("n_paired_environments") != 20
                or summary.get("status") not in {"complete", "complete_with_fallbacks"}):
            raise ValueError(f"A6c paired summary is incomplete: {task_id}")
        for field in ("mean_classical_nmse_db", "mean_oracle_current_nmse_db",
                      "mean_paired_delta_db_classical_minus_neural", "median_paired_delta_db", "sd_paired_delta_db"):
            if summary.get(field) is None:
                raise ValueError(f"A6c paired summary lacks {field}: {task_id}")
            _finite(summary[field], f"{task_id}.{field}")
        ci = summary.get("paired_delta_bootstrap_95pct_ci_db")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"A6c paired 95% interval is malformed: {task_id}")
        low, high = (_finite(ci[0], f"{task_id} ci low"), _finite(ci[1], f"{task_id} ci high"))
        if low > high or summary.get("bootstrap_replicates") != 10000:
            raise ValueError(f"A6c paired 95% interval metadata differs: {task_id}")
        summary_by_task[task_id] = summary
    if set(summary_by_task) != set(expected):
        raise ValueError("A6c paired summaries do not cover the exact nine-task grid")

    if not rows or not REQUIRED_CSV_FIELDS.issubset(rows[0]):
        raise ValueError("A6c strict source CSV is missing required columns")
    if len(rows) != 180:
        raise ValueError(f"A6c strict source CSV must contain 180 rows, got {len(rows)}")
    run_fingerprint = report.get("source_bindings", {}).get("run_fingerprint")
    seen = set()
    per_task: dict[str, dict[str, int]] = {
        task_id: {"complete_cells": 0, "fallback_cells": 0, "fallback_prediction_rows": 0,
                 "unavailable_cells": 0, "paired_rows": 0}
        for task_id in expected
    }
    envs_by_task: dict[str, set[str]] = {task_id: set() for task_id in expected}
    paired_values: dict[str, dict[str, list[float]]] = {
        task_id: {"esprit": [], "anchor": [], "delta": []} for task_id in expected
    }
    for row in rows:
        task_id = row["task_id"]
        if task_id not in expected or not row["environment_id"]:
            raise ValueError("A6c source CSV contains an unknown task or empty environment ID")
        family, horizon_ms, dynamic_horizon = expected[task_id]
        dynamic_text = str(row.get("dynamic_horizon", "")).strip().lower()
        if (row["family"] != family or dynamic_text not in {"true", "false"}
                or dynamic_text != str(dynamic_horizon).lower()):
            raise ValueError(f"A6c row family or dynamic/static label differs: {task_id}")
        try:
            step_counts = json.loads(row["horizon_step_counts"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"A6c horizon-step count is malformed: {task_id}") from exc
        if (not isinstance(step_counts, dict) or not step_counts
                or any(not isinstance(key, str) or type(value) is not int or value <= 0
                       for key, value in step_counts.items())):
            raise ValueError(f"A6c horizon-step counts must be positive integers: {task_id}")
        horizon_text = str(row.get("horizon_ms", "")).strip()
        if dynamic_horizon:
            if horizon_text or set(step_counts) != {"3", "6"}:
                raise ValueError("Dynamic D2 task must retain its 3/6-step bins")
        else:
            expected_step = str(round(float(horizon_ms) / 0.5))
            if set(step_counts) != {expected_step}:
                raise ValueError(f"A6c step counts differ from the fixed horizon plan: {task_id}")
            if (horizon_text and not math.isclose(
                    _finite(horizon_text, f"{task_id} horizon_ms"),
                    float(horizon_ms), rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"A6c fixed horizon differs from its task definition: {task_id}")
        key = (task_id, row["environment_id"])
        if key in seen:
            raise ValueError(f"Duplicate A6c source row {key}")
        seen.add(key)
        envs_by_task[task_id].add(row["environment_id"])
        if row["run_fingerprint"] != run_fingerprint:
            raise ValueError(f"A6c source row run fingerprint differs: {key}")
        status = row["status"]
        fallback_count = _csv_int(row["n_fallback_rows"], f"{key} n_fallback_rows")
        if status == "complete":
            if fallback_count != 0:
                raise ValueError(f"Complete A6c row has nonzero fallback count: {key}")
            per_task[task_id]["complete_cells"] += 1
        elif status == "complete_with_fallbacks":
            if fallback_count <= 0:
                raise ValueError(f"Fallback status lacks fallback rows: {key}")
            per_task[task_id]["fallback_cells"] += 1
        else:
            per_task[task_id]["unavailable_cells"] += 1
            raise ValueError(f"A6c unavailable or incomplete cell blocks integration: {key}")
        per_task[task_id]["fallback_prediction_rows"] += fallback_count
        per_task[task_id]["paired_rows"] += 1
        nmse = _finite(row["nmse_db"], f"{key} nmse_db")
        anchor = _finite(row["oracle_current_nmse_db"], f"{key} oracle_current_nmse_db")
        delta = _finite(row["paired_delta_db"], f"{key} paired_delta_db")
        if not math.isclose(delta, nmse - anchor, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"A6c paired delta differs from its same-environment values: {key}")
        paired_values[task_id]["esprit"].append(nmse)
        paired_values[task_id]["anchor"].append(anchor)
        paired_values[task_id]["delta"].append(delta)
    if len(seen) != 180:
        raise ValueError("A6c source CSV does not cover the exact 180 cells")
    for task_id, family, _horizon, _dynamic in TASKS:
        if len(envs_by_task[task_id]) != 20 or per_task[task_id]["paired_rows"] != 20:
            raise ValueError(f"A6c task lacks exactly 20 unique paired environments: {task_id}")
        if (per_task[task_id]["complete_cells"] != summary_by_task[task_id]["n_complete_cells"]
                or per_task[task_id]["fallback_cells"] != summary_by_task[task_id]["n_complete_with_fallbacks"]):
            raise ValueError(f"A6c summary fallback cell counts differ from source rows: {task_id}")
        deltas = paired_values[task_id]["delta"]
        values_to_check = {
            "mean_classical_nmse_db": math.fsum(paired_values[task_id]["esprit"]) / 20,
            "mean_oracle_current_nmse_db": math.fsum(paired_values[task_id]["anchor"]) / 20,
            "mean_paired_delta_db_classical_minus_neural": math.fsum(deltas) / 20,
            "median_paired_delta_db": statistics.median(deltas),
            "sd_paired_delta_db": statistics.stdev(deltas),
        }
        for field, value in values_to_check.items():
            audited = _finite(summary_by_task[task_id][field], f"{task_id} audited {field}")
            if not math.isclose(audited, value, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"A6c paired summary does not reproduce source rows for {task_id}: {field}")
        if family_envs is not None:
            expected_envs = set(family_envs[family])
            if envs_by_task[task_id] != expected_envs:
                raise ValueError(f"A6c source row environment set differs from fixed panel: {task_id}")
    return [summary_by_task[task_id] for task_id, *_ in TASKS], per_task


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{_finite(value, 'table value'):.{digits}f}"


def make_candidate_outputs(
    report: dict[str, Any],
    summaries: list[dict[str, Any]],
    fallback: dict[str, dict[str, int]],
    *,
    report_sha256: str,
    source_csv_sha256: str,
) -> dict[str, bytes]:
    """Create table, facts, and mapping from audited summaries only; add no inference."""
    import io

    task_specs = _expected_task_specs()
    ordered = {summary["task_id"]: summary for summary in summaries}
    summary_fields = (
        "fact_id", "task_id", "family", "task_label", "horizon", "dynamic_horizon_step_bins",
        "n_expected_environments", "n_paired_environments", "n_complete_cells", "n_fallback_cells",
        "n_fallback_prediction_rows", "n_unavailable_cells", "mean_esprit_nmse_db",
        "mean_oracle_current_nmse_db", "mean_paired_delta_db_classical_minus_neural",
        "median_paired_delta_db", "sd_paired_delta_db", "paired_delta_ci95_low_db",
        "paired_delta_ci95_high_db", "bootstrap_replicates", "bootstrap_seed", "status",
    )
    summary_records = []
    fact_records = []
    map_records = []
    tex_rows = []
    fact_source_report = "revision/paper/remaining_fixes_20261029/classical_analysis/a6c_heldout_v3/verification.json"
    fact_source_csv = "revision/paper/remaining_fixes_20261029/classical_analysis/a6c_heldout_v3/source_data.csv"
    for task_id, family, horizon_ms, is_dynamic in TASKS:
        summary = ordered[task_id]
        counts = fallback[task_id]
        label = {
            "D2_static_1p5ms": "D2 static",
            "D2_static_3ms": "D2 static",
            "D2_dynamic_1p5_3ms": "D2 dynamic",
            "D3_variable_speed_2ms": "D3 variable speed",
            "D3_variable_speed_4ms": "D3 variable speed",
            "D3_smooth_curve_turn_2ms": "D3 curved turn",
            "D3_smooth_curve_turn_4ms": "D3 curved turn",
            "D3_cluster_transition_2ms": "D3 scenario switch",
            "D3_cluster_transition_4ms": "D3 scenario switch",
        }[task_id]
        horizon = "1.5/3" if is_dynamic else f"{horizon_ms:g}"
        ci_low, ci_high = summary["paired_delta_bootstrap_95pct_ci_db"]
        fact_prefix = f"a6c_{task_id}"
        fact_ids = [
            f"{fact_prefix}_mean_esprit_nmse_db", f"{fact_prefix}_mean_oracle_current_nmse_db",
            f"{fact_prefix}_mean_paired_delta_db", f"{fact_prefix}_paired_delta_ci95_low_db",
            f"{fact_prefix}_paired_delta_ci95_high_db", f"{fact_prefix}_n_paired_environments",
            f"{fact_prefix}_fallback_cells", f"{fact_prefix}_fallback_prediction_rows",
        ]
        summary_records.append({
            "fact_id": f"{fact_prefix}_row", "task_id": task_id, "family": family,
            "task_label": label, "horizon": horizon,
            "dynamic_horizon_step_bins": "3,6" if is_dynamic else "",
            "n_expected_environments": summary["n_expected_environments"],
            "n_paired_environments": summary["n_paired_environments"],
            "n_complete_cells": counts["complete_cells"],
            "n_fallback_cells": counts["fallback_cells"],
            "n_fallback_prediction_rows": counts["fallback_prediction_rows"],
            "n_unavailable_cells": counts["unavailable_cells"],
            "mean_esprit_nmse_db": format(float(summary["mean_classical_nmse_db"]), ".17g"),
            "mean_oracle_current_nmse_db": format(float(summary["mean_oracle_current_nmse_db"]), ".17g"),
            "mean_paired_delta_db_classical_minus_neural": format(float(summary["mean_paired_delta_db_classical_minus_neural"]), ".17g"),
            "median_paired_delta_db": format(float(summary["median_paired_delta_db"]), ".17g"),
            "sd_paired_delta_db": format(float(summary["sd_paired_delta_db"]), ".17g"),
            "paired_delta_ci95_low_db": format(float(ci_low), ".17g"),
            "paired_delta_ci95_high_db": format(float(ci_high), ".17g"),
            "bootstrap_replicates": summary["bootstrap_replicates"], "bootstrap_seed": summary["bootstrap_seed"],
            "status": summary["status"],
        })
        metric_facts = [
            (fact_ids[0], "mean_esprit_nmse_db", summary["mean_classical_nmse_db"], "dB", "equal-environment mean NMSE dB from audited paired-summary output", fact_source_report, f"metric_audit.paired_summaries[{task_id}].mean_classical_nmse_db"),
            (fact_ids[1], "mean_oracle_current_nmse_db", summary["mean_oracle_current_nmse_db"], "dB", "equal-environment mean same-target oracle-current NMSE dB from audited paired-summary output", fact_source_report, f"metric_audit.paired_summaries[{task_id}].mean_oracle_current_nmse_db"),
            (fact_ids[2], "mean_paired_delta_db_classical_minus_neural", summary["mean_paired_delta_db_classical_minus_neural"], "dB", "mean paired environment delta; positive means ESPRIT NMSE is worse", fact_source_report, f"metric_audit.paired_summaries[{task_id}].mean_paired_delta_db_classical_minus_neural"),
            (fact_ids[3], "paired_delta_bootstrap_95pct_ci_lower", ci_low, "dB", "pointwise environment-bootstrap 95% CI; 10000 replicates", fact_source_report, f"metric_audit.paired_summaries[{task_id}].paired_delta_bootstrap_95pct_ci_db[0]"),
            (fact_ids[4], "paired_delta_bootstrap_95pct_ci_upper", ci_high, "dB", "pointwise environment-bootstrap 95% CI; 10000 replicates", fact_source_report, f"metric_audit.paired_summaries[{task_id}].paired_delta_bootstrap_95pct_ci_db[1]"),
            (fact_ids[5], "n_paired_environments", summary["n_paired_environments"], "environments", "paired environments; unit is environment", fact_source_report, f"metric_audit.paired_summaries[{task_id}].n_paired_environments"),
            (fact_ids[6], "n_fallback_cells", counts["fallback_cells"], "cells", "all task cells with one or more prediction fallbacks", fact_source_csv, f"task_id={task_id};status=complete_with_fallbacks"),
            (fact_ids[7], "fallback_prediction_rows", counts["fallback_prediction_rows"], "prediction rows", "sum of retained n_fallback_rows across all task environments", fact_source_csv, f"task_id={task_id};sum(n_fallback_rows)"),
        ]
        for fact_id, quantity, value, unit, aggregation, source_file, selector in metric_facts:
            fact_records.append({
                "fact_id": fact_id, "quantity": quantity, "method": METHOD_LABEL,
                "speed_kmh": "", "horizon_ms": horizon_ms if horizon_ms is not None else "",
                "snr_db": "", "value": format(float(value), ".17g") if isinstance(value, (int, float)) else str(value),
                "unit": unit, "aggregation": aggregation, "source_file": source_file,
                "source_selector": selector,
            })
        map_records.append({
            "asset": "tables/esprit_a6c_challenge.tex", "row_selector": task_id,
            "source_csv": fact_source_csv,
            "fact_ids": ";".join(fact_ids),
        })
        tex_rows.append(
            f"{label} & {horizon} & {_fmt(summary['mean_classical_nmse_db'])} & "
            f"{_fmt(summary['mean_oracle_current_nmse_db'])} & "
            f"{_fmt(summary['mean_paired_delta_db_classical_minus_neural'])} "
            f"[{_fmt(ci_low)}, {_fmt(ci_high)}] & "
            f"{counts['fallback_cells']}/{summary['n_expected_environments']} / "
            f"{counts['fallback_prediction_rows']} \\\\"
        )

    tex = "\n".join([
        "% Candidate only. Values are copied from the audited A6c verification.json.",
        "% Do not insert until root reviews the completed audit and source bindings.",
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Post-test ESPRIT challenge using the fixed D1-validation-selected rank-4, six-row setting, with no D2/D3 fitting or retuning and no oracle-speed input to ESPRIT. Each row pairs 20 environments with the oracle-current reference on the same target. The dynamic D2 request contains both 3-step and 6-step bins. NMSE is in dB; paired $\\Delta$ is ESPRIT minus oracle-current NMSE, so positive values are worse. Brackets give pointwise 95\\% environment-bootstrap intervals (10,000 resamples). Fallback entries give affected cells out of 20 and retained prediction rows.}",
        "\\label{tab:esprit-a6c-challenge}",
        "\\begin{tabular}{l r r r r l}", "\\toprule",
        "Task & Horizon (ms) & ESPRIT & Oracle-current & Paired $\\Delta$ [95\\% CI] & Fallback (cells / rows) \\\\ \\midrule",
        *tex_rows, "\\bottomrule", "\\end{tabular}", "\\end{table*}",
    ]) + "\n"

    summary_buffer = io.StringIO(newline="")
    summary_writer = csv.DictWriter(summary_buffer, fieldnames=summary_fields, lineterminator="\n")
    summary_writer.writeheader()
    summary_writer.writerows(summary_records)
    fact_buffer = io.StringIO(newline="")
    fact_writer = csv.DictWriter(fact_buffer, fieldnames=FACT_FIELDS, lineterminator="\n")
    fact_writer.writeheader()
    fact_writer.writerows(fact_records)
    map_buffer = io.StringIO(newline="")
    map_writer = csv.DictWriter(map_buffer, fieldnames=MAP_FIELDS, lineterminator="\n")
    map_writer.writeheader()
    map_writer.writerows(map_records)

    outputs: dict[str, bytes] = {
        "esprit_a6c_challenge_summary.csv": summary_buffer.getvalue().encode("utf-8"),
        "facts_candidate.csv": fact_buffer.getvalue().encode("utf-8"),
        "table_statistic_map_candidate.csv": map_buffer.getvalue().encode("utf-8"),
        "esprit_a6c_challenge.tex": tex.encode("utf-8"),
    }
    verification = {
        "status": "candidate_ready_review_only",
        "audit_schema": report["schema"],
        "audit_status": report["status"],
        "integration_eligible": report["integration_eligible"],
        "audit_report_sha256": report_sha256,
        "audit_source_data_sha256": source_csv_sha256,
        "summary_rows": len(summary_records), "fact_rows": len(fact_records), "table_map_rows": len(map_records),
        "new_inference": False,
        "uses_audited_paired_summary_without_recomputing_ci": True,
        "manuscript_modified": False,
        "root_review_required_before_integration": True,
        "claims": {
            "configuration": "fixed rank 4, six Hankel rows; selected on D1 validation only",
            "d2_d3_fit_or_retuning": False,
            "oracle_speed_used_as_esprit_input": False,
            "oracle_current_role": "paired same-target reference only",
            "formal_test_access_boundary": report.get("test_access_disclosure"),
            "scope": "nine tasks, 20 environments per task; challenge result does not establish the ESPRIT family",
        },
        "candidate_files": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()},
    }
    outputs["verification.json"] = (json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return outputs


def build_from_files(audit_json: Path = AUDIT_JSON, audit_csv: Path = AUDIT_CSV,
                     output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    report_sha = sha256(audit_json)
    csv_sha = sha256(audit_csv)
    source_record = report.get("source_data_csv", {})
    if source_record.get("path") != audit_csv.relative_to(PROJECT_ROOT).as_posix():
        raise ValueError("Audit receipt points to an unexpected source-data path")
    with audit_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    summaries, fallback = validate_audit_package(report, rows, source_csv_sha256=csv_sha)
    outputs = make_candidate_outputs(report, summaries, fallback,
                                     report_sha256=report_sha, source_csv_sha256=csv_sha)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = output_dir / name
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError(f"Refusing to overwrite a different candidate output: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
        temporary.replace(path)
    return {"status": "candidate_ready_review_only", "output_dir": str(output_dir),
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()},
            "summary_rows": 9, "facts": 72, "table_map_rows": 9}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_from_files(args.audit_json, args.audit_csv, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
