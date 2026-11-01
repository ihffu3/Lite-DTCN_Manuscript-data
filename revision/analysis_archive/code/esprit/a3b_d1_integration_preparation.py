"""Prepare, but do not integrate, the reviewed A3b/D1 ESPRIT comparison.

The command is deliberately fail-closed. It reads only committed CSV/JSON
evidence and current text/fact files. It never opens test MAT files, edits the
paper, or overwrites an existing candidate artifact with different bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
REMAINING = ROOT / "revision/paper/remaining_fixes_20261029"
HERE = Path(__file__).resolve().parent
SELECTION_DIR = REMAINING / "classical_analysis/d1_validation_selection"
D1_AUDIT_DIR = REMAINING / "classical_analysis/d1_heldout_v1"
D1_RUNNER_PATH = REMAINING / "classical/esprit_d1_heldout_runner_v1.py"
D1_PROTOCOL_PATH = REMAINING / "classical/esprit_d1_heldout_protocol_v1.json"
BASELINE_DIR = ROOT / "revision/submission/PHYCOM_CORE_TECHNICAL_20261020/evidence/formal165"
BASELINE_VERIFY_PATH = BASELINE_DIR / "verification_complete.json"
BASELINE_DATA_PATH = BASELINE_DIR / "classical_d1_environment_metrics.csv"
WORKING_SOURCE = ROOT / "revision/paper/figure_redraw_20261029"
EXISTING_FACTS_PATH = WORKING_SOURCE / "analysis/facts.csv"
OUTPUT_DIR = HERE / "candidate_outputs_post_result_review"

SELECTION_VERIFICATION_SHA256 = "4696a308d874349468e7dc099af009f44789d11b60d3599930e0840defa0e7d2"
BASELINE_VERIFICATION_SHA256 = "9a6022bfe3f51007db60d1b787b12e3ebb9e066f8fac0a956a6f658320acc08d"
D1_AUDIT_SCHEMA = "d1-esprit-heldout-strict-audit-v1"
SPEEDS_KMH = (30, 120, 300)
HORIZONS = (1, 2, 4, 8)
ENVS_PER_SPEED = 30
EXPECTED_CONFIG = {"rank": 4, "hankel_rows": 6, "gain_fit_ridge": 1e-8}
ORIGINAL_CONFIG = {"rank": 2, "hankel_rows": 5, "gain_fit_ridge": 1e-8}
ORIGIN_START = 19
N_TIME = 4001
N_USERS = 4
NO_FIT_ID = "not_applicable_esprit_has_no_learned_fit"
FACT_FIELDS = ["fact_id", "quantity", "method", "speed_kmh", "horizon_ms", "snr_db",
               "value", "unit", "aggregation", "source_file", "source_selector"]
TABLE_MAP_FIELDS = ["asset", "row_selector", "source_csv", "fact_ids"]
REL_TOL = 1e-12
ABS_TOL = 1e-14


class IntegrationGateError(ValueError):
    """Raised when source evidence is incomplete or inconsistent."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationGateError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrationGateError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise IntegrationGateError(f"Cannot read {label}: {path}") from exc
    if not rows:
        raise IntegrationGateError(f"{label} is empty")
    return rows


def _as_float(value: Any, label: str, *, allow_neg_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationGateError(f"{label} is not numeric") from exc
    if math.isnan(number) or number == math.inf or (number == -math.inf and not allow_neg_inf):
        raise IntegrationGateError(f"{label} is non-finite")
    return number


def _as_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationGateError(f"{label} is not an integer") from exc
    if isinstance(value, bool) or str(number) != str(value).strip():
        raise IntegrationGateError(f"{label} is not an exact integer")
    return number


def _close(actual: Any, expected: float, label: str, *, rel: float = REL_TOL,
           abs_tol: float = ABS_TOL) -> float:
    value = _as_float(actual, label, allow_neg_inf=expected == -math.inf)
    if not math.isclose(value, expected, rel_tol=rel, abs_tol=abs_tol):
        raise IntegrationGateError(f"{label} differs: actual={value!r}, expected={expected!r}")
    return abs(value - expected)


def _json_value(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise IntegrationGateError(f"{label} is not a JSON object")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise IntegrationGateError(f"{label} is invalid JSON") from exc
    if not isinstance(result, dict):
        raise IntegrationGateError(f"{label} is not a JSON object")
    return result


def _configuration_key(rank: int, rows: int, ridge: float) -> tuple[int, int, float]:
    return rank, rows, ridge


def _config_id(rank: int, rows: int, ridge: float) -> str:
    ridge_text = format(ridge, ".0e").replace("e-0", "e-").replace("e+0", "e+")
    return f"r{rank}_m{rows}_g{ridge_text}"


def _metric_db(error_energy: float, target_energy: float, stored_linear: Any,
               stored_db: Any, label: str) -> tuple[float, float]:
    if not math.isfinite(error_energy) or error_energy < 0:
        raise IntegrationGateError(f"{label} error energy is invalid")
    if not math.isfinite(target_energy) or target_energy <= 0:
        raise IntegrationGateError(f"{label} target energy is invalid")
    linear = error_energy / target_energy
    _close(stored_linear, linear, f"{label} linear NMSE")
    db = -math.inf if linear == 0 else 10.0 * math.log10(linear)
    _close(stored_db, db, f"{label} NMSE dB")
    return linear, db


def _validate_prediction_status(row: dict[str, Any], expected_rows: int, label: str) -> int:
    statuses = _json_value(row.get("prediction_status_counts"), f"{label} status counts")
    if not statuses or any(type(count) is not int or count < 0 for count in statuses.values()):
        raise IntegrationGateError(f"{label} prediction status counts are invalid")
    if sum(statuses.values()) != expected_rows:
        raise IntegrationGateError(f"{label} status counts do not cover all prediction rows")
    fallback = sum(count for status, count in statuses.items() if status != "ok")
    if _as_int(row.get("n_fallback_rows"), f"{label} fallback rows") != fallback:
        raise IntegrationGateError(f"{label} fallback count differs from its status counts")
    expected_status = "complete_with_fallbacks" if fallback else "complete"
    if row.get("status") != expected_status:
        raise IntegrationGateError(f"{label} completion status differs from its fallback count")
    return fallback


def _validate_selection(selection_verification: dict[str, Any],
                        selection_rows: list[dict[str, Any]],
                        selection_csv_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = selection_verification
    if (report.get("status") != "root_verified_complete_validation_scan"
            or report.get("candidate_count") != 20
            or report.get("validation_cells") != 1920
            or report.get("independent_validation_environments") != 24
            or report.get("paired_origins_per_cell") != 4
            or report.get("paired_users_per_origin") != 4
            or report.get("all_candidate_targets_match") is not True
            or report.get("reviewed_source_hashes_verified") is not True
            or report.get("test_MAT_opened_by_this_audit") != 0):
        raise IntegrationGateError("A3b validation scan lacks the root-verified exact 20×24×4 evidence")
    if report.get("source_data_sha256") != selection_csv_sha256:
        raise IntegrationGateError("A3b validation CSV SHA differs from its root verification")
    if len(selection_rows) != 20:
        raise IntegrationGateError(f"Expected 20 A3b validation candidates, found {len(selection_rows)}")

    prepared: list[dict[str, Any]] = []
    seen: set[tuple[int, int, float]] = set()
    for i, raw in enumerate(selection_rows, 1):
        rank = _as_int(raw.get("rank"), f"validation candidate {i} rank")
        hankel_rows = _as_int(raw.get("hankel_rows"), f"validation candidate {i} Hankel rows")
        ridge = _as_float(raw.get("gain_fit_ridge"), f"validation candidate {i} ridge")
        score = _as_float(raw.get("mean_validation_environment_horizon_nmse_db"),
                           f"validation candidate {i} score")
        fallbacks = _as_int(raw.get("fallback_rows"), f"validation candidate {i} fallback rows")
        cells = _as_int(raw.get("validation_cells"), f"validation candidate {i} cells")
        key = _configuration_key(rank, hankel_rows, ridge)
        if key in seen or rank < 1 or hankel_rows < 2 or ridge <= 0 or fallbacks < 0 or cells != 96:
            raise IntegrationGateError(f"Duplicate or malformed A3b candidate: {key!r}")
        seen.add(key)
        prepared.append({
            "rank": rank, "hankel_rows": hankel_rows, "gain_fit_ridge": ridge,
            "score_nmse_db": score, "fallback_rows": fallbacks,
            "validation_cells": cells,
            "candidate_id": _config_id(rank, hankel_rows, ridge),
            "selected": False,
        })

    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise IntegrationGateError("Root-verified selection metadata is missing")
    selected_key = _configuration_key(
        _as_int(selected.get("rank"), "selected rank"),
        _as_int(selected.get("hankel_rows"), "selected Hankel rows"),
        _as_float(selected.get("gain_fit_ridge"), "selected gain-fit ridge"),
    )
    if selected_key != (4, 6, 1e-8):
        raise IntegrationGateError("Root-reviewed A3b winner differs from rank 4 / six rows / ridge 1e-8")
    winner = [row for row in prepared
              if _configuration_key(row["rank"], row["hankel_rows"], row["gain_fit_ridge"]) == selected_key]
    if len(winner) != 1:
        raise IntegrationGateError("Selected A3b configuration is absent or duplicated")
    _close(winner[0]["score_nmse_db"],
           _as_float(selected.get("selection_score_nmse_db"), "selected validation score"),
           "selected validation score")
    winner[0]["selected"] = True
    return prepared, selected


def _validate_result_report(d1_verification: dict[str, Any], *,
                            d1_csv_sha256: str, d1_row_count: int,
                            d1_runner_sha256: str, d1_protocol_sha256: str,
                            selection_result_sha256: str) -> None:
    report = d1_verification
    if report.get("schema") != D1_AUDIT_SCHEMA or report.get("status") != "verified_complete":
        raise IntegrationGateError("D1 held-out result has not passed the complete strict audit")
    coverage = report.get("coverage", {})
    if (coverage.get("expected_environments") != 90
            or coverage.get("observed_environments") != 90
            or coverage.get("expected_cells") != 360
            or coverage.get("observed_cells") != 360
            or coverage.get("speeds_kmh") != list(SPEEDS_KMH)
            or coverage.get("horizon_steps") != list(HORIZONS)
            or coverage.get("cells_per_speed_horizon") != ENVS_PER_SPEED
            or coverage.get("exact_environment_horizon_grid") is not True):
        raise IntegrationGateError("D1 strict audit coverage does not match the registered 90×4 grid")
    metrics = report.get("metric_audit", {})
    if (metrics.get("equal_environment_groups_recomputed") != 12
            or metrics.get("energy_ratio_formula") !=
            "linear_nmse = physical_error_energy_complex128 / physical_target_energy_complex128"
            or metrics.get("dB_formula") != "nmse_db = 10*log10(linear_nmse)"):
        raise IntegrationGateError("D1 strict audit metric algebra or 12-group recomputation is missing")
    fallback = report.get("fallback_audit", {})
    if fallback.get("all_rows_scored_including_fallbacks") is not True:
        raise IntegrationGateError("D1 strict audit does not confirm fallback rows remain scored")
    selection = report.get("selection_boundary", {})
    if (selection.get("configuration") != EXPECTED_CONFIG
            or selection.get("configuration_source") != "A3b D1 validation only"
            or selection.get("selection_result_sha256") != selection_result_sha256
            or selection.get("test_selection_or_fit") is not False):
        raise IntegrationGateError("D1 result selection boundary is not validation-only")
    disclosure = report.get("formal_test_access_disclosure", "")
    if not isinstance(disclosure, str) or "post-test-access" not in disclosure.lower() \
            or "not preregistered" not in disclosure.lower():
        raise IntegrationGateError("D1 post-test-access disclosure is missing")
    bindings = report.get("source_bindings", {})
    if (bindings.get("runner_sha256") != d1_runner_sha256
            or bindings.get("protocol_sha256") != d1_protocol_sha256
            or bindings.get("selection_result_sha256") != selection_result_sha256
            or not bindings.get("run_fingerprint")
            or not bindings.get("generation_fingerprint")
            or not bindings.get("root_review_receipt_sha256")):
        raise IntegrationGateError("D1 frozen runner/protocol/receipt bindings are inconsistent")
    csv_binding = report.get("source_data_csv", {})
    if (csv_binding.get("sha256") != d1_csv_sha256
            or csv_binding.get("rows") != 360 or d1_row_count != 360):
        raise IntegrationGateError("D1 strict-audit CSV hash or row count differs")


def _validate_cell_rows(rows: list[dict[str, Any]], *, tuned: bool,
                        run_fingerprint: str = "", generation_fingerprint: str = "",
                        formal_protocol_sha256: str = "") -> dict[tuple[int, int, str], dict[str, Any]]:
    expected_ids = {f"FORMAL_RL_20261015_D1_S{speed:03d}_TEST_G{group:03d}"
                    for speed in SPEEDS_KMH for group in range(1, ENVS_PER_SPEED + 1)}
    expected_keys = {(speed, horizon, env_id)
                     for speed in SPEEDS_KMH for horizon in HORIZONS for env_id in expected_ids
                     if f"_S{speed:03d}_" in env_id}
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        speed = _as_int(row.get("speed_kmh"), "row speed")
        horizon = _as_int(row.get("horizon_steps"), "row horizon")
        env_id = row.get("environment_id")
        key = (speed, horizon, env_id)
        if key not in expected_keys or key in indexed:
            raise IntegrationGateError(f"D1 row grid has a duplicate or unexpected address: {key!r}")
        if row.get("source") != "esprit" or row.get("split", "test") != "test":
            raise IntegrationGateError(f"Unexpected ESPRIT family/split identity at {key!r}")
        if tuned:
            if (row.get("family") != "D1_main"
                    or row.get("fit_id") != "not_applicable_esprit_has_no_learned_fit"
                    or row.get("configuration_source") != "A3b D1 validation only"
                    or row.get("run_fingerprint") != run_fingerprint
                    or row.get("generation_fingerprint") != generation_fingerprint
                    or row.get("formal_protocol_sha256") != formal_protocol_sha256):
                raise IntegrationGateError(f"Tuned ESPRIT config/source binding mismatch at {key!r}")
            config = _json_value(row.get("configuration"), f"configuration {key}")
            if (config.get("rank") != 4 or config.get("hankel_rows") != 6
                    or not math.isclose(float(config.get("gain_fit_ridge", math.nan)), 1e-8,
                                        rel_tol=0, abs_tol=0)):
                raise IntegrationGateError(f"Tuned ESPRIT configuration differs at {key!r}")

        expected_ms = horizon * 0.5
        _close(row.get("horizon_ms"), expected_ms, f"horizon ms {key!r}")
        n_prediction_rows = (N_TIME - horizon - ORIGIN_START) * N_USERS
        if _as_int(row.get("n_prediction_rows"), f"prediction rows {key!r}") != n_prediction_rows:
            raise IntegrationGateError(f"D1 row geometry differs at {key!r}")
        error_energy = _as_float(row.get("physical_error_energy_complex128"), f"error energy {key!r}")
        target_energy = _as_float(row.get("physical_target_energy_complex128"), f"target energy {key!r}")
        if target_energy <= 0:
            raise IntegrationGateError(f"Nonpositive target energy at {key!r}")
        linear, db = _metric_db(error_energy, target_energy,
                                row.get("linear_nmse"), row.get("nmse_db"), f"metric {key!r}")
        if linear < 0 or not isinstance(row.get("data_sha256"), str) or not row.get("data_sha256"):
            raise IntegrationGateError(f"D1 data identity/metric invalid at {key!r}")
        if not isinstance(row.get("qa_receipt_sha256"), str) or not row.get("qa_receipt_sha256"):
            raise IntegrationGateError(f"D1 QA identity missing at {key!r}")
        fallback_count = _validate_prediction_status(row, n_prediction_rows, f"prediction {key!r}")
        indexed[key] = {
            **row, "speed_kmh": speed, "horizon_steps": horizon,
            "horizon_ms": expected_ms, "n_prediction_rows": n_prediction_rows,
            "physical_error_energy_complex128": error_energy,
            "physical_target_energy_complex128": target_energy,
            "linear_nmse": linear, "nmse_db": db,
            "n_fallback_rows": fallback_count,
        }
    if set(indexed) != expected_keys:
        missing = len(expected_keys - set(indexed))
        raise IntegrationGateError(f"D1 exact speed×horizon×environment grid is incomplete ({missing} missing)")
    return indexed


def _check_paired_baseline(old: dict[tuple[int, int, str], dict[str, Any]],
                           tuned: dict[tuple[int, int, str], dict[str, Any]]) -> None:
    for key in sorted(tuned):
        new = tuned[key]
        prior = old[key]
        for field in ("data_sha256", "qa_receipt_sha256", "n_prediction_rows"):
            if new[field] != prior[field]:
                raise IntegrationGateError(f"Original/tuned D1 paired identity mismatch {field}: {key!r}")
        if str(new.get("data_bytes", "")) != str(prior.get("data_bytes", "")):
            raise IntegrationGateError(f"Original/tuned D1 paired environment metadata mismatch: {key!r}")
        _close(new.get("environment_seed"),
               _as_float(prior.get("environment_seed"), f"original seed {key!r}"),
               f"paired environment seed {key!r}", rel=0, abs_tol=0)
        _close(new["physical_target_energy_complex128"],
               prior["physical_target_energy_complex128"],
               f"paired target energy {key!r}")


def original_fact_id(speed_kmh: int, horizon_ms: float) -> str:
    horizon = format(float(horizon_ms), ".1f")
    return f"nmse_esprit_v{int(speed_kmh)}_h{horizon}"


def _group_stats(indexed: dict[tuple[int, int, str], dict[str, Any]],
                 speed: int, horizon: int) -> dict[str, Any]:
    group = [row for (v, h, _), row in indexed.items() if v == speed and h == horizon]
    group.sort(key=lambda row: row["environment_id"])
    if len(group) != ENVS_PER_SPEED:
        raise IntegrationGateError(f"Expected 30 environments at {speed}/{horizon}, got {len(group)}")
    dbs = [row["nmse_db"] for row in group]
    mean_db = -math.inf if any(value == -math.inf for value in dbs) else math.fsum(dbs) / len(dbs)
    return {
        "mean_nmse_db": mean_db,
        "mean_linear_nmse": math.fsum(row["linear_nmse"] for row in group) / len(group),
        "fallback_rows": sum(row["n_fallback_rows"] for row in group),
        "n_environments": len(group),
        "n_prediction_rows": sum(row["n_prediction_rows"] for row in group),
        "all_rows_including_fallbacks_scored": True,
    }


def _validate_original_facts(existing_facts: dict[str, dict[str, Any]],
                             old_stats: dict[tuple[int, int], dict[str, Any]]) -> None:
    for (speed, horizon), stats in old_stats.items():
        fact_id = original_fact_id(speed, horizon * 0.5)
        fact = existing_facts.get(fact_id)
        if not isinstance(fact, dict):
            raise IntegrationGateError(f"Existing original ESPRIT fact is missing: {fact_id}")
        if (fact.get("quantity") != "nmse_db_env_mean"
                or fact.get("method") != "esprit"
                or fact.get("unit") != "dB"
                or fact.get("aggregation") != "equal_env_mean"):
            raise IntegrationGateError(f"Existing original ESPRIT fact metadata differs: {fact_id}")
        _close(fact.get("value"), stats["mean_nmse_db"], f"existing fact {fact_id}",
               rel=0, abs_tol=5.1e-7)


def _build_facts(validation: list[dict[str, Any]],
                 comparison: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    validation_facts: list[dict[str, str]] = []
    for row in validation:
        fact_id = f"a3b_esprit_validation_score_{row['candidate_id']}"
        selector = (f"rank={row['rank']};hankel_rows={row['hankel_rows']};"
                    f"gain_fit_ridge={format(row['gain_fit_ridge'], '.17g')}")
        validation_facts.append({
            "fact_id": fact_id, "quantity": "mean_validation_environment_horizon_nmse_db",
            "method": "esprit_a3b_validation_candidate", "speed_kmh": "",
            "horizon_ms": "", "snr_db": "", "value": format(row["score_nmse_db"], ".17g"),
            "unit": "dB", "aggregation": "mean_validation_environment_horizon_nmse_db",
            "source_file": "revision/paper/remaining_fixes_20261029/classical_analysis/d1_validation_selection/source_data.csv",
            "source_selector": selector,
        })

    tuned_facts: list[dict[str, str]] = []
    for row in comparison:
        speed, horizon = row["speed_kmh"], row["horizon_steps"]
        h_ms = horizon * 0.5
        base = f"a3b_esprit_tuned_v{speed}_h{h_ms:.1f}"
        selector = f"speed_kmh={speed};horizon_steps={horizon};configuration=rank4_rows6_ridge1e-8"
        common = {
            "method": "esprit_a3b_validation_selected",
            "speed_kmh": str(speed), "horizon_ms": f"{h_ms:.1f}", "snr_db": "",
            "source_file": "revision/paper/remaining_fixes_20261029/classical_analysis/d1_heldout_v1/source_data.csv",
            "source_selector": selector,
        }
        tuned_facts.append({
            **common, "fact_id": base + "_mean_nmse_db", "quantity": "nmse_db_env_mean",
            "value": format(row["tuned_mean_nmse_db"], ".17g"), "unit": "dB",
            "aggregation": "equal_env_mean_30_test_environments",
        })
        tuned_facts.append({
            **common, "fact_id": base + "_fallback_rows", "quantity": "fallback_prediction_rows",
            "value": str(row["tuned_fallback_rows"]), "unit": "prediction_rows",
            "aggregation": "sum_over_30_test_environments;included_in_error_score",
        })
    ids = [row["fact_id"] for row in validation_facts + tuned_facts]
    if len(ids) != len(set(ids)):
        raise IntegrationGateError("Candidate fact identifiers are not unique")
    return validation_facts, tuned_facts


def _validation_tex(validation: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!t]\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        (r"\caption{D1 validation scan of the 20 fixed temporal ESPRIT configurations. "
         r"Scores are mean environment-by-horizon NMSE in dB across 24 validation environments "
         r"and four horizons; the score, with its frozen selection ordering, selected the displayed configuration. "
         r"Fallback rows are retained. No held-out test value was used for selection.}\label{tab:esprit-a3b-validation-scan}"),
        r"\begin{tabular}{rrrrrrl}", r"\toprule",
        r"Hankel rows & Rank & Gain-fit ridge & Mean validation NMSE (dB) & Fallback rows & Cells & Selection\\",
        r"\midrule",
    ]
    for row in validation:
        selected = r"\textbf{Selected}" if row["selected"] else ""
        lines.append(
            f"{row['hankel_rows']} & {row['rank']} & ${row['gain_fit_ridge']:.0e}$ & "
            f"{row['score_nmse_db']:.5f} & {row['fallback_rows']} & "
            f"{row['validation_cells']} & {selected} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _fmt_db_cell(value: float, fallback_rows: int) -> str:
    text = r"$-\infty$" if value == -math.inf else f"${value:.3f}$"
    return f"{text} ({fallback_rows})"


def _comparison_tex(comparison: list[dict[str, Any]]) -> str:
    by_key = {(row["speed_kmh"], row["horizon_steps"]): row for row in comparison}
    lines = [
        r"\begin{table}[!t]\centering\footnotesize",
        r"\setlength{\tabcolsep}{2.6pt}",
        (r"\caption{Complete paired D1 comparison of the original temporal ESPRIT setting "
         r"(rank 2, five Hankel rows) and the D1-validation-selected setting (rank 4, six rows; "
         r"gain-fit ridge $10^{-8}$). Each entry is mean environment-level NMSE in dB over the same "
         r"30 test environments, followed in parentheses by the total persistence-fallback prediction rows. "
         r"All scored rows, including fallbacks, are retained; repeated horizons reuse environments. "
         r"This is descriptive and adds no hypothesis tests or uncertainty intervals.}\label{tab:esprit-d1-rank-replay}"),
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Configuration & Speed (km/h) & 0.5 ms & 1 ms & 2 ms & 4 ms\\",
        r"\midrule",
    ]
    for speed in SPEEDS_KMH:
        for label in ("Original rank 2, rows 5", "Validation-selected rank 4, rows 6"):
            cells = []
            for horizon in HORIZONS:
                row = by_key[(speed, horizon)]
                mean = row["original_mean_nmse_db"] if label.startswith("Original") else row["tuned_mean_nmse_db"]
                fallback = row["original_fallback_rows"] if label.startswith("Original") else row["tuned_fallback_rows"]
                cells.append(_fmt_db_cell(mean, fallback))
            lines.append(f"{label} & {speed} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _build_table_map(validation: list[dict[str, Any]],
                     comparison: list[dict[str, Any]]) -> list[dict[str, str]]:
    maps: list[dict[str, str]] = []
    validation_path = "revision/paper/remaining_fixes_20261029/classical_analysis/d1_validation_selection/source_data.csv"
    comparison_path = "revision/paper/remaining_fixes_20261029/classical_analysis/integration_preparation/candidate_outputs_post_result_review/d1_esprit_comparison_source_data.csv"
    for row in validation:
        maps.append({
            "asset": "tables/esprit_a3b_validation_scan.tex",
            "row_selector": row["candidate_id"], "source_csv": validation_path,
            "fact_ids": f"a3b_esprit_validation_score_{row['candidate_id']}",
        })
    for row in comparison:
        h_ms = row["horizon_steps"] * 0.5
        fact_ids = [
            original_fact_id(row["speed_kmh"], h_ms),
            f"a3b_esprit_tuned_v{row['speed_kmh']}_h{h_ms:.1f}_mean_nmse_db",
            f"a3b_esprit_tuned_v{row['speed_kmh']}_h{h_ms:.1f}_fallback_rows",
        ]
        maps.append({
            "asset": "tables/esprit_d1_rank_replay.tex",
            "row_selector": f"speed{row['speed_kmh']}_h{row['horizon_steps']}",
            "source_csv": comparison_path,
            "fact_ids": ";".join(fact_ids),
        })
    return maps


def prepare_candidate_data(*, selection_verification: dict[str, Any],
                           selection_rows: list[dict[str, Any]], selection_csv_sha256: str,
                           d1_verification: dict[str, Any], d1_rows: list[dict[str, Any]],
                           d1_csv_sha256: str, d1_runner_sha256: str,
                           d1_protocol_sha256: str, old_verification: dict[str, Any],
                           old_rows: list[dict[str, Any]], old_csv_sha256: str,
                           existing_facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate evidence and compute candidate artifacts entirely in memory."""
    validation, selected = _validate_selection(
        selection_verification, selection_rows, selection_csv_sha256,
    )
    _validate_result_report(
        d1_verification, d1_csv_sha256=d1_csv_sha256, d1_row_count=len(d1_rows),
        d1_runner_sha256=d1_runner_sha256, d1_protocol_sha256=d1_protocol_sha256,
        selection_result_sha256=selection_verification["source_result_sha256"],
    )
    if (old_verification.get("complete") is not True
            or old_verification.get("files", {}).get("classical_d1_environment_metrics.csv") != old_csv_sha256):
        raise IntegrationGateError("Original ESPRIT baseline CSV does not match its frozen evidence manifest")

    bindings = d1_verification["source_bindings"]
    tuned = _validate_cell_rows(
        d1_rows, tuned=True, run_fingerprint=bindings["run_fingerprint"],
        generation_fingerprint=bindings["generation_fingerprint"],
        formal_protocol_sha256=bindings["formal_data_protocol_sha256"],
    )
    original_rows = [row for row in old_rows if row.get("source") == "esprit"]
    if len(original_rows) != 360:
        raise IntegrationGateError(f"Original ESPRIT D1 baseline must have 360 rows, found {len(original_rows)}")
    original = _validate_cell_rows(original_rows, tuned=False)
    _check_paired_baseline(original, tuned)

    original_stats: dict[tuple[int, int], dict[str, Any]] = {}
    tuned_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for speed in SPEEDS_KMH:
        for horizon in HORIZONS:
            original_stats[(speed, horizon)] = _group_stats(original, speed, horizon)
            tuned_stats[(speed, horizon)] = _group_stats(tuned, speed, horizon)
    _validate_original_facts(existing_facts, original_stats)

    comparison: list[dict[str, Any]] = []
    for speed in SPEEDS_KMH:
        for horizon in HORIZONS:
            prior, current = original_stats[(speed, horizon)], tuned_stats[(speed, horizon)]
            comparison.append({
                "speed_kmh": speed, "horizon_steps": horizon, "horizon_ms": horizon * 0.5,
                "n_environments": ENVS_PER_SPEED,
                "configuration_label": "Original rank 2, rows 5",
                "original_rank": ORIGINAL_CONFIG["rank"],
                "original_hankel_rows": ORIGINAL_CONFIG["hankel_rows"],
                "original_gain_fit_ridge": ORIGINAL_CONFIG["gain_fit_ridge"],
                "original_mean_nmse_db": prior["mean_nmse_db"],
                "original_mean_linear_nmse": prior["mean_linear_nmse"],
                "original_fallback_rows": prior["fallback_rows"],
                "tuned_rank": EXPECTED_CONFIG["rank"],
                "tuned_hankel_rows": EXPECTED_CONFIG["hankel_rows"],
                "tuned_gain_fit_ridge": EXPECTED_CONFIG["gain_fit_ridge"],
                "tuned_mean_nmse_db": current["mean_nmse_db"],
                "tuned_mean_linear_nmse": current["mean_linear_nmse"],
                "tuned_fallback_rows": current["fallback_rows"],
                "all_rows_including_fallbacks_scored": True,
            })
    validation_facts, tuned_facts = _build_facts(validation, comparison)
    table_map = _build_table_map(validation, comparison)
    comparison_source = [{key: value for key, value in row.items()
                          if key != "configuration_label"} for row in comparison]
    return {
        "validation": validation, "selected": selected,
        "comparison_rows": comparison,
        "validation_facts": validation_facts,
        "tuned_facts": tuned_facts,
        "comparison_source_rows": comparison_source,
        "table_map_rows": table_map,
        "validation_table_tex": _validation_tex(validation),
        "comparison_table_tex": _comparison_tex(comparison),
        "integration_plan": integration_plan_text(),
    }


def integration_plan_text() -> str:
    """Return evidence-matched review text proposals, without manuscript edits."""
    return """# A3b/D1 ESPRIT integration preparation

Status: candidate wording only. No manuscript, fact table, table map, or response source is modified by this tool.

## Main-text proposal (two sentences)

After formal test data had already been opened, a limited D1 validation-only follow-up selected temporal ESPRIT with rank 4, six Hankel rows, and gain-fit ridge $10^{-8}$. Compared with the original rank-2, five-row setting, the selected configuration had lower equal-environment mean NMSE-dB at all four horizons at 30 and 120 km/h, but higher means at all four horizons at 300 km/h (at 2 ms, $-5.58$ to $-23.17$ dB at 30 km/h and $0.75$ to $1.09$ dB at 300 km/h); the complete 12-cell comparison is reported in Supplementary Table~\\ref{tab:esprit-d1-rank-replay}.

## Reviewer 1 and merged-response proposal

The follow-up was limited and performed after formal test artifacts had previously been opened, so it is non-preregistered and non-blind. The rank-4, six-row configuration was selected using the D1 validation scan only; held-out test values were used only to evaluate that fixed configuration. Compared with the original rank-2, five-row setting, mean NMSE-dB was lower at all four horizons at 30 and 120 km/h, but higher at all four horizons at 300 km/h. At 2 ms, the original and selected means were $-5.58$ and $-23.17$ dB at 30 km/h, and $0.75$ and $1.09$ dB at 300 km/h. The supplementary comparison reports all three speeds and four horizons, with all 30 environments per cell and fallback counts; all prediction rows were retained and no fallback rows were triggered in either configuration. These descriptive results qualify the restricted-configuration score without establishing performance of the ESPRIT method family.

## Candidate SI tables

- `tab:esprit-a3b-validation-scan`: all 20 configurations and the validation score used for selection.
- `tab:esprit-d1-rank-replay`: original versus validation-selected ESPRIT, 3 speeds × 4 horizons, mean environment-level NMSE-dB and fallback row counts. No fallback rows were triggered.

The table values are descriptive. No p values, confidence intervals, or new inferential family are introduced. The existing D1 grid and cost-front evidence remain separate and unchanged.
"""


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key, "")) for key in fields})
    return stream.getvalue().encode("utf-8")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, float) and value == -math.inf:
        return "-Infinity"
    return value


def _comparison_csv_fields() -> list[str]:
    return [
        "speed_kmh", "horizon_steps", "horizon_ms", "n_environments",
        "original_rank", "original_hankel_rows", "original_gain_fit_ridge",
        "original_mean_nmse_db", "original_mean_linear_nmse", "original_fallback_rows",
        "tuned_rank", "tuned_hankel_rows", "tuned_gain_fit_ridge",
        "tuned_mean_nmse_db", "tuned_mean_linear_nmse", "tuned_fallback_rows",
        "all_rows_including_fallbacks_scored",
    ]


def _artifact_bytes(candidate: dict[str, Any]) -> dict[str, bytes]:
    validation_facts = candidate["validation_facts"]
    tuned_facts = candidate["tuned_facts"]
    table_map = candidate["table_map_rows"]
    facts = validation_facts + tuned_facts
    return {
        "validation_facts_candidate.csv": _csv_bytes(validation_facts, FACT_FIELDS),
        "tuned_heldout_facts_candidate.csv": _csv_bytes(tuned_facts, FACT_FIELDS),
        "facts_candidate.csv": _csv_bytes(facts, FACT_FIELDS),
        "d1_esprit_comparison_source_data.csv": _csv_bytes(
            candidate["comparison_source_rows"], _comparison_csv_fields()),
        "table_statistic_map_candidate.csv": _csv_bytes(table_map, TABLE_MAP_FIELDS),
        "esprit_a3b_validation_scan.tex": candidate["validation_table_tex"].encode("utf-8"),
        "esprit_d1_rank_replay.tex": candidate["comparison_table_tex"].encode("utf-8"),
        "integration_plan.md": candidate["integration_plan"].encode("utf-8"),
    }


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace existing candidate artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
    temporary.replace(path)


def load_and_prepare(project_root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve(strict=True)
    if root != ROOT.resolve(strict=True):
        raise IntegrationGateError("Preparation is bound to the current project root")

    selection_verify_path = SELECTION_DIR / "verification.json"
    selection_csv_path = SELECTION_DIR / "source_data.csv"
    if sha256_file(selection_verify_path) != SELECTION_VERIFICATION_SHA256:
        raise IntegrationGateError("Root-verified A3b selection verification changed; review it again")
    selection_verification = _read_json(selection_verify_path, "A3b root verification")
    selection_csv_sha = sha256_file(selection_csv_path)
    selection_rows = _read_csv(selection_csv_path, "A3b validation source data")

    d1_report_path = D1_AUDIT_DIR / "verification.json"
    d1_csv_path = D1_AUDIT_DIR / "source_data.csv"
    d1_verification = _read_json(d1_report_path, "D1 strict-audit report")
    d1_csv_sha = sha256_file(d1_csv_path)
    d1_rows = _read_csv(d1_csv_path, "D1 strict-audit source data")
    d1_runner_sha = sha256_file(D1_RUNNER_PATH)
    d1_protocol_sha = sha256_file(D1_PROTOCOL_PATH)

    if sha256_file(BASELINE_VERIFY_PATH) != BASELINE_VERIFICATION_SHA256:
        raise IntegrationGateError("Original D1 baseline verification manifest changed")
    old_verification = _read_json(BASELINE_VERIFY_PATH, "original D1 evidence manifest")
    old_csv_sha = sha256_file(BASELINE_DATA_PATH)
    old_rows = _read_csv(BASELINE_DATA_PATH, "original D1 environment metrics")

    facts_rows = _read_csv(EXISTING_FACTS_PATH, "current fact ledger")
    existing_facts = {row["fact_id"]: row for row in facts_rows}
    if len(existing_facts) != len(facts_rows):
        raise IntegrationGateError("Current fact ledger has duplicate fact identifiers")

    candidate = prepare_candidate_data(
        selection_verification=selection_verification,
        selection_rows=selection_rows,
        selection_csv_sha256=selection_csv_sha,
        d1_verification=d1_verification, d1_rows=d1_rows,
        d1_csv_sha256=d1_csv_sha, d1_runner_sha256=d1_runner_sha,
        d1_protocol_sha256=d1_protocol_sha,
        old_verification=old_verification, old_rows=old_rows,
        old_csv_sha256=old_csv_sha, existing_facts=existing_facts,
    )
    bindings = {
        "selection_verification_sha256": sha256_file(selection_verify_path),
        "selection_csv_sha256": selection_csv_sha,
        "selection_result_sha256": selection_verification["source_result_sha256"],
        "selection_protocol_sha256": selection_verification["protocol_sha256"],
        "d1_strict_audit_sha256": sha256_file(d1_report_path),
        "d1_source_data_sha256": d1_csv_sha,
        "d1_runner_sha256": d1_runner_sha,
        "d1_protocol_sha256": d1_protocol_sha,
        "d1_run_fingerprint": d1_verification["source_bindings"]["run_fingerprint"],
        "d1_generation_fingerprint": d1_verification["source_bindings"]["generation_fingerprint"],
        "original_baseline_verification_sha256": sha256_file(BASELINE_VERIFY_PATH),
        "original_baseline_csv_sha256": old_csv_sha,
        "existing_fact_ledger_sha256": sha256_file(EXISTING_FACTS_PATH),
        "input_test_mat_opened_by_preparation": 0,
        "manuscript_modified": False,
    }
    return candidate, bindings


def write_candidate_artifacts(candidate: dict[str, Any], bindings: dict[str, Any],
                              output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    outputs = _artifact_bytes(candidate)
    output_dir = Path(output_dir).resolve()
    if output_dir != OUTPUT_DIR.resolve():
        raise IntegrationGateError("Candidate outputs must remain in the independent preparation directory")
    output_hashes = {name: sha256_bytes(content) for name, content in outputs.items()}
    report = {
        "schema": "a3b-d1-esprit-integration-preparation-v1",
        "status": "prepared_for_root_review_not_integrated",
        "input_bindings": bindings,
        "coverage": {
            "validation_candidates": len(candidate["validation"]),
            "validation_cells": 1920,
            "heldout_environments": 90,
            "heldout_cells": 360,
            "heldout_groups_recomputed": 12,
            "heldout_comparison_rows": len(candidate["comparison_rows"]),
            "heldout_cells_per_speed_horizon": 30,
        },
        "selection": {
            "rank": EXPECTED_CONFIG["rank"],
            "hankel_rows": EXPECTED_CONFIG["hankel_rows"],
            "gain_fit_ridge": EXPECTED_CONFIG["gain_fit_ridge"],
            "validation_score_nmse_db": candidate["selected"]["selection_score_nmse_db"],
            "split": "D1 validation only",
            "test_used_for_selection": False,
        },
        "summary_policy": {
            "heldout_estimate": "equal-environment mean of environment NMSE-dB, n=30 per speed/horizon",
            "fallback_rows_included_in_scored_result": True,
            "fallback_counts_reported_for_both_configurations": True,
            "new_p_values_or_confidence_intervals": False,
            "existing_D1_grid_and_cost_front_replaced": False,
        },
        "planned_paper_locations": {
            "main_text": "sections/06_results_ab.tex, two sentences only",
            "supplementary_tables": [
                "tables/esprit_a3b_validation_scan.tex",
                "tables/esprit_d1_rank_replay.tex",
            ],
            "reviewer_response": "R1-M5 in response_reviewer_1.tex and reply_ab.tex",
        },
        "generated_artifacts": output_hashes,
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for name, content in outputs.items():
        _write_immutable(output_dir / name, content)
    _write_immutable(output_dir / "verification.json", report_bytes)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-candidates", action="store_true",
                        help="prepare independent candidate data only after both strict evidence gates pass")
    args = parser.parse_args(argv)
    if not args.prepare_candidates:
        parser.error("This source-only preparation requires explicit --prepare-candidates")
    try:
        candidate, bindings = load_and_prepare()
        report = write_candidate_artifacts(candidate, bindings)
    except IntegrationGateError as exc:
        print(f"NOT_READY: {exc}")
        return 2
    print(json.dumps({
        "status": report["status"],
        "output_directory": str(OUTPUT_DIR),
        "validation_candidates": report["coverage"]["validation_candidates"],
        "heldout_cells": report["coverage"]["heldout_cells"],
        "manuscript_modified": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
