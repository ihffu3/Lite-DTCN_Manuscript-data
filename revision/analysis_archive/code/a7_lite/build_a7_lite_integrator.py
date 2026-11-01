#!/usr/bin/env python
"""Recompute and integrate the approved A7b fresh Lite history results."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REMAINING = ROOT / "revision/paper/remaining_fixes_20261029"
REPLAY = (REMAINING / "fixed_checkpoint_replay_v1/results/lite_history/"
          "53928327483e7826951a8b7cdc2fc5a2dbac0b44c57868be5ce119dba232c355")
REVIEW_PATH = REMAINING / "audit/ROOT_A7_LITE_RESULT_REVIEW.json"
ROOT_POINTS_PATH = REMAINING / "audit/A7_LITE_POINTS_ROOT.json"
TRAINING_PROTOCOL_PATH = (REMAINING / "training/a7b_lite_history_steps4_v1/"
                           "training_protocol_v1.json")
SOURCE_CODE = ROOT / "revision/paper/figure_redraw_20261029"
FACTS_PATH = SOURCE_CODE / "analysis/facts.csv"
STAT_MAP_PATH = SOURCE_CODE / "analysis/table_statistic_map.csv"
POINT_MAP_PATH = SOURCE_CODE / "analysis/point_map.csv"
MAIN_PATH = SOURCE_CODE / "src/sections/06_results_ab.tex"
SI_PATH = SOURCE_CODE / "src/supplementary.tex"
RESPONSE_PATH = SOURCE_CODE / "src/response_reviewer_1.tex"
REPLY_PATH = SOURCE_CODE / "src/reply_ab.tex"
TABLE_PATH = SOURCE_CODE / "src/tables/a7b_lite_paired.tex"
PARAMETER_COUNT_PATH = HERE / "a7_lite_parameter_counts.csv"
BACKUP_DIR = SOURCE_CODE / "backups/a7b_lite_preintegration_20261029"
SPEEDS = (30, 120, 300)
HISTORIES = (2, 5, 10)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED_BASE = 20_261_029
PRETEST_SHA256 = "53928327483e7826951a8b7cdc2fc5a2dbac0b44c57868be5ce119dba232c355"
FACT_FIELDS = ["fact_id", "quantity", "method", "speed_kmh", "horizon_ms", "snr_db",
               "value", "unit", "aggregation", "source_file", "source_selector"]
MAP_FIELDS = ["asset", "row_selector", "source_csv", "fact_ids"]
POINT_FIELDS = ["speed_kmh", "history_length", "run_id", "n_environments",
                "mean_nmse_db", "ci95_low_db", "ci95_high_db", "bootstrap_replicates",
                "bootstrap_seed", "resampling_unit"]
PAIRED_FIELDS = ["speed_kmh", "contrast", "history_length", "reference_history_length",
                 "mean_delta_nmse_db", "ci95_low_db", "ci95_high_db",
                 "n_paired_environments", "bootstrap_replicates", "bootstrap_seed",
                 "interpretation"]
PARAMETER_COUNT_FIELDS = ["history_length", "trainable_parameters", "model", "n_tx",
                          "n_freq", "node_dim", "temporal_hidden", "residual",
                          "source_file", "source_sha256"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=atol):
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _fmt(value: float) -> str:
    return format(float(value), ".17g")


def compute_results() -> dict:
    """Verify frozen inputs and independently recompute all means and intervals."""
    import numpy as np

    review = _read_json(REVIEW_PATH)
    root_points = _read_json(ROOT_POINTS_PATH)
    protocol = _read_json(TRAINING_PROTOCOL_PATH)
    identity_path = REPLAY / "run_identity.json"
    manifest_path = REPLAY / "run_manifest.json"
    summary_path = REPLAY / "summary.json"
    per_env_path = REPLAY / "per_environment.csv"
    identity = _read_json(identity_path)
    summary = _read_json(summary_path)

    if review.get("schema") != "a7b-lite-heldout-result-review-v1" or review.get("status") != "approved_for_figure_render":
        raise PermissionError("A7 Lite results lack the required root held-out-result approval")
    if review.get("reviewed_by") != "root":
        raise PermissionError("A7 Lite result review is not recorded as a root review")
    if identity.get("pretest_manifest_sha256") != PRETEST_SHA256:
        raise ValueError("Replay identity is bound to a different pretest manifest")
    if review.get("pretest_manifest_sha256") != PRETEST_SHA256:
        raise ValueError("Root review is bound to a different pretest manifest")
    input_hashes = {
        "pretest_manifest_sha256": PRETEST_SHA256,
        "run_identity_sha256": sha256(identity_path),
        "run_manifest_sha256": sha256(manifest_path),
        "summary_sha256": sha256(summary_path),
        "per_environment_csv_sha256": sha256(per_env_path),
        "root_review_sha256": sha256(REVIEW_PATH),
        "root_points_sha256": sha256(ROOT_POINTS_PATH),
        "training_protocol_sha256": sha256(TRAINING_PROTOCOL_PATH),
    }
    for review_key, hash_key in (
        ("run_manifest_sha256", "run_manifest_sha256"),
        ("summary_sha256", "summary_sha256"),
        ("per_environment_csv_sha256", "per_environment_csv_sha256"),
    ):
        if review.get(review_key) != input_hashes[hash_key]:
            raise ValueError(f"Root result review no longer matches {review_key}")

    if protocol.get("schema") != "remaining-reviewer-a7b-lite-history-protocol-v1":
        raise ValueError("Unexpected A7 Lite training protocol")
    run_specs = protocol.get("run_plan", {}).get("fresh_training_runs", [])
    expected_ids = {f"lite_fullwindow_s{speed}_L{history}_steps4_seed42"
                    for speed in SPEEDS for history in HISTORIES}
    if {row.get("run_id") for row in run_specs} != expected_ids or len(run_specs) != 9:
        raise ValueError("A7 protocol does not define exactly the nine reviewed fresh fits")
    if protocol["training_recipe"].get("seed") != 42 or protocol["horizon_convention"].get("horizon_steps") != 4:
        raise ValueError("A7 result recipe or horizon differs from the frozen protocol")
    if protocol["historical_test_disclosure"].get("formal_test_previously_accessed") is not True:
        raise ValueError("Expected historical formal-test access disclosure is missing")

    with per_env_path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if len(raw_rows) != 270:
        raise ValueError(f"Expected 270 complete A7 Lite result rows, found {len(raw_rows)}")
    by_cell: dict[tuple[int, int], list[dict]] = {(s, h): [] for s in SPEEDS for h in HISTORIES}
    seen = set()
    for source in raw_rows:
        speed, history = int(source["speed_kmh"]), int(source["history_length"])
        run_id = source["run_id"]
        expected_run_id = f"lite_fullwindow_s{speed}_L{history}_steps4_seed42"
        if (speed, history) not in by_cell or run_id != expected_run_id:
            raise ValueError(f"Unexpected A7 result row identity: {run_id}")
        if (source.get("family") != "lite_history" or source.get("condition") != "C0"):
            raise ValueError(f"Unexpected family/condition for {run_id}")
        key = (run_id, source["environment_id"])
        if key in seen:
            raise ValueError(f"Duplicate A7 Lite environment row: {key}")
        seen.add(key)
        row = dict(source)
        for name in ("n_origins", "n_rows"):
            row[name] = int(source[name])
        for name in ("error_energy", "target_energy", "prediction_energy", "nmse", "nmse_db"):
            row[name] = float(source[name])
            if not math.isfinite(row[name]):
                raise ValueError(f"Non-finite {name} for {key}")
        if row["target_energy"] <= 0 or row["error_energy"] < 0 or row["nmse"] <= 0:
            raise ValueError(f"Invalid physical-energy metric for {key}")
        _close(row["nmse"], row["error_energy"] / row["target_energy"], f"NMSE energy identity {key}", 2e-12)
        _close(row["nmse_db"], 10.0 * math.log10(row["nmse"]), f"NMSE dB identity {key}", 2e-12)
        by_cell[(speed, history)].append(row)

    for speed in SPEEDS:
        reference = None
        for history in HISTORIES:
            rows = sorted(by_cell[(speed, history)], key=lambda row: row["environment_id"])
            if len(rows) != 30 or len({row["environment_id"] for row in rows}) != 30:
                raise ValueError(f"Expected 30 unique environments for speed={speed}, L={history}")
            ids = [row["environment_id"] for row in rows]
            if reference is None:
                reference = ids
            elif ids != reference:
                raise ValueError(f"A7 length conditions are not paired on the same environments at speed={speed}")
            if any(row["n_origins"] != 3978 or row["n_rows"] != 15912 for row in rows):
                raise ValueError(f"A7 per-environment origin/row counts differ for speed={speed}, L={history}")
            if history == HISTORIES[0]:
                ref_identity = {
                    row["environment_id"]: (row["mat_sha256"], row["job_sha256"],
                                             row["qa_receipt_sha256"], row["target_energy"])
                    for row in rows
                }
            else:
                for row in rows:
                    expected_identity = ref_identity[row["environment_id"]]
                    row_identity = (row["mat_sha256"], row["job_sha256"],
                                    row["qa_receipt_sha256"], row["target_energy"])
                    if row_identity[:3] != expected_identity[:3]:
                        raise ValueError(f"MAT/job/QA identity differs across lengths for {row['environment_id']}")
                    _close(row_identity[3], expected_identity[3], f"target energy {row['environment_id']}", 2e-12)

    points, paired = [], []
    for speed in SPEEDS:
        data: dict[int, tuple[list[dict], object]] = {}
        for history in HISTORIES:
            rows = sorted(by_cell[(speed, history)], key=lambda row: row["environment_id"])
            values = np.asarray([row["nmse_db"] for row in rows], dtype=np.float64)
            if values.shape != (30,) or not np.isfinite(values).all():
                raise ValueError("A7 Lite point grid is incomplete or non-finite")
            data[history] = (rows, values)
        ids = [row["environment_id"] for row in data[2][0]]
        seed = BOOTSTRAP_SEED_BASE + speed
        rng = np.random.Generator(np.random.PCG64(seed))
        sample_indices = rng.integers(0, len(ids), size=(BOOTSTRAP_N, len(ids)))
        for history in HISTORIES:
            rows, values = data[history]
            mean = math.fsum(float(value) for value in values) / len(values)
            boot_means = values[sample_indices].mean(axis=1)
            low, high = np.quantile(boot_means, [0.025, 0.975], method="linear")
            run_id = rows[0]["run_id"]
            summary_row = summary.get("models", {}).get(run_id)
            if not summary_row:
                raise ValueError(f"Fixed replay summary lacks {run_id}")
            _close(mean, summary_row["mean_environment_nmse_db"], f"fixed summary mean {run_id}")
            point = {
                "speed_kmh": speed, "history_length": history, "run_id": run_id,
                "n_environments": len(ids), "mean_nmse_db": mean,
                "ci95_low_db": float(low), "ci95_high_db": float(high),
                "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": seed,
                "resampling_unit": "environment_id; paired across L within speed",
            }
            points.append(point)

        anchor = data[2][1]
        for history in (5, 10):
            delta = data[history][1] - anchor
            mean_delta = math.fsum(float(value) for value in delta) / len(delta)
            boot_delta = delta[sample_indices].mean(axis=1)
            low, high = np.quantile(boot_delta, [0.025, 0.975], method="linear")
            contrast = f"L{history} minus L2"
            fixed = [row for row in summary.get("paired_descriptive_contrasts", [])
                     if row.get("speed_kmh") == speed and row.get("contrast") == contrast]
            if len(fixed) != 1:
                raise ValueError(f"Fixed summary does not contain one {contrast} contrast for {speed}")
            _close(mean_delta, fixed[0]["mean_delta_nmse_db"], f"fixed paired mean {speed} {contrast}")
            paired.append({
                "speed_kmh": speed, "contrast": contrast,
                "history_length": history, "reference_history_length": 2,
                "mean_delta_nmse_db": mean_delta, "ci95_low_db": float(low),
                "ci95_high_db": float(high), "n_paired_environments": len(ids),
                "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": seed,
                "interpretation": "paired descriptive contrast; one training seed; no inferential test",
            })

    if len(points) != 9 or len(paired) != 6:
        raise AssertionError("A7 Lite integration grid changed unexpectedly")
    _assert_matches_root_points(points, paired, root_points)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from revision.code.models.candidates import FullWindowLiteDTCN

    expected_counts = protocol["model"]["expected_parameter_count"]
    parameter_counts = {}
    for history in HISTORIES:
        model = FullWindowLiteDTCN(
            history_length=history, n_tx=8, n_freq=32, node_dim=16,
            temporal_hidden=32, residual=False,
        )
        count = sum(parameter.numel() for parameter in model.parameters())
        expected = expected_counts[f"L{history}"]
        if count != expected:
            raise ValueError(f"Parameter count L{history}={count}, protocol expects {expected}")
        parameter_counts[f"L{history}"] = count
    model_source_path = ROOT / "revision/code/models/candidates.py"
    input_hashes["parameter_model_source_sha256"] = sha256(model_source_path)

    point_map_validation = _validate_merged_point_map(points)
    input_hashes["merged_point_map_sha256"] = point_map_validation["sha256"]

    return {
        "points": points, "paired": paired, "parameter_counts": parameter_counts,
        "root_points": root_points, "summary": summary, "protocol": protocol,
        "hashes": input_hashes, "point_map_validation": point_map_validation,
    }


def _assert_matches_root_points(points: list[dict], paired: list[dict], root_points: dict) -> None:
    expected_points = root_points.get("points", [])
    expected_pairs = root_points.get("contrasts", [])
    if len(expected_points) != 9 or len(expected_pairs) != 6:
        raise ValueError("Root A7 points file must contain nine means and six paired contrasts")
    for expected in expected_points:
        match = [row for row in points if (row["speed_kmh"], row["history_length"])
                 == (expected["speed_kmh"], expected["history_length"])]
        if len(match) != 1:
            raise ValueError("Root point grid has a missing or duplicate cell")
        for key in ("mean_nmse_db", "ci95_low_db", "ci95_high_db"):
            _close(match[0][key], expected[key], f"root point {expected['speed_kmh']} L{expected['history_length']} {key}")
    for expected in expected_pairs:
        match = [row for row in paired if (row["speed_kmh"], row["contrast"])
                 == (expected["speed_kmh"], expected["contrast"])]
        if len(match) != 1:
            raise ValueError("Root paired grid has a missing or duplicate contrast")
        for key in ("mean_delta_nmse_db", "ci95_low_db", "ci95_high_db"):
            _close(match[0][key], expected[key], f"root pair {expected['speed_kmh']} {expected['contrast']} {key}")


def _validate_merged_point_map(points: list[dict]) -> dict:
    with POINT_MAP_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        f"a7b_lite_mean_nmse_v{point['speed_kmh']}_L{point['history_length']}": point
        for point in points
    }
    selected = [row for row in rows if row.get("source_fact_id") in expected]
    if len(selected) != 9 or {row.get("source_fact_id") for row in selected} != set(expected):
        raise ValueError("Merged point_map.csv must contain exactly the nine A7 Lite mean fact IDs")
    for row in selected:
        fact_id = row["source_fact_id"]
        point = expected[fact_id]
        _close(float(row["y_value"]), point["mean_nmse_db"], f"merged point map value {fact_id}")
        if row["x_value"] != str(point["history_length"]):
            raise ValueError(f"Merged point-map history length differs for {fact_id}")
        if row["panel"] != f"lite_speed_{point['speed_kmh']}":
            raise ValueError(f"Merged point-map panel differs for {fact_id}")
        if row["figure_asset"] != "a7b_history_length.pdf":
            raise ValueError(f"Merged point-map figure differs for {fact_id}")
    return {
        "path": POINT_MAP_PATH.relative_to(ROOT).as_posix(),
        "sha256": sha256(POINT_MAP_PATH),
        "a7_lite_mean_rows_verified": len(selected),
    }


def build_fact_rows(result: dict) -> list[dict]:
    point_source = "revision/paper/remaining_fixes_20261029/a7_lite_integration/a7_lite_points.csv"
    pair_source = "revision/paper/remaining_fixes_20261029/a7_lite_integration/a7_lite_paired.csv"
    rows = []
    for point in result["points"]:
        speed, history = point["speed_kmh"], point["history_length"]
        common = {
            "method": "a7b_lite_history", "speed_kmh": str(float(speed)),
            "horizon_ms": "2.0", "snr_db": "", "unit": "dB",
            "source_file": point_source,
        }
        for stat, quantity, aggregation, value in (
            ("mean", "nmse_db_env_mean", "equal_environment_mean_seed42", point["mean_nmse_db"]),
            ("ci95_low", "nmse_db_ci95_low", "pointwise_environment_bootstrap_95", point["ci95_low_db"]),
            ("ci95_high", "nmse_db_ci95_high", "pointwise_environment_bootstrap_95", point["ci95_high_db"]),
        ):
            rows.append({
                "fact_id": f"a7b_lite_{stat}_nmse_v{speed}_L{history}",
                "quantity": quantity, "aggregation": aggregation, "value": _fmt(value),
                "source_selector": f"run_id={point['run_id']};stat={stat};seed=42",
                **common,
            })
    for pair in result["paired"]:
        speed, history = pair["speed_kmh"], pair["history_length"]
        slug = f"v{speed}_L{history}_minus_L2"
        common = {
            "method": "a7b_lite_history_paired", "speed_kmh": str(float(speed)),
            "horizon_ms": "2.0", "snr_db": "", "unit": "dB",
            "source_file": pair_source,
        }
        for stat, quantity, aggregation, value in (
            ("mean", "paired_delta_nmse_db_mean", "paired_environment_mean_seed42", pair["mean_delta_nmse_db"]),
            ("ci95_low", "paired_delta_nmse_db_ci95_low", "paired_environment_bootstrap_95", pair["ci95_low_db"]),
            ("ci95_high", "paired_delta_nmse_db_ci95_high", "paired_environment_bootstrap_95", pair["ci95_high_db"]),
        ):
            rows.append({
                "fact_id": f"a7b_lite_paired_delta_{stat}_nmse_{slug}",
                "quantity": quantity, "aggregation": aggregation, "value": _fmt(value),
                "source_selector": f"speed_kmh={speed};contrast={pair['contrast']};stat={stat};seed=42",
                **common,
            })
    parameter_source = "revision/paper/remaining_fixes_20261029/a7_lite_integration/a7_lite_parameter_counts.csv"
    for history in HISTORIES:
        rows.append({
            "fact_id": f"a7b_lite_trainable_parameters_L{history}",
            "quantity": "trainable_parameters",
            "method": "a7b_lite_history",
            "speed_kmh": "",
            "horizon_ms": "",
            "snr_db": "",
            "value": str(result["parameter_counts"][f"L{history}"]),
            "unit": "parameters",
            "aggregation": "constructed_model_parameter_count",
            "source_file": parameter_source,
            "source_selector": (
                f"history_length={history};model=FullWindowLiteDTCN;n_tx=8;n_freq=32;"
                "node_dim=16;temporal_hidden=32;residual=False"
            ),
        })
    if len(rows) != 48 or len({row["fact_id"] for row in rows}) != 48:
        raise AssertionError("A7 Lite facts must comprise 45 result statistics and three parameter counts")
    return rows


def build_table_map_rows(result: dict) -> list[dict]:
    rows = []
    pair_source = "revision/paper/remaining_fixes_20261029/a7_lite_integration/a7_lite_paired.csv"
    for pair in result["paired"]:
        speed, history = pair["speed_kmh"], pair["history_length"]
        slug = f"v{speed}_L{history}_minus_L2"
        rows.append({
            "asset": "tables/a7b_lite_paired.tex",
            "row_selector": slug,
            "source_csv": pair_source,
            "fact_ids": ";".join((
                f"a7b_lite_paired_delta_mean_nmse_{slug}",
                f"a7b_lite_paired_delta_ci95_low_nmse_{slug}",
                f"a7b_lite_paired_delta_ci95_high_nmse_{slug}",
            )),
        })
    return rows


def _csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_points_csv_rows(result: dict) -> list[dict]:
    return [{key: (str(row[key]) if isinstance(row[key], int) else _fmt(row[key])
                   if isinstance(row[key], float) else row[key])
             for key in POINT_FIELDS} for row in result["points"]]


def build_paired_csv_rows(result: dict) -> list[dict]:
    return [{key: (str(row[key]) if isinstance(row[key], int) else _fmt(row[key])
                   if isinstance(row[key], float) else row[key])
             for key in PAIRED_FIELDS} for row in result["paired"]]


def build_parameter_count_rows(result: dict) -> list[dict]:
    source_file = "revision/code/models/candidates.py"
    source_sha = result["hashes"]["parameter_model_source_sha256"]
    return [{
        "history_length": str(history),
        "trainable_parameters": str(result["parameter_counts"][f"L{history}"]),
        "model": "FullWindowLiteDTCN",
        "n_tx": "8",
        "n_freq": "32",
        "node_dim": "16",
        "temporal_hidden": "32",
        "residual": "False",
        "source_file": source_file,
        "source_sha256": source_sha,
    } for history in HISTORIES]


def build_paired_table(result: dict) -> str:
    lines = [
        r"\begin{table*}[!t]\centering\footnotesize\setlength{\tabcolsep}{6pt}",
        r"\caption{Paired fresh Lite-DTCN history-length differences at 2~ms (seed 42). Each length is compared with $L=2$ over the same 30 test environments at each speed. Brackets give descriptive pointwise 95\% percentile intervals from 10,000 paired environment resamples using PCG64 seed $20261029+v$. No new hypothesis test is introduced.}",
        r"\label{tab:a7b-lite-paired}",
        r"\begin{tabular}{llrr}\toprule",
        r"Speed & Contrast & Mean difference (dB) & Pointwise 95\% CI (dB)\\",
        r"\midrule",
    ]
    for pair in result["paired"]:
        speed, history = pair["speed_kmh"], pair["history_length"]
        mean = pair["mean_delta_nmse_db"]
        low, high = pair["ci95_low_db"], pair["ci95_high_db"]
        lines.append(
            f"{speed}~km/h & $L={history}$ $-$ $L=2$ & {mean:+.3f} & $[{low:+.3f},\,{high:+.3f}]$" + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


MAIN_OLD = ("A separately fitted PC-Wiener scan over two to ten taps gives lower mean error with longer histories at all three speeds "
            "(Supplementary Fig.~\\ref{fig:a7b-history-length}); it does not establish an optimal neural history length.")
MAIN_NEW = (
    "A separately fitted PC-Wiener scan over two to ten taps gives lower mean error with longer histories at all three speeds "
    "(Supplementary Fig.~\\ref{fig:a7b-history-length}). A fresh Lite-DTCN refit at $L=2,5,10$ gives lower mean NMSE at "
    "$L=10$ than at $L=2$ by 2.36, 0.42 and 0.22~dB at 30, 120 and 300~km/h, respectively. At 30~km/h, however, $L=5$ "
    "is worse than $L=2$ ($-5.93$ versus $-6.57$~dB); these finite single-seed refits do not establish an optimal history "
    "length (Supplementary Fig.~\\ref{fig:a7b-history-length}; Table~\\ref{tab:a7b-lite-paired})."
)
SI_METHOD_MARKER = r"\paragraph{Fresh neural history-length scan}"
SI_METHOD = (
    r"\paragraph{Fresh neural history-length scan}"
    "\nFresh Lite-DTCN fits were trained independently at $L=2,5,10$ and 30, 120 and 300~km/h (one seed, 42, per cell). "
    "The model was \\texttt{FullWindowLiteDTCN} with $n_{\\rm tx}=8$, $n_{\\rm freq}=32$, node dimension 16, temporal hidden "
    "dimension 32 and no residual; the resulting trainable parameter counts are 22,448, 22,832 and 23,472 for $L=2,5,10$. "
    "Only history length changes, but this also changes parameter count. All fits use the same direct four-step (2-ms) target, "
    "C0 normalized real/imaginary MSE, Adam at learning rate 0.001, batch size 128, 32 training steps per epoch, no scheduler, "
    "a 100-epoch cap and patience 15. Training uses the 30 speed-matched D1 training environments with stride-one origins; "
    "one RI mean and standard deviation are fitted on the 90 D1 training environments only. Checkpoints are selected by the "
    "strict minimum equal-environment physical linear NMSE over the eight speed-matched D1 validation environments; test "
    "metrics do not select checkpoints or alter the recipe.\n"
    "After all checkpoint choices were fixed, each model was evaluated on the same 30 speed-matched D1 test environments, "
    "using the same test origins starting at index 19 across lengths. The test files had been accessed in earlier work, so "
    "this comparison is post hoc and non-blind. Figure~\\ref{fig:a7b-history-length} shows the nine equal-environment mean "
    "NMSE-dB values without repeating them in a table. The paired differences in Table~\\ref{tab:a7b-lite-paired} are $L=5$ "
    "or $L=10$ minus $L=2$ within each speed. Pointwise 95\\% intervals resample the 30 environment IDs with replacement "
    "10,000 times (NumPy PCG64 seed $20261029+v$ for speed $v$ km/h), using the same sampled IDs across lengths within each "
    "speed. These descriptive intervals introduce no p-values or new inferential family. One fitted seed per cell and the "
    "changing parameter count limit conclusions to this finite recipe-specific scan; it does not establish an optimal length, "
    "a stable ranking, or the contribution of every lag."
)
SI_TABLE_INPUT = r"\input{tables/a7b_lite_paired}"
CAPTION_OLD = (r"\caption{PC-Wiener history-length scan at a 2-ms horizon. Each panel has an independent vertical scale. Points average "
               r"environment-level NMSE-dB over 30 test environments per speed; bars are post-hoc pointwise 95\% percentile "
               r"intervals from 10,000 environment resamples (PCG64 seed 20261031), with the same sampled identities across "
               r"lengths within speed. These descriptive intervals do not expand the original inferential families.}")
CAPTION_NEW = (r"\caption{History-length scans at a direct 2-ms horizon. Panels (a)--(c) show PC-Wiener and (d)--(f) fresh seed-42 "
               r"Lite-DTCN refits; columns show 30, 120 and 300~km/h. Points are means over 30 test environments and bars are "
               r"descriptive pointwise 95\% environment-bootstrap intervals. Each panel has an independent vertical scale.}")
REPLY_OLD = ("A neural history-length sweep remains incomplete, and neither the classical scan nor fixed-weight perturbations "
             "prove the usefulness of every neural lag or an optimal neural history length.")
REPLY_NEW = (
    "The fresh Lite-DTCN scan gives lower mean NMSE at $L=10$ than at $L=2$ by 2.36, 0.42 and 0.22~dB at 30, 120 and "
    "300~km/h; at 30~km/h, $L=5$ is worse than $L=2$ ($-5.93$ versus $-6.57$~dB) (Supplementary Fig.~"
    r"\ref{s-fig:a7b-history-length}; Table~\ref{tab:a7b-lite-paired}). These are finite single-seed refits and do not "
    "establish an optimal neural history length, stable ranking, or the usefulness of every lag."
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1)
    if count == 0 and new in text:
        return text
    raise ValueError(f"Expected exactly one {label} target or an already integrated replacement; found {count}")


def _backup_original_sources() -> dict:
    source_files = [FACTS_PATH, STAT_MAP_PATH, MAIN_PATH, SI_PATH, RESPONSE_PATH, REPLY_PATH]
    backup_manifest_path = BACKUP_DIR / "backup_manifest.json"
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if backup_manifest_path.exists():
        manifest = _read_json(backup_manifest_path)
        for item in manifest["files"]:
            backup_path = BACKUP_DIR / item["backup_file"]
            if item["existed"]:
                if not backup_path.is_file() or sha256(backup_path) != item["sha256"]:
                    raise ValueError("A7 pre-integration backup no longer matches its manifest")
        return manifest

    items = []
    for source in source_files:
        if not source.is_file():
            raise FileNotFoundError(f"Required manuscript source is missing: {source}")
        name = source.relative_to(SOURCE_CODE).as_posix().replace("/", "__")
        target = BACKUP_DIR / name
        shutil.copyfile(source, target)
        items.append({
            "source_file": source.relative_to(ROOT).as_posix(),
            "backup_file": name,
            "existed": True,
            "sha256": sha256(target),
        })
    items.append({
        "source_file": TABLE_PATH.relative_to(ROOT).as_posix(),
        "backup_file": "src__tables__a7b_lite_paired.tex",
        "existed": TABLE_PATH.exists(),
        "sha256": sha256(TABLE_PATH) if TABLE_PATH.exists() else None,
    })
    manifest = {"schema": "a7b-lite-preintegration-backup-v1", "files": items}
    backup_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _append_rows(path: Path, fields: list[str], new_rows: list[dict], key_field: str | tuple[str, ...]) -> int:
    existing_bytes = path.read_bytes()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected CSV header in {path}: {reader.fieldnames}")
        existing_rows = list(reader)
    by_key = {}
    key_of = (lambda row: tuple(row[field] for field in key_field)
              if isinstance(key_field, tuple) else row[key_field])
    for row in existing_rows:
        key = key_of(row)
        by_key.setdefault(key, row)
    missing = []
    for expected in new_rows:
        key = key_of(expected)
        previous = by_key.get(key)
        if previous is None:
            missing.append(expected)
        elif any(previous.get(field, "") != str(expected.get(field, "")) for field in fields):
            raise ValueError(f"Existing A7 integration row conflicts for {key!r} in {path}")
    if not missing:
        return 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        if existing_bytes and not existing_bytes.endswith((b"\n", b"\r")):
            stream.write("\n")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writerows(missing)
    return len(missing)


def _prepare_text_outputs(result: dict) -> dict[Path, str]:
    main = MAIN_PATH.read_text(encoding="utf-8")
    main = _replace_once(main, MAIN_OLD, MAIN_NEW, "main-results history-length sentence")

    si = SI_PATH.read_text(encoding="utf-8")
    if SI_METHOD_MARKER not in si:
        anchor = r"\paragraph{Classical history-length control}"
        if si.count(anchor) != 1:
            raise ValueError("Could not locate the single classical history-length control insertion point")
        si = si.replace(anchor, SI_METHOD + "\n" + SI_TABLE_INPUT + "\n\n" + anchor, 1)
    elif SI_METHOD not in si:
        raise ValueError("A7 Lite SI method marker exists with conflicting text")
    if si.count(SI_TABLE_INPUT) == 0:
        anchor = r"\paragraph{Classical history-length control}"
        if si.count(anchor) != 1:
            raise ValueError("Could not locate the classical section for the paired-table input")
        si = si.replace(anchor, SI_TABLE_INPUT + "\n\n" + anchor, 1)
    elif si.count(SI_TABLE_INPUT) != 1:
        raise ValueError("The A7 Lite paired table must be input exactly once")
    if not (si.index(SI_METHOD_MARKER) < si.index(SI_TABLE_INPUT)
            < si.index(r"\paragraph{Classical history-length control}")):
        raise ValueError("The paired table input must follow the neural scan and precede classical control")
    pcw_means = (
        "Figure~\\ref{fig:a7b-history-length} shows decreasing mean error over the tested lengths: from $-11.41$ to $-36.48$~dB at 30~km/h, "
        "$-1.55$ to $-5.69$~dB at 120~km/h and $-1.17$ to $-1.73$~dB at 300~km/h."
    )
    pcw_means_with_seed = pcw_means + " The PC-Wiener pointwise 95\\% environment-bootstrap intervals use 10,000 resamples and NumPy PCG64 seed 20261031."
    if pcw_means in si:
        si = si.replace(pcw_means, pcw_means_with_seed, 1)
    elif pcw_means_with_seed not in si:
        raise ValueError("Could not restore the PC-Wiener bootstrap seed in the SI")
    si = si.replace(
        "This classical control does not identify a preferred neural history length; no neural history-length sweep is claimed.",
        "This classical control does not identify a preferred neural history length.",
    )
    if "no neural history-length sweep is claimed" in si:
        raise ValueError("Stale SI claim says the neural history-length sweep is absent")
    si = _replace_once(si, CAPTION_OLD, CAPTION_NEW, "A7 history figure caption")

    outputs = {MAIN_PATH: main, SI_PATH: si}
    for path in (RESPONSE_PATH, REPLY_PATH):
        text = path.read_text(encoding="utf-8")
        if "\\RevisedExcerpt{Fixed-weight history perturbations are evaluated with a common backend in Supplementary Table~\\ref{tab:history_core163}; they measure input sensitivity rather than the usefulness of every lag.}" not in text:
            raise ValueError(f"The fixed-intervention response quote changed in {path.name}")
        text = _replace_once(text, REPLY_OLD, REPLY_NEW, f"R1-M1 neural history response in {path.name}")
        if "A neural history-length sweep remains incomplete" in text:
            raise ValueError(f"Stale incomplete-sweep claim remains in {path.name}")
        old_evidence = r"\ST{tab:history_core163}; \SF{fig:a7b-history-length}."
        new_evidence = r"\ST{tab:history_core163}; \ST{tab:a7b-lite-paired}; \SF{fig:a7b-history-length}."
        if old_evidence in text:
            text = text.replace(old_evidence, new_evidence, 1)
        elif new_evidence not in text:
            raise ValueError(f"Could not update or verify the R1-M1 evidence line in {path.name}")
        outputs[path] = text
    table_text = build_paired_table(result)
    if TABLE_PATH.exists() and TABLE_PATH.read_text(encoding="utf-8") != table_text:
        raise ValueError("Existing A7 Lite paired table conflicts with the deterministic integration output")
    return outputs | {TABLE_PATH: table_text}


def run_integration() -> dict:
    result = compute_results()
    fact_rows = build_fact_rows(result)
    map_rows = build_table_map_rows(result)
    point_rows = build_points_csv_rows(result)
    pair_rows = build_paired_csv_rows(result)
    parameter_rows = build_parameter_count_rows(result)
    point_path = HERE / "a7_lite_points.csv"
    paired_path = HERE / "a7_lite_paired.csv"
    parameter_path = PARAMETER_COUNT_PATH
    point_bytes = _csv_bytes(POINT_FIELDS, point_rows)
    paired_bytes = _csv_bytes(PAIRED_FIELDS, pair_rows)
    parameter_bytes = _csv_bytes(PARAMETER_COUNT_FIELDS, parameter_rows)
    for path, expected in ((point_path, point_bytes), (paired_path, paired_bytes),
                           (parameter_path, parameter_bytes)):
        if path.exists() and path.read_bytes() != expected:
            raise ValueError(f"Existing derived A7 CSV conflicts with recomputed values: {path.name}")
    text_outputs = _prepare_text_outputs(result)
    backup_manifest = _backup_original_sources()

    for path, content in ((point_path, point_bytes), (paired_path, paired_bytes),
                          (parameter_path, parameter_bytes)):
        if not path.exists():
            path.write_bytes(content)
    added_facts = _append_rows(FACTS_PATH, FACT_FIELDS, fact_rows, "fact_id")
    added_maps = _append_rows(STAT_MAP_PATH, MAP_FIELDS, map_rows, ("asset", "row_selector"))
    for path, content in text_outputs.items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")

    output_files = [
        point_path, paired_path, parameter_path, FACTS_PATH, STAT_MAP_PATH, MAIN_PATH, SI_PATH,
        RESPONSE_PATH, REPLY_PATH, TABLE_PATH,
        BACKUP_DIR / "backup_manifest.json",
    ]
    outputs = {path.relative_to(ROOT).as_posix(): sha256(path) for path in output_files}
    validation = {
        "schema": "a7b-lite-history-integration-validation-v1",
        "status": "validated_no_build",
        "input_hashes": result["hashes"],
        "integration_script_sha256": sha256(Path(__file__)),
        "backup_manifest_sha256": sha256(BACKUP_DIR / "backup_manifest.json"),
        "backup_manifest": backup_manifest,
        "recomputed": {
            "per_environment_rows": 270, "points": 9, "pointwise_ci_bounds": 18,
            "paired_contrasts": 6, "paired_ci_bounds": 12,
            "facts_new_this_invocation": added_facts,
            "table_map_rows_new_this_invocation": added_maps,
            "facts_integrated": len(fact_rows), "paired_table_map_rows_integrated": len(map_rows),
            "parameter_count_rows": len(parameter_rows),
            "merged_point_map_rows_verified": result["point_map_validation"]["a7_lite_mean_rows_verified"],
            "parameter_counts": result["parameter_counts"],
            "bootstrap": {"method": "PCG64", "replicates": BOOTSTRAP_N,
                          "seed_rule": "20261029 + speed_kmh", "resampling_unit": "environment_id"},
        },
        "outputs_sha256": outputs,
        "verified_shared_figure_map": result["point_map_validation"],
        "checks": [
            "root approval hash bindings verified",
            "270 unique per-environment rows and 9 x 30 paired cells verified",
            "physical NMSE energy and dB identities recomputed",
            "fixed-run summary and root points/paired intervals independently matched",
            "model parameter counts reconstructed from the named implementation",
            "merged figure point_map.csv contains the nine exact A7 Lite mean IDs and values",
            "facts and table-statistic mapping written with resolvable repository-relative paths",
            "both R1-M1 response files updated while fixed-intervention excerpt remained exact",
            "paired table is included exactly once between the neural and classical SI sections",
            "parameter-count facts resolve to the separate CSV bound to the model-source hash",
            "figure assets and point_map.csv were not written by this integrator; no build was run",
        ],
    }
    validation_path = HERE / "validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "validation_path": validation_path.relative_to(ROOT).as_posix(),
        "facts_added": added_facts,
        "table_map_rows_added": added_maps,
        "point_count": len(result["points"]),
        "paired_count": len(result["paired"]),
        "parameter_counts": result["parameter_counts"],
        "outputs_sha256": outputs,
    }


if __name__ == "__main__":
    print(json.dumps(run_integration(), ensure_ascii=False, indent=2))
