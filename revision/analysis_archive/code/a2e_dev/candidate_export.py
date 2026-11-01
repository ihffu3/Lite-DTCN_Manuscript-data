"""Prepare an isolated SI-table/facts bundle from the reviewed A2e dev search.

This module does not train models or read formal-test outputs. Its CLI is
deliberately pinned to one complete development receipt, one root review, and
one protocol revision. Do not run the real export until the source is reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
RECEIPT_REL = Path("revision/runs/A2e_itransformer_dev_search_20261029_r02/selection_receipt.json")
REVIEW_REL = Path("revision/paper/remaining_fixes_20261029/audit/ROOT_A2E_DEVELOPMENT_SELECTION_REVIEW_R02.json")
PROTOCOL_REL = Path(
    "revision/paper/remaining_fixes_20261029/training/"
    "a2e_itransformer_development_search_v1/training_protocol_v2.json"
)
OUTPUT_REL = Path(
    "revision/paper/remaining_fixes_20261029/training/"
    "a2e_dev_search_analysis_v1/candidate_exports/r02"
)

EXPECTED_RECEIPT_SHA256 = "661c3c41621b47b87e43ab17f65a8e2999998b19949bee02817a05e20016ef9a"
EXPECTED_REVIEW_SHA256 = "1473525b35eac44e5fb2e58a82f15a5e04652c4971e804636d0e22ca7b7bdbb8"
EXPECTED_PROTOCOL_SHA256 = "87db59b27f6a0cf7ff4a9cf565aab231e312282ae99b77ef9943297aac88340d"
EXPECTED_REVIEW_SOURCE_SHA256 = "7b1bf39984d488569d57188414033dc96564aa63d0b33c0221d3a7da7c1e283b"

FACT_FIELDS = (
    "fact_id", "quantity", "method", "speed_kmh", "horizon_ms", "snr_db",
    "value", "unit", "aggregation", "source_file", "source_selector",
)
MAP_FIELDS = ("asset", "row_selector", "source_csv", "fact_ids")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _config_grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grid = protocol.get("search_grid")
    if not isinstance(grid, dict):
        raise ValueError("protocol search_grid is missing")
    dimensions = ("d_model", "encoder_layers", "learning_rates")
    if any(not isinstance(grid.get(key), list) for key in dimensions):
        raise ValueError("protocol search grid dimensions are malformed")
    prefix = grid.get("config_id_prefix")
    heads = grid.get("heads")
    dropout = _finite_number(grid.get("dropout"), "protocol dropout")
    if not isinstance(prefix, str) or not prefix or isinstance(heads, bool) or not isinstance(heads, int):
        raise ValueError("protocol configuration identity is malformed")
    if grid.get("d_ff_rule") != "2 * d_model":
        raise ValueError("unsupported protocol d_ff rule")

    configurations: list[dict[str, Any]] = []
    for d_model in grid["d_model"]:
        for layers in grid["encoder_layers"]:
            for learning_rate in grid["learning_rates"]:
                if isinstance(d_model, bool) or not isinstance(d_model, int):
                    raise ValueError("protocol d_model values must be integers")
                if isinstance(layers, bool) or not isinstance(layers, int):
                    raise ValueError("protocol encoder layer counts must be integers")
                lr = _finite_number(learning_rate, "protocol learning rate")
                lr_tag = {0.0003: "3e4", 0.001: "1e3"}.get(lr)
                if lr_tag is None:
                    raise ValueError(f"unsupported protocol learning rate: {lr}")
                config_id = f"{prefix}_d{d_model:03d}_l{layers}_lr{lr_tag}"
                configurations.append({
                    "config_id": config_id,
                    "d_model": d_model,
                    "heads": heads,
                    "encoder_layers": layers,
                    "d_ff": 2 * d_model,
                    "dropout": dropout,
                    "learning_rate": lr,
                })
    expected_count = grid.get("configuration_count")
    if len(configurations) != expected_count or len({item["config_id"] for item in configurations}) != len(configurations):
        raise ValueError("protocol configuration count or identity is inconsistent")
    return configurations


def _task_plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = protocol.get("task_plan")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("protocol must define the three representative validation tasks")
    result: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("malformed protocol task")
        key = task.get("task_key")
        if not isinstance(key, str) or not key:
            raise ValueError("protocol task key is malformed")
        speed = _finite_number(task.get("speed_kmh"), f"{key} speed")
        horizon = _finite_number(task.get("horizon_ms"), f"{key} horizon")
        result.append({**task, "speed_kmh": speed, "horizon_ms": horizon})
    if len({task["task_key"] for task in result}) != 3:
        raise ValueError("protocol task keys must be unique")
    return result


def _score_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-15)


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _method_note(protocol: dict[str, Any]) -> str:
    grid = protocol["search_grid"]
    opportunity = str(grid["training_opportunity"])
    if "ceil(task training rows / batch size)" not in opportunity or "patience 15" not in opportunity:
        raise ValueError("protocol training-opportunity description is missing expected constraints")
    return (
        "The development search evaluated eight iTransformer configurations: "
        f"d_model={grid['d_model']}, encoder layers={grid['encoder_layers']}, "
        f"learning rate={grid['learning_rates']}, with {grid['heads']} heads, "
        f"d_ff=2d_model, and dropout={grid['dropout']}. For each task, all eight "
        "configurations received the same replacement-sampled row opportunity "
        "(ceil(n_train/32) updates per epoch, up to 100 epochs, patience 15). "
        "This approximates the Lite search opportunity but is not equal compute, "
        "updates, or wall time: batch size 32 with ceiling steps differs from Lite "
        "batch size 128, and early stopping can reduce realized updates. The selected "
        "configuration minimizes the equal-weight arithmetic mean of the three "
        "best validation mean-environment physical linear NMSE scores. This is a "
        "development-only, nonblind/post-hoc search; formal test data were not opened "
        "for these fits, while formal test access has occurred historically."
    )


def build_candidate_data(
    receipt: dict[str, Any],
    review: dict[str, Any],
    protocol: dict[str, Any],
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the 24-fit development contract and build non-writing outputs."""
    if receipt.get("status") != "complete":
        raise ValueError("selection receipt is not complete")
    if receipt.get("schema") != "a2e-itransformer-development-search-receipt-v2":
        raise ValueError("unexpected selection receipt schema")
    if review.get("status") != "root_verified_completed_development_selection":
        raise ValueError("root review is not a completed development-selection review")
    if review.get("review_source_sha256") != EXPECTED_REVIEW_SOURCE_SHA256:
        raise ValueError("root review source pin does not match the reviewed search source")
    if not receipt.get("historical_formal_test_access") or receipt.get("blind_status") != "nonblind_posthoc":
        raise ValueError("receipt does not disclose the historical nonblind/post-hoc context")
    if receipt.get("test_files_opened") != [] or review.get("formal_test_MAT_files_opened") != []:
        raise ValueError("formal test files were recorded as opened for the development search")

    task_plan = _task_plan(protocol)
    tasks = [task["task_key"] for task in task_plan]
    configs = _config_grid(protocol)
    if protocol.get("configuration_selection", {}).get("task_weighting") != "Equal weight, one third per task cell.":
        raise ValueError("protocol selection weighting is not the expected equal-task rule")
    if protocol.get("configuration_selection", {}).get("formal_test_used") is True:
        raise ValueError("protocol indicates formal test use during development selection")

    expected_pairs = {(config["config_id"], task) for config in configs for task in tasks}
    results = receipt.get("results")
    if not isinstance(results, list) or len(results) != 24:
        raise ValueError("selection receipt must contain 24 unique completed fits")
    result_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    result_run_ids: list[str] = []
    for result in results:
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise ValueError("each development result must be complete")
        pair = (result.get("config_id"), result.get("task_key"))
        if pair not in expected_pairs or pair in result_by_pair:
            raise ValueError("development receipt has an unexpected or duplicate fit")
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("development result run_id is missing")
        result_run_ids.append(run_id)
        validated_result = dict(result)
        validated_result["_validated_score"] = _finite_number(result.get("best_validation_score"), f"{run_id} score")
        result_by_pair[pair] = validated_result
    if set(result_by_pair) != expected_pairs or len(set(result_run_ids)) != 24:
        raise ValueError("selection receipt must contain 24 unique completed fits")
    if receipt.get("run_ids") != result_run_ids or len(set(receipt.get("run_ids", []))) != 24:
        raise ValueError("receipt run_ids do not match its result rows")

    root_fits = review.get("fits")
    if review.get("verified_fit_count") != 24 or not isinstance(root_fits, list) or len(root_fits) != 24:
        raise ValueError("root review does not verify all 24 development fits")
    root_by_run = {row.get("run_id"): row for row in root_fits if isinstance(row, dict)}
    if len(root_by_run) != 24 or set(root_by_run) != set(result_run_ids):
        raise ValueError("root review fit identities do not match the receipt")
    for result in results:
        root_fit = root_by_run[result["run_id"]]
        reviewed_score = _finite_number(
            root_fit.get("best_validation_linear_nmse"),
            f"root review score for {result['run_id']}",
        )
        receipt_score = _finite_number(result.get("best_validation_score"), f"receipt score for {result['run_id']}")
        if not _score_close(receipt_score, reviewed_score):
            raise ValueError(f"root review score differs from receipt for {result['run_id']}")

    receipt_scores = receipt.get("selection", {}).get("configuration_scores")
    if not isinstance(receipt_scores, list) or len(receipt_scores) != 8:
        raise ValueError("receipt must contain the eight configuration selection scores")
    selection_by_config = {item.get("config_id"): item for item in receipt_scores if isinstance(item, dict)}
    if len(selection_by_config) != 8 or set(selection_by_config) != {item["config_id"] for item in configs}:
        raise ValueError("receipt configuration selection identities are incomplete or duplicated")

    source_hashes = source_hashes or {}
    receipt_sha = source_hashes.get("receipt_sha256", "")
    review_sha = source_hashes.get("review_sha256", "")
    protocol_sha = source_hashes.get("protocol_sha256", "")
    if source_hashes:
        if receipt.get("protocol_sha256") != protocol_sha:
            raise ValueError("receipt protocol SHA does not match protocol source")
        if review.get("selection_receipt_sha256") != receipt_sha:
            raise ValueError("root review receipt SHA does not match selection receipt")

    rank_order = []
    for config in configs:
        config_id = config["config_id"]
        score_record = selection_by_config[config_id]
        task_scores = score_record.get("task_scores")
        if not isinstance(task_scores, dict) or set(task_scores) != set(tasks):
            raise ValueError(f"selection task score set is incomplete for {config_id}")
        for task in tasks:
            receipt_score = _finite_number(task_scores[task], f"{config_id}/{task} selection score")
            fit_score = result_by_pair[(config_id, task)]["_validated_score"]
            if not _score_close(receipt_score, fit_score):
                raise ValueError(f"selection task score does not match fit result for {config_id}/{task}")
        aggregate = math.fsum(float(task_scores[task]) for task in tasks) / len(tasks)
        recorded_aggregate = _finite_number(score_record.get("equal_task_mean"), f"{config_id} equal-task mean")
        if not _score_close(aggregate, recorded_aggregate):
            raise ValueError(f"equal-task mean does not match task scores for {config_id}")
        rank_order.append((aggregate, len(rank_order), config_id))

    rank_order.sort(key=lambda item: (item[0], item[1]))
    expected_rank = {config_id: rank for rank, (_, _, config_id) in enumerate(rank_order, start=1)}
    for config_id, score_record in selection_by_config.items():
        if score_record.get("selection_rank") != expected_rank[config_id]:
            raise ValueError(f"selection rank does not match equal-task scores for {config_id}")

    selected_id = receipt.get("selection", {}).get("selected_config_id")
    if selected_id != rank_order[0][2]:
        raise ValueError("selected configuration does not match the minimum equal-task mean")
    selected_config = next(item for item in configs if item["config_id"] == selected_id)
    review_config = review.get("selected_config")
    if not isinstance(review_config, dict):
        raise ValueError("root review selected configuration is missing")
    for key, expected in selected_config.items():
        actual = review_config.get(key)
        if isinstance(expected, float):
            if not _score_close(_finite_number(actual, f"review selected {key}"), expected):
                raise ValueError(f"root review selected configuration differs at {key}")
        elif actual != expected:
            raise ValueError(f"root review selected configuration differs at {key}")

    results_by_config: dict[str, dict[str, float]] = {}
    for config in configs:
        config_id = config["config_id"]
        selection = selection_by_config[config_id]
        results_by_config[config_id] = {
            task: _finite_number(selection["task_scores"][task], f"{config_id}/{task}")
            for task in tasks
        }

    result_rows = []
    fact_rows = []
    map_rows = []
    receipt_source = RECEIPT_REL.as_posix()
    results_source = (OUTPUT_REL / "candidate_results.csv").as_posix()
    asset = "tables/a2e_development_search.tex"
    for config in configs:
        config_id = config["config_id"]
        task_values = results_by_config[config_id]
        aggregate = math.fsum(task_values[task] for task in tasks) / len(tasks)
        selection = selection_by_config[config_id]
        row = {
            **config,
            **{f"{task}_best_validation_linear_nmse": task_values[task] for task in tasks},
            "equal_task_mean_linear_nmse": aggregate,
            "selection_rank": expected_rank[config_id],
            "selected": config_id == selected_id,
        }
        result_rows.append(row)

        fact_ids = []
        for task in task_plan:
            task_key = task["task_key"]
            result = result_by_pair[(config_id, task_key)]
            fact_id = f"a2e_dev_linear_nmse_{config_id}_{task_key}"
            fact_ids.append(fact_id)
            selector = (
                f"run_id={result['run_id']};config_id={config_id};task_key={task_key};"
                f"field=best_validation_score;receipt_sha256={receipt_sha};review_sha256={review_sha}"
            )
            fact_rows.append({
                "fact_id": fact_id,
                "quantity": "best_validation_linear_nmse",
                "method": "itransformer_development_search",
                "speed_kmh": task["speed_kmh"],
                "horizon_ms": task["horizon_ms"],
                "snr_db": "",
                "value": format(task_values[task_key], ".17g"),
                "unit": "linear_NMSE",
                "aggregation": "best_validation_mean_environment_seed42",
                "source_file": receipt_source,
                "source_selector": selector,
            })

        aggregate_id = f"a2e_dev_equal_task_mean_linear_nmse_{config_id}"
        fact_ids.append(aggregate_id)
        fact_rows.append({
            "fact_id": aggregate_id,
            "quantity": "equal_task_mean_linear_nmse",
            "method": "itransformer_development_search",
            "speed_kmh": "",
            "horizon_ms": "",
            "snr_db": "",
            "value": format(aggregate, ".17g"),
            "unit": "linear_NMSE",
            "aggregation": "arithmetic_mean_three_development_task_cells",
            "source_file": receipt_source,
            "source_selector": (
                f"config_id={config_id};field=selection.configuration_scores.equal_task_mean;"
                f"selection_rank={expected_rank[config_id]};receipt_sha256={receipt_sha};review_sha256={review_sha}"
            ),
        })
        map_rows.append({
            "asset": asset,
            "row_selector": f"config_id={config_id}",
            "source_csv": results_source,
            "fact_ids": ";".join(fact_ids),
        })

    method_note = _method_note(protocol)
    tex_table = _render_tex_table(result_rows, task_plan, selected_id)
    return {
        "result_rows": result_rows,
        "fact_rows": fact_rows,
        "map_rows": map_rows,
        "method_note": method_note,
        "tex_table": tex_table,
        "selection": {"selected_config_id": selected_id, "scores": selection_by_config},
        "task_plan": task_plan,
        "source_hashes": source_hashes,
        "source_paths": {
            "selection_receipt": receipt_source,
            "root_review": REVIEW_REL.as_posix(),
            "protocol": PROTOCOL_REL.as_posix(),
        },
    }


def _render_tex_table(result_rows: list[dict[str, Any]], task_plan: list[dict[str, Any]], selected_id: str) -> str:
    task_keys = [task["task_key"] for task in task_plan]
    task_labels = [
        rf"\shortstack{{{task['speed_kmh']:g} km/h\\{task['horizon_ms']:g} ms}}" for task in task_plan
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Eight iTransformer configurations evaluated in development. Values are best validation mean-environment physical linear NMSE for the three task cells and their equal-weight mean; all values in bold identify the selected configuration. The sampled-row opportunity was approximately matched, not compute, updates, or wall time. The search was nonblind/post-hoc.}",
        r"\label{tab:a2e-development-search}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        "$d_{\\mathrm{model}}$ & $L$ & LR & "
        + " & ".join(task_labels)
        + " & Equal-task mean " + "\\" * 2,
        r"\midrule",
    ]
    for row in result_rows:
        config_id = row["config_id"]
        selected = config_id == selected_id
        values = [row[f"{key}_best_validation_linear_nmse"] for key in task_keys]
        cells = [
            str(row["d_model"]),
            str(row["encoder_layers"]),
            _format_number(row["learning_rate"]),
            *(_format_number(value) for value in values),
            _format_number(row["equal_task_mean_linear_nmse"]),
        ]
        if selected:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_bundle(output_dir: Path, bundle: dict[str, Any]) -> None:
    """Write candidate-only artifacts; refuse overwrite of an existing bundle."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate bundle: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir()
    result_fields = [
        "config_id", "d_model", "encoder_layers", "learning_rate", "heads", "d_ff", "dropout",
        "s030_h01_best_validation_linear_nmse", "s300_h04_best_validation_linear_nmse",
        "s300_h08_best_validation_linear_nmse", "equal_task_mean_linear_nmse", "selection_rank", "selected",
    ]
    _write_csv(output_dir / "candidate_results.csv", result_fields, bundle["result_rows"])
    _write_csv(output_dir / "candidate_facts.csv", FACT_FIELDS, bundle["fact_rows"])
    _write_csv(output_dir / "candidate_table_statistic_map.csv", MAP_FIELDS, bundle["map_rows"])
    (tables_dir / "a2e_development_search.tex").write_text(bundle["tex_table"], encoding="utf-8", newline="\n")
    (output_dir / "candidate_method_note.md").write_text(bundle["method_note"] + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema": "a2e-development-search-candidate-export-v1",
        "status": "candidate_only_root_review_required",
        "exporter_source_sha256": _sha256(Path(__file__).resolve()),
        "source_hashes": bundle["source_hashes"],
        "source_paths": bundle["source_paths"],
        "selected_config_id": bundle["selection"]["selected_config_id"],
        "candidate_count": len(bundle["result_rows"]),
        "fit_score_fact_count": len(bundle["fact_rows"]),
        "table_map_row_count": len(bundle["map_rows"]),
        "formal_test_data_used": False,
        "active_manuscript_modified": False,
        "created_by": "candidate_export.py",
    }
    (output_dir / "candidate_export.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def load_reviewed_inputs(repo_root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Load only the three hash-pinned development-selection artifacts."""
    paths = {
        "receipt": Path(repo_root) / RECEIPT_REL,
        "review": Path(repo_root) / REVIEW_REL,
        "protocol": Path(repo_root) / PROTOCOL_REL,
    }
    expected = {
        "receipt": EXPECTED_RECEIPT_SHA256,
        "review": EXPECTED_REVIEW_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
    for key, digest in hashes.items():
        if digest != expected[key]:
            raise ValueError(f"pinned {key} SHA mismatch: {digest}")
    review = _read_json(paths["review"])
    if review.get("review_source_sha256") != EXPECTED_REVIEW_SOURCE_SHA256:
        raise ValueError("root review source SHA does not match the reviewed development runner")
    source_hashes = {
        "receipt_sha256": hashes["receipt"],
        "review_sha256": hashes["review"],
        "protocol_sha256": hashes["protocol"],
    }
    return (
        _read_json(paths["receipt"]),
        review,
        _read_json(paths["protocol"]),
        source_hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-reviewed", action="store_true", help="write the hash-pinned R02 candidate bundle")
    args = parser.parse_args()
    if not args.export_reviewed:
        parser.error("pass --export-reviewed to write the isolated candidate bundle")
    receipt, review, protocol, hashes = load_reviewed_inputs()
    bundle = build_candidate_data(receipt, review, protocol, hashes)
    output_dir = REPO_ROOT / OUTPUT_REL
    write_candidate_bundle(output_dir, bundle)
    print(f"candidate bundle written: {output_dir}")


if __name__ == "__main__":
    main()
