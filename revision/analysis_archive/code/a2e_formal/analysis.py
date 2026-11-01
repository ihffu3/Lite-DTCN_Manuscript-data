"""Read complete A2e saved receipts; report seed dispersion without new inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
REPLAY_SHA = "56780005014707041589ab7118565cd994630f5d85687b716df41c27721dbfc8"
PROTOCOL_SHA = "c818dc71b17821538a26531eca0be73d95e5ea661105da44142cb8ae97b0120c"
HORIZONS, SEEDS = (1, 2, 4, 8), (42, 43, 44)


def summarize(rows):
    """Use 30 equally weighted environments per seed, then SD of three means."""
    by_key = {}
    for row in rows:
        h, s, e = row["horizon_steps"], row["seed"], row["environment_id"]
        if (type(h) is not int or h not in HORIZONS or type(s) is not int
                or s not in SEEDS or not isinstance(e, str) or not e):
            raise ValueError("Unexpected horizon, seed or environment identity")
        key = (h, s, e)
        if key in by_key:
            raise ValueError("Duplicate environment row")
        value, energy = row["metrics"]["nmse_db"], row["metrics"]["target_energy"]
        if any(isinstance(v, bool) or not isinstance(v, (float, int))
               or not math.isfinite(v) for v in (value, energy)) or energy <= 0:
            raise ValueError("Nonfinite/extended-real result requires separate reporting; no filtering")
        mat = row["data_identity"]["mat_sha256"]
        if not isinstance(mat, str) or not mat:
            raise ValueError("Missing MAT identity")
        by_key[key] = (float(value), float(energy), mat)
    envs = sorted({key[2] for key in by_key})
    if len(envs) != 30 or set(by_key) != {(h, s, e) for h in HORIZONS for s in SEEDS for e in envs}:
        raise ValueError("Expected complete 12-fit by 30-environment grid")
    for h in HORIZONS:
        for e in envs:
            if len({by_key[(h, s, e)][1:] for s in SEEDS}) != 1:
                raise ValueError("Targets or MAT identities differ across paired seeds")
    seed_rows, horizon_rows = [], []
    for h in HORIZONS:
        means = []
        for s in SEEDS:
            mean = math.fsum(by_key[(h, s, e)][0] for e in envs) / 30
            means.append(mean)
            seed_rows.append(dict(horizon_steps=h, horizon_ms=h / 2, seed=s,
                                  n_environments=30, mean_environment_nmse_db=mean))
        mean = math.fsum(means) / 3
        horizon_rows.append(dict(
            horizon_steps=h, horizon_ms=h / 2, n_seeds=3, n_environments_per_seed=30,
            mean_seed_environment_nmse_db=mean,
            sample_sd_across_three_seed_means_db=math.sqrt(math.fsum((v - mean) ** 2 for v in means) / 2),
            minimum_seed_mean_nmse_db=min(means), maximum_seed_mean_nmse_db=max(means),
            seed_range_db=max(means) - min(means)))
    return seed_rows, horizon_rows


def execute(manifest_path):
    # The replay's immutable, reviewed reader validates all exact IDs, hashes,
    # physical energies, target geometry, saved summaries and CSVs. No evaluator
    # or MAT loader is called from this analysis.
    from revision.paper.remaining_fixes_20261029.training.a2e_formal_replay_v2 import runner as replay

    if replay.sha256(Path(replay.__file__)) != REPLAY_SHA or replay.sha256(replay.PROTOCOL_PATH) != PROTOCOL_SHA:
        raise ValueError("Reviewed replay source or protocol changed")
    protocol = replay.read_json(replay.PROTOCOL_PATH)
    replay.validate_protocol(protocol)
    manifest, digest = replay._read_pretest_manifest(manifest_path, protocol)
    snapshot = manifest["snapshot"]
    approval_path = ROOT / protocol["artifacts"]["pretest_manifest_directory"] / protocol["artifacts"]["root_approval_filename"]
    approval = replay.require_root_approval(manifest, digest, PROTOCOL_SHA, approval_path, protocol)
    output = ROOT / protocol["artifacts"]["output_directory"] / digest
    bundles, groups = snapshot["fit_bundles"], snapshot["test_groups"]
    identity = {
        "schema": "a2e-formal-replay-run-identity-v1", "pretest_manifest_sha256": digest,
        "snapshot_sha256": manifest["snapshot_sha256"], "approval_file_sha256": approval["approval_file_sha256"],
        "replay_runner_sha256": REPLAY_SHA, "replay_protocol_sha256": PROTOCOL_SHA,
        "selection_receipt_sha256": snapshot["selection_receipt_sha256"],
        "verification_closeout_sha256": snapshot["verification_closeout_sha256"],
        "run_ids": snapshot["plan_run_ids"],
        "test_environment_ids": [row["environment_id"] for row in groups],
        "n_environment_rows": 360, "device": "cuda:0",
        "gpu_memory_cap_gib": protocol["execution"]["gpu_memory_cap_gib"],
        "test_based_selection": False, "test_based_tuning": False,
    }
    rows = replay._verify_completed_replay(
        output, identity, output / "run_identity.json", bundles, groups,
        {**manifest, "manifest_sha256": digest}, snapshot["selection_receipt_sha256"],
        approval["approval_file_sha256"], "cuda:0", protocol)
    seeds, horizons = summarize(rows)
    destination = HERE / "results" / digest[:16]
    if destination.exists():
        raise FileExistsError("Preserving prior analysis; inspect its verification before rerunning")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".analysis-", dir=destination.parent))
    for name, table in (("seed_means.csv", seeds), ("horizon_seed_dispersion.csv", horizons)):
        with (staging / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    review = dict(
        status="complete_saved_results_verified_pending_root_numeric_review",
        analysis_source_sha256=replay.sha256(Path(__file__)),
        pretest_manifest_sha256=digest, root_approval_sha256=approval["approval_file_sha256"],
        inputs={name: replay.sha256(output / name) for name in
                ("run_manifest.json", "run_identity.json", "row_hash_ledger.json", "summary.json", "per_environment.csv")},
        outputs={p.name: replay.sha256(p) for p in staging.glob("*.csv")},
        complete_fits=12, complete_environment_rows=360,
        aggregation="Within seed: equal mean of 30 environment NMSE-dB values. Across seeds: mean and sample SD of three seed means (ddof=1).",
        limitations="Three training seeds; no confidence intervals, p-values, pooling of 90 rows as independent environments, or new hypothesis family.",
        test_mat_files_opened=[], new_training=False, new_inference=False)
    (staging / "verification.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.rename(destination)
    return dict(result_directory=str(destination), **review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analyze-complete-saved-results", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.manifest), indent=2))


if __name__ == "__main__":
    main()
