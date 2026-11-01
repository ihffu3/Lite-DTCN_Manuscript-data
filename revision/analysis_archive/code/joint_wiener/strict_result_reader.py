"""Verify saved joint-Wiener results before downstream paper aggregation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SPEEDS_KMH = (30, 300)
HISTORY_LENGTH = 10
HORIZON_STEPS = 4
HORIZON_MS = 2.0
ORIGIN_START = 19
TEST_STRIDE = 1
EXPECTED_GROUPS_PER_SPEED = 30
EXPECTED_ORIGINS = 3978
EXPECTED_ROWS = 15912
_ENV_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _same_number(actual, expected: float, label: str) -> None:
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError(f"{label} differs from the verified physical metric")


def _source_bindings(root: Path, approval: dict, directory: Path) -> dict[str, str]:
    bindings = approval.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Root test approval has no source bindings")
    root = root.resolve()
    normalized = {}
    protocol_path = (directory / "protocol_v1.json").resolve()
    try:
        protocol_relative = protocol_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Protocol must be inside the bound project root") from exc
    for relative, expected in bindings.items():
        relative_text = str(relative).replace("\\", "/")
        relative_path = Path(relative_text)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Source binding escapes the project root: " + str(relative))
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Source binding escapes the project root: " + str(relative)) from exc
        expected_sha = _digest(expected, "Source binding")
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError("Source binding SHA mismatch: " + relative_text)
        key = relative_path.as_posix()
        if key in normalized:
            raise ValueError("Duplicate normalized source binding: " + key)
        normalized[key] = expected_sha
    if normalized.get(protocol_relative) != _sha256(protocol_path):
        raise ValueError("Protocol is not included in the approved source bindings")
    return bindings


def _validate_protocol(root: Path, directory: Path, protocol: dict, manifest: dict) -> None:
    if (protocol.get("speeds_kmh") != list(SPEEDS_KMH)
            or protocol.get("history_length") != HISTORY_LENGTH
            or protocol.get("horizon_steps") != HORIZON_STEPS
            or protocol.get("horizon_ms") != HORIZON_MS
            or protocol.get("origin_start") != ORIGIN_START
            or protocol.get("test_stride") != TEST_STRIDE
            or protocol.get("training_environments_per_speed") != EXPECTED_GROUPS_PER_SPEED
            or protocol.get("validation_environments_per_speed") != 8
            or protocol.get("test_environments_per_speed") != EXPECTED_GROUPS_PER_SPEED
            or protocol.get("frequency_block_width") != 4
            or protocol.get("relative_ridges") != [1e-8, 1e-6, 1e-4, 1e-2]
            or protocol.get("post_test_access") is not True
            or protocol.get("preregistered") is not False):
        raise ValueError("Joint-Wiener protocol differs from the frozen fixed-test scope")
    manifest_path = root / "revision/data/formal/manifest.json"
    if _digest(protocol.get("formal_manifest_sha256"), "Protocol manifest SHA") != _sha256(manifest_path):
        raise ValueError("Formal manifest SHA differs from the frozen protocol")
    if not isinstance(manifest.get("groups"), dict):
        raise ValueError("Formal manifest has no groups map")


def _expected_test_groups(manifest: dict) -> dict[int, list[str]]:
    groups = manifest["groups"]
    expected = {}
    for speed in SPEEDS_KMH:
        ids = sorted(
            env_id for env_id, group in groups.items()
            if isinstance(group, dict) and group.get("family") == "D1_main"
            and group.get("split") == "test" and group.get("speed_kmh") == speed
        )
        if (len(ids) != EXPECTED_GROUPS_PER_SPEED
                or len(ids) != len(set(ids))
                or any(_ENV_ID.fullmatch(env_id) is None for env_id in ids)):
            raise ValueError(f"Manifest must contain 30 unique D1 test environments at {speed} km/h")
        for env_id in ids:
            group = groups[env_id]
            if (group.get("status") != "complete" or group.get("qa", {}).get("status") != "passed"
                    or not group.get("output_path") or not group.get("job_path")):
                raise ValueError("Manifest test group is incomplete or lacks QA/input identity: " + env_id)
            _digest(group.get("output_sha256"), "Manifest MAT SHA")
            _digest(group.get("job_sha256"), "Manifest job SHA")
            _digest(group.get("qa", {}).get("receipt_sha256"), "Manifest QA receipt SHA")
            if (isinstance(group.get("output_bytes"), bool)
                    or not isinstance(group.get("output_bytes"), int) or group["output_bytes"] < 1):
                raise ValueError("Manifest MAT byte count is invalid: " + env_id)
        expected[speed] = ids
    return expected


def _validate_fit_receipts(directory: Path, approval: dict, source_bindings: dict,
                           protocol: dict) -> dict[str, str]:
    approved_fit = approval.get("fit_receipt_sha256")
    if not isinstance(approved_fit, dict) or set(approved_fit) != {str(s) for s in SPEEDS_KMH}:
        raise ValueError("Root test approval must bind one fit receipt per speed")
    receipt_hashes = {}
    for speed in SPEEDS_KMH:
        folder = directory / "results" / f"s{speed}"
        receipt_path = folder / "fit_receipt.json"
        receipt_sha = _sha256(receipt_path) if receipt_path.is_file() else None
        if receipt_sha != _digest(approved_fit[str(speed)], "Approved fit receipt SHA"):
            raise ValueError(f"Fit receipt SHA is not approved at {speed} km/h")
        receipt = _read_json(receipt_path, "fit receipt")
        scores = receipt.get("validation_linear_nmse")
        ridges = receipt.get("relative_ridges")
        selected = receipt.get("selected_index")
        if (receipt.get("status") != "complete" or receipt.get("speed_kmh") != speed
                or receipt.get("horizon_steps") != HORIZON_STEPS
                or receipt.get("test_files_opened") != []
                or receipt.get("source_bindings") != source_bindings
                or ridges != protocol["relative_ridges"]
                or not isinstance(scores, list) or len(scores) != len(ridges)
                or any(isinstance(score, bool) or not isinstance(score, (int, float))
                       or not math.isfinite(float(score)) for score in scores)
                or isinstance(selected, bool) or not isinstance(selected, int)
                or selected != min(range(len(scores)), key=scores.__getitem__)
                or receipt.get("selected_ridge") != ridges[selected]):
            raise ValueError(f"Fit receipt identity or validation selection is invalid at {speed} km/h")
        heads_path = folder / "selected_heads.npz"
        if (not heads_path.is_file()
                or _sha256(heads_path) != _digest(receipt.get("selected_heads_sha256"), "Selected-head SHA")):
            raise ValueError(f"Selected head artifact SHA mismatch at {speed} km/h")
        receipt_hashes[str(speed)] = receipt_sha
    return receipt_hashes


def _validate_result_row(row: dict, env_id: str, speed: int, group: dict,
                         fit_sha: str, source_bindings: dict) -> None:
    identity = row.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Environment result has no identity object: " + env_id)
    expected_identity = {
        "environment_id": env_id, "split": "test", "family": "D1_main",
        "speed_kmh": group["speed_kmh"], "mat_sha256": group["output_sha256"],
        "job_sha256": group["job_sha256"],
        "qa_receipt_sha256": group["qa"]["receipt_sha256"],
        "mat_bytes": group["output_bytes"],
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Environment result identity differs from manifest: " + env_id)
    if (row.get("environment_id") != env_id
            or row.get("fit_receipt_sha256") != fit_sha
            or row.get("source_bindings") != source_bindings
            or row.get("horizon_steps") != HORIZON_STEPS
            or row.get("horizon_ms") != HORIZON_MS
            or row.get("stride") != TEST_STRIDE
            or row.get("n_origins") != EXPECTED_ORIGINS
            or row.get("n_rows") != EXPECTED_ROWS):
        raise ValueError("Environment result identity or dense row coverage differs: " + env_id)
    energies = [row.get("error_energy"), row.get("target_energy"), row.get("prediction_energy")]
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) for value in energies)
            or energies[0] < 0 or energies[1] <= 0 or energies[2] < 0):
        raise ValueError("Environment result energy is invalid: " + env_id)
    linear_nmse = float(energies[0]) / float(energies[1])
    if not math.isfinite(linear_nmse) or linear_nmse <= 0:
        raise ValueError("Environment result linear NMSE is invalid: " + env_id)
    nmse_db = 10.0 * math.log10(linear_nmse)
    for key in ("nmse", "pooled_linear_nmse"):
        _same_number(row.get(key), linear_nmse, env_id + " " + key)
    for key in ("nmse_db", "pooled_nmse_db"):
        _same_number(row.get(key), nmse_db, env_id + " " + key)


def read_verified_results(root: Path = ROOT, directory: Path = HERE) -> tuple[list[dict], dict]:
    """Return only manifest-bound, exact-coverage saved results for paper use.

    This checks the stored evaluation artifacts; it does not reopen CSI or rerun
    the scientific evaluation.
    """
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    protocol = _read_json(directory / "protocol_v1.json", "protocol")
    approval = _read_json(directory / "root_test_approval.json", "root test approval")
    manifest_path = root / "revision/data/formal/manifest.json"
    manifest = _read_json(manifest_path, "formal manifest")
    if approval.get("status") != "approved_for_fixed_test_replay":
        raise ValueError("Root test approval is missing or has the wrong status")
    _validate_protocol(root, directory, protocol, manifest)
    source_bindings = _source_bindings(root, approval, directory)
    expected = _expected_test_groups(manifest)
    fit_receipt_hashes = _validate_fit_receipts(directory, approval, source_bindings, protocol)
    if fit_receipt_hashes != approval["fit_receipt_sha256"]:
        raise ValueError("Fit receipt bindings differ from root test approval")

    summary_path = directory / "results" / "summary.json"
    summary = _read_json(summary_path, "summary")
    if (summary.get("status") != "complete"
            or summary.get("test_for_fitting_or_selection") is not False
            or summary.get("source_bindings") != source_bindings
            or summary.get("fit_receipt_sha256") != fit_receipt_hashes):
        raise ValueError("Summary identity or fixed-test disclosure differs from approved sources")

    rows = []
    expected_ids = set()
    for speed in SPEEDS_KMH:
        folder = directory / "results" / f"s{speed}" / "test_environments"
        expected_ids.update(expected[speed])
        actual_paths = list(folder.glob("*.json")) if folder.is_dir() else []
        if {path.stem for path in actual_paths} != set(expected[speed]) or len(actual_paths) != len(expected[speed]):
            raise ValueError(f"Saved test result files do not exactly cover 30 manifest groups at {speed} km/h")
        for env_id in expected[speed]:
            group = manifest["groups"][env_id]
            path = folder / f"{env_id}.json"
            row = _read_json(path, "per-environment result")
            _validate_result_row(row, env_id, speed, group,
                                 fit_receipt_hashes[str(speed)], source_bindings)
            rows.append(row)
    if len(rows) != 60 or len({row["environment_id"] for row in rows}) != 60:
        raise ValueError("Saved test results do not contain the exact 60 unique manifest environments")

    summary_rows = summary.get("per_environment")
    if not isinstance(summary_rows, list) or len(summary_rows) != 60:
        raise ValueError("Summary does not contain exactly 60 environment rows")
    summary_by_id = {}
    for row in summary_rows:
        env_id = row.get("environment_id") if isinstance(row, dict) else None
        if not isinstance(env_id, str) or env_id in summary_by_id:
            raise ValueError("Summary contains a missing or duplicate environment ID")
        summary_by_id[env_id] = row
    if set(summary_by_id) != expected_ids:
        raise ValueError("Summary environment IDs differ from the exact manifest test panel")
    for row in rows:
        if summary_by_id[row["environment_id"]] != row:
            raise ValueError("Summary row differs from its saved environment result: " + row["environment_id"])

    expected_means = {
        str(speed): math.fsum(row["nmse_db"] for row in rows
                              if row["identity"]["speed_kmh"] == speed) / EXPECTED_GROUPS_PER_SPEED
        for speed in SPEEDS_KMH
    }
    reported_means = summary.get("mean_environment_nmse_db")
    if not isinstance(reported_means, dict) or set(reported_means) != set(expected_means):
        raise ValueError("Summary speed-level means do not cover the frozen speed grid")
    for speed, expected_mean in expected_means.items():
        _same_number(reported_means.get(speed), expected_mean, "Summary mean NMSE dB at " + speed + " km/h")
    return rows, summary
