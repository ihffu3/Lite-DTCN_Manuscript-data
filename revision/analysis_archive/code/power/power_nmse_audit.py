"""Posthoc environment-level power/error association; no new forecasts or tests."""
from pathlib import Path
import hashlib
import json
import math
import platform
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OUT = BASE / 'power_audit'
NEURAL = ROOT / 'revision/reports/local_results53/main_per_environment.csv'
PANELS = ROOT / 'revision/submission/PHYCOM_CORE_TECHNICAL_20261020/evidence/formal165/core_main_nmse.csv'
CONTROLS = ROOT / 'revision/paper/reviewer_fix_20261028/analysis/local_control_amplitude_environment_metrics.csv'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def correlation(x, y, ranks=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if ranks:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])

def summarize(frame, keys, energy, nmse, rows, collection):
    results = []
    for key, g in frame.groupby(keys, sort=True):
        assert len(g) == 30 and g.environment_id.nunique() == 30
        assert (g[energy] > 0).all() and (g[rows] > 0).all()
        power = g[energy].to_numpy(float) / (g[rows].to_numpy(float) * 256)
        log_power = 10 * np.log10(power)
        y = g[nmse].to_numpy(float)
        k = key if isinstance(key, tuple) else (key,)
        result = dict(zip(keys, k))
        result.update(collection=collection, n_environments=30,
                      min_mean_coefficient_power=float(power.min()),
                      max_mean_coefficient_power=float(power.max()),
                      pearson_log_power_nmse_db=correlation(log_power, y),
                      spearman_power_nmse_db=correlation(log_power, y, ranks=True))
        result['status'] = ('undefined_nonfinite_or_constant_response'
                            if result['spearman_power_nmse_db'] is None else 'complete')
        results.append(result)
    return results

def main():
    assert math.isclose(correlation([1, 2, 3], [3, 2, 1], True), -1)
    assert math.isclose(correlation([1, 2, 3], [2, 4, 6]), 1)
    assert correlation([1, 1, 1], [1, 2, 3]) is None
    OUT.mkdir(exist_ok=True)
    protocol = {
        'scope': 'A2-d optional posthoc descriptive association, all existing panels',
        'independent_unit': 'environment; 30 environments per panel; no pooling of repeated conditions/seeds',
        'x': '10*log10(target_energy/(n_rows*256 complex coefficients)); each row is one forecast origin-user pair, not one unique user',
        'y': 'existing per-environment NMSE-dB',
        'estimators': ['Pearson correlation with log power', 'Spearman rank correlation'],
        'undefined_policy': 'retain panel and mark undefined; do not drop or replace nonfinite/reference-zero values',
        'inference': 'No p-values, new significance family, causal attribution, or test-driven tuning',
        'access': 'Existing formal test results already accessed; no raw MAT or weights opened',
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [NEURAL, PANELS, CONTROLS]},
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    df = pd.read_csv(NEURAL)
    assert len(df) == 3870 and set(df.status) == {'ok'}
    assert np.allclose(df.physical_error_energy_complex128 / df.physical_target_energy_complex128,
                       df.linear_nmse, rtol=1e-12, atol=1e-15)
    assert np.allclose(10*np.log10(df.linear_nmse), df.nmse_db, rtol=1e-12, atol=1e-12)
    bindings = df[['result_path', 'result_sha256']].drop_duplicates()
    assert bindings.result_path.nunique() == len(bindings)
    for row in bindings.itertuples():
        assert sha(ROOT / row.result_path) == row.result_sha256
    panels = pd.read_csv(PANELS)
    # Classical rows have no neural model identity; they are not part of this lookup.
    identity = panels.loc[panels.model_identity.notna(), ['model_identity', 'family']].drop_duplicates()
    assert not identity.model_identity.duplicated().any()
    df = df.merge(identity, left_on='model_run_id', right_on='model_identity', how='left', validate='many_to_one')
    assert df.family.notna().all()
    results = summarize(df, ['family', 'model_run_id', 'training_seed', 'speed_kmh', 'horizon_ms'],
                        'physical_target_energy_complex128', 'nmse_db', 'n_rows', 'original_neural')
    controls = pd.read_csv(CONTROLS)
    assert len(controls) == 720
    results += summarize(controls, ['method', 'speed_kmh', 'horizon_ms'], 'target_energy',
                         'raw_nmse_db', 'n_rows', 'local_control_amplitude')
    out = pd.DataFrame(results)
    assert len(out) == 153
    out.to_csv(OUT / 'panel_correlations.csv', index=False)
    summary = {'rows': len(out), 'original_neural_panels': 129, 'control_panels': 24,
               'undefined_panels': int((out.status != 'complete').sum()),
               'source_rows': 4590, 'runtime_python': platform.python_version(),
               'output_sha256': sha(OUT / 'panel_correlations.csv')}
    (OUT / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
