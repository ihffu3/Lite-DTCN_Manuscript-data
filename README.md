# Lite-DTCN_Manuscript-data

This repository contains a selected, traceable release of first-party code and summary data for the clean-history CSI-prediction baseline. It is an archival research artifact, not a turnkey reproduction package.

## Scope and status

The clean-history baseline uses noise-free CSI histories as prediction inputs. The repository includes selected model, baseline, metric, and link-evaluation source files, QuaDRiGa data-generation adapters, frozen protocol/configuration records, and summary CSVs. Additional noisy-history, amplitude-diagnostic and training follow-ups are not included in this clean-history baseline.

Raw CSI datasets and trained checkpoints/weights are not included. The summary ledger points to upstream aggregate filenames in its `source_file` column, but the underlying `core_main_nmse.csv` and `core_link_metric_panels.csv` files were not available in this release directory. Some configuration paths are preserved as archival provenance and may refer to the original author's workspace; they are not portable job files.

## Contents

- `revision/code/core`, `models`, `baselines`, and `link`: selected first-party implementations.
- `revision/code/data/*.m`: first-party MATLAB adapters for generating development/formal groups and densifying QuaDRiGa tracks. The adapters require an explicit execution flag and external QuaDRiGa installation. QuaDRiGa source code is not redistributed here.
- `config/`: data contracts, formal protocol, and resource-limited neural/classical candidate choices.
- `data/summary/`: the fact ledger, development-candidate validation scores, complete shared-model grid, classical/neural rate gains and derived comparison tables.
- `scripts/replot_public_summary.py`: replot the included NMSE–spectral-efficiency trade-off CSV without CSI or neural-inference packages.
- `scripts/verify_manifest.py` and `manifest.json`: verify the byte size and SHA-256 digest of released source/data files; generated figures are excluded.

## Metric definitions

The CSV method key `lmmse` is a legacy machine-readable alias for **PC-Wiener** (the per-coefficient empirical Wiener predictor); figures and manuscript prose use the name PC-Wiener.

For each test environment, complex squared prediction error is summed over the valid prediction origins and channel elements, then divided by the corresponding summed target energy. This yields one linear NMSE per environment. The reported `nmse_db_env_mean` is the arithmetic mean of those environment-level NMSE values after conversion to dB, with equal weight per environment. It is not a pooled-energy NMSE. Where the fact ledger says `equal_env_mean_seed42`, the displayed statistic is the equally weighted environment mean for seed 42.

Candidate selection uses a separate validation metric. `validation_score` is the inverse-normalized, physical-scale linear NMSE, averaged equally across environments within each validation task and then equally across the three validation-task means; smaller is better. It is not training MSE.

In `nmse_se_300_snr10_rzf.csv`, `nmse_gain_db` is outdated-CSI NMSE in dB minus predicted-CSI NMSE in dB, so positive values indicate lower NMSE. `sum_se_gain` is predicted-CSI RZF sum spectral efficiency minus outdated-CSI RZF sum spectral efficiency, in bit/s/Hz, so positive values indicate a rate gain. Perfect CSI is an evaluation reference under the same link implementation; it is not an asserted capacity bound.

## Environment-level uncertainty update

The added NMSE tables report environment means, sample SDs and post-hoc pointwise 95% percentile-bootstrap intervals, conditional on each fitted model. The independent units are 30 environment groups per D1 condition; users and overlapping prediction windows are nested. Training seeds remain separate. The NMSE summaries use 10,000 environment resamples and PCG64 seed 20261028. `registered_paired_nmse_contrasts_reused.csv` retains the original primary-three and secondary-31 contrast intervals without replacing their analysis.

The BER files contain 36 cell intervals, 24 Lite-minus-comparator paired-difference intervals and the underlying 20-environment error/bit counts per cell. Each panel uses the same 10,000 sampled environment indices across methods (PCG64 seed 20261029). These are pointwise descriptive intervals, with no simultaneous-coverage or new significance-test claim. The reanalysis used Python 3.13.5 and NumPy 2.1.3. These tables reuse completed clean-history predictions; they contain no result from the local follow-up training or noisy-history evaluation that is still in progress.

## Environment and data generation

The recorded reference environment used Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, and PyTorch 2.5.1 (the recorded installation used the CUDA 12.1 wheel). `requirements.txt` lists the corresponding Python package versions. The MATLAB adapters were recorded with MATLAB R2025b (25.2.0.2998904) and QuaDRiGa 2.8.1-0. Channel generation requires a separately obtained and installed QuaDRiGa distribution and its MATLAB dependencies. These environment notes identify the recorded setup; they do not establish portability across platforms.

The included CSV-only plot can be reproduced with:

```text
python scripts/replot_public_summary.py
```

The output is written to `figures/nmse_se_tradeoff.pdf` and `figures/nmse_se_tradeoff.svg`. To verify the release files against the manifest, run:

```text
python scripts/verify_manifest.py
```

Neither command regenerates CSI, retrains models, or reproduces the full manuscript analysis.

## Reuse

No open-source license has been selected for this repository. Reuse permissions remain subject to applicable copyright until the repository owner adds license terms.
