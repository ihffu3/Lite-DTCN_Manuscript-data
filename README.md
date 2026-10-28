# Lite-DTCN_Manuscript-data

This repository contains a selected, traceable release of first-party code and summary data for the clean-history CSI-prediction baseline. It is an archival research artifact, not a turnkey reproduction package.

## Scope and status

The clean-history baseline uses noise-free CSI histories as prediction inputs. The repository includes selected model, baseline, metric, and link-evaluation source files, QuaDRiGa data-generation adapters, frozen protocol/configuration records, and summary CSVs. The completed A7 and C2/A4 clean-history summaries are included below. A1/A6, noisy-history analyses, and the additional A4 extension using newly restored weights remain outside this release.

Raw CSI datasets and trained checkpoints/weights are not included. The summary ledger points to upstream aggregate filenames in its `source_file` column, but the underlying `core_main_nmse.csv` and `core_link_metric_panels.csv` files were not available in this release directory. Some configuration paths are preserved as archival provenance and may refer to the original author's workspace; they are not portable job files.

## Contents

- `revision/code/core`, `models`, `baselines`, and `link`: selected first-party implementations.
- `revision/code/data/*.m`: first-party MATLAB adapters for generating development/formal groups and densifying QuaDRiGa tracks. The adapters require an explicit execution flag and external QuaDRiGa installation. QuaDRiGa source code is not redistributed here.
- `config/`: data contracts, formal protocol, and resource-limited neural/classical candidate choices.
- `data/summary/`: the fact ledger, development-candidate validation scores, complete shared-model grid, classical/neural rate gains, derived comparison tables, A7a/A7b follow-up summaries, and C2/A4 training-control/amplitude-diagnostic summaries.
- `scripts/replot_public_summary.py`: replot the included NMSE–spectral-efficiency trade-off CSV without CSI or neural-inference packages.
- `scripts/verify_manifest.py` and `manifest.json`: verify the byte size and SHA-256 digest of released source/data files; generated figures are excluded.

## Metric definitions

The CSV method key `lmmse` is a legacy machine-readable alias for **PC-Wiener** (the per-coefficient empirical Wiener predictor); figures and manuscript prose use the name PC-Wiener.

For each test environment, complex squared prediction error is summed over the valid prediction origins and channel elements, then divided by the corresponding summed target energy. This yields one linear NMSE per environment. The reported `nmse_db_env_mean` is the arithmetic mean of those environment-level NMSE values after conversion to dB, with equal weight per environment. It is not a pooled-energy NMSE. Where the fact ledger says `equal_env_mean_seed42`, the displayed statistic is the equally weighted environment mean for seed 42.

Candidate selection uses a separate validation metric. `validation_score` is the inverse-normalized, physical-scale linear NMSE, averaged equally across environments within each validation task and then equally across the three validation-task means; smaller is better. It is not training MSE.

In `nmse_se_300_snr10_rzf.csv`, `nmse_gain_db` is outdated-CSI NMSE in dB minus predicted-CSI NMSE in dB, so positive values indicate lower NMSE. `sum_se_gain` is predicted-CSI RZF sum spectral efficiency minus outdated-CSI RZF sum spectral efficiency, in bit/s/Hz, so positive values indicate a rate gain. Perfect CSI is an evaluation reference under the same link implementation; it is not an asserted capacity bound.

## Environment-level uncertainty update

The added NMSE tables report environment means, sample SDs and post-hoc pointwise 95% percentile-bootstrap intervals, conditional on each fitted model. The independent units are 30 environment groups per D1 condition; users and overlapping prediction windows are nested. Training seeds remain separate. The NMSE summaries use 10,000 environment resamples and PCG64 seed 20261028. `registered_paired_nmse_contrasts_reused.csv` retains the original primary-three and secondary-31 contrast intervals without replacing their analysis.

The BER files contain 36 cell intervals, 24 Lite-minus-comparator paired-difference intervals and the underlying 20-environment error/bit counts per cell. Each panel uses the same 10,000 sampled environment indices across methods (PCG64 seed 20261029). These are pointwise descriptive intervals, with no simultaneous-coverage or new significance-test claim. The reanalysis used Python 3.13.5 and NumPy 2.1.3. These tables reuse completed clean-history predictions. The C2/A4 and A7 follow-ups below are separate additions; A1/A6 and the additional A4 extension using newly restored weights are not included.

## C2 training controls and A4 amplitude diagnostics

The C2 summary covers six predictor variants at 30 and 300 km/h for a 2-ms horizon (12 method-speed panels), with 30 D1 test environments per panel. The A4 reference results add 120 environment records: 60 Outdated-H-origin and 60 Perfect-target observations across the same two speeds. Neural variants use one fit (seed 42) per method-speed condition, with checkpoint selection based on validation data only. The weighted-loss variant uses inverse training-environment channel energy as a loss weight; it does not normalize using test windows or targets. For the PC-Wiener-plus-residual hybrid, epoch 0 is the zero-residual baseline and is excluded from checkpoint selection; eligible epochs begin at epoch 1.

The A4 summaries report raw and target-aware diagnostic quantities. NMSE is computed as an error-energy/target-energy ratio within each environment, converted to dB per environment, and averaged equally across environments. The oracle complex scalar uses target-channel information and is non-deployable. The reported $\rho$ is averaged over forecast-user rows within each environment and then equally across environments. The joint-matrix-normalized RZF10 diagnostic applies the same train-referenced full-matrix scaling to predictor outputs and both Outdated-H-origin and Perfect-target references; it is separate from raw RZF10 rate.

C2/A4 95% intervals are posthoc pointwise percentile-bootstrap intervals from 10,000 environment resamples (PCG64 seed 42); paired intervals resample matched environments. They are descriptive, conditional on the single neural fit per method-speed condition, and introduce no new hypothesis-test family. The original two families of three primary and 31 secondary contrasts are unchanged.

## A7 follow-up summaries

- **A7a: fixed-weight history-input sensitivity.** The release includes the summary and environment-level CSVs for three specified history inputs evaluated with the same backend, checkpoint and metric path on the same 30 D1 test environments (300 km/h, 2 ms, seed 42). This describes those perturbations only; it does not establish a general history-distribution effect, optimal history length or individual-lag utility. Intervals are posthoc, pointwise paired environment-bootstrap intervals (10,000 resamples; PCG64 seed 20261030).
- **A7b: train-only PC-Wiener tap-length scan.** The release includes the 15-cell summary and 450 environment-level rows for tap lengths $L\in\{2,4,6,8,10\}$ at 30, 120 and 300 km/h (30 matched D1 test environments per speed and tap; 4-step/2-ms horizon). Estimates use equal-environment means of environment-level NMSE-dB. Intervals are posthoc, pointwise environment-cluster bootstrap intervals (10,000 resamples; PCG64 seed 20261031), paired across tap lengths within speed. This classical scan does not establish an optimal neural history length; intervals are not simultaneous and add no hypothesis-test family.

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
