# Lite-DTCN_Manuscript-data

This repository contains a selected, traceable release of first-party code and summary data for the clean-history CSI-prediction baseline and its completed A1/A6/A7/C2/A4 analyses. It is an archival research artifact, not a turnkey reproduction package.

## Scope and status

The clean-history baseline uses noise-free CSI histories as prediction inputs. The repository includes selected model, baseline, metric, and link-evaluation source files, QuaDRiGa data-generation adapters, frozen protocol/configuration records, and summary CSVs. The release includes completed A1 noise-sensitivity, A6 classical-challenge, A7, and C2/A4 follow-ups. These are separate analyses with their own stated scopes.

Raw CSI datasets, channel/prediction matrices, and trained checkpoints/weights are not included or uploaded. The CSVs contain aggregate or per-environment metrics and provenance hashes. The summary ledger points to upstream aggregate filenames in its `source_file` column, but the underlying `core_main_nmse.csv` and `core_link_metric_panels.csv` files were not available in this release directory. Some configuration paths are preserved as archival provenance and may refer to the original author's workspace; they are not portable job files.

## Contents

- `revision/code/core`, `models`, `baselines`, and `link`: selected first-party implementations.
- `revision/code/data/*.m`: first-party MATLAB adapters for generating development/formal groups and densifying QuaDRiGa tracks. The adapters require an explicit execution flag and external QuaDRiGa installation. QuaDRiGa source code is not redistributed here.
- `config/`: data contracts, formal protocol, and resource-limited neural/classical candidate choices.
- `data/summary/`: the fact ledger, development-candidate validation scores, complete shared-model grid, classical/neural rate gains, derived comparison tables, A1/A6/A7/C2/A4 summaries and environment-level metrics, and `figure_point_map.csv` with 425 figure-point mappings.
- `scripts/replot_public_summary.py`: replot the included NMSE–spectral-efficiency trade-off CSV without CSI or neural-inference packages.
- `scripts/verify_manifest.py` and `manifest.json`: verify the byte size and SHA-256 digest of released source/data files; generated figures are excluded.

## Metric definitions

The CSV method key `lmmse` is a legacy machine-readable alias for **PC-Wiener** (the per-coefficient empirical Wiener predictor); figures and manuscript prose use the name PC-Wiener.

For each test environment, complex squared prediction error is summed over the valid prediction origins and channel elements, then divided by the corresponding summed target energy. This yields one linear NMSE per environment. The reported `nmse_db_env_mean` is the arithmetic mean of those environment-level NMSE values after conversion to dB, with equal weight per environment. It is not a pooled-energy NMSE. Where the fact ledger says `equal_env_mean_seed42`, the displayed statistic is the equally weighted environment mean for seed 42.

Candidate selection uses a separate validation metric. `validation_score` is the inverse-normalized, physical-scale linear NMSE, averaged equally across environments within each validation task and then equally across the three validation-task means; smaller is better. It is not training MSE.

In `nmse_se_300_snr10_rzf.csv`, `nmse_gain_db` is outdated-CSI NMSE in dB minus predicted-CSI NMSE in dB, so positive values indicate lower NMSE. `sum_se_gain` is predicted-CSI RZF sum spectral efficiency minus outdated-CSI RZF sum spectral efficiency, in bit/s/Hz, so positive values indicate a rate gain. Perfect CSI is an evaluation reference under the same link implementation; it is not an asserted capacity bound.

## Environment-level uncertainty update

The added NMSE tables report environment means, sample SDs and post-hoc pointwise 95% percentile-bootstrap intervals, conditional on each fitted model. For the D1 panels, the independent units are 30 environment groups per condition; users and overlapping prediction windows are nested. Training seeds remain separate. The descriptive environment-only NMSE summaries added in the uncertainty update use 10,000 resamples and PCG64 seed 20261028. `registered_paired_nmse_contrasts_reused.csv` retains the original primary-three and secondary-31 contrast intervals without replacing their analysis. A1, A6, A7 and the later C2/A4 extensions use their own pointwise intervals and seeds as described below; they do not add a hypothesis-test family.

The BER files contain 36 cell intervals, 24 Lite-minus-comparator paired-difference intervals and the underlying 20-environment error/bit counts per cell. Each panel uses the same 10,000 sampled environment indices across methods (PCG64 seed 20261029). These are pointwise descriptive intervals, with no simultaneous-coverage or new significance-test claim. The reanalysis used Python 3.13.5 and NumPy 2.1.3. These tables reuse completed clean-history predictions. A1, A6, C2/A4 and A7 are separate analyses with their own units and intervals; see the sections below.

## C2 training controls and A4 amplitude diagnostics

The C2 summary covers six predictor variants at 30 and 300 km/h for a 2-ms horizon (12 method-speed panels), with 30 D1 test environments per panel. The A4 reference results add 120 environment records: 60 Outdated-H-origin and 60 Perfect-target observations at 30 and 300 km/h (2 ms). A further A4 follow-up adds 240 records for Original-F26-Lite, PC-Wiener, Outdated-H-origin and Perfect-target at 120 km/h (2 ms) and 300 km/h (4 ms). Together with the 360 C2 records, `local_control_amplitude_environment_metrics.csv` contains 720 rows across four speed/horizon conditions: 30/2 ms, 300/2 ms, 120/2 ms and 300/4 ms. Each method-condition panel has 30 environments. Outdated-H-origin repeats available past CSI; Perfect-target uses the unknown future target and is non-deployable. Neural variants use one fit (seed 42) per method-speed condition, with checkpoint selection based on validation data only. The weighted-loss variant uses inverse training-environment channel energy as a loss weight; it does not normalize using test windows or targets. For the PC-Wiener-plus-residual hybrid, epoch 0 is the zero-residual baseline and is excluded from checkpoint selection; eligible epochs begin at epoch 1.

The A4 summaries report raw and target-aware diagnostic quantities. NMSE is computed as an error-energy/target-energy ratio within each environment, converted to dB per environment, and averaged equally across environments. The oracle complex scalar uses target-channel information and is non-deployable. The reported $\rho$ is averaged over forecast-user rows within each environment and then equally across environments. The joint-matrix-normalized RZF10 diagnostic applies the same train-referenced full-matrix scaling to predictor outputs and both Outdated-H-origin and Perfect-target references; it is separate from raw RZF10 rate.

C2/A4 95% intervals are posthoc pointwise percentile-bootstrap intervals from 10,000 environment resamples (PCG64 seed 42); paired intervals resample matched environments. They are descriptive, conditional on the single neural fit per method-speed condition, and introduce no new hypothesis-test family. The original two families of three primary and 31 secondary contrasts are unchanged.

The release also includes the byte-preserved weighted-loss and PC-Wiener residual model definitions at `revision/code/algorithm_upgrade_20261020/losses.py` and `revision/code/models/hybrid_residual.py`. These are model/loss definitions, not complete training entry points; the width-128 variant uses the existing candidate `temporal_hidden` parameter. The existing cost-front table and figure retain the original main-grid comparison set. A separate same-MAC weighted-loss control dominates Original-F26-Lite at the two tested points; those points were not retrofitted into the historical grid or its figure.

## A1 noise sensitivity

A1 covers six speed-by-noise conditions (30 or 300 km/h; nominal train-referenced SNR of 10, 20 or 30 dB). Six methods are evaluated per condition, yielding 36 method-condition summaries and 1,080 environment-method rows (30 environments per panel). The release includes the six-cell expected-history-SNR summary, the 36-cell environment-bootstrap summary, and paired Lite-versus-PC-Wiener contrasts.

The nominal train-referenced SNR is a noise-variance setting, not the realized receive SNR. Expected actual history SNR is computed from unique clean observed-history samples and the prescribed complex-noise variance. Philox draws are keyed by seed, environment, prediction origin and user; the same draw is shared across methods for a given window, while overlapping windows receive independent draws. Prediction targets remain clean. PC-Wiener and Ridge are refitted separately at each speed and nominal SNR using noisy training histories and clean future targets. Lite retains its clean-trained checkpoint; AR retains its clean fit, Kalman uses its clean fitted state with the prescribed measurement variance, and persistence repeats the latest noisy observation. Pointwise intervals for the 36 plotted means use 10,000 environment resamples (PCG64 seed 20261101), sharing draws across methods and noise levels within speed. The six paired Lite-minus-PC-Wiener intervals use the per-condition seeds in their CSV and add no hypothesis-test family.

## A6 paired classical challenge

A6 reports five classical methods (AR, Kalman, PC-Wiener/LMMSE, Persistence and recursive Ridge (h=1)) across nine declared tasks. There are 20 paired environment-task rows per method-task panel, 900 environment rows and 45 method-task summaries with matched-environment intervals. The nine tasks reuse 80 unique environment IDs across four families (20 per family); tasks within a family share those environment groups, so repeated rows are not additional independent environments. The 180-row shared_v_tau oracle-current result is included as the neural reference on the same challenge environments.

All five classical states use the frozen 90-environment D1 mixed-speed training panel. PC-Wiener and Ridge recursively deploy a one-step head; AR, clean-mode Kalman and persistence use native forecasts. The neural reference receives exact origin-time simulated speed metadata. These are fixed deployment recipes. Pointwise paired intervals use 10,000 PCG64 resamples with per-cell seeds in the CSV.

The corrected summary fixes a source-key lookup error only: the 900 environment rows are preserved, and no prediction or fitting was rerun by that correction. Earlier AR, Kalman and Persistence challenge rows use the same 80 environment IDs; their repeat check is not an additional sample.

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
