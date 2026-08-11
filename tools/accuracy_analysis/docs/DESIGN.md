# 20260803-184537 Multi-Control Repeatability Analysis Design

## Objective

Quantify which properties distinguish the low-accuracy `20260803-184537`
sequence from two independently low-error controls, and explain why repeated
OKVIS2-X runs on the target sequence are unstable. Treat `bak0` as a separately
documented backend outlier and focus repeatability statistics on `bak1` through
`bak4`.

## Inputs

- Dataset: `/home/chenguyuan/data/20260803/20260803-184537_euroc`
- Mocap: `/home/chenguyuan/data/20260803/mocap_ego2_20260803/mocap_20260803_184540.log`
- Config: `config/okvis2_eucm_EGO2.yaml`
- Runs: primary analysis on `bak1` through `bak4`; `bak0` excluded as an outlier
- Maps: each run's final CSV and G2O map
- Control 1: `workspace/ego2_results/20260803-183537`, dataset
  `/home/chenguyuan/data/20260803/20260803-183537_euroc`
- Control 2: `workspace/ego2_results/20260803-184027`, dataset
  `/home/chenguyuan/data/20260803/20260803-184027_euroc`
- Target comparison run: `workspace/ego2_results/20260803-184537/bak3`

## Architecture

Represent every compared sequence with an explicit `SequenceSpec` containing
its name, role (`control` or `target`), dataset, result directory, mocap log and
plot color. Helpers partition contexts by role and validate at least one
control and exactly one target; no result depends on list order or object
identity.

All three sequences use identical APE, motion, IMU, mocap, camera, sampled
image-quality and G2O definitions. Camera-specific low-sharpness thresholds are
the p5 of the pooled control samples, with equal deterministic sampling per
sequence and camera. Individual control medians and p5 values remain visible so
the pooled threshold cannot hide scene-specific disagreement.

The final diagnosis combines two evidence levels:

1. Dataset-level contrast: a target property is called distinctive only when
   it lies outside both controls or the pooled control range.
2. Estimator-level repeatability: all `bak1`-`bak4` runs share the same sensor
   stream, so their APE, map topology and retained-observation spread quantify
   amplification inside OKVIS2-X rather than acquisition differences.

## Metrics

1. evo APE translation after timestamp association within 10 ms and SE(3)
   Umeyama alignment: RMSE, mean, median, maximum and percentiles.
2. Normalized APE: `APE RMSE / associated mocap path length * 100%`.
3. Mocap motion: cumulative distance, finite-difference linear speed and
   quaternion-difference angular speed.
4. IMU dynamics: gyroscope magnitude, acceleration magnitude, sample interval
   and gap statistics.
5. Estimated motion: CSV velocity magnitude, pose-difference speed and angular
   speed, plus adjacent-position jump magnitude.
6. Repeatability: pairwise aligned trajectory RMSE, first sustained 1 cm
   divergence, map graph size differences, observations per landmark, distinct
   states per landmark and landmark state-time span for every primary run.
7. Image acquisition: per-camera timestamps, cross-camera matching rate and a
   deterministic sampled Laplacian sharpness time series.

## Outputs

- `analyze_repeatability.py`: self-contained command-line analysis.
- `test_analyze_repeatability.py`: unit tests for numerical primitives.
- `run_summary.csv`, `reference_summary.csv`, `pairwise_repeatability.csv`,
  `motion_summary.csv`, `imu_summary.csv`, `camera_summary.csv`.
- Figures 10-14 generalized to two controls plus one target, and Figure 15
  comparing the control envelope against every target run's APE and retained
  map observations.
- `REPORT.md`: evidence-backed diagnosis, limitations and controlled
  experiments needed to isolate each nondeterministic mechanism.

## Interpretation Rules

- A different map topology before evo proves estimator nondeterminism; evo
  cannot create different keyframes or landmarks.
- APE divided by path length is a descriptive normalization, not a substitute
  for APE and not an official evo metric.
- Correlation between high motion and APE does not by itself prove a sensor
  fault. IMU gaps, camera drops, blur and online trajectory continuity are
  checked independently.
- Existing outputs do not contain run logs, so exact loop-closure pairs and
  Ceres iteration counts cannot be reconstructed. Source-level nondeterministic
  mechanisms are reported separately from empirically observed failures.
- Pooled-control p5 is a relative low-detail threshold, not an absolute blur
  classifier. A lower Laplacian variance can reflect blur, texture or view.
- Correlations across four target runs are descriptive only; `n=4` is too small
  for inferential claims.
- Physical camera-IMU synchronization is not proven by the software
  `image_delay` offset.
