# Repeatability Quantitative Analysis Implementation Plan

> **For agentic workers:** Execute inline with test-driven development. Keep all
> generated artifacts under this analysis directory.

**Goal:** Build and run a reproducible quantitative analysis of repeated
OKVIS2-X runs, excluding the obvious `bak0` backend outlier from primary
statistics, and compare against the matched `20260803-183537` sequence.

**Architecture:** One Python module owns deterministic parsing, timestamp
association, SE(3) alignment, motion metrics and report generation. Unit tests
exercise numerical primitives with synthetic trajectories before the full
dataset is processed. evo supplies APE and trajectory semantics; NumPy/SciPy,
Pillow and Matplotlib supply statistics and figures.

**Tech Stack:** Python 3.10, evo 1.37, NumPy, SciPy, Pillow, Matplotlib, unittest.

---

### Task 1: Numerical contracts

**Files:**
- Create: `test_analyze_repeatability.py`
- Create: `analyze_repeatability.py`

- [x] Test exact path length and finite-difference linear speed.
- [x] Test quaternion angular speed for a known constant-rate rotation.
- [x] Test rigid SE(3) alignment and translation APE.
- [x] Test first sustained threshold crossing.
- [x] Test G2O record counting.
- [x] Run tests and verify they fail because the module is absent.

### Task 2: Minimal implementation

**Files:**
- Create: `analyze_repeatability.py`

- [x] Implement the tested numerical primitives.
- [x] Run focused tests until all pass.
- [x] Add validated OKVIS, mocap, IMU and camera readers.
- [x] Re-run the full test file.

### Task 3: Dataset analysis and figures

**Files:**
- Modify: `analyze_repeatability.py`
- Generate: CSV, PNG and Markdown artifacts in this directory.

- [x] Associate and SE(3)-align all primary run stages with mocap using evo.
- [x] Calculate per-run dynamics, APE and map topology.
- [x] Calculate pairwise final-BA repeatability.
- [x] Calculate IMU, camera synchronization and sampled sharpness metrics.
- [x] Process `20260803-183537` with identical definitions.
- [x] Render combined figures and write machine-readable CSV files.
- [x] Write `REPORT.md` from computed values, not hard-coded conclusions.

### Task 4: Verification

- [x] Run the unit tests in the `okvis2x` environment.
- [x] Run the analysis from a clean output state without exceptions.
- [x] Independently recompute all four primary final-BA APE values and the
  matched reference with `evo_ape`.
- [x] Verify every PNG opens, has non-constant pixels and readable layout.
- [x] Verify report claims match the generated CSV values.

### Task 5: Matched-sequence sensor and motion comparison

**Files:**
- Modify: `test_analyze_repeatability.py`
- Modify: `analyze_repeatability.py`
- Generate: `sequence_*.csv`, `10_*.png` through `14_*.png`
- Modify: `REPORT.md`

- [x] Add failing tests for duration-weighted motion thresholds, tolerant
  four-camera synchronization, image exposure metrics, and prefix-aligned
  drift error.
- [x] Run the focused tests and verify each new contract fails because the
  comparison helper is absent.
- [x] Analyze `20260803-183537` and `20260803-184537/bak3` with identical IMU,
  motion, camera, image-quality, APE, and map-topology definitions.
- [x] Quantify high-speed/high-angular-rate exposure, sustained motion,
  camera/IMU gaps, sharpness conditioned on angular rate, clipping, contrast,
  frame difference, and observations per landmark.
- [x] Align error, motion, sampled sharpness, and camera-gap events on a common
  sequence-time axis and render side-by-side comparison figures.
- [x] Update `REPORT.md` with observed evidence, ranked contributors, and
  explicit limits on causal claims.

### Task 6: Matched-sequence verification

- [x] Run all unit tests in the `okvis2x` environment.
- [x] Regenerate every CSV, PNG, and report from the raw datasets.
- [x] Independently recompute the two final-BA APE values with `evo_ape`.
- [x] Verify all generated PNG files decode, are nonblank, and are at least
  1200 by 800 pixels.
- [x] Cross-check report numbers against machine-readable CSV output.

### Task 7: Explicit sequence roles and pooled controls

**Files:**
- Modify: `test_analyze_repeatability.py`
- Modify: `analyze_repeatability.py`

- [x] Add tests constructing three shuffled contexts and assert that
  `partition_sequence_contexts()` returns two controls and the unique target.
- [x] Add tests that missing target, multiple targets and missing controls raise
  `ValueError` with role-specific messages.
- [x] Add a test where two controls and one target have synthetic per-camera
  sharpness values; assert `pooled_control_sharpness_thresholds()` excludes the
  target and returns the pooled-control p5.
- [x] Run only the new tests and verify RED because the role and threshold
  helpers do not exist.
- [x] Add immutable `SequenceSpec` and implement the two helpers with explicit
  validation, then run the focused tests to GREEN.

### Task 8: Three-sequence configuration and quality semantics

**Files:**
- Modify: `test_analyze_repeatability.py`
- Modify: `analyze_repeatability.py`

- [x] Add a test for `grouped_bar_layout(3)` asserting three symmetric offsets,
  positive width and bars contained inside one category.
- [x] Run the layout test and verify RED because the helper is absent.
- [x] Add the `20260803-184027` control dataset/result/mocap defaults and update
  the target mocap default to the current `mocap_ego2_20260803` path.
- [x] Build all comparison contexts from `SequenceSpec`; attach `role` to each
  context and remove order-dependent target selection.
- [x] Replace `reference_*p5` output fields with pooled-control fields and keep
  all three sequence rows in summary, camera, image-bin, gap, timeline, IMU,
  mocap and map CSV files.
- [x] Implement `grouped_bar_layout()` and use it in Figures 10-12 and 14; make
  Figure 13 height scale with three sequence rows.
- [x] Run the full unit test file and keep all prior contracts green.

### Task 9: Target-run map persistence and dynamic report

**Files:**
- Modify: `analyze_repeatability.py`
- Generate: `15_control_envelope_and_target_runs.png`
- Modify: `REPORT.md`

- [x] Populate every `bak1`-`bak4` run row with the same full
  `map_topology_summary()` used for sequence comparison.
- [x] Add Figure 15 with APE, observations/landmark, distinct states/landmark
  and median landmark span for both controls and all four primary target runs.
- [x] Rewrite the matched comparison report to render three sequence rows,
  target/control APE ratios, pooled-control image-quality semantics and
  control-envelope conclusions without hard-coded cam1-cam3 claims.
- [x] Add a repeatability diagnosis using the four target APE values, pairwise
  divergence, map-count spread and retained-observation ranges; label all
  four-run correlations descriptive.
- [x] Run the full analysis and inspect the report for stale two-sequence
  wording, missing values and conclusions unsupported by both controls.

### Task 10: Multi-control verification

- [x] Run all unit tests in the `okvis2x` environment.
- [x] Independently recompute final-BA evo APE for `183537`, `184027` and
  `184537/bak3` with 10 ms association, SE(3) alignment and no scale correction.
- [x] Verify `sequence_comparison_summary.csv` has 3 rows,
  `sequence_camera_quality.csv` has 12 rows and `reference_summary.csv` has 2
  rows; verify `run_summary.csv` has all map-persistence fields for 4 runs.
- [x] Decode every PNG, require nonconstant pixels and minimum 1200x800 size,
  and visually inspect Figures 10-15.
- [x] Cross-check report values against CSV and obtain an independent final
  review with no open Critical or Important issue.
