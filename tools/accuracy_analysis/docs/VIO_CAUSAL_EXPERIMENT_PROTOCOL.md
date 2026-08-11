# VIO Causal Experiment Protocol

## Purpose

This protocol separates four candidate mechanisms affecting VIO accuracy:
camera-center displacement, actual exposure, scene feature density and IMU time
offset. It uses controlled acquisition for Geometry, Image and Texture, and
archived deterministic replay for Timing. A correlation from unmatched runs is
not a causal result.

Every matrix cell requires five complete repetitions. A repetition is a fresh
OKVIS process with a new output directory. The binary build, diagnostic schema
and all non-target settings remain fixed within a comparison.

## Experiment Matrices

### Geometry

Acquire the same scene under the same illumination, actual exposure and angular
motion. Select events by mocap camera-center displacement between matched
frames. The manipulated mediator is translation/parallax.

| Cell | Camera-center displacement | Repetitions |
|---|---:|---:|
| G-near | 0-2 cm | 5 |
| G-mid | 10-20 cm | 5 |
| G-far | >=30 cm | 5 |

The exclusion gap between 2 and 10 cm and between 20 and 30 cm prevents
borderline events from changing cells under mocap uncertainty. Exposure,
illumination, scene content, angular-speed peak and integrated rotation must
overlap across cells within frozen tolerances.

### Image

Acquire the same scene and motion at three actual exposure levels. `1x` is a
fixed nominal physical exposure selected before collection; `0.5x` and `2x`
are physical exposure changes, not post-processing gains. Record the realized
exposure in microseconds for every frame or exposure block.

| Cell | Actual exposure target | Required record | Repetitions |
|---|---:|---:|---:|
| I-short | 0.5x | realized exposure [us] | 5 |
| I-nominal | 1x | realized exposure [us] | 5 |
| I-long | 2x | realized exposure [us] | 5 |

Camera-center translation, angular motion, scene, focus and illumination are
non-target variables. Do not substitute digital brightness or synthetic blur
for actual exposure.

### Texture

Acquire matched scene layouts with a preregistered feature-density target. Use
the same detector and frozen detector settings to measure density before VIO
outcomes are inspected.

| Cell | Feature-density target | Repetitions |
|---|---:|---:|
| T-low | low | 5 |
| T-medium | medium | 5 |
| T-high | high | 5 |

Pilot data define non-overlapping numeric feature-density ranges for low,
medium and high. Freeze those ranges before outcome collection. Exposure,
illumination, camera-center translation, angular-speed peak and integrated
rotation must overlap within the non-target tolerances.

### Timing

Use one archived raw dataset for all cells. Generate immutable variants with
`prepare_imu_time_offset_variants.py`; only the integer nanosecond timestamp in
`imu0/data.csv` changes. Images and every other sensor payload remain linked to
the archived source.

| Cell | IMU timestamp offset | Repetitions |
|---|---:|---:|
| Tm-10 | -10 ms | 5 |
| Tm-5 | -5 ms | 5 |
| Tm-2 | -2 ms | 5 |
| Tm0 | 0 ms | 5 |
| Tp2 | +2 ms | 5 |
| Tp5 | +5 ms | 5 |
| Tp10 | +10 ms | 5 |

Create the full archived replay set in one command:

```bash
python3 tools/accuracy_analysis/scripts/prepare_imu_time_offset_variants.py \
  --source-dataset /path/to/sequence_euroc \
  --output-root /path/to/imu_offset_variants \
  --offsets-ms -10 -5 -2 0 2 5 10
```

The generated `dataset_manifest.csv` is the input to
`run_vio_diagnostics.py --dataset-manifest`.

## Required Records

Record the following before inspecting VIO outcome metrics. Paths alone are
not provenance; hash raw data, mocap and configuration content.

| Required field | Definition |
|---|---|
| `experiment_id` | Unique matrix-cell identity used by the replay runner |
| `operator` | Person responsible for acquisition or replay |
| `config_sha256` | SHA-256 of the complete OKVIS configuration |
| `scene` | Fixed scene identifier and layout revision |
| `motion` | Motion-profile identifier and execution method |
| `exposure_us` | Realized physical exposure in microseconds |
| `illumination` | Measured illumination and measurement location |
| `translation` | Mocap camera-center displacement and interval |
| `rotation` | Angular-speed peak and integrated absolute rotation |
| `repeat` | Integer 1 through 5 |
| `raw` | Raw dataset path, immutable archive ID and SHA-256 |
| `mocap` | Mocap path, calibration ID, coverage and SHA-256 |
| `deviation` | Protocol deviation or explicit `none` |

Also retain intervention requested/realized values, variant manifest hash,
binary build ID, randomization seed and order, run command, UTC start/end,
return code, produced files, diagnostic completion status, online/final-BA APE
and map-persistence metrics.

## Session Randomization

Randomize cell order within each acquisition or replay session using a recorded
seed. Use five complete randomized blocks: each block contains every cell once
and supplies one of its five repetitions. Do not collect all five repeats of a
cell consecutively. Balance cells across operator, host and time-of-session;
none may be confounded with a single session position. Freeze the order before
reading outcome metrics. Record interruptions and resumed-order deviations.

## Manipulation Acceptance Checks

Acceptance is decided without APE or map outcomes. Freeze all numeric
tolerances in excluded pilot sessions.

All matrices must pass these checks:

1. Angular-speed peak distributions and integrated-rotation distributions
   overlap across cells within the preregistered tolerance.
2. Every non-target variable is within its frozen tolerance: scene, motion,
   exposure, illumination, translation, rotation, raw input and mocap, except
   where that variable is the matrix's manipulated mediator.
3. All five repetitions are present, use unique output directories, exit zero,
   report `run_complete=true`, retain the completion sentinel and contain both
   online and final-BA trajectories.
4. The manipulated mediator changes monotonically in the requested direction;
   a requested label alone is not evidence that the manipulation occurred.
5. Config, binary, raw and mocap hashes match the preregistered values, and all
   protocol deviations are reported before inclusion decisions.

Matrix-specific checks:

| Matrix | Required acceptance evidence |
|---|---|
| Geometry | Mocap camera-center displacement lies inside 0-2 cm, 10-20 cm or >=30 cm as assigned; displacement distributions increase G-near to G-mid to G-far; exposure, illumination and rotation overlap. |
| Image | Hardware metadata confirms realized exposure in microseconds near 0.5x, 1x and 2x; exposure increases I-short to I-long; scene, illumination, translation and both angular measures overlap; image count and timestamps are unchanged. |
| Texture | Frozen-detector feature density lies in its pilot-defined low, medium or high range and increases T-low to T-high; exposure, illumination, translation and both angular measures overlap. |
| Timing | Every IMU timestamp changes by exactly the requested integer nanoseconds; shifted timestamps are nonnegative and strictly increasing; all non-timestamp IMU columns are byte-identical; camera and non-IMU entries are symlinks to the archived source; raw and mocap hashes are identical across offsets. |

Reject a cell that misses its mediator range or a non-target tolerance. Keep its
records in the ledger and do not silently replace it. A replacement receives a
new run ID and deviation record. Report individual runs by randomized block;
any added interaction or post-hoc matrix is explicitly exploratory.
