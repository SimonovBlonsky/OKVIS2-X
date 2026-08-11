# OKVIS2 Online Camera-IMU Time-Offset Calibration

## Objective

Add causal online estimation of the residual camera-IMU time offset to OKVIS2-X.
The estimator runs inside the existing sequential SLAM system and feeds accepted
updates back into subsequent IMU propagation, map projection, and GP3P initial
pose estimation. Dataset replay may run slower than wall-clock time, but the
algorithm must not inspect future frames or perform full-sequence SfM-style
optimization.

The four EGO2 cameras are hardware-synchronized and therefore share one time
offset relative to the IMU. The existing Kalibr value remains the nominal
calibration. Bundle adjustment estimates only a bounded residual:

```text
d_effective = d_nominal + delta_d
t_nominal   = t_camera_raw - d_nominal
t_effective = t_nominal - delta_d
```

Positive `delta_d` means that the image exposure is earlier than the current
nominal corrected timestamp.

The primary success criterion is improved robustness to camera-IMU timing error
during high-angular-motion events. Code-change count and execution speed are
secondary constraints. The implementation must preserve the existing OKVIS2
preprocessing, feature extraction, data association, GP3P, graph state,
outlier-removal, landmark-cleanup, and loop-closure workflows rather than
replace them with a separate pipeline.

## Current Behavior

`CameraParameters::image_delay` is parsed once and subtracted in
`ThreadedSlam::addImages()`. The corrected timestamp becomes the immutable
`MultiFrame` and graph-state timestamp. Frontend propagation and every
`ImuError` then integrate to these fixed times. The graph contains pose,
speed/bias, landmark, and per-camera extrinsic parameter blocks, but no temporal
calibration block.

Consequently, changing `image_delay` currently requires replaying the sequence.
Final BA cannot estimate or repair the offset, and a backend-only change would
not prevent the earlier projection error and GP3P failures in the frontend.

## Scope

The first version includes:

- one constant residual time offset shared by all configured cameras;
- a prior, bounds, observability checks, and accepted-value rollback;
- time-shifted IMU preintegration with interpolated moving boundaries;
- causal feedback into all frontend and graph propagation calls;
- read-only frontend delay hypotheses to protect map association before BA has
  converged;
- staged interaction with the existing online extrinsic calibration;
- structured diagnostics and synthetic plus EGO2 validation.

The first version excludes:

- per-camera offsets, because the cameras are hardware-synchronized;
- per-frame offsets or random-walk delay;
- affine clock skew;
- rolling-shutter or exposure-duration estimation;
- photometric blur correction;
- a non-causal full-sequence time calibration pass;
- copying VINS-Fusion code or adopting its KLT feature-velocity model.

Windowed estimates and diagnostics will determine whether a later clock-skew
model is justified. A sequence-dependent offline best delay is not treated as
ground truth.

## Selected Architecture

Use a time-offset-dependent IMU factor as the physical estimation model. Add a
read-only delay-hypothesis scorer immediately before the existing map matching
stage so that incorrect nominal propagation does not remove all old-map visual
support before the backend can estimate the offset.

The existing visual reprojection factors remain unchanged. This avoids a local
constant-image-velocity approximation and keeps the EUCM and other camera
models on their existing projection paths.

```text
image preprocessing and BRISK detection
    -> propagate candidate poses for several delta_d hypotheses
    -> read-only initialized-map matching and GP3P scoring
    -> select current-frame hypothesis
    -> existing map matching and GP3P
    -> existing realtime BA with shared delta_d
    -> accept or reject the candidate time-offset update
    -> existing outlier removal, cleanup, and loop closure
    -> accepted delta_d feeds the next frame
```

## Timestamp Semantics

`MultiFrame::timestamp()` and `ViGraph::State::timestamp` keep their current
meaning and ordering: `t_nominal = t_raw - d_nominal`. They are never rewritten
when `delta_d` changes. Existing state IDs, queue ordering, callbacks, and
timestamp equality assertions therefore remain valid.

Code that needs a physical integration boundary calls one shared helper:

```text
effectiveTime(t_nominal, delta_d) = t_nominal - delta_d
```

The raw timestamp does not need a new field because it can be reconstructed as
`t_nominal + d_nominal`. Diagnostics record nominal time, effective time, the
nominal delay, and the accepted residual used for the frame.

Existing trajectory output retains nominal timestamps for API compatibility.
Diagnostics additionally provide the effective-timestamp mapping required by
evaluation tools. This avoids time regressions in live callbacks when the
global estimate changes.

## Configuration

Add backward-compatible optional fields below
`camera_parameters.online_calibration`:

```yaml
do_time_offset: false
time_offset_prior_sigma: 0.005       # seconds
time_offset_bound: 0.020             # symmetric residual bound, seconds
time_offset_numeric_diff_epsilon: 0.00001
time_offset_hypothesis_count: 9
time_offset_min_initialized_matches: 30
time_offset_min_cameras: 2
time_offset_max_posterior_sigma: 0.004
time_offset_stable_update_count: 5
```

Missing fields preserve the current behavior exactly. Time-offset calibration
is rejected at startup when IMU use is disabled. Bounds apply to `delta_d`, not
to the absolute `image_delay`.

## Parameter Ownership

Add a one-dimensional `TimeOffsetParameterBlock` to each `ViGraph`. It is
global to the graph and is not stored in `State`, so state removal and landmark
cleanup cannot delete it.

`realtimeGraph_` is the authoritative online estimator. `fullGraph_` owns a
separate block because the two Ceres problems may be optimized independently.
During normal operation the full-graph block is fixed and synchronized to the
last accepted realtime value at existing backend synchronization points. The
first version does not release this block in final BA.

The block stores `delta_d` in seconds and starts at zero. A scalar Gaussian
prior constrains it to the Kalibr nominal calibration, and Ceres lower and upper
bounds enforce the configured residual interval.

## Time-Shifted IMU Preintegration

Each IMU link retains nominal endpoints `t0_` and `t1_`. For a candidate
`delta_d`, it integrates over:

```text
t0_effective = t0_ - delta_d
t1_effective = t1_ - delta_d
```

Both endpoints move by the same amount, so `t1_ - t0_` is unchanged. The IMU
waveform segment, preintegrated rotation, velocity, position, covariance, and
bias Jacobians change.

Extract the mutable preintegration calculations into a local result type:

```text
PreintegrationResult
  Delta_q
  C_integral and C_doubleintegral
  acceleration integrals
  bias Jacobians
  covariance and square-root information
  integration-step count and validity
```

A calculation takes explicit nominal endpoints, `delta_d`, IMU measurements,
IMU parameters, and reference biases. It linearly interpolates gyro and
accelerometer measurements at both moving boundaries and uses the existing
integration scheme inside the interval. It never extrapolates beyond available
measurements and does not mutate the main cached result.

The Ceres IMU factor receives a fifth parameter block:

```text
pose_i, speed_bias_i, pose_j, speed_bias_j, delta_d
```

Existing analytic pose and speed/bias Jacobians remain. The first version uses
a central-difference time Jacobian computed from independent results at
`delta_d + epsilon` and `delta_d - epsilon`. Synthetic tests sweep epsilon
around the configured default and require a stable derivative. Runtime is not
a release criterion for this version.

When calibration is disabled, the time-offset block is fixed at zero and the
factor must reproduce the legacy residual and Jacobians within numerical
tolerance.

## IMU Buffer Coverage

Every IMU link and frontend propagation must have samples spanning all allowed
shifted boundaries:

```text
[t0_nominal - bound - guard,
 t1_nominal + bound + guard]
```

`guard` is two measured IMU periods, computed from recent timestamps. Replace
the fixed 20 ms overlap with the maximum of the legacy overlap and the required
time-offset coverage. A missing boundary sample freezes time-offset estimation
for the frame and records an error; the system must not extrapolate IMU data.

IMU-link append and merge operations retain the measurements needed to
recompute the merged interval at any allowed residual offset.

## Frontend Feedback

All pose propagation paths use the latest accepted residual:

- first-frame IMU initialization window;
- detection-time pose prediction;
- map-matching and GP3P initial pose;
- `ViGraph::addStatesPropagate()` initial state;
- each read-only delay hypothesis.

The estimate produced while processing frame `k` only affects frame `k+1` and
later, except that the current frame may use a read-only hypothesis selected
before its actual data association. No future measurements are used.

## Delay-Hypothesis Protection

The hypothesis stage runs after feature detection and before the existing
mutating map-matching call. It must not add observations, change landmark IDs,
modify graph states, or invoke cleanup.

When confidence is low, candidates are evenly spaced across the configured
residual bounds. Once a posterior standard deviation is available, candidates
cover the accepted estimate and its clipped `+/-1 sigma` and `+/-2 sigma`
neighborhood, supplemented to the configured count within the bounds.

For each candidate:

1. propagate the current pose using shifted IMU boundaries;
2. project initialized map landmarks using the existing camera geometry;
3. run the existing descriptor threshold and spatial eligibility logic in
   read-only form;
4. optionally compute a non-central GP3P model when at least ten
   correspondences exist;
5. produce a score ordered by GP3P inliers, accepted initialized-map matches,
   number of contributing cameras, and median predicted reprojection error.

Ties within measurement noise retain the last accepted residual. The selected
hypothesis initializes current-frame propagation and the realtime BA scalar. It
does not become the accepted global estimate until backend validation passes.
The actual map matching, observation insertion, and GP3P execution then run
once through the existing mutating path.

## Calibration State Machine

The backend controller has three states:

```text
WARMUP
  delta_d fixed at zero
  wait for an initialized map and adequate old-map support

ESTIMATING
  extrinsics fixed
  delay hypotheses enabled
  delta_d variable in realtime BA when temporally observable

HOLD
  delta_d fixed at lastAcceptedDelta
  retain or restore the existing online-extrinsic policy after delay stability
```

The system enters `ESTIMATING` when at least the configured number of accepted
initialized-map matches span the configured number of cameras and the
time-offset data information is nonzero. It enters `HOLD` when visual support
falls below those thresholds, primary RANSAC fails repeatedly, IMU coverage is
insufficient, or a candidate update is rejected.

The posterior scalar covariance is computed after marginalizing the other
active variables. Effective data information is estimated as:

```text
I_data = max(0, 1 / sigma_posterior^2 - 1 / sigma_prior^2)
```

This prevents the Gaussian prior from making an unobservable interval appear
informative. Constant angular velocity or static motion should leave the
posterior dominated by the prior and therefore keep the estimate fixed.

Five consecutive accepted updates with changes below the posterior sigma mark
the delay stable. At that point `delta_d` is held during low-information
periods and the existing online-extrinsic policy may resume. A later informative
window can reopen delay estimation while temporarily fixing extrinsics again.

## Update Acceptance and Rollback

A realtime BA time-offset update is accepted only when all conditions hold:

- the solver terminates successfully and returns finite values;
- the estimate remains strictly inside the configured bounds;
- posterior sigma is below the configured maximum;
- effective data information is positive;
- initialized-map observations still meet the support thresholds;
- total robustified cost does not increase;
- no IMU factor reports invalid coverage or integration.

Before a tentative solve, save the current scalar and the state estimates that
the solve may change. If validation rejects the update, restore
`lastAcceptedDelta` and rerun the normal realtime optimization with the scalar
fixed. This keeps pose, speed/bias, landmarks, and delay mutually consistent.
Do not merely reset the scalar after a rejected solve.

An estimate that reaches a bound is rejected and reported as a model/initial
calibration warning rather than accepted as a valid calibration.

## Interaction With Extrinsic Calibration

The existing EGO2 configuration optimizes camera extrinsics online. Rotation
extrinsics, gyro bias, and time offset are coupled during sustained rotation.
The first version therefore uses staged calibration:

- fix all camera extrinsics in `WARMUP` and `ESTIMATING`;
- estimate the shared residual delay only in informative, visually healthy
  windows;
- after delay stability, hold delay and restore the existing extrinsic policy;
- when delay estimation reopens, fix extrinsics again.

The final BA extrinsic behavior remains unchanged; time offset stays fixed to
the last accepted realtime value in the first version.

## Failure Handling

- Missing IMU coverage: skip the candidate, freeze the update, and log the
  required and available time ranges.
- Non-finite integration or Jacobian: reject the solve, restore the previous
  graph state, and rerun with delay fixed.
- Ceres failure: use the same rollback path.
- Flat or tied frontend hypothesis scores: retain the previous estimate.
- Fewer than ten GP3P correspondences: score from map matches and reprojection
  error but do not treat GP3P as successful.
- Fragmented visual support: hold the last accepted estimate; do not calibrate
  against a map dominated by new or uninitialized landmarks.
- realtime/full synchronization conflict: realtime remains authoritative and
  full graph receives the value only at the existing synchronization barrier.

All fallbacks preserve the legacy fixed-delay behavior with the most recent
accepted effective delay. No failure path silently changes sign conventions or
uses raw camera time as an IMU time.

## Diagnostics

Extend the existing opt-in diagnostics rather than adding unstructured logs.
Record per frame:

- nominal and effective image timestamps;
- nominal, candidate, and accepted delay values;
- state-machine state and transition reason;
- posterior sigma and effective data information;
- IMU coverage interval and boundary interpolation status;
- hypothesis values and their GP3P/match/camera/reprojection scores;
- whether BA released the scalar;
- update acceptance or rejection reason;
- whether extrinsics were fixed by the calibration controller.

These fields are bounded per frame and do not add records to the existing
multi-gigabyte landmark-event stream.

## Testing

### Unit tests

1. Verify the scalar parameter block, prior, bounds, plus/minus operations, and
   sign convention.
2. Compare zero residual offset against legacy preintegration.
3. Verify interpolation and integration for positive and negative offsets using
   analytic constant and linearly changing IMU signals.
4. Sweep numeric-difference epsilon and compare against a high-accuracy
   reference derivative.
5. Confirm that shifting a constant gyro/acceleration interval produces no
   false time information.
6. Recover a known offset under angular acceleration and visually constrained
   poses.
7. Verify insufficient-buffer rejection at both bounds.
8. Exercise IMU-link append/merge and graph state elimination with the global
   block present.
9. Verify realtime/full scalar synchronization and rejected-update rollback.
10. Verify the calibration state transitions and staged extrinsic fixation.
11. Verify hypothesis scoring is read-only and selects the correct synthetic
    projection shift.
12. With calibration disabled, compare trajectories, residuals, and frontend
    decisions against the existing fixed-delay path.

### Dataset tests

Run repeated causal replays of all available 20260803-20260806 sequences with
the same build and configuration, comparing fixed-delay and online-delay modes.
Do not use any sequence-specific offline best delay as truth.

Evaluate:

- APE RMSE and trajectory completion;
- predicted reprojection-error distribution;
- large-reprojection RANSAC triggers and primary GP3P failures;
- initialized-map matches and GP3P inlier ratio;
- observation removals, active initialized landmarks, and landmark lifetime;
- delay estimate repeatability, posterior sigma, bound hits, and state changes;
- whether improvements precede rather than merely follow visual recovery.

Normal sequences must not systematically degrade. As an initial regression
guard, APE may not worsen by more than 10% without a corresponding improvement
in trajectory completion or a documented reference-alignment limitation.

Synthetic known-offset tests establish numerical correctness. Real sequences
establish robustness and causal benefit, not an absolute ground-truth delay.

## Acceptance Criteria

The feature is accepted when:

- known synthetic offsets are recovered within 1 ms under observable motion;
- static and constant-rate motion retain the Kalibr prior rather than drift;
- all four cameras use one shared estimate;
- frontend propagation and backend factors use the same accepted value and sign;
- high-angular sequences show repeatable improvement in old-map support,
  reprojection consistency, GP3P behavior, or APE without broad normal-sequence
  regression;
- rejected updates leave the graph internally consistent;
- disabling calibration reproduces legacy behavior;
- no implementation reads future frames or performs sequence-wide batch
  calibration.

## Implementation Boundary

The implementation may add focused parameter, factor, controller, hypothesis,
and diagnostic helpers. It may extend existing propagation APIs and IMU error
parameter lists. It must not replace the OKVIS2 frontend/backend pipeline,
change unrelated graph states, or refactor unrelated preprocessing, mapping,
cleanup, or loop-closure behavior.

No git commit is part of this design task. Repository changes require explicit
user authorization and remain uncommitted unless requested separately.
