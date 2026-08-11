# VIO Angular-Motion to Visual-Fragmentation Causal Diagnostics

## Objective

Add opt-in diagnostics to OKVIS2-X and extend the EGO2 analysis so that the
currently missing middle of the failure chain can be tested directly:

```text
high angular motion
    -> image/feature degradation, weak triangulation geometry, or prediction error
    -> loss of accepted 3D-2D consistency
    -> observation rejection and insufficient landmark replenishment
    -> persistent weak map support (offline "visual fragmentation" state)
    -> trajectory error
```

The diagnostics must determine which middle mechanism is present, when it
starts, whether it recovers, and whether it precedes GP3P failures and landmark
loss. They must not assume that short landmark life or low final-map quality is
an independent cause.

The first validation cohort is the designed angular-impulse subgroup:

- `20260806-175103`: high-peak negative control that recovers;
- `20260806-175304`: angular impulse followed by catastrophic divergence;
- `20260806-175539`: angular impulse followed by persistent degradation.

After validating that instrumentation does not change the result, run the same
diagnostics over all 24 sequences from 20260803 through 20260806.

## Causal Hypotheses

The instrumentation must keep three competing, potentially simultaneous paths
separate.

### H1: Image and feature degradation

Angular motion during image exposure produces large pixel displacement or blur.
The expected signature is a drop in detected keypoints, spatial coverage,
repeatable descriptors, or accepted matches before the GP3P failure burst.

This path is supported only if the image/feature measures deteriorate after the
angular event and before map consistency fails. A stable keypoint population
with stable match acceptance would count against this path even if an offline
Laplacian sharpness proxy is low.

### H2: Weak triangulation geometry

High rotation with insufficient useful translation reduces accepted temporal
ray angles and depth observability. The expected signature is a decrease in ray
angle or baseline, an increase in `isParallel`/invalid triangulations, and a
drop in initialisable landmark births before the existing landmark population
collapses.

Body-origin mocap translation is not accepted-feature parallax. This hypothesis
therefore uses the actual bearing rays and camera centers passed to
`triangulateFast()`, separated by temporal and synchronous multi-camera call
sites.

### H3: Prediction and calibration sensitivity

At high angular rate, camera-IMU timing error, exposure timing, extrinsic error,
IMU integration error, or sensor limits can amplify the difference between the
predicted pose and the visual pose. The expected signature is stable image
features and usable triangulation geometry, but increased map reprojection
error, fewer map matches, or a large GP3P correction relative to the propagated
pose.

This path is not uniquely identified by one diagnostic. It is supported by the
combination of raw IMU timing/limit checks, visual-prediction residuals, and
controlled timing/exposure/calibration sensitivity runs.

### H4: Map-support feedback loop

Once a frame loses 3D-2D consistency, GP3P and reprojection filtering remove
observations. This can reduce the landmark pool available to the next frame,
causing repeated matching failures even after angular velocity returns to
normal. The expected signature is:

```text
first correspondence/inlier collapse
    -> observation removals
    -> lower active initialized-landmark support
    -> inadequate landmark births/initializations
    -> repeated correspondence/inlier collapse
```

This path explains persistence. It must be distinguished from ordinary state
marginalization and loop-closure landmark merging.

## Selected Architecture

Use an opt-in, structured CSV diagnostics pipeline rather than adding more
unstructured `LOG(INFO)` messages or tracing every feature to disk.

Diagnostics are disabled by default. Setting `OKVIS_DIAGNOSTICS_DIR` to an
explicit per-run directory enables them. Existing application arguments and
production configurations remain unchanged.

Add a small diagnostics writer to `okvis_common`, which is already a dependency
of both `okvis_frontend` and `okvis_ceres`. The writer owns the output streams,
schema headers, synchronization, and one-time error reporting. It accepts plain
record structures and has no dependency on frontend or backend implementation
types.

Frontend code owns per-frame accumulators. Matching threads update thread-local
accumulators only; the caller merges them after `join()` and writes one record
per frame or per RANSAC invocation. No filesystem writes are allowed inside the
feature-matching inner loops.

Backend code records only landmark lifecycle/rejection events that cannot be
classified correctly in the frontend. Mutation paths append to graph-local
memory queues; safe batch boundaries move those queues to the writer. This
prevents normal graph maintenance from being conflated with visual failure and
keeps CSV I/O out of backend mutation bursts.

## Output Contract

Every file uses `schema_version=1` and integer nanosecond timestamps where a
current frame exists. Frame, initialisation-pair, and lifecycle streams use
their specific `frame_id`, `current_frame_id/older_frame_id`, or
`event/subject/birth_frame_id` fields. The run directory is the run identity; analysis code
must also store dataset and run names in a separate manifest rather than infer
them from timestamps. The manifest also stores resolved mocap/config paths and
the parsed camera `image_delay`, because neither the EuRoC input nor the
diagnostics directory contains the reference log by contract.

### `vio_diag_frame.csv`

One row per multiframe after data association and initialization:

- initialization, data-association success, tracking-quality threshold, and
  keyframe flags;
- keypoint count per camera and total;
- keypoint response median and lower quantile per camera;
- occupied-cell fraction on a fixed image grid and convex-hull coverage;
- per-camera projected eligible landmarks, descriptor comparisons,
  below-threshold candidates, epipolar/divergent-ray rejects, accepted
  initialized/uninitialized matches, and total observations attached to the
  current frame;
- best descriptor distance before later geometric rejection, in addition to
  accepted-match descriptor distance;
- mean/quantiles of predicted reprojection error before optimization;
- tracking quality;
- active landmark totals split by initialized/uninitialized status;
- landmark births, initialization transitions, and observations added during
  the frame;
- observation removals split by reason;
- motion-stereo return count.

The fixed grid dimensions and quantile definitions are constants recorded in
the metadata file. Per-camera columns are fixed from the configured camera
count, with EGO2 expected to have four cameras.

### `vio_diag_triangulation.csv`

One row per frame and triangulation source category, not one row per attempt.
Source categories distinguish at least:

- temporal motion stereo;
- same-frame multi-camera/spatial stereo;
- re-triangulation of an uninitialized map landmark.

Fields include:

- attempts and descriptor-accepted candidate count;
- valid, invalid, parallel, and initialisable counts;
- camera-pair or camera-source counts;
- camera-center baseline p10/median/p90;
- bearing-ray angle p10/median/p90;
- pixel displacement p10/median/p90 where two image measurements exist;
- inferred depth p10/median/p90 for finite valid results;
- rejection counts for back-projection failure, descriptor threshold,
  epipolar-plane check, divergent rays, depth/near-camera check, projection
  failure, and final reprojection threshold;
- new landmark and initialized-landmark counts.

Ray angle is computed from the normalized world-frame bearing rays actually
used by triangulation. Baseline is the distance between the two camera centers,
not body-origin translation.

### `vio_diag_initialisation.csv`

One row per `runRansac2d2d()` camera/frame-pair comparison, including the
fewer-than-10-correspondence path:

- current and older frame id, camera and invocation;
- correspondence count;
- `computeModel()` status, inlier count and ratio for both rotation-only and
  relative-pose models;
- selected branch, selected inliers and final function success.

This stream diagnoses the initialization path only. A forced fallback
`rotationOnly=true` after complete failure is serialized as `selection=none`,
not a successful pure-rotation estimate. The constrained OpenGV model is not
used to claim that an estimated essential matrix norm literally becomes zero.

### `vio_diag_ransac.csv`

One row for every `runRansac3d2d()` invocation, including invocations with too
few correspondences:

- invocation index within the frame;
- primary trigger plus a trigger bitmask: no-IMU initialization, large
  reprojection error, too few accepted matches, or retry with uninitialized
  landmarks;
- correspondence, inlier, outlier, and removed-observation counts;
- inlier ratio and success flag;
- correspondence and inlier counts per camera;
- occupied-cell fraction of correspondences and inliers;
- immutable state pose at data-association entry and immediate pre-invocation
  pose;
- GP3P pose when a valid model exists;
- rotation and translation corrections from both recorded poses to GP3P;
- model-computation status, including no-prior-frame and `<10
  correspondences` early returns.

The entry pose is named for its exact capture point after `addStates()`. The
second GP3P invocation follows visual optimization, so its immediate
pre-invocation pose must not be described as raw IMU propagation.

The current boolean return is ambiguous because `<10 correspondences` is
converted to `bool`. The diagnostic record must report this case explicitly,
but production behavior is not changed as part of instrumentation.

### `vio_diag_landmark_events.csv`

One row per lifecycle event that changes map support:

- globally increasing event sequence, landmark id and event type;
- event frame/time, changed subject frame/time, and landmark birth frame/time
  as distinct fields;
- initialized status before and after the event;
- observation count before and after the event;
- current quality where available;
- event reason.

Event types include `birth`, `initialized`, `deinitialized`, `observation_added`,
`observation_removed`, `landmark_removed`, and `landmark_merged`. Removal
reasons must distinguish:

- GP3P outlier;
- post-optimization reprojection outlier;
- uninitialized-landmark reprojection rejection;
- loop-closure reassociation or merge;
- state marginalization/IMU merge;
- realtime/full-graph synchronization;
- explicit landmark merge;
- unknown, which is retained rather than guessed.

High-frequency `observation_added` rows are controlled by
`OKVIS_DIAGNOSTICS_OBSERVATION_ADDS=1` and are off by default because the frame
summary already contains their count. Phase 1 enables them for one short run to
validate the count, then leaves them off for population replay. Removal and
lifecycle-transition events remain enabled because they establish temporal
ordering.

### `vio_diag_metadata.csv`

A `schema_version,key,value` event stream for run-level metadata, containing:

- schema version and build/git revision when available;
- process start time;
- camera count and relevant frontend thresholds;
- diagnostic feature flags;
- output status and first write error, if any;
- column definitions that affect interpretation, including grid dimensions,
  quantiles, RANSAC thresholds, and triangulation source categories.

The writer emits startup keys after all files open successfully and appends
`run_complete=true` only after all diagnostic streams have been flushed during
normal shutdown. A crash or write failure therefore cannot masquerade as a
complete run. Duplicate keys use the last value, allowing a final status to
supersede the startup status without rewriting the file.

## Source Integration Points

The implementation must preserve the existing uncommitted EUCM-related change
in `okvis_frontend/src/Frontend.cpp`.

- `Frontend::detectAndDescribe()`: keypoint count, response, and spatial
  coverage inputs.
- `Frontend::dataAssociationAndInitialization()`: final per-frame summary and
  high-level state flags.
- `Frontend::matchToMap()` and its worker methods: eligible landmarks,
  descriptor acceptance, predicted reprojection error, accepted observations,
  and rejection categories.
- `Frontend::matchMotionStereo()`: temporal ray geometry, pixel displacement,
  triangulation outcome, birth, and initialization transition counts.
- same-frame stereo and uninitialized-landmark triangulation call sites:
  corresponding spatial/re-triangulation categories.
- `Frontend::runRansac2d2d()`: initialization-only rotation-versus-relative
  model support and explicit complete-failure state.
- `Frontend::runRansac3d2d()`: trigger, correspondence/inlier distribution,
  observation removals, frame-entry pose, immediate pre-invocation pose, and
  both corrections to the GP3P model.
- `Frontend::removeOutliers()`: post-optimization reprojection rejection.
- `ViGraph::updateLandmarks()`: initialization transitions and current
  accepted-ray geometry/quality summary.
- `ViGraphEstimator` and `ViSlamBackend` observation/landmark removal paths:
  reasoned lifecycle events, especially marginalization, merge, and graph
  synchronization.

Do not instrument `triangulateFast()` with direct file writes. It lacks caller
context and is used in hot paths. Its callers already have the source category,
camera/frame identities, pixel measurements, and output flags required for the
aggregate record.

## Runtime and Failure Behavior

- With `OKVIS_DIAGNOSTICS_DIR` unset, diagnostics add only a cheap disabled
  branch and do not allocate per-feature diagnostic vectors.
- With diagnostics enabled, filesystem writes happen after worker joins and
  outside matching inner loops.
- The writer creates only the explicitly supplied diagnostics directory. It
  does not delete or truncate unrelated paths.
- The writer atomically claims an active sentinel and refuses any existing
  diagnostic CSV, active sentinel, or complete sentinel. It never truncates a
  prior run even when started manually outside the rerun harness.
- Directory/open/write failure logs one clear error, marks diagnostics failed,
  and disables further writes without terminating VIO.
- CSV output uses the classic locale, fixed headers, finite-number checks, and
  empty fields for unavailable quantities. `nan` and guessed sentinel values
  are not used as measurements.
- The analysis pipeline rejects a run if metadata is missing, schema versions
  differ, frame timestamps are not monotonic, landmark event sequences are not
  increasing, or diagnostics report a write failure. Subject timestamps may
  be older than event timestamps by design.

## Analysis Design

### Event alignment

Join diagnostics to raw IMU and mocap using `timestamp_ns`. Apply the configured
OKVIS `image_delay` correction to raw camera-index timestamps before joining
offline image metrics. Define angular
events using the established `>3 rad/s`, minimum 50 ms, and 250 ms merge-gap
rule. Preserve continuous angular speed, event duration, peak speed, and
integrated angle rather than reducing each event to a binary label.

For the impulse subgroup, produce event-aligned traces over at least a
pre-event baseline, the event, and a post-event recovery interval. The exact
window is bounded by available data and recorded in the output table; the
primary view should include approximately 5 s before and 10 s after the main
event.

### Stage-specific mediators

Compute frame-level mediator families without combining them into one score:

- feature availability and spatial coverage;
- map-match acceptance and predicted reprojection error;
- accepted temporal/spatial ray geometry and triangulation outcome;
- initialized-landmark replenishment versus observation-removal rate;
- GP3P correspondence/inlier support and pose correction;
- active initialized-map support and recovery time.

Use absolute values and within-sequence change from the pre-event baseline.
The baseline-normalized view helps compare cameras and sequences, but it never
replaces physical units.

### Temporal ordering and recovery

For each event, identify change points or threshold crossings for:

1. feature/geometry/prediction mediators;
2. first GP3P correspondence or inlier collapse;
3. observation-removal burst;
4. active-map support decline;
5. persistent trajectory-error onset against mocap.

Report onset uncertainty at the camera-frame resolution. A proposed mediator
does not explain the chain if it changes only after GP3P/map failure.

Pre-register baseline `[start-5,start-1] s`, mediator
`[start,end+0.5] s`, GP3P onset scan `[start,end+2] s`, lagged GP3P persistence
`[end+0.5,end+2] s`, map-support `[end+2,end+5] s`, and late-recovery
`[end+5,end+10] s` windows. The onset scan, not the post-event persistence
window, determines whether GP3P failed before a proposed mediator. Estimate the
online-trajectory rigid alignment only from the healthy pre-event window and
freeze it for later error; whole-trajectory alignment is reserved for
sequence-level APE.

The `175103` negative control is decisive for recovery: determine which
mediators return to their pre-event range before a feedback loop starts, then
compare the same recovery time and residual deficit with `175304/175539`.

### Statistical model

Frame rows are not independent samples. Use sequence/event-clustered bootstrap
confidence intervals or a mixed-effects/distributed-lag model. At minimum test:

1. angular exposure predicts a subsequent mediator change;
2. lagged mediator state predicts subsequent GP3P/map-support failure after
   controlling for angular exposure and pre-event map health;
3. the direct angular-exposure coefficient is attenuated when the mediator is
   included;
4. conclusions retain direction when impulse experiments and known mocap
   correction sequences are excluded.

Before model fitting, match each angular event to low-angular windows from the
same sequence/run on pre-event map support, external mocap
translation/rotation, keypoint count, and image sharpness. Report unmatched
events rather than substituting unrelated controls. The implementation plan
fixes exact windows, model equations, calipers, and minimum sample counts.

Coefficient attenuation is mediation evidence, not proof. Results from the 24
non-orthogonal sequences remain observational.

## Controlled Experiments

After replay validation, collect orthogonal experiments to separate the three
causal paths:

1. Hold scene, angular profile, exposure, and lighting fixed; vary translation
   baseline from near-zero to moderate.
2. Hold scene and six-degree-of-freedom motion fixed; vary exposure time or
   illumination to alter exposure-period image motion.
3. Hold motion fixed; vary scene texture/feature density.
4. Repeat matched angular impulses with deliberate camera-IMU timestamp offsets
   of `-10/-5/-2/0/+2/+5/+10 ms` in offline replay to measure prediction
   sensitivity. These are analysis-only replays and do not alter acquisition
   timestamps in the archived source dataset.

Each cell needs repeated runs. Do not infer a universal angular-speed threshold
from the three existing impulse sequences.

## Evidence Upgrade Criteria

A middle link can be reported as strongly supported only when all applicable
conditions hold:

1. **Exposure/dose relation:** larger angular exposure produces a larger or
   more persistent mediator change.
2. **Temporal precedence:** the mediator changes before GP3P failure,
   observation-removal bursts, and active-map collapse.
3. **Recovery contrast:** the mediator recovers in `175103` but does not recover
   in the two failed impulse sequences.
4. **Specificity:** high angular motion without mediator degradation does not
   cause the same persistent failure, and mediator degradation from another
   controlled source produces the expected downstream state.
5. **Intervention:** changing exposure, translation baseline, texture, or
   timing while holding angular motion fixed changes the downstream failure
   probability in the predicted direction.
6. **Replication:** direction and timing repeat across runs and are not driven
   by one date, one sequence, or one camera.

Until intervention and replication are available, report the result as a
time-ordered mechanism consistent with the data, not a proven universal cause.

## Rollout and Validation

### Phase 1: instrumentation validation

- Add unit tests for CSV headers, disabled behavior, row escaping/finite-value
  handling, and write-failure behavior.
- Add focused frontend tests for accumulator merging and quantiles without
  requiring a full dataset.
- Build the affected OKVIS libraries with existing warnings enabled.
- Run one short healthy sequence once with diagnostics disabled and once
  enabled; confirm output completeness and measure runtime overhead.
- Confirm diagnostics-disabled trajectory output is unchanged relative to the
  same source build.

### Phase 2: impulse experiment replay

- Replay `175103/175304/175539` twice each into fresh result directories.
- Recompute APE and confirm the qualitative baseline outcome: `175103`
  recovers, while `175304/175539` retain their failure modes.
- Generate event-aligned mediator and downstream-state plots.
- Identify which hypotheses survive temporal ordering and recovery checks.

### Phase 3: all-sample replay

- Replay all 24 sequences, retaining run identity and diagnostics manifest.
- Produce population distributions, event-level models, sensitivity cohorts,
  and per-camera checks.
- Update the department report without replacing direct metrics with the
  offline visual-fragmentation label.

### Phase 4: controlled acquisition/replay

- Execute the orthogonal experiment matrix.
- Estimate failure probability and mediator response with uncertainty.
- Upgrade or reject each causal path using the evidence criteria above.

## Non-Goals

- No tracking, RANSAC, triangulation, or estimator threshold changes are part
  of this instrumentation task.
- No recovery mechanism or production alarm is designed before the causal
  diagnostics are analyzed.
- No claim that `E`-matrix degeneracy explains runtime `RANSAC FAIL`; the
  runtime path remains non-central 3D-2D GP3P.
- No use of final-map landmark quality as a standalone trigger measurement.
- No per-feature image or descriptor dump for all 24 sequences in the first
  implementation. A bounded detailed trace can be added later if aggregate
  diagnostics leave a specific ambiguity.
