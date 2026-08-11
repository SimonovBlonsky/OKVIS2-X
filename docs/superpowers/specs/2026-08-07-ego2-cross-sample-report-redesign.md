# EGO2 Cross-Sample Analysis and Report Redesign

## Objective

Rebuild the 20260803-20260806 EGO2 analysis as one population-level study of
24 sequences and 48 OKVIS runs. Dates are provenance fields, not analysis
groups. The only separately interpreted subset is the explicitly designed
`20260806-175103 / 175304 / 175539` stationary-plus-angular-impulse experiment.

The department-facing report must answer three questions:

1. Which sequences are inaccurate, visually fragmented, or scale-unstable?
2. Which measured factors are associated with those outcomes across all
   samples?
3. How strongly does the current evidence support each proposed trigger or
   failure-chain component?

## Design Principles

- Design the artifact set from a blank slate. Existing figures and tables are
  data-validation references only; their count, numbering, filenames,
  dimensions, panel composition, date grouping and figure/table boundaries
  impose no constraint on the replacement bundle.
- Derive each artifact from a report claim and its required evidence. Merge
  views when that improves comparison, and split them when density or scale
  would obscure the evidence. There is no target, minimum or maximum number
  of figures, tables, panels or trajectory pages.
- Use all 24 sequences as the primary cohort and all 48 runs for run-level
  metrics.
- Use semantic filenames without numeric prefixes.
- Keep `APE > 10 mm`, visual fragmentation, and scale instability as distinct
  labels.
- Keep candidate causes in the report even when evidence is weak, but state
  their support level and missing evidence explicitly.
- Do not present a proxy as a direct measurement. In particular, mocap body
  translation is not measured camera parallax, and Laplacian variance cannot
  distinguish blur from low scene texture.
- Preserve the three impulse experiments as a designed subgroup, not as a
  special interpretation of the whole 20260806 date.

## Scope and Evaluation Contract

- Results: `workspace/ego2_results/20260803` through `20260806`.
- Sensor datasets: `/home/chenguyuan/data/20260803` through `20260806`.
- Primary trajectory: each run's
  `okvis2-slam-calib-final-ba_trajectory.csv`.
- Primary accuracy: translation APE RMSE after timestamp association within
  10 ms and rigid SE(3) alignment without scale correction.
- Sequence value: median of run1 and run2.
- Scale diagnosis: Sim(3) is diagnostic only and never replaces the primary
  SE(3) APE.
- The confirmed mocap rigid-body configuration correction remains limited to
  `20260805-122310 / 123231 / 123752` and is identified on every affected row.
- Image sampling: 80 deterministic, uniformly spaced indexed frames per
  camera per sequence, matching the established 20260805 analysis default.

## Cohorts and Sensitivity Analysis

Every reported cross-sequence relationship has a primary full-cohort result
and the following sensitivity views when the metric is available:

1. `all`: all 24 sequences;
2. `without_impulse`: exclude `175103 / 175304 / 175539`;
3. `without_mocap_correction`: exclude `122310 / 123231 / 123752`;
4. `natural_uncorrected_subset`: exclude both special subsets.

The full cohort remains the headline result. Subsets test whether a conclusion
depends on experimental design or evaluation correction; they do not replace
the full-cohort result.

Outcome labels are:

- `ape_over_10mm`: corrected sequence median APE exceeds 10 mm;
- `visual_fragmentation`: recomputed for every sequence from run-median
  metrics, requiring both RANSAC FAIL rate at least `15/min` and final-map
  landmark median time span at most `3 s`; these cut points lie inside the two
  observed cross-sample gaps and are reported as offline descriptive labels,
  not deployment alarm thresholds;
- `scale_instability`: recomputed from corrected trajectories, requiring both
  sequence-median Sim(3) APE improvement of at least 25% over SE(3) and
  sequence-median scale deviation `|scale - 1|` of at least `0.10`;
- `impulse_experiment`: the three designed angular-impulse sequences.

## Unified Metrics

### Accuracy and repeatability

For all 48 runs and 24 sequence aggregates:

- raw and corrected APE;
- run1/run2 spread and ratio;
- associated poses and evaluation duration;
- online/final/final-BA stage metrics where available;
- pairwise run divergence where available.

### Angular motion

For every sequence:

- angular-speed median, p95, p99 and maximum;
- duration fractions above 1/2/3/4 rad/s;
- debounced 3 rad/s event count per 5 min;
- cumulative orientation path;
- impulse event peak, duration and integrated angular displacement for the
  designed impulse subset.

The dedicated APE-angular-speed analysis reports Spearman rho and sample count
for p95, p99, maximum, >3 rad/s fraction and event frequency across every
cohort. It must make low-duty-cycle dilution visible rather than treating p95
as the sole motion statistic.

### Tracking and loop closure

For every run, then aggregated by sequence:

- RANSAC FAIL count and rate;
- large-reprojection-error count and rate;
- uninitialised-landmark RANSAC count;
- dropped camera correspondence count;
- loop descriptor matches, attempts, acceptances, rates and rejection
  fraction.

### Landmark survival and triangulation

For every final G2O map:

- states, landmarks, observations and keypoints;
- observations per landmark and per state;
- distinct states per landmark;
- landmark time-span median and p95;
- single-state landmark fraction and landmark creation rate;
- `VERTEX_TRACKXYZ` quality count, median, p90 and p95;
- fraction above `0.001` and initialized fraction above the OKVIS threshold
  `0.04`.

The final G2O quality is explicitly interpreted as an estimator-coupled
posterior geometry measure, not an independent trigger.

### Image and sensor proxies

Use the same deterministic image sampling for all 24 datasets:

- Laplacian variance p5, p10 and median;
- intensity standard deviation and clipping fractions;
- previous-frame mean absolute difference;
- keypoints per camera frame from final G2O;
- camera indexed-file completeness and synchronization metrics;
- mocap tracking and interval integrity where logs expose them.

Laplacian variance and contrast are reported as image edge-content/sharpness
proxies. The report may discuss blur or low texture only as candidate
interpretations because these statistics do not separate the two.

### Observability and scale

Join the verified all-sample observability and Sim(3) tables:

- 0.5 s body-origin translation and rotation proxies;
- high-rotation/low-translation exposure;
- representative failure-window metrics;
- stereo support and local stereo support;
- SE(3)/Sim(3) APE and scale;
- local scale-deviation onset.

The body-origin baseline and ideal 3 m disparity remain proxy quantities. The
report states that actual accepted feature parallax and `isParallel` timing are
not present in current runtime logs.

## Evidence Model

Generate machine-readable evidence rows for every factor and cohort. Each row
contains metric coverage, Spearman rho, p-value for reference only, Cliff's
delta where a binary outcome is evaluated, group ranges, overlap status,
sensitivity-direction consistency, temporal evidence and counterexamples.

The grading is auditable rather than inferred from prose. For population
associations, use absolute Spearman rho of `0.60` as strong, `0.35` as
moderate and `0.20` as weak; for binary-group separation, use absolute
Cliff's delta of `0.80`, `0.474` and `0.147`, respectively. A cohort result is
eligible for grading only with at least 80% metric coverage and 12 sequences.
The population grade cannot exceed the full-cohort grade and is the lowest
threshold met across eligible sensitivity cohorts. A direction reversal
causes an explicit downgrade, while any required view below the coverage floor
caps population evidence at weak. Event-level temporal evidence is graded
separately and can be strong only when failure onset is aligned in at least two
designed or naturally repeated events and a documented counterexample is used
to delimit sufficiency. P-values are shown as small-sample context and do not
promote or demote a grade mechanically.

Support levels are assigned as follows:

- **Strong support:** meets the strong population threshold with stable
  sensitivity direction, or has strong replicated event-level timing.
- **Moderate support:** meets the moderate population threshold with stable
  direction, but retains group overlap or lacks direct timing evidence.
- **Weak support:** meets only the weak population threshold, is an indirect
  proxy, has partial coverage, is unstable across subsets, or is restricted to
  a small number of windows.
- **Currently not supported:** stays below the weak thresholds, reverses
  direction, or has direct counterexamples without positive timing evidence.
  The factor remains documented with the result that weakened it and the
  measurement needed to retest it.

No support label by itself establishes causality. The report separates:

1. the strongly supported downstream estimator failure chain;
2. candidate upstream triggers with graded evidence.

The downstream chain can include tracking inconsistency, landmark collapse,
weak posterior geometry, loop-closure overload and scale/trajectory
divergence only to the extent supported by the unified tables. High angular
speed, weak temporal parallax and image degradation are not inserted as a
proven starting node unless their cross-sample and event evidence warrants it.

## Software Architecture

### Cross-sample analyzer

Create
`tools/accuracy_analysis/scripts/analyze_cross_sample_diagnostics.py`.
It imports existing discovery, APE, motion, alarm, log, camera and G2O helpers
rather than duplicating them. Its responsibilities are:

1. discover and validate exactly 24 sequences and 48 runs;
2. load the regenerated multiday accuracy results;
3. collect uniform run-level tracking, topology and quality metrics;
4. collect uniform sequence-level image and sensor proxies;
5. join observability and scale inputs by sequence/run with cardinality checks;
6. produce the unified run and sequence CSVs;
7. compute cohort correlations, separation ranges and evidence rows;
8. render question-led figures from generated CSV rows.

Parsing reusable G2O landmark quality belongs in
`analyze_repeatability.py`, adjacent to existing map topology helpers.

### Trajectory visualizer

Create a grouped multiday entry point under `tools/traj_visualization/` that
reuses the established trajectory loading and plotting utilities. It discovers
the current `day/[group/]sequence/run` layout and renders every sequence with
run1/run2 and mocap reference in a consistent panel. Pagination or multiple
overview files are allowed when needed for legibility; no fixed page or figure
count is imposed.

### Artifact design and output bundle

The canonical bundle remains
`workspace/ego2_results/20260803_20260805_accuracy_analysis`. Artifact design
starts from the report's claims after the unified metrics have been computed,
using this decision process:

1. state the claim or comparison the artifact must support;
2. identify the rows, coverage and uncertainty needed to audit that claim;
3. choose a figure, table or prose result according to which form communicates
   the evidence most clearly;
4. determine panels and pagination from readability at final document size;
5. assign a descriptive filename after the content is settled.

Required analytical coverage includes overall APE, the dedicated all-sample
APE-angular-speed relationship, tracking and loop behavior, landmark survival,
triangulation quality, image and sensor proxies, observability and scale,
trajectories, and the designed angular-impulse timeline. This is a coverage
contract, not a prescribed mapping from topics to files: related questions may
share an artifact, and one question may require multiple artifacts.

Machine-readable outputs must expose the unified run rows, sequence rows,
correlations, evidence grades and the contributing metric families. Their
normalization may use one or several CSV files based on stable keys and audit
clarity; no legacy table boundary or filename is preserved merely for
continuity.

After the replacement bundle passes verification, copied 20260806-only figures
and other superseded consolidated artifacts are removed from the canonical
figure directory. Their original verified sources remain under
`workspace/ego2_results/20260806_analysis`.

## Report Structure

Rewrite `report_20260803to20260806.md` around analytical questions:

1. executive conclusions for the 24-sequence population;
2. scope, evaluation contract, outcome labels and cohort definitions;
3. overall accuracy and repeatability;
4. confirmed mocap configuration correction;
5. all-sample tracking fragmentation and loop behavior;
6. all-sample landmark survival and triangulation;
7. dedicated all-sample APE-angular-speed analysis;
8. candidate trigger evidence synthesis, including weak and unsupported
   factors;
9. stationary-plus-angular-impulse subgroup;
10. OKVIS source path, observability and scale interpretation;
11. sensor integrity, alarm guidance and engineering actions;
12. limitations and a self-contained artifact index.

Dates appear in the sequence table and provenance only. The prose does not use
“new 0806 result” framing. The impulse subgroup is justified by experimental
design, not by date.

## Error Handling

The analyzer fails with the exact sequence/run and missing artifact when any
required final trajectory, final map, mocap log, dataset, camera index or
joined row is absent. It rejects duplicate sequence/run keys, empty metric
arrays, non-finite required values and output cardinalities other than 24/48.
Optional metrics are allowed only when the output records an explicit
availability field and the report states the reduced coverage.

## Testing and Verification

Use test-first development for all new behavior.

Focused tests cover:

- `VERTEX_TRACKXYZ` quality parsing and thresholds;
- 48-run to 24-sequence aggregation;
- cohort membership and sensitivity subsets;
- correlation/evidence rows and support-level boundary cases;
- join rejection for missing or duplicate sequence/run keys;
- trajectory discovery and grouping for flat and grouped day layouts;
- claim-driven artifact selection, descriptive output naming and stable
  ordering without asserting an artifact count.

Final artifact verification requires:

- 24 unique sequence rows and 48 unique run rows;
- 24-sequence coverage for every primary cross-sample metric;
- explicit reduced-coverage metadata for any optional metric;
- every population diagnostic to cover all sequences for which its metric is
  available and to state the coverage explicitly;
- any main-report view limited to the three impulse sequences to answer the
  angular-impulse timing question explicitly; no other main diagnostic may
  silently substitute a date or hand-picked subset for the full population;
- successful CSV parsing and no empty table;
- every Markdown link resolving inside the canonical bundle;
- every PNG decoding with nonzero pixel variance and legible dimensions;
- visual inspection of trajectory pages and dense labels;
- report values and evidence grades recomputed from generated CSVs;
- no current report references to superseded date-only figures, numeric
  artifact slots or legacy figure/table groupings;
- focused accuracy-analysis, evaluator and trajectory-visualization tests.

The known unrelated `analyze_repeatability.py` default-result-path test failure
is reported separately and is not silently treated as a passing suite.

## Non-Goals

- Do not modify OKVIS estimator source or configuration.
- Do not infer actual accepted-feature parallax when it was not logged.
- Do not call Laplacian variance a direct blur or texture measurement.
- Do not estimate a causal failure probability from the three impulse trials.
- Do not preserve an old artifact count, numbering scheme or date-based layout.
- Do not modify original trajectories, maps, logs, images or mocap data.
