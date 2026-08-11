# EGO2 Four-Day Department Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the 20260806 EGO2 results into the existing 20260803-20260805 analysis directory as one self-contained four-day report with regenerated tables and figures.

**Architecture:** Extend the existing multiday analyzer so its default dataset range is 20260803-20260806 and all plot titles derive from the selected days. Regenerate the cross-day artifacts in place, copy the already verified 20260806 diagnostics into a non-conflicting consolidated naming scheme, then build one canonical Markdown report whose figures and tables all resolve inside the consolidated directory.

**Tech Stack:** Python 3.10, NumPy, SciPy, evo, Matplotlib, CSV, Markdown, unittest.

---

### Task 1: Make the multiday analyzer four-day aware

**Files:**
- Modify: `tools/accuracy_analysis/scripts/test_analyze_multiday.py`
- Modify: `tools/accuracy_analysis/scripts/analyze_multiday.py`

- [ ] **Step 1: Add failing default-range and title tests**

Add tests asserting that `DEFAULT_DAYS` includes `20260806`, the default expected sequence count is 24, `DAY_COLORS`/`DAY_MARKERS` include `20260806`, and `analysis_period_label()` returns `Four-day (20260803-20260806)` for the default range.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
conda run -n okvis2x python -m unittest discover -s tools/accuracy_analysis/scripts -p 'test_analyze_multiday.py'
```

Expected: failure because the default range is still three days and `analysis_period_label()` does not exist.

- [ ] **Step 3: Implement the minimal four-day behavior**

Set `DEFAULT_DAYS = ("20260803", "20260804", "20260805", "20260806")`, add 0806 visual styles, introduce `DEFAULT_EXPECTED_SEQUENCES = 24`, and derive figure titles from `analysis_period_label(sequence_rows)` instead of hard-coded `Three-day` text. Preserve the 0805-only comparison columns for compatibility with existing CSV consumers.

- [ ] **Step 4: Run focused and complete tool tests**

Run the focused multiday test, then all accuracy-analysis and trajectory-visualization tests. Expected: new tests pass; record any unrelated pre-existing failure separately without changing its ownership surface.

### Task 2: Regenerate the consolidated multiday artifacts

**Files:**
- Regenerate: `workspace/ego2_results/20260803_20260805_accuracy_analysis/tables/multiday_*.csv`
- Regenerate: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/01_*.png`
- Regenerate: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/02_*.png`
- Regenerate: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/03_*.png`
- Regenerate: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/04_*.png`

- [ ] **Step 1: Run the analyzer with the four default days**

```bash
conda run -n okvis2x python tools/accuracy_analysis/scripts/analyze_multiday.py
```

Expected: 24 sequence progress rows, 48 run rows, and output written to the existing consolidated directory.

- [ ] **Step 2: Validate regenerated cardinalities and classifications**

Assert 24 sequence rows, 48 run rows, four distinct days, two classification rows, and no empty CSV. Recompute the APE threshold counts and 3 rad/s confusion matrix directly from CSV.

- [ ] **Step 3: Inspect all four regenerated figures**

Verify each PNG is nonblank and that the multiday titles show the four-day range rather than `Three-day`.

### Task 3: Consolidate 20260806 diagnostics without filename collisions

**Files:**
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/05_20260806_accuracy_summary.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/06_20260806_prefix_drift_timeline.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/07_20260806_failure_factor_correlations.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/08_20260806_angular_velocity_exposure.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/09_20260806_triangulation_quality.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/10_observability_motion_proxies.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/11_observability_failure_and_scale.png`
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/figures/12_20260806_trajectories.png`
- Create: corresponding `tables/20260806_*.csv` and `tables/observability_*.csv`

- [ ] **Step 1: Copy verified 20260806 figures with consolidated numbering**

Map the existing 0806 figures to 05-12 so 01-04 remain owned by the multiday analyzer.

- [ ] **Step 2: Copy the 0806 and observability CSVs**

Preserve source filenames for tables because they do not collide with the `multiday_*` outputs.

- [ ] **Step 3: Compare source and destination hashes**

Use SHA-256 to prove each consolidated copy exactly matches its verified source artifact.

### Task 4: Build and verify the department-facing report

**Files:**
- Create: `workspace/ego2_results/20260803_20260805_accuracy_analysis/report_20260803to20260806.md`
- Modify: `workspace/ego2_results/20260803_20260805_accuracy_analysis/report_20260803to20260805.md`

- [ ] **Step 1: Generate the four-day summary numbers from CSV**

Calculate per-day APE ranges, total `APE > 10 mm` count, true visual-failure group, RANSAC/lifetime separation, four-day Spearman coefficients, alarm confusion matrices, 0806 severity groups, and Sim(3) medians.

- [ ] **Step 2: Write the canonical report**

Cover scope and evaluation rules, four-day totals, all 24 sequences, confirmed bad-mocap correction, visual failure chain, 0806 findings, geometric observability/scale diagnosis, alarm guidance, sensor integrity, limitations, and a self-contained figure/table index.

- [ ] **Step 3: Mark the old three-day report as superseded**

Add a prominent top-of-file note linking to the four-day canonical report while retaining the historical content for traceability.

- [ ] **Step 4: Run artifact verification**

Parse every consolidated CSV, resolve every Markdown link, verify PNG dimensions and pixel variance, confirm 24 sequences/48 runs, and scan consolidated figure titles for stale `Three-day`/0803-0805 wording.

- [ ] **Step 5: Re-run final tests and report the exact results**

Run the multiday focused tests, evaluator tests, trajectory visualization tests, and the artifact verifier. Do not claim full completion if any report-owned check fails.
