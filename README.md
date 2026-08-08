# DETACH-ROCKET on the FordB Dataset

## Overview

This project reproduces and extends the DETACH-ROCKET feature-selection pipeline for binary time-series classification on the FordB dataset.

ROCKET transforms each time series into a high-dimensional representation using random convolutional kernels. DETACH then performs sequential feature detachment to identify a compact subset of informative ROCKET features.

The project evaluates five approaches:

1. Full ROCKET
2. DETACH-ROCKET
3. SelectKBest
4. Random Pruning
5. RidgeClassifierCV on the original time-series samples

Two complementary experiments are reported:

- a **main equal-budget comparison** using a fixed budget of 789 features for DETACH, SelectKBest, and Random Pruning;
- a **five-seed matched-budget comparison** between DETACH and SelectKBest, where SelectKBest uses exactly the number of features selected by DETACH for each ROCKET seed.

---

## Dataset

The experiments use the **FordB** dataset from the UCR Time Series Archive.

| Property | Value |
|---|---:|
| Training samples | 3,636 |
| Test samples | 810 |
| Time points per sample | 500 |
| Classes | -1 and +1 |
| Missing values | None |

Class distribution:

| Partition | Class -1 | Class +1 |
|---|---:|---:|
| Train | 1,860 | 1,776 |
| Test | 401 | 409 |

The classes are approximately balanced.

---

## ROCKET Transformation

ROCKET applies **10,000 random convolutional kernels** to each time series.

Each kernel produces two features:

- **MAX** — maximum convolution response
- **PPV** — proportion of positive values

This results in **20,000 transformed features per sample**.

```text
Training: 3,636 × 500 → 3,636 × 20,000
Test:       810 × 500 →   810 × 20,000
```

---

## Leakage-Free Experimental Pipeline

The final experimental pipeline is designed to avoid information leakage.

1. Load the original FordB training and test partitions.
2. Generate the ROCKET representation.
3. Split the transformed training data into stratified sub-training and validation partitions.
4. Fit `StandardScaler` only on the sub-training partition.
5. Transform validation and test data using the fitted scaler.
6. Run DETACH using sub-training and validation data.
7. Fit comparison feature-selection methods on the same sub-training data.
8. Train Ridge classifiers on the selected feature subsets.
9. Evaluate final predictions on the independent FordB test set.

Final split dimensions:

```text
Sub-training: 2,908 × 20,000
Validation:     728 × 20,000
Test:           810 × 20,000
```

The independent test partition is not used to fit the scaler or select features.

---

## DETACH Configuration

The DETACH configuration used in the experiments is:

```python
DetachMatrix(
    trade_off=0.1,
    recompute_alpha=False,
    verbose=True
)
```

DETACH performs sequential feature detachment based on classifier-derived feature importance, while validation performance guides the feature-selection process.

---

## Results

### Main Evaluation

The main controlled feature-selection experiment uses a budget of **789 features** for DETACH, SelectKBest, and Random Pruning.

The Full ROCKET and Raw Ridge results are included as additional reference baselines.

| Method | Features | Accuracy | F1-score |
|---|---:|---:|---:|
| Full ROCKET | 20,000 | 80.25% | 80.58% |
| **DETACH-ROCKET** | **789** | **81.48%** | **81.53%** |
| SelectKBest | 789 | 80.62% | 80.92% |
| Random Pruning | 789 | 78.91% ± 1.01% | 78.85% ± 0.95% |
| Raw Ridge | 500 | 48.89% | 47.06% |

Random Pruning is reported as mean ± standard deviation over 10 independent random seeds.

### Equal-Budget Feature-Selection Comparison

DETACH, SelectKBest, and Random Pruning all retain **789 features** in this comparison.

| Method | Features | Accuracy | F1-score |
|---|---:|---:|---:|
| **DETACH-ROCKET** | **789** | **81.48%** | **81.53%** |
| SelectKBest | 789 | 80.62% | 80.92% |
| Random Pruning | 789 | 78.91% ± 1.01% | 78.85% ± 0.95% |

Observed DETACH improvement over SelectKBest:

- Accuracy: **+0.86 percentage points**
- F1-score: **+0.60 percentage points**

Observed DETACH improvement over Random Pruning:

- Accuracy: **+2.57 percentage points**
- F1-score: **+2.68 percentage points**

The three feature-selection methods use the same 789-feature budget. Random Pruning is summarized over 10 random seeds, while the displayed DETACH and SelectKBest values correspond to the main experiment.

Full ROCKET is reported separately and should not be interpreted as an equal-budget comparison with DETACH because it retains all 20,000 ROCKET features.

---

### Feature Compression

In the main experiment, DETACH reduced the ROCKET representation from 20,000 to 789 features.

| Metric | Value |
|---|---:|
| Initial features | 20,000 |
| Selected features | 789 |
| Removed features | 19,211 |
| Retained features | 3.945% |
| Feature reduction | 96.055% |
| Compression ratio | 25.35× |

DETACH therefore produced a representation approximately **25.35 times smaller** than the full ROCKET representation.

The observed test accuracies were:

```text
Full ROCKET:    80.25%
DETACH-ROCKET:  81.48%
```

Because Full ROCKET and DETACH use different feature counts, this comparison is descriptive rather than an equal-budget feature-selection comparison.

---

## Multi-Seed Matched-Budget Evaluation

A second experiment was performed to evaluate sensitivity to different random ROCKET representations.

Five ROCKET random seeds were evaluated:

```text
0, 1, 2, 3, 4
```

For each seed:

1. A new 20,000-feature ROCKET representation was generated.
2. The same stratified sub-training/validation split procedure was used.
3. `StandardScaler` was fitted only on the sub-training partition.
4. DETACH selected a feature subset.
5. SelectKBest was restricted to **exactly the same number of features selected by DETACH for that seed**.
6. Both methods were evaluated on the same independent FordB test set.

Therefore, the DETACH–SelectKBest multi-seed experiment is a **paired matched-budget comparison**.

### Selected Feature Budgets

| ROCKET Seed | DETACH Features | SelectKBest Features |
|---:|---:|---:|
| 0 | 153 | 153 |
| 1 | 405 | 405 |
| 2 | 405 | 405 |
| 3 | 497 | 497 |
| 4 | 1,253 | 1,253 |

For every ROCKET seed, SelectKBest uses exactly the same feature budget as DETACH.

The number of features retained by DETACH ranged from 153 to 1,253 across the five ROCKET representations.

---

### Per-Seed Matched-Budget Results

| Seed | Features | DETACH Accuracy | SelectKBest Accuracy | DETACH F1 | SelectKBest F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 153 | 80.99% | 75.80% | 81.22% | 78.03% |
| 1 | 405 | 80.86% | 76.79% | 80.93% | 77.99% |
| 2 | 405 | 82.47% | 77.78% | 82.38% | 78.57% |
| 3 | 497 | 79.51% | 77.65% | 79.61% | 78.63% |
| 4 | 1,253 | 80.74% | 79.75% | 80.69% | 80.10% |

DETACH achieved higher observed Accuracy and F1-score than matched-budget SelectKBest in all five paired runs.

---

### Multi-Seed Summary

| Method | Accuracy | F1-score |
|---|---:|---:|
| **DETACH** | **80.91% ± 1.05%** | **80.97% ± 1.00%** |
| SelectKBest — matched budget | 77.56% ± 1.46% | 78.66% ± 0.86% |

Mean paired difference in favor of DETACH:

- Accuracy: **+3.36 percentage points**
- F1-score: **+2.30 percentage points**

---

### Statistical Comparison

A paired t-test was applied to the five matched ROCKET-seed results.

| Metric | t-statistic | df | p-value |
|---|---:|---:|---:|
| Accuracy | 4.085 | 4 | **0.015** |
| F1-score | 3.609 | 4 | **0.023** |

For this five-seed experiment, the paired t-test detected statistically significant differences between DETACH and matched-budget SelectKBest for both Accuracy and F1-score at the conventional 0.05 significance level.

Because the statistical comparison contains only five paired observations, these inferential results should be interpreted cautiously and as evidence specific to the current FordB experiment.

---

## Key Observations

The experiments support the following observations on FordB:

1. **ROCKET feature extraction is important.**  
   Ridge classification on the original 500-point signals achieved 48.89% accuracy, whereas ROCKET-based approaches achieved approximately 80%.

2. **The ROCKET representation contains substantial redundancy on FordB.**  
   In the main experiment, DETACH retained only 789 of 20,000 features while maintaining strong classification performance.

3. **Feature-selection strategy matters.**  
   Randomly retaining the same 789-feature budget did not reproduce DETACH performance in the main controlled comparison.

4. **DETACH achieved higher observed performance than SelectKBest under matched feature budgets across all five evaluated ROCKET seeds.**

5. **The DETACH-selected feature count varied across ROCKET realizations.**  
   The selected feature count ranged from 153 to 1,253 across the five seeds.

---

## Extensions Added in This Project

Beyond reproducing the core DETACH-ROCKET workflow, this project adds:

- leakage-free scaling
- F1-score evaluation
- Raw Ridge baseline
- SelectKBest baseline
- Random Pruning baseline
- 10-seed Random Pruning evaluation
- multi-seed ROCKET evaluation
- matched-budget DETACH–SelectKBest evaluation
- paired statistical testing
- feature-compression analysis
- reusable preprocessing and evaluation utilities
- lightweight unit tests
- pinned core dependencies
- reproducible CSV result files
- a command-line entry point for smoke testing and baseline execution
- continuous integration with GitHub Actions
- explicit documentation of changes relative to the upstream repository

---

## Project Structure

```text
detach-rocket-fordb/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── CHANGES.md
├── LICENSE
├── README.md
├── requirements.txt
├── run_experiment.py
├── setup.py
│
├── src/
│   └── pipeline.py
│
├── tests/
│   └── test_pipeline.py
│
├── examples/
│   ├── Detach_ROCKET_example_UCR.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── detach_multiseed_5seeds.csv
│   ├── selectkbest_matched_budget_5seeds.csv
│   ├── matched_budget_detach_vs_selectkbest_5seeds.csv
│   └── multiseed_matched_budget_summary.csv
│
└── detach_rocket/
```

---

## Installation

The experiments were run using **Python 3.10.11**.

Install the pinned core dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

The root-level `requirements.txt` contains the dependencies required for the project environment.

---

## Computational Requirements

ROCKET generates 20,000 features per sample in the current configuration, and the multi-seed experiments repeat this transformation for several random kernel realizations. As a result, the full experimental workflow can require substantial memory and computation time.

Exact runtime and hardware benchmarks are not reported because the experiments were not executed under a controlled benchmarking setup. Runtime can vary substantially depending on CPU performance, available memory, parallelism, and software environment.

Users with limited computational resources may run the lightweight smoke test or the main single-seed experiment before attempting the complete multi-seed analysis.

---

## Tests

Run the test suite from the project root:

```bash
python -m pytest
```

The current lightweight test suite covers:

- FordB dataset loading
- stratified splitting
- scaler fitting behavior
- held-out test data not being used to fit the scaler
- Accuracy and F1-score evaluation utilities

The tests are also executed automatically through GitHub Actions on pushes and pull requests to the `main` branch.

---

## Main Notebook

The complete experimental analysis, including DETACH and the multi-seed matched-budget experiments, is available in:

```text
examples/Detach_ROCKET_example_UCR.ipynb
```

---

## Command-Line Entry Point

A command-line entry point is provided at:

```text
run_experiment.py
```

For a lightweight smoke test:

```bash
python run_experiment.py
```

This loads the FordB dataset and verifies the expected training and test shapes without running the computationally expensive ROCKET transformation.

To run the ROCKET + SelectKBest baseline:

```bash
python run_experiment.py --full
```

The `--full` option performs the 10,000-kernel ROCKET transformation, applies the leakage-free preprocessing pipeline, selects 789 features with SelectKBest, trains the Ridge classifier, and reports the resulting Accuracy and F1-score.

The complete DETACH and multi-seed analyses remain available in the main experimental notebook.

---

## Result Files

### Multi-Seed Matched-Budget Results

```text
examples/detach_multiseed_5seeds.csv
examples/selectkbest_matched_budget_5seeds.csv
examples/matched_budget_detach_vs_selectkbest_5seeds.csv
examples/multiseed_matched_budget_summary.csv
```

### Main Experiment Outputs

```text
examples/clean_detach_results.csv
examples/select_kbest_results.csv
examples/clean_random_pruning_10_runs.csv
examples/final_clean_feature_selection_comparison.csv
examples/detach_compression_analysis.csv
```

These files are retained as reproducible result artifacts associated with the reported experiments.

---

## Reproducibility

The main experiment uses:

```text
ROCKET random_state = 42
Train-validation split random_state = 42
```

The multi-seed ROCKET experiment uses:

```text
0, 1, 2, 3, 4
```

Random Pruning was evaluated using:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The independent FordB test partition is not used to fit the scaler or select features.

Core dependency versions are pinned in the root-level `requirements.txt`.

An explicit internal DETACH random seed was not independently verified in the final implementation. Therefore, claims regarding DETACH's internal deterministic behavior are kept limited.

---

## Upstream Code and Project Changes

The `detach_rocket/` directory is based on the original DETACH-ROCKET implementation by Gonzalo Uribarri and Federico Barone.

The primary contributions in this repository concern the FordB experimental workflow, leakage-free evaluation, additional baselines, multi-seed matched-budget analysis, statistical evaluation, testing, reproducibility, and project-level tooling.

A detailed summary of the changes relative to the upstream repository is provided in:

```text
CHANGES.md
```

This repository does not claim to provide a redesigned implementation of the core DETACH algorithm.

---

## Current Limitations

The current project has several limitations:

- The analysis is restricted to a single dataset, FordB.
- The matched-budget DETACH–SelectKBest multi-seed experiment contains only five paired ROCKET seeds.
- The paired t-test is based on only five paired observations, so its inferential results should be interpreted cautiously.
- The number of features selected by DETACH varies across ROCKET seeds.
- Random Pruning was evaluated over 10 random seeds, whereas the matched-budget DETACH–SelectKBest experiment uses five ROCKET seeds.
- Full ROCKET and DETACH use different feature counts; therefore, their direct performance comparison is descriptive rather than equal-budget.
- In contrast, the DETACH–SelectKBest comparisons described as equal-budget or matched-budget explicitly use the same feature count within each comparison.
- The command-line entry point currently reproduces the ROCKET + SelectKBest baseline rather than the complete DETACH and multi-seed workflow.
- The lightweight unit tests cover core preprocessing and evaluation utilities but do not execute the full computationally expensive experimental pipeline.
- The findings are specific to FordB and should not be generalized to other time-series datasets without additional experiments.
- A broader multi-dataset benchmark has not yet been performed.

---

## License and Attribution

This project includes and modifies code from the official
[Detach-ROCKET repository](https://github.com/gon-uri/detach_rocket),
which is distributed under the BSD 3-Clause License.

The original DETACH-ROCKET implementation is credited to its authors.

The FordB experiments, additional baselines, leakage-free evaluation,
multi-seed matched-budget comparison, statistical analysis, tests,
command-line tooling, CI configuration, and result files were added in this project.

See the `LICENSE` file for the full license text.

---

## References

1. Dempster, A., Petitjean, F., and Webb, G. I.  
   **ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.**  
   *Data Mining and Knowledge Discovery*, 34, 1454–1495, 2020.

2. Dau, H. A., et al.  
   **The UCR Time Series Archive.**  
   *IEEE/CAA Journal of Automatica Sinica*, 6(6), 1293–1305, 2019.

The DETACH-ROCKET implementation used in this project is linked in the License and Attribution section above.

---

## Conclusion

On the FordB dataset, DETACH-ROCKET produced compact ROCKET feature subsets while maintaining strong classification performance.

In the main experiment, DETACH retained:

```text
789 / 20,000 features
```

corresponding to:

- **96.055% feature reduction**
- **25.35× compression**

with:

- **81.48% Accuracy**
- **81.53% F1-score**

In the controlled 789-feature comparison, DETACH achieved higher observed Accuracy and F1-score than both SelectKBest and Random Pruning.

The additional five-seed experiment compared DETACH and SelectKBest under a matched feature budget for every ROCKET seed.

DETACH achieved:

- **80.91% ± 1.05% Accuracy**
- **80.97% ± 1.00% F1-score**

Matched-budget SelectKBest achieved:

- **77.56% ± 1.46% Accuracy**
- **78.66% ± 0.86% F1-score**

DETACH achieved higher observed Accuracy and F1-score in all five paired runs. The paired t-test produced p-values of **0.015 for Accuracy** and **0.023 for F1-score**.

These findings provide evidence that DETACH can identify compact and effective ROCKET feature subsets on FordB. Because the current evaluation is limited to one dataset and five paired ROCKET seeds for the matched-budget analysis, additional datasets and random initializations are needed before drawing broader conclusions.