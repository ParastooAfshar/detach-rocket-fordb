# DETACH-ROCKET on the FordB Dataset

## Overview

This project reproduces and extends the DETACH-ROCKET feature-selection pipeline for binary time-series classification on the FordB dataset.

ROCKET transforms each time series into a high-dimensional representation using random convolutional kernels. DETACH then performs sequential, coefficient-based feature pruning to identify a compact subset of informative ROCKET features.

The project evaluates:

1. Full ROCKET using all 20,000 transformed features
2. DETACH-ROCKET feature selection
3. SelectKBest
4. Random feature pruning
5. RidgeClassifierCV directly on the raw time-series samples

In addition to the original single-run analysis, the project includes a **multi-seed matched-budget evaluation** in which DETACH and SelectKBest are compared using the same number of selected features for each ROCKET seed.

---

## Dataset

The experiments use the **FordB** binary time-series classification dataset from the UCR archive.

| Property | Value |
|---|---:|
| Training samples | 3,636 |
| Test samples | 810 |
| Time points per sample | 500 |
| Classes | -1, +1 |
| Missing values | None |

Class distribution:

| Partition | Class -1 | Class +1 |
|---|---:|---:|
| Train | 1,860 | 1,776 |
| Test | 401 | 409 |

The two classes are approximately balanced.

---

## ROCKET Transformation

ROCKET applies **10,000 random convolutional kernels** to each time series.

Two features are extracted from each kernel response:

- **MAX** — maximum convolution response
- **PPV** — proportion of positive values

This produces **20,000 features per sample**.

```text
Training: 3,636 × 500 → 3,636 × 20,000
Test:       810 × 500 →   810 × 20,000
```

---

## Experimental Pipeline

The final pipeline avoids information leakage:

1. Load the original FordB training and test partitions.
2. Generate ROCKET features.
3. Split the transformed training set into stratified sub-training and validation partitions.
4. Fit `StandardScaler` only on the sub-training partition.
5. Apply the fitted scaler to validation and test data.
6. Run DETACH using only sub-training and validation data.
7. Fit comparison feature-selection methods on the same sub-training data.
8. Train Ridge classifiers on the selected feature subsets.
9. Use the independent FordB test partition only for final evaluation.

Final split sizes:

```text
Sub-training: 2,908 × 20,000
Validation:     728 × 20,000
Test:           810 × 20,000
```

---

## DETACH Configuration

The main DETACH configuration is:

```python
DetachMatrix(
    trade_off=0.1,
    recompute_alpha=False,
    verbose=True
)
```

DETACH sequentially removes low-importance features according to the absolute coefficients of a Ridge classifier. Validation performance is used during the feature-detachment process.

---

# Results

## Single-Run Results

The original controlled experiment used a 789-feature budget.

| Method | Features | Accuracy | F1-score |
|---|---:|---:|---:|
| Full ROCKET | 20,000 | 80.25% | 80.58% |
| **DETACH-ROCKET** | **789** | **81.48%** | **81.53%** |
| SelectKBest | 789 | 80.62% | 80.92% |
| Random Pruning | 789 | 78.91% ± 1.01% | 78.85% ± 0.95% |
| Raw Ridge | 500 | 48.89% | 47.06% |

Random Pruning is reported as mean ± standard deviation over 10 random seeds.

For the single-run equal-budget comparison, DETACH produced the highest observed Accuracy and F1-score.

---

## Feature Compression

In the main single-run experiment, DETACH reduced the feature space from 20,000 to 789 features.

| Metric | Value |
|---|---:|
| Initial features | 20,000 |
| Selected features | 789 |
| Removed features | 19,211 |
| Retained features | 3.945% |
| Feature reduction | 96.055% |
| Compression ratio | 25.35× |

Thus, the selected representation was approximately **25 times smaller** than the full ROCKET representation.

The Full ROCKET and DETACH accuracy values are descriptive results and should not be interpreted as a controlled equal-budget comparison.

---

# Multi-Seed Matched-Budget Evaluation

To evaluate whether the result was dependent on a single ROCKET initialization, ROCKET was repeated using five random seeds:

```text
0, 1, 2, 3, 4
```

For every seed:

- ROCKET generated a new 20,000-dimensional representation.
- The same stratified sub-training/validation split procedure was used.
- The scaler was fitted only on the corresponding sub-training partition.
- DETACH selected its feature subset.
- SelectKBest was then restricted to **exactly the same number of features selected by DETACH for that seed**.

This provides a paired, matched-feature-budget comparison.

## Selected Feature Counts

| ROCKET Seed | DETACH Feature Budget |
|---:|---:|
| 0 | 153 |
| 1 | 405 |
| 2 | 405 |
| 3 | 497 |
| 4 | 1,253 |

The variation in selected feature count indicates that DETACH's selected representation depends on the ROCKET feature realization.

---

## Multi-Seed Performance

| Method | Accuracy | F1-score |
|---|---:|---:|
| **DETACH** | **80.91% ± 1.05%** | **80.97% ± 1.00%** |
| SelectKBest — matched budget | 77.56% ± 1.46% | 78.66% ± 0.86% |

Across the five paired ROCKET seeds, DETACH achieved higher observed test Accuracy and F1-score than matched-budget SelectKBest in all five runs.

Mean paired improvement:

- **Accuracy:** +3.36 percentage points
- **F1-score:** +2.30 percentage points

---

## Statistical Comparison

A paired t-test was applied to the five seed-level paired results.

| Metric | t-statistic | df | p-value |
|---|---:|---:|---:|
| Accuracy | 4.085 | 4 | **0.015** |
| F1-score | 3.609 | 4 | **0.023** |

Under this five-seed experiment, the paired t-test indicates a statistically significant difference between DETACH and matched-budget SelectKBest for both metrics.

Because the analysis contains only five paired ROCKET seeds, these significance results should be interpreted as limited-sample evidence rather than a general conclusion across datasets or experimental settings.

---

## Key Observations

The experiments support the following observations for FordB:

1. **ROCKET representations are substantially more effective than raw Ridge classification.**  
   Raw Ridge achieved 48.89% accuracy, compared with approximately 80% using ROCKET-derived representations.

2. **The 20,000-dimensional ROCKET representation contains substantial redundancy.**  
   DETACH retained only a small fraction of the generated features while preserving strong test performance.

3. **Feature selection matters.**  
   Randomly retaining the same number of features did not reproduce the DETACH result in the original controlled experiment.

4. **DETACH remained stronger than SelectKBest under matched feature budgets across five ROCKET seeds.**

5. **The number of DETACH-selected features is not constant across ROCKET realizations.**  
   The five-seed experiment selected between 153 and 1,253 features.

---

## Extensions Added in This Project

Beyond reproducing the core DETACH-ROCKET workflow, this project adds:

- leakage-free preprocessing
- F1-score evaluation
- Raw Ridge baseline
- SelectKBest baseline
- Random Pruning baseline
- 10-seed Random Pruning evaluation
- multi-seed ROCKET evaluation
- matched-budget DETACH–SelectKBest comparison
- paired statistical testing
- feature-compression analysis
- reproducible CSV result files
- lightweight reusable pipeline functions
- unit tests
- pinned core dependencies
- updated documentation

---

## Project Structure

```text
detach-rocket-fordb/
│
├── README.md
├── LICENSE
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

Python 3.10 was used for the experiments.

Create and activate a virtual environment, then install the required dependencies:

```bash
python -m pip install -r examples/requirements.txt
```

Core versions are pinned in `examples/requirements.txt`.

---

## Running the Tests

From the project root:

```bash
python -m pytest
```

The current lightweight test suite checks core preprocessing, evaluation, dataset-loading, and leakage-related behavior.

---

## Main Notebook

The experimental workflow is available in:

```text
examples/Detach_ROCKET_example_UCR.ipynb
```

---

## Main Result Files

### Multi-Seed Evaluation

```text
examples/detach_multiseed_5seeds.csv
examples/selectkbest_matched_budget_5seeds.csv
examples/matched_budget_detach_vs_selectkbest_5seeds.csv
examples/multiseed_matched_budget_summary.csv
```

### Original Experiment Outputs

```text
examples/clean_detach_results.csv
examples/select_kbest_results.csv
examples/clean_random_pruning_10_runs.csv
examples/final_clean_feature_selection_comparison.csv
examples/detach_compression_analysis.csv
```

---

## Reproducibility

The project uses explicit random seeds where possible.

For the main pipeline:

```text
Training-validation split random_state = 42
ROCKET main experiment random_state = 42
```

The multi-seed ROCKET evaluation uses:

```text
0, 1, 2, 3, 4
```

Random Pruning was evaluated using:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The test set is not used to fit the scaler or perform feature selection.

An explicit internal DETACH random seed was not independently verified in the final implementation, so reproducibility claims regarding DETACH's internal behavior should be interpreted accordingly.

---

## Current Limitations

The current project has several limitations:

- The analysis is restricted to a single dataset, FordB.
- The DETACH–SelectKBest multi-seed comparison contains only five ROCKET seeds.
- Five paired observations provide limited statistical power and limited evidence about the distribution of performance across random initializations.
- DETACH selected substantially different feature counts across ROCKET seeds.
- Random Pruning was evaluated over 10 seeds, whereas the matched-budget DETACH–SelectKBest comparison currently uses five.
- The Full ROCKET and DETACH results are not a controlled equal-budget comparison.
- The current statistical results should not be generalized to other UCR datasets without additional experiments.
- The project does not yet include a broad benchmark across multiple datasets.

---

## License and Attribution

This project includes and modifies code from the official
[Detach-ROCKET repository](https://github.com/gon-uri/detach_rocket),
distributed under the BSD 3-Clause License.

The original DETACH-ROCKET implementation is credited to its authors.

The FordB experiments, additional baselines, leakage-free evaluation pipeline,
multi-seed matched-budget analysis, statistical testing, documentation, tests,
and result files were added in this project.

See `LICENSE` for the full license text.

---

## References

1. Dempster, A., Petitjean, F., and Webb, G. I.  
   **ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.**  
   *Data Mining and Knowledge Discovery*, 34, 1454–1495, 2020.

2. **DETACH-ROCKET: Sequential feature selection for time-series classification with random convolutional kernels.**

3. Dau, H. A., et al.  
   **The UCR Time Series Archive.**  
   *IEEE/CAA Journal of Automatica Sinica*, 6(6), 1293–1305, 2019.

---

## Conclusion

On FordB, DETACH-ROCKET produced a highly compressed ROCKET representation while maintaining strong classification performance.

In the original experiment, DETACH retained **789 of 20,000 features**, corresponding to a **96.055% reduction** and a **25.35× compression ratio**, while achieving **81.48% Accuracy** and **81.53% F1-score**.

In the five-seed matched-budget experiment, DETACH achieved:

- **80.91% ± 1.05% Accuracy**
- **80.97% ± 1.00% F1-score**

compared with:

- **77.56% ± 1.46% Accuracy**
- **78.66% ± 0.86% F1-score**

for matched-budget SelectKBest.

The paired differences favored DETACH for all five ROCKET seeds, with paired t-test p-values of **0.015 for Accuracy** and **0.023 for F1-score**. Given the small number of seeds and the use of a single dataset, these results should be interpreted cautiously and validated on additional datasets.