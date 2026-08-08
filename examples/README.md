# DETACH-ROCKET on the FordB Dataset

## Project Overview

This project reproduces and extends the DETACH-ROCKET feature-selection
pipeline for binary time-series classification on the FordB dataset.

ROCKET transforms each time series into a high-dimensional representation
using random convolutional kernels. DETACH then performs sequential,
coefficient-based feature pruning to identify a compact subset of informative
ROCKET features.

The project evaluates five approaches:

1. Full ROCKET using all 20,000 transformed features
2. DETACH-ROCKET feature selection
3. SelectKBest with the same feature budget as DETACH
4. Random Pruning with the same feature budget as DETACH
5. RidgeClassifierCV applied directly to the raw time-series samples

The main controlled feature-selection comparison is between DETACH,
SelectKBest, and Random Pruning. These methods use the same feature budget
and the same leakage-free preprocessing protocol.

---

## Dataset

FordB is a binary time-series classification dataset from the UCR archive.

- Training samples: 3,636
- Test samples: 810
- Time points per sample: 500
- Classes: -1 and +1
- Missing values: none

Class distribution:

| Partition | Class -1 | Class +1 |
| --------- | -------: | -------: |
| Train     |    1,860 |    1,776 |
| Test      |      401 |      409 |

The two classes are approximately balanced.

---

## ROCKET Transformation

ROCKET applies 10,000 random convolutional kernels to each time series.

Two features are extracted from each kernel response:

- MAX: maximum convolution response
- PPV: proportion of positive values

This produces 20,000 features for each sample.

Transformation dimensions:

- Training set: 3,636 × 500 → 3,636 × 20,000
- Test set: 810 × 500 → 810 × 20,000

The final ROCKET transformation uses:

```python
Rocket(
    num_kernels=10_000,
    random_state=42
)
```

---

## Leakage-Free Experimental Pipeline

The final experiment follows a leakage-free preprocessing protocol:

1. Load the original FordB training and test partitions.
2. Generate 20,000 ROCKET features.
3. Split the ROCKET training features into stratified sub-training and
   validation partitions.
4. Fit `StandardScaler` only on the sub-training partition.
5. Transform the validation and test partitions using the fitted scaler.
6. Perform feature selection using only the training-side data.
7. Evaluate the selected models on the independent test set.
8. Use the test set only for final evaluation.

The final stratified split uses:

```python
test_size=0.2
random_state=42
```

Final split dimensions:

- Sub-training: 2,908 × 20,000
- Validation: 728 × 20,000
- Test: 810 × 20,000

After correcting the scaling order, the reported DETACH test result remained
unchanged.

---

## DETACH Configuration

The reported final DETACH experiment used:

```python
DetachMatrix(
    trade_off=0.1,
    recompute_alpha=False,
    verbose=True
)
```

DETACH sequentially removes low-importance features based on the magnitude
of Ridge classifier coefficients and uses validation performance to determine
a compact feature representation.

---

## Final Results

### Overall Model Summary

| Method         | Features |       Accuracy |       F1-score |
| -------------- | -------: | -------------: | -------------: |
| Full ROCKET    |   20,000 |         80.25% |         80.58% |
| DETACH-ROCKET  |      789 |     **81.48%** |     **81.53%** |
| SelectKBest    |      789 |         80.62% |         80.92% |
| Random Pruning |      789 | 78.91% ± 1.01% | 78.85% ± 0.95% |
| Raw Ridge      |      500 |         48.89% |         47.06% |

Random Pruning results are reported as mean ± standard deviation over
10 independent random seeds.

The Full ROCKET result is included as a reference baseline. Its comparison
with DETACH is descriptive rather than a fully controlled equal-budget
feature-selection comparison.

---

## Controlled Feature-Selection Comparison

DETACH, SelectKBest, and Random Pruning were compared using the same
789-feature budget and the same clean preprocessing protocol.

| Method         | Features |       Accuracy |       F1-score |
| -------------- | -------: | -------------: | -------------: |
| DETACH-ROCKET  |      789 |     **81.48%** |     **81.53%** |
| SelectKBest    |      789 |         80.62% |         80.92% |
| Random Pruning |      789 | 78.91% ± 1.01% | 78.85% ± 0.95% |

Observed DETACH improvement over SelectKBest:

- Accuracy: +0.86 percentage points
- F1-score: +0.60 percentage points

Observed DETACH improvement over Random Pruning:

- Accuracy: +2.57 percentage points
- F1-score: +2.68 percentage points

These values describe the observed test-set differences under the current
experimental configuration. No statistical significance test was performed
for the DETACH-versus-SelectKBest comparison.

---

## Feature Compression

DETACH reduced the ROCKET feature space from 20,000 to 789 features.

- Initial features: 20,000
- Selected features: 789
- Removed features: 19,211
- Retained features: 3.945%
- Feature reduction: 96.055%
- Compression ratio: 25.35×

The selected representation was approximately 25.35 times smaller than the
original ROCKET representation.

In the reported experiments, test accuracy was 80.25% for Full ROCKET and
81.48% for DETACH-ROCKET. Because these two configurations do not constitute
the main controlled equal-budget comparison, this difference should be
interpreted descriptively.

---

## Interpretation

The results support four main observations:

1. In the linear Ridge classification pipeline studied here, the ROCKET
   representation was important for achieving strong FordB performance.
   Ridge classification directly on the raw 500-point signals achieved
   48.89% accuracy.

2. For FordB and the configuration evaluated here, more than 96% of the
   20,000 ROCKET features could be removed while maintaining the observed
   test performance.

3. Randomly retaining 789 ROCKET features did not reproduce the observed
   DETACH performance.

4. Under the controlled 789-feature comparison, DETACH achieved higher
   observed Accuracy and F1-score than SelectKBest and Random Pruning.

---

## Extensions Added in This Project

Beyond reproducing the core DETACH-ROCKET workflow, this project adds:

- F1-score evaluation
- Ridge classification on the raw time-series values
- SelectKBest with the same 789-feature budget
- Random Pruning evaluated over 10 independent seeds
- Leakage-free scaling
- Controlled equal-budget feature-selection comparison
- Feature-compression analysis
- Reproducible CSV result tables
- Updated performance and feature-count figures
- Explicit separation between the original paper results and project
  extensions
- Reusable Python utilities in `src/`
- Automated unit tests in `tests/`

---

## Project Structure

```text
detach-rocket-fordb/
│
├── detach_rocket/
│   └── Original DETACH-ROCKET implementation
│
├── examples/
│   ├── Detach_ROCKET_example_UCR.ipynb
│   ├── requirements.txt
│   ├── CSV result files
│   └── result figures
│
├── src/
│   └── pipeline.py
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── LICENSE
└── setup.py
```

`src/pipeline.py` contains reusable utilities for:

- loading FordB
- stratified train-validation splitting
- leakage-safe scaling
- prediction evaluation

The notebook remains the main experimental workflow, while reusable
preprocessing and evaluation logic is progressively separated into Python
modules.

---

## Installation

The project was developed using:

```text
Python 3.10.11
```

Create and activate a Python virtual environment, then install the required
packages:

```bash
python -m pip install -r examples/requirements.txt
```

The main dependencies are pinned to the versions used in the project to
improve reproducibility.

---

## Running the Tests

The project includes unit tests for the reusable pipeline utilities.

Run:

```bash
python -m pytest
```

The current test suite checks:

- leakage-safe scaler fitting
- stratified train-validation splitting
- exclusion of held-out data from scaler fitting
- prediction evaluation
- FordB dataset loading

These are lightweight unit tests for the reusable utilities and are not a
replacement for rerunning the complete experimental notebook.

---

## Notebook

The main experimental notebook is:

```text
examples/Detach_ROCKET_example_UCR.ipynb
```

---

## Main Output Files

### Final Result Tables

- `clean_detach_results.csv`
- `select_kbest_results.csv`
- `clean_random_pruning_10_runs.csv`
- `final_clean_feature_selection_comparison.csv`
- `final_clean_improvements.csv`
- `updated_comparison_5_models.csv`
- `updated_report_table_percent.csv`
- `detach_compression_analysis.csv`

### Final Figures

- `final_clean_feature_selection_comparison.png`
- `updated_five_model_accuracy.png`
- `updated_five_model_f1.png`
- `updated_five_model_feature_count.png`
- `detach_vs_selectkbest_accuracy.png`
- `detach_vs_selectkbest_f1.png`

### Documentation

- `updated_project_summary.txt`
- `environment_versions.txt`
- `requirements.txt`

These output files are located under the `examples/` directory where
applicable.

Some earlier exploratory files may remain in the project directory. The
files listed above correspond to the final results used in the report and
presentation.

---

## Reproducibility

The following reproducibility controls are explicitly used in the final
workflow:

- ROCKET kernel generation: `random_state=42`
- Stratified training-validation split: `random_state=42`
- Single-run random feature selection: NumPy random generator seed `42`
- Random Pruning repeated evaluation: seeds `0` through `9`

Random Pruning seeds:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Package versions are pinned in:

```text
examples/requirements.txt
```

The test set is reserved for final evaluation and is not used for fitting
the scaler or performing feature selection.

A final explicit random seed for the internal DETACH implementation is not
documented here because it was not independently verified from the current
notebook version.

---

## Current Limitations

The current project has several limitations that should be considered when
interpreting the results:

- The analysis focuses on a single UCR dataset, FordB.
- The multi-seed comparison currently uses five ROCKET random seeds, which is
  still a relatively small sample for statistical inference.
- In the matched-budget multi-seed analysis, DETACH selected a different number
  of features for each ROCKET seed, indicating that the selected feature count
  is sensitive to the experimental configuration.
- DETACH achieved higher observed Accuracy and F1-score than matched-budget
  SelectKBest across all five paired ROCKET seeds. The paired t-test indicated
  statistically significant differences for Accuracy (p = 0.015) and F1-score
  (p = 0.023). However, these results should be interpreted cautiously because
  of the small number of seeds.
- Random Pruning was evaluated over 10 seeds, whereas the matched-budget
  DETACH–SelectKBest comparison currently uses five seeds.
- The Full ROCKET and DETACH results should not be interpreted as a fully
  controlled equal-budget comparison.
- The conclusions are specific to the FordB dataset and should not be
  generalized to other time-series datasets without further evaluation.

---

## Conclusion

DETACH-ROCKET reduced the FordB ROCKET feature space from 20,000 to 789
features, corresponding to a 96.055% reduction and a 25.35-fold compression.

The reported DETACH model achieved:

- 81.48% Accuracy
- 81.53% F1-score

In the controlled 789-feature comparison, DETACH achieved higher observed
test scores than SelectKBest and Random Pruning.

For FordB and the experimental configuration evaluated here, targeted
feature selection produced a substantially more compact ROCKET
representation while maintaining the observed classification performance.