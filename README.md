# DETACH-ROCKET on the FordB Dataset

## Project Overview

This project reproduces and extends the DETACH-ROCKET feature-selection pipeline for binary time-series classification on the FordB dataset.

ROCKET transforms each time series into a high-dimensional representation using random convolutional kernels. DETACH then performs sequential, coefficient-based feature pruning to identify a compact subset of informative ROCKET features.

The project evaluates five approaches:

1. Full ROCKET using all 20,000 transformed features
2. DETACH-ROCKET feature selection
3. SelectKBest with the same feature budget as DETACH
4. Random feature pruning with the same feature budget as DETACH
5. RidgeClassifierCV applied directly to the raw time-series samples

The main controlled feature-selection comparison uses the same sub-training partition, validation partition, fitted scaler, test set, and feature budget for DETACH, SelectKBest, and Random Pruning.

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
|---|---:|---:|
| Train | 1,860 | 1,776 |
| Test | 401 | 409 |

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

---

## Leakage-Free Experimental Pipeline

The final experiment uses a leakage-free preprocessing pipeline:

1. Load the original FordB train and test partitions.
2. Generate 20,000 ROCKET features.
3. Split the ROCKET training features into stratified sub-training and validation partitions.
4. Fit `StandardScaler` only on the sub-training partition.
5. Transform the validation and test partitions using the fitted scaler.
6. Run DETACH using the sub-training and validation partitions.
7. Fit SelectKBest only on the same sub-training partition.
8. Run Random Pruning using the same 789-feature budget.
9. Evaluate the final models on the independent test set.
10. Use the test set only for final evaluation.

Final split dimensions:

- Sub-training: 2,908 × 20,000
- Validation: 728 × 20,000
- Test: 810 × 20,000

After correcting the scaling order, the DETACH result remained unchanged.

---

## DETACH Configuration

The final DETACH model was configured as follows:

```python
DetachMatrix(
    trade_off=0.1,
    recompute_alpha=False,
    verbose=True
)
```

DETACH sequentially removes low-importance features based on the absolute coefficients of a Ridge classifier. Validation performance is used to select the final model size.

---

## Final Results

### Overall Model Summary

| Method | Features | Accuracy | F1-score |
|---|---:|---:|---:|
| Full ROCKET | 20,000 | 80.25% | 80.58% |
| DETACH-ROCKET | 789 | **81.48%** | **81.53%** |
| SelectKBest | 789 | 80.62% | 80.92% |
| Random Pruning | 789 | 78.91% ± 1.01% | 78.85% ± 0.95% |
| Raw Ridge | 500 | 48.89% | 47.06% |

Random Pruning results are reported as mean ± standard deviation over 10 independent random seeds.

---

## Controlled Feature-Selection Comparison

DETACH, SelectKBest, and Random Pruning were compared under the same experimental conditions and with exactly 789 selected features.

| Method | Features | Accuracy | F1-score |
|---|---:|---:|---:|
| DETACH-ROCKET | 789 | **81.48%** | **81.53%** |
| SelectKBest | 789 | 80.62% | 80.92% |
| Random Pruning | 789 | 78.91% | 78.85% |

DETACH improvement over SelectKBest:

- Accuracy: +0.86 percentage points
- F1-score: +0.60 percentage points

DETACH improvement over Random Pruning:

- Accuracy: +2.57 percentage points
- F1-score: +2.68 percentage points

Because the three methods used the same feature budget and preprocessing pipeline, the results provide a controlled comparison of the selected feature subsets.

---

## Feature Compression

DETACH reduced the ROCKET feature space from 20,000 to 789 features.

- Initial features: 20,000
- Selected features: 789
- Removed features: 19,211
- Retained features: 3.945%
- Feature reduction: 96.055%
- Compression ratio: 25.35×

The final representation was approximately 25.35 times smaller than the original ROCKET representation.

Test accuracy increased numerically from 80.25% for Full ROCKET to 81.48% for DETACH-ROCKET.

---

## Interpretation

The results support four main observations:

1. ROCKET feature extraction is essential for FordB. Ridge classification on the raw 500-point signals achieved only 48.89% accuracy.
2. The 20,000-dimensional ROCKET representation contains substantial redundancy for this dataset.
3. Randomly retaining 789 features did not reproduce DETACH performance.
4. Under the controlled 789-feature comparison, DETACH achieved higher Accuracy and F1-score than both SelectKBest and Random Pruning.

---

## Extensions Added in This Project

Beyond reproducing the core DETACH-ROCKET pipeline, this project adds:

- F1-score evaluation
- Ridge classification on the raw time-series values
- SelectKBest with the same 789-feature budget
- Clean Random Pruning over 10 independent seeds
- Leakage-free scaling
- A controlled equal-budget feature-selection comparison
- Feature-compression analysis
- Reproducible CSV result tables
- Updated performance and feature-count figures
- Explicit separation between the original paper and the project extensions

---

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r examples/requirements.txt
```

The experiments were developed using Python 3.10.

---

## Notebook

The main experimental notebook is:

```text
examples/Detach_ROCKET_example_UCR.ipynb
```

---

## Main Output Files

### Final Result Tables

- `examples/clean_detach_results.csv`
- `examples/select_kbest_results.csv`
- `examples/clean_random_pruning_10_runs.csv`
- `examples/final_clean_feature_selection_comparison.csv`
- `examples/final_clean_improvements.csv`
- `examples/updated_comparison_5_models.csv`
- `examples/updated_report_table_percent.csv`
- `examples/detach_compression_analysis.csv`

### Final Figures

- `examples/final_clean_feature_selection_comparison.png`
- `examples/updated_five_model_accuracy.png`
- `examples/updated_five_model_f1.png`
- `examples/updated_five_model_feature_count.png`
- `examples/detach_vs_selectkbest_accuracy.png`
- `examples/detach_vs_selectkbest_f1.png`

### Documentation

- `examples/updated_project_summary.txt`
- `examples/environment_versions.txt`
- `examples/requirements.txt`

Some earlier exploratory files may remain in the local project directory, but the files listed above contain the final leakage-free results used in the report and presentation.

---

## Reproducibility

Fixed random seeds were used for:

- Training-validation splitting
- Random feature selection
- ROCKET kernel generation in repeated exploratory experiments

The final clean split used stratification and `random_state=42`.

Random Pruning was evaluated using 10 independent seeds:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The test set was not used for fitting the scaler, selecting features, or choosing the final DETACH model size.

---

## License and Attribution

This project includes and modifies code from the official
[Detach-ROCKET repository](https://github.com/gon-uri/detach_rocket),
which is distributed under the BSD 3-Clause License.

The original Detach-ROCKET implementation is credited to its authors.
The FordB experiments, leakage-free evaluation pipeline, additional
baselines, result tables, and figures were added in this project.

See the `LICENSE` file for the full license text.

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

DETACH-ROCKET reduced the FordB ROCKET feature space from 20,000 to 789 features, corresponding to a 96.055% reduction and a 25.35-fold compression.

The final DETACH model achieved:

- 81.48% Accuracy
- 81.53% F1-score

In the controlled comparison with the same 789-feature budget, DETACH outperformed both SelectKBest and Random Pruning on the FordB test set.

These results show that targeted feature selection can produce a substantially more compact ROCKET representation without sacrificing classification performance on FordB.
