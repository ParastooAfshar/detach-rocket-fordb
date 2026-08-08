# Changes from the Upstream DETACH-ROCKET Repository

This repository is based on the original DETACH-ROCKET implementation by Gonzalo Uribarri and Federico Barone.

The core `detach_rocket/` package is retained from the upstream project. The main changes in this repository focus on the FordB experimental workflow, evaluation methodology, reproducibility, testing, and documentation.

## Added in This Repository

- FordB-specific experimental workflow
- Leakage-free preprocessing with `StandardScaler` fitted only on the sub-training partition
- Accuracy and F1-score evaluation
- Raw Ridge baseline
- SelectKBest baseline
- Random Pruning baseline
- 10-seed Random Pruning evaluation
- Feature-compression analysis
- Multi-seed ROCKET evaluation
- Matched-budget DETACH–SelectKBest comparison
- Paired statistical testing
- Reproducible CSV result files
- Reusable preprocessing and evaluation utilities in `src/pipeline.py`
- Lightweight unit tests in `tests/test_pipeline.py`
- Pinned dependency versions
- Updated project-level README and installation instructions

## Core Package Status

The `detach_rocket/` directory is based on the upstream DETACH-ROCKET implementation.

This project does not claim to provide a redesigned or independently maintained version of the DETACH algorithm itself. The primary contributions are in the experimental pipeline and evaluation built around the upstream implementation.

## Upstream Repository

Original repository:

https://github.com/gon-uri/detach_rocket

## This Repository

https://github.com/ParastooAfshar/detach-rocket-fordb