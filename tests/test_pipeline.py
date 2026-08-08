from src.pipeline import split_and_scale
from src.pipeline import split_and_scale, evaluate_predictions
from src.pipeline import split_and_scale, evaluate_predictions, load_fordb

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_scaler_fits_only_on_subtrain():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 20))
    y = np.array([0, 1] * 50)

    X_sub_scaled, X_val_scaled, y_sub, y_val, scaler = split_and_scale(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    assert X_sub_scaled.shape == (80, 20)
    assert X_val_scaled.shape == (20, 20)

    np.testing.assert_allclose(
        X_sub_scaled.mean(axis=0),
        np.zeros(20),
        atol=1e-10
    )
        
def test_stratified_split_preserves_class_ratio():
    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 60 + [1] * 40)

    X_sub_scaled, X_val_scaled, y_sub, y_val, scaler = split_and_scale(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    assert (y_sub == 0).sum() == 48
    assert (y_sub == 1).sum() == 32

    assert (y_val == 0).sum() == 12
    assert (y_val == 1).sum() == 8
    
def test_test_set_is_not_used_for_scaler_fit():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 10))
    y = np.array([0, 1] * 50)

    X_sub_scaled, X_val_scaled, y_sub, y_val, scaler = split_and_scale(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_sub, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    np.testing.assert_allclose(
        scaler.mean_,
        X_sub.mean(axis=0)
    )
    
def test_evaluate_predictions():
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([1, -1, -1, -1])

    results = evaluate_predictions(y_true, y_pred)

    assert results["accuracy"] == 0.75
    assert 0 <= results["f1"] <= 1
    
def test_load_fordb():
    dataset = load_fordb()

    assert dataset is not None