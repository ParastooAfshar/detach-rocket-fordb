from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_sub, X_val, y_sub, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()

    X_sub_scaled = scaler.fit_transform(X_sub)
    X_val_scaled = scaler.transform(X_val)

    return X_sub_scaled, X_val_scaled, y_sub, y_val, scaler

from sklearn.metrics import accuracy_score, f1_score


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    
from detach_rocket.utils_datasets import fetch_ucr_dataset


def load_fordb():
    return fetch_ucr_dataset("FordB")