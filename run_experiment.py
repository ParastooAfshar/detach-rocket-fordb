import argparse
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sktime.transformations.panel.rocket import Rocket

from src.pipeline import load_fordb, split_and_scale


def smoke_test():
    """Load FordB and verify the expected dataset shapes."""
    dataset = load_fordb()

    X_train = dataset["data_train"]
    y_train = dataset["target_train"]
    X_test = dataset["data_test"]
    y_test = dataset["target_test"]

    print("FordB loaded successfully")
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)


def run_selectkbest_experiment():
    """Run the main ROCKET + SelectKBest baseline on FordB."""
    dataset = load_fordb()

    X_train = dataset["data_train"]
    y_train = dataset["target_train"]
    X_test = dataset["data_test"]
    y_test = dataset["target_test"]

    X_train_rocket = X_train[:, np.newaxis, :]
    X_test_rocket = X_test[:, np.newaxis, :]

    print("Generating ROCKET features...")

    rocket = Rocket(
        num_kernels=10_000,
        random_state=42,
    )

    rocket.fit(X_train_rocket)

    Z_train = rocket.transform(X_train_rocket)
    Z_test = rocket.transform(X_test_rocket)

    print("ROCKET transformation complete")
    print("Train features:", Z_train.shape)
    print("Test features:", Z_test.shape)

    Z_sub, Z_val, y_sub, y_val, scaler = split_and_scale(
        Z_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )

    Z_test_scaled = scaler.transform(Z_test)

    selector = SelectKBest(
        score_func=f_classif,
        k=789,
    )

    Z_sub_selected = selector.fit_transform(
        Z_sub,
        y_sub,
    )

    Z_test_selected = selector.transform(
        Z_test_scaled,
    )

    model = RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 10),
    )

    model.fit(
        Z_sub_selected,
        y_sub,
    )

    y_pred = model.predict(
        Z_test_selected,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
    )

    print("\nSelectKBest baseline")
    print("--------------------")
    print("Selected features: 789")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Ridge alpha: {model.alpha_}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FordB DETACH-ROCKET project entry point."
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the ROCKET + SelectKBest experiment. "
            "Without this flag, only a lightweight dataset smoke test is run."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.full:
        run_selectkbest_experiment()
    else:
        smoke_test()
        print(
            "\nSmoke test completed. "
            "Use `python run_experiment.py --full` "
            "to run the ROCKET + SelectKBest baseline."
        )


if __name__ == "__main__":
    main()