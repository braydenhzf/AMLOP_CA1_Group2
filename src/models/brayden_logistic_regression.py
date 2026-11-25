"""Brayden's reusable logistic regression training function."""

from time import perf_counter

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def train_logistic_regression(
    X_train,
    y_train,
    *,
    C=1.0,
    class_weight=None,
    max_iter=1000,
):
    """Fit preprocessing and logistic regression on training data only."""
    model = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
        )),
    ])

    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    return model, training_seconds