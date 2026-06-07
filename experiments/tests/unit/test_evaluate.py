import math

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import evaluate


@pytest.mark.unit
def test_evaluate_dummy(tiny_dataset):
    """A prior DummyClassifier scores near the binary entropy of the prior."""
    X_train, y_train, X_val, y_val = tiny_dataset
    model = DummyClassifier(strategy="prior").fit(X_train, y_train)

    metrics = evaluate(model, X_val, y_val)
    assert set(metrics) == {"val_logloss", "val_acc", "val_auc"}

    p = float(np.mean(y_train))  # prior P(class == 1)
    expected = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert metrics["val_logloss"] == pytest.approx(expected, rel=0.05)
    # A constant predictor has no ranking power: AUC is ~0.5.
    assert metrics["val_auc"] == pytest.approx(0.5, abs=0.05)


@pytest.mark.unit
def test_evaluate_rejects_no_proba(tiny_dataset):
    """A model without predict_proba raises a clear error."""

    class NoProbaModel:
        classes_ = [0, 1]

        def predict(self, X):
            return [0] * len(X)

    _, _, X_val, y_val = tiny_dataset
    with pytest.raises(ValueError, match="predict_proba"):
        evaluate(NoProbaModel(), X_val, y_val)


@pytest.mark.unit
def test_evaluate_multiclass():
    """evaluate() handles >2 classes (macro one-vs-rest AUC)."""
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=6, n_redundant=1,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    X_train, X_val, y_train, y_val = X[:240], X[240:], y[:240], y[240:]
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)

    metrics = evaluate(model, X_val, y_val)
    assert set(metrics) == {"val_logloss", "val_acc", "val_auc"}
    assert all(math.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["val_auc"] <= 1.0
    assert metrics["val_logloss"] < math.log(3)  # better than uniform over 3 classes


@pytest.mark.unit
def test_evaluate_rejects_single_class():
    """A degenerate single-class model is rejected."""

    class OneClass:
        classes_ = [0]

        def predict_proba(self, X):
            return np.ones((len(X), 1))

        def predict(self, X):
            return [0] * len(X)

    with pytest.raises(ValueError, match="at least 2 classes"):
        evaluate(OneClass(), [[1.0]], [0])


@pytest.mark.unit
def test_evaluate_handles_pipeline(tiny_dataset):
    """evaluate() works on a fitted sklearn Pipeline (delegated proba/classes_)."""
    X_train, y_train, X_val, y_val = tiny_dataset
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    ).fit(X_train, y_train)

    metrics = evaluate(pipeline, X_val, y_val)
    assert set(metrics) == {"val_logloss", "val_acc", "val_auc"}
    assert all(math.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["val_auc"] <= 1.0
    # A real classifier should beat the prior on this separable synthetic set.
    assert metrics["val_logloss"] < 0.69
