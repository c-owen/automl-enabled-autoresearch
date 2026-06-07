"""Family adapters — shared measurement plumbing for the experiment harness.

A single place that turns a *typed config dict* into a fitted-able model for each
of the four families. Consumed by the BO tool (Step 7), the reference arms
(Step 9), and the family-ceiling check (Step 5). Every adapter:

* applies the **same fixed default preprocessing** as the family baselines in
  ``tests/fixtures/family_baselines/`` (and ``train.py`` for xgboost) — so a trial
  measured through an adapter is comparable to an agent trial by construction;
* returns an *unfitted* estimator with the uniform contract
  ``model.fit(X_df, y)`` / ``model.predict_proba`` / ``model.predict`` /
  ``model.classes_`` over the RAW labels ``load_task`` returns, so the locked
  ``prepare.evaluate`` scores it identically to an agent's own trial.

``PARAM_SPECS[family]`` is the typed *superset* of tunable hyperparameters (used
to validate an agent-declared ``--space``; the agent may search any subset).
``DEFAULTS[family]`` are the family-baseline constants used for any param the
caller does not override.

This module is **locked**, like ``prepare.py`` and ``logging_lib.py``: it is
measurement plumbing, and the autoresearch agent never edits it.
"""

import numpy as np

from prepare import RANDOM_SEED

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AdapterError(ValueError):
    """A config key/value the adapter cannot honor (unknown or out of spec)."""


# ---------------------------------------------------------------------------
# Typed tunable spec (the superset the agent's --space may draw from) and the
# family-baseline default constants (used for any unspecified param).
# ---------------------------------------------------------------------------

PARAM_SPECS = {
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 12},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.5, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 40},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
    },
    "logistic_regression": {
        "C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        "max_iter": {"type": "int", "low": 100, "high": 2000},
    },
    "mlp": {
        "hidden_sizes": {
            "type": "categorical",
            "choices": ["256,128", "128,64", "64,32", "128"],
        },
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "max_epochs": {"type": "int", "low": 10, "high": 100},
    },
}

# Family-baseline constants (mirror tests/fixtures/family_baselines/ and, for
# xgboost, train.py). Params absent from PARAM_SPECS (e.g. mlp `patience`) are
# fixed here and not tunable.
DEFAULTS = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
    },
    "mlp": {
        "hidden_sizes": (128, 64),
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 30,
        "patience": 5,
    },
}

ALLOWED_FAMILIES = list(PARAM_SPECS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(family: str, config: dict) -> None:
    """Raise ``AdapterError`` if any config entry is unknown or out of spec.

    Validates only the caller-supplied keys (defaults are trusted). The error
    always names the offending key.
    """
    if family not in PARAM_SPECS:
        raise AdapterError(
            f"unknown family {family!r}; expected one of {sorted(PARAM_SPECS)}"
        )
    specs = PARAM_SPECS[family]
    for key, value in config.items():
        if key not in specs:
            raise AdapterError(
                f"unknown hyperparameter {key!r} for family {family!r}; "
                f"allowed: {sorted(specs)}"
            )
        spec = specs[key]
        kind = spec["type"]
        if kind == "categorical":
            if value not in spec["choices"]:
                raise AdapterError(
                    f"{key!r}={value!r} is not an allowed choice {spec['choices']} "
                    f"for family {family!r}"
                )
        elif kind in ("int", "float"):
            # bool is an int subclass — reject it explicitly as a wrong type.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AdapterError(
                    f"{key!r} must be a {kind}, got {type(value).__name__}"
                )
            if kind == "int" and not float(value).is_integer():
                raise AdapterError(f"{key!r} must be an integer, got {value!r}")
            low, high = spec["low"], spec["high"]
            if value < low or value > high:
                raise AdapterError(
                    f"{key!r}={value!r} out of range [{low}, {high}] for "
                    f"family {family!r}"
                )
        else:  # pragma: no cover — guards against a malformed spec
            raise AdapterError(f"malformed spec for {key!r}: type {kind!r}")


def _resolved_params(family: str, config: dict) -> dict:
    validate_config(family, config)
    return {**DEFAULTS[family], **config}


# ---------------------------------------------------------------------------
# Shared preprocessing (matches the family-baseline fixtures exactly)
# ---------------------------------------------------------------------------


def _make_preprocessor(*, scale: bool, dense: bool):
    """ColumnTransformer mirroring the fixtures: median-impute (+ optional scale)
    numerics; constant-impute + one-hot categoricals. Columns are selected by
    dtype at fit time, so no ``X`` is needed at build time."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps),
             make_column_selector(dtype_exclude=["object", "category"])),
            ("cat", cat_pipe,
             make_column_selector(dtype_include=["object", "category"])),
        ]
    )


def _parse_hidden(value) -> tuple:
    """Normalize a hidden-layer spec (tuple/list, "256,128", or int) to a tuple."""
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        return tuple(int(x) for x in value.split(",") if x.strip())
    if isinstance(value, int) and not isinstance(value, bool):
        return (int(value),)
    raise AdapterError(f"cannot parse hidden_sizes={value!r}")


# ---------------------------------------------------------------------------
# Wrapper models for the two families whose fit isn't a plain sklearn Pipeline
# ---------------------------------------------------------------------------


class _XGBAdapterModel:
    """XGBoost with train.py's preprocessing: category dtypes + label-encoded
    targets, re-exposing predict_proba/classes_ over the ORIGINAL labels so
    ``prepare.evaluate`` measures it exactly like the agent's xgboost trial."""

    def __init__(self, params: dict):
        self._params = params
        self._model = None
        self._le = None
        self.classes_ = None

    @staticmethod
    def _categorize(X):
        X = X.copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category")
        return X

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        self._model = XGBClassifier(
            **self._params,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="logloss",
        )
        self._model.fit(self._categorize(X), y_enc)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(self._categorize(X))

    def predict(self, X):
        enc = np.asarray(self._model.predict(self._categorize(X))).astype(int)
        return self.classes_[enc]


class _TorchMLPAdapterModel:
    """A minimal PyTorch MLP — a faithful port of the mlp family-baseline fixture
    (median-impute+scale / impute+one-hot preprocessing, Adam, cross-entropy,
    early stopping on a carved-out split), parameterized by ``config``."""

    def __init__(self, params: dict):
        self._params = params
        self._net = None
        self._preprocessor = None
        self.classes_ = None

    @staticmethod
    def _build_net(input_dim, hidden_sizes, dropout, n_classes):
        import torch.nn as nn

        layers = []
        prev = input_dim
        for width in hidden_sizes:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import copy

        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        p = self._params
        hidden = _parse_hidden(p["hidden_sizes"])
        dropout = float(p["dropout"])
        lr = float(p["learning_rate"])
        weight_decay = float(p["weight_decay"])
        batch_size = int(p["batch_size"])
        max_epochs = int(p["max_epochs"])
        patience = int(p["patience"])

        torch.manual_seed(RANDOM_SEED)

        self._preprocessor = _make_preprocessor(scale=True, dense=True)
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_

        X_fit_raw, X_es_raw, y_fit, y_es = train_test_split(
            X, y_enc, test_size=0.1, random_state=RANDOM_SEED, stratify=y_enc,
        )
        X_fit = np.asarray(self._preprocessor.fit_transform(X_fit_raw), dtype=np.float32)
        X_es = np.asarray(self._preprocessor.transform(X_es_raw), dtype=np.float32)

        X_fit_t = torch.from_numpy(X_fit)
        y_fit_t = torch.from_numpy(y_fit.astype(np.int64))
        X_es_t = torch.from_numpy(X_es)
        y_es_t = torch.from_numpy(y_es.astype(np.int64))

        net = self._build_net(X_fit.shape[1], hidden, dropout, len(label_encoder.classes_))
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        n = X_fit_t.shape[0]
        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        epochs_without_improvement = 0

        for _epoch in range(max_epochs):
            net.train()
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(net(X_fit_t[idx]), y_fit_t[idx])
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_es_t), y_es_t).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = copy.deepcopy(net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        net.load_state_dict(best_state)
        self._net = net
        return self

    def predict_proba(self, X):
        import torch

        self._net.eval()
        Xt = np.asarray(self._preprocessor.transform(X), dtype=np.float32)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(Xt))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build(family: str, config: dict = None):
    """Return an unfitted model for ``family`` configured by ``config``.

    ``config`` overrides the family DEFAULTS; every key/value is validated
    against ``PARAM_SPECS[family]`` first (``AdapterError`` on violation, naming
    the offending key). The result fits on the raw ``(X_df, y)`` from
    ``load_task`` and is scored by ``prepare.evaluate``.
    """
    config = dict(config or {})
    params = _resolved_params(family, config)

    if family == "xgboost":
        return _XGBAdapterModel(params)

    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        return Pipeline(
            [
                ("preprocess", _make_preprocessor(scale=False, dense=False)),
                ("model", RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    max_features=params["max_features"],
                    n_jobs=-1,
                    random_state=0,
                )),
            ]
        )

    if family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        return Pipeline(
            [
                ("preprocess", _make_preprocessor(scale=True, dense=False)),
                ("model", LogisticRegression(
                    C=float(params["C"]),
                    max_iter=int(params["max_iter"]),
                    penalty="l2",
                    solver="lbfgs",
                )),
            ]
        )

    if family == "mlp":
        return _TorchMLPAdapterModel(params)

    # Unreachable: _resolved_params already rejected unknown families.
    raise AdapterError(f"unhandled family {family!r}")
