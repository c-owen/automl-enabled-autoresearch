"""Step 4: family adapters fit + score, and reproduce the family baselines.

The parity tests reconstruct each fixture's exact preprocessing + model inline
(mirroring tests/fixtures/family_baselines/ and train.py for xgboost) and assert
that ``build(family, defaults)`` yields the same val_logloss on tiny_synthetic —
the executable form of the protocol's "same preprocessing as the baselines".
"""

import copy

import numpy as np
import pandas as pd
import pytest

import family_adapters as fa
import prepare


def _as_frames(tiny):
    """tiny_synthetic ships as numpy arrays; adapters consume DataFrames."""
    X_train, y_train, X_val, y_val = tiny
    return pd.DataFrame(X_train), y_train, pd.DataFrame(X_val), y_val


@pytest.mark.integration
@pytest.mark.parametrize("family", fa.ALLOWED_FAMILIES)
def test_adapter_builds_each_family(family, tiny_dataset):
    X_train, y_train, X_val, y_val = _as_frames(tiny_dataset)

    model = fa.build(family)  # default (family-baseline) config
    model.fit(X_train, y_train)
    scores = prepare.evaluate(model, X_val, y_val)

    assert set(scores) == {"val_logloss", "val_acc", "val_auc"}
    assert np.isfinite(scores["val_logloss"])
    assert np.isfinite(scores["val_acc"])
    assert np.isfinite(scores["val_auc"])


# --- inline reconstructions of the frozen family baselines -----------------


def _ref_xgboost(Xtr, ytr, Xv, yv):
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    Xtr, Xv = Xtr.copy(), Xv.copy()
    for col in Xtr.select_dtypes(include=["object", "category"]).columns:
        Xtr[col] = Xtr[col].astype("category")
        Xv[col] = Xv[col].astype("category")
    le = LabelEncoder()
    ytr_e, yv_e = le.fit_transform(ytr), le.transform(yv)
    m = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,
        tree_method="hist", enable_categorical=True, eval_metric="logloss",
    )
    m.fit(Xtr, ytr_e)
    return prepare.evaluate(m, Xv, yv_e)["val_logloss"]


def _ref_random_forest(Xtr, ytr, Xv, yv):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat = list(Xtr.select_dtypes(include=["object", "category"]).columns)
    num = [c for c in Xtr.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    m = Pipeline([("preprocess", pre), ("model", RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, random_state=0))])
    m.fit(Xtr, ytr)
    return prepare.evaluate(m, Xv, yv)["val_logloss"]


def _ref_logistic_regression(Xtr, ytr, Xv, yv):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = list(Xtr.select_dtypes(include=["object", "category"]).columns)
    num = [c for c in Xtr.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    m = Pipeline([("preprocess", pre), ("model", LogisticRegression(
        C=1.0, max_iter=1000, penalty="l2", solver="lbfgs"))])
    m.fit(Xtr, ytr)
    return prepare.evaluate(m, Xv, yv)["val_logloss"]


def _ref_mlp(Xtr, ytr, Xv, yv):
    import torch
    import torch.nn as nn
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    torch.manual_seed(prepare.RANDOM_SEED)
    cat = list(Xtr.select_dtypes(include=["object", "category"]).columns)
    num = [c for c in Xtr.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat),
    ])
    le = LabelEncoder()
    ytr_e = le.fit_transform(ytr)
    Xf_raw, Xe_raw, yf, ye = train_test_split(
        Xtr, ytr_e, test_size=0.1, random_state=prepare.RANDOM_SEED, stratify=ytr_e)
    Xf = np.asarray(pre.fit_transform(Xf_raw), dtype=np.float32)
    Xe = np.asarray(pre.transform(Xe_raw), dtype=np.float32)

    def build_net(d, hs, dr, k):
        layers, prev = [], d
        for w in hs:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(dr)]
            prev = w
        layers.append(nn.Linear(prev, k))
        return nn.Sequential(*layers)

    Xf_t, yf_t = torch.from_numpy(Xf), torch.from_numpy(yf.astype(np.int64))
    Xe_t, ye_t = torch.from_numpy(Xe), torch.from_numpy(ye.astype(np.int64))
    net = build_net(Xf.shape[1], (128, 64), 0.1, len(le.classes_))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = Xf_t.shape[0]
    best, best_state, bad = float("inf"), copy.deepcopy(net.state_dict()), 0
    for _ in range(30):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xf_t[idx]), yf_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xe_t), ye_t).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)

    class _W:
        classes_ = le.classes_

        def predict_proba(self, X):
            net.eval()
            Xt = np.asarray(pre.transform(X), dtype=np.float32)
            with torch.no_grad():
                return torch.softmax(net(torch.from_numpy(Xt)), dim=1).numpy()

        def predict(self, X):
            return le.classes_[self.predict_proba(X).argmax(axis=1)]

    return prepare.evaluate(_W(), Xv, yv)["val_logloss"]


_REFS = {
    "xgboost": _ref_xgboost,
    "random_forest": _ref_random_forest,
    "logistic_regression": _ref_logistic_regression,
    "mlp": _ref_mlp,
}
# Deterministic families match to float noise; the torch MLP is "within noise".
_TOL = {
    "xgboost": 1e-6,
    "random_forest": 1e-6,
    "logistic_regression": 1e-6,
    "mlp": 1e-3,
}


@pytest.mark.integration
@pytest.mark.parametrize("family", fa.ALLOWED_FAMILIES)
def test_adapter_fixture_parity(family, tiny_dataset):
    Xtr, ytr, Xv, yv = _as_frames(tiny_dataset)

    model = fa.build(family)  # default config == the fixture's constants
    model.fit(Xtr, ytr)
    adapter_ll = prepare.evaluate(model, Xv, yv)["val_logloss"]

    ref_ll = _REFS[family](Xtr, ytr, Xv, yv)
    assert abs(adapter_ll - ref_ll) <= _TOL[family], (
        f"{family}: adapter val_logloss {adapter_ll} != fixture {ref_ll}"
    )
