"""Numerical tests: the pipeline and the MLP actually learn.

These use a small synthetic frame with the real column names rather than the
32k-row table, so the suite stays fast and runs without network access.
"""
import numpy as np
import pandas as pd
import pytest

from api.training import MLP, build_preprocessor, calibration_curve, compute_metrics, train_model
from shared.features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS


def make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A learnable signal: older people working longer hours earn more.
    Includes NULLs so the imputer is exercised."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "education_num": rng.integers(1, 16, n),
        "capital_gain": rng.integers(0, 500, n),
        "capital_loss": np.zeros(n, dtype=int),
        "hours_per_week": rng.integers(10, 70, n),
        "workclass": rng.choice(["Private", "State-gov", None], n),
        "marital_status": rng.choice(["Married-civ-spouse", "Never-married"], n),
        "occupation": rng.choice(["Sales", "Exec-managerial", None], n),
        "relationship": rng.choice(["Husband", "Not-in-family"], n),
        "race": rng.choice(["White", "Black"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "native_country": rng.choice(["United-States", "Mexico"], n),
    })
    score = 0.06 * df["age"] + 0.05 * df["hours_per_week"] - 5.0
    df["label"] = (score + rng.normal(0, 0.3, n) > 0).astype(int)
    df["split"] = np.where(rng.random(n) < 0.2, "test", "train")
    return df


def test_preprocessor_imputes_and_encodes():
    df = make_frame()
    pre = build_preprocessor()
    X = pre.fit_transform(df[FEATURE_COLS])
    assert not np.isnan(X).any(), "NULLs survived the imputer"
    # One-hot expansion means more columns out than in.
    assert X.shape[1] > len(NUMERIC_COLS) + len(CATEGORICAL_COLS)
    assert X.shape[0] == len(df)


def test_unseen_category_does_not_crash():
    """handle_unknown='ignore' keeps serving when a new category appears."""
    df = make_frame()
    pre = build_preprocessor()
    pre.fit(df[FEATURE_COLS])
    novel = df.head(1).copy()
    novel["native_country"] = "Atlantis"
    out = pre.transform(novel[FEATURE_COLS])
    assert out.shape[0] == 1


def test_mlp_requires_two_hidden_layers():
    with pytest.raises(ValueError):
        MLP(input_dim=10, hidden_sizes=[16])


def test_model_learns_the_signal():
    """The headline numerical test: accuracy and AUC must beat chance by a
    wide margin on data with a known relationship."""
    df = make_frame()
    config = {"name": "test", "hidden_sizes": [32, 16], "activation": "relu",
              "dropout": 0.0, "lr": 0.005, "weight_decay": 0.0,
              "batch_size": 64, "epochs": 15, "patience": 5, "seed": 0}
    _, _, results = train_model(df, config)
    m = results["metrics"]
    assert m["accuracy"] > 0.75, f"accuracy only {m['accuracy']:.3f}"
    assert m["roc_auc"] > 0.80, f"roc_auc only {m['roc_auc']:.3f}"
    assert results["n_test"] > 0


def test_metrics_are_internally_consistent():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y, proba)
    cm = m["confusion"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)
    assert m["accuracy"] == pytest.approx((cm["tn"] + cm["tp"]) / len(y))


def test_calibration_bins_cover_predictions():
    y = np.random.default_rng(0).integers(0, 2, 200)
    proba = np.random.default_rng(1).random(200)
    bins = calibration_curve(y, proba, n_bins=5)
    assert sum(b["count"] for b in bins) == len(proba)