"""Tests the assignment calls for specifically:

  * /predict_batch returns exactly as many predictions as rows it was given
  * a frozen reference row produces a stable probability (+/- 1e-3)

The regression test pins the model's behaviour. If retraining, a library
upgrade, or an accidental change to the preprocessing pipeline shifts the
output on a known input, this fails -- which is the point. A silent drift in
a deployed classifier is far worse than a loud test failure.
"""
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from shared.features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_BOUNDS, NUMERIC_COLS

# A fixed, ordinary record. Nothing about it is special except that it never
# changes.
REFERENCE_ROW = {
    "age": 39,
    "education_num": 13,
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "workclass": "State-gov",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native_country": "United-States",
}

FIXTURE_PATH = Path(__file__).parent / "reference_probability.json"
TOLERANCE = 1e-3


def _sample_rows(n: int) -> pd.DataFrame:
    """n copies of the reference row with varied ages, as a CSV-shaped frame."""
    rows = []
    for i in range(n):
        row = dict(REFERENCE_ROW)
        row["age"] = 25 + (i % 40)
        rows.append(row)
    return pd.DataFrame(rows)[FEATURE_COLS]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("n_rows", [1, 7, 50])
def test_batch_returns_same_row_count(client, n_rows):
    df = _sample_rows(n_rows)
    buf = io.BytesIO(df.to_csv(index=False).encode())

    resp = client.post("/predict_batch", files={"file": ("rows.csv", buf, "text/csv")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")

    assert resp.status_code == 200
    body = resp.json()
    assert body["n"] == n_rows, f"sent {n_rows} rows, got {body['n']} back"
    assert len(body["predictions"]) == n_rows
    for pred in body["predictions"]:
        assert 0.0 <= pred["proba"] <= 1.0
        assert pred["label"] in (0, 1)


def test_batch_rejects_missing_columns(client):
    df = _sample_rows(3).drop(columns=["occupation"])
    buf = io.BytesIO(df.to_csv(index=False).encode())
    resp = client.post("/predict_batch", files={"file": ("bad.csv", buf, "text/csv")})
    assert resp.status_code == 422
    assert "occupation" in resp.text


def test_reference_row_probability_is_stable(client):
    """On first run this records the current probability and passes; every run
    afterwards compares against it. The fixture file is committed, so the
    pinned value travels with the repository."""
    resp = client.post("/predict", json={"record": REFERENCE_ROW})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    assert resp.status_code == 200
    proba = resp.json()["proba"]

    if not FIXTURE_PATH.exists():
        FIXTURE_PATH.write_text(json.dumps(
            {"record": REFERENCE_ROW, "proba": proba, "tolerance": TOLERANCE}, indent=2))
        pytest.skip(f"recorded reference probability {proba:.6f} -- commit "
                    f"{FIXTURE_PATH.name} so future runs compare against it")

    expected = json.loads(FIXTURE_PATH.read_text())["proba"]
    assert abs(proba - expected) < TOLERANCE, (
        f"reference probability drifted: expected {expected:.6f}, got {proba:.6f}. "
        f"If this is an intentional retrain, delete {FIXTURE_PATH.name} and re-run."
    )


def test_reference_row_is_deterministic(client):
    """Two identical requests must return identical probabilities -- dropout
    must be off at inference, and the preprocessor must not be refitting."""
    r1 = client.post("/predict", json={"record": REFERENCE_ROW})
    if r1.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    r2 = client.post("/predict", json={"record": REFERENCE_ROW})
    assert r1.json()["proba"] == r2.json()["proba"]