"""Request-schema tests for /predict.

A malformed request must be rejected with a clear error rather than scored on
whatever happens to be present -- a prediction made from partial input is
worse than no prediction, because it looks the same as a real one.
"""
from fastapi.testclient import TestClient

from api.main import app
from shared.features import CATEGORICAL_COLS, NUMERIC_BOUNDS, NUMERIC_COLS

VALID_RECORD = {
    **{c: int(NUMERIC_BOUNDS[c]["default"]) for c in NUMERIC_COLS},
    "workclass": "Private",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "native_country": "United-States",
}


def test_missing_record_key_is_rejected():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"not_a_record": {}})
    assert resp.status_code == 422


def test_incomplete_record_is_rejected():
    partial = {k: v for k, v in VALID_RECORD.items() if k != "occupation"}
    with TestClient(app) as client:
        resp = client.post("/predict", json={"record": partial})
    assert resp.status_code == 422
    assert "occupation" in resp.text


def test_wrong_type_is_rejected():
    bad = dict(VALID_RECORD, source_row_id="not-an-integer")
    with TestClient(app) as client:
        resp = client.post("/predict", json={"record": VALID_RECORD, "source_row_id": "abc"})
    assert resp.status_code == 422


def test_valid_record_scores_in_range():
    """Only runs if a checkpoint is present; otherwise the API returns 503."""
    with TestClient(app) as client:
        resp = client.post("/predict", json={"record": VALID_RECORD})
    if resp.status_code == 503:
        import pytest
        pytest.skip("no model checkpoint loaded")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["proba"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["predicted_class"] in ("<=50K", ">50K")