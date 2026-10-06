"""The /schema endpoint is the contract the Streamlit form is built from.

If it drifts from what the model was trained on, the UI silently sends the
wrong columns, so this test pins the contract in both directions.
"""
from fastapi.testclient import TestClient

from api.main import app
from shared.features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, PROTECTED_COLS


def test_schema_matches_feature_contract():
    with TestClient(app) as client:
        resp = client.get("/schema")
    assert resp.status_code == 200
    body = resp.json()

    numeric = [f["name"] for f in body["numeric"]]
    categorical = [f["name"] for f in body["categorical"]]

    assert numeric == NUMERIC_COLS
    assert categorical == CATEGORICAL_COLS
    assert len(numeric) + len(categorical) == len(FEATURE_COLS)


def test_schema_exposes_protected_attributes():
    """The bias audit can only group by attributes the schema declares."""
    with TestClient(app) as client:
        body = client.get("/schema").json()
    assert set(PROTECTED_COLS).issubset(set(body["protected"]))
    assert "sex" in body["protected"]


def test_categoricals_offer_choices():
    with TestClient(app) as client:
        body = client.get("/schema").json()
    for field in body["categorical"]:
        assert len(field["choices"]) > 0, f"{field['name']} has no choices"