"""Live Supabase round trip.

Inserts a fixture row into adult_income, scores it through the API so a
prediction is logged against it, confirms the join the fairness audit depends
on, then deletes everything it created. Skips without credentials so the rest
of the suite still runs offline.
"""
import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

from api.main import app
from shared.features import NUMERIC_BOUNDS, NUMERIC_COLS

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")),
    reason="Supabase credentials not set",
)

FIXTURE = {
    **{c: int(NUMERIC_BOUNDS[c]["default"]) for c in NUMERIC_COLS},
    "workclass": "Private",
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "native_country": "United-States",
    "income": "<=50K",
    "label": 0,
    "split": "test",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_prediction_round_trip(client):
    from api.main import get_client
    sb = get_client()

    created = sb.table("adult_income").insert(FIXTURE).execute()
    row_id = created.data[0]["id"]

    try:
        record = {k: v for k, v in FIXTURE.items()
                  if k not in ("income", "label", "split")}
        resp = client.post("/predict", json={"record": record, "source_row_id": row_id})
        if resp.status_code == 503:
            pytest.skip("no model checkpoint loaded")
        assert resp.status_code == 200

        logged = (sb.table("predictions").select("*")
                  .eq("source_row_id", row_id).execute())
        assert logged.data, "no prediction row was written"

        pred = logged.data[0]
        assert pred["true_label"] == FIXTURE["label"], \
            "ground truth was not carried across; the fairness audit would be wrong"
        assert pred["run_id"] is not None
        assert 0.0 <= pred["proba"] <= 1.0
    finally:
        sb.table("predictions").delete().eq("source_row_id", row_id).execute()
        sb.table("adult_income").delete().eq("id", row_id).execute()


def test_audit_function_is_callable(client):
    """The audit aggregates in Postgres; this proves the function exists and
    returns the fields the UI expects."""
    resp = client.get("/audit", params={"attribute": "sex"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["attribute"] == "sex"
    for group in body["groups"]:
        assert {"group_value", "n", "fpr", "fnr"}.issubset(group.keys())