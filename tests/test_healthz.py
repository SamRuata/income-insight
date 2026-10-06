"""Smoke test: the service starts and reports its dependencies."""
from fastapi.testclient import TestClient

from api.main import app


def test_healthz_ok():
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    # The endpoint must always answer, even degraded -- Render uses it as the
    # health check, and a 500 here would take the deployment down.
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body
    assert "supabase" in body


def test_version_reports_stack():
    with TestClient(app) as client:
        resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "torch" in body and "sklearn" in body