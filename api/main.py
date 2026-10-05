"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Owns the model. The Streamlit tier never imports torch or sklearn; it calls
these endpoints over HTTPS. Supabase is the only persistence layer, reached
here with the service-role key.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
import sklearn
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

from api.training import MLP, predict_proba
from shared.features import FEATURE_COLS, PROTECTED_COLS, schema_payload

MODELS_DIR = Path("models")
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "baseline")

app = FastAPI(title="Income-Insight API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    record: Dict = Field(..., description="One row keyed by feature name")
    source_row_id: Optional[int] = Field(
        None, description="adult_income.id, when scoring a known row (enables fairness audit)"
    )


class PredictResponse(BaseModel):
    label: int
    proba: float
    predicted_class: str
    run_id: int


class BatchResponse(BaseModel):
    n: int
    run_id: int
    predictions: List[Dict]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_state: Dict = {"model": None, "pre": None, "meta": None, "run_id": None, "client": None}


def get_client():
    if _state["client"] is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
        _state["client"] = create_client(url, key)
    return _state["client"]


def load_model(name: str = DEFAULT_MODEL) -> None:
    """Rebuild the MLP from its checkpoint and pair it with its own fitted
    preprocessor. Weights and preprocessor always travel together; loading one
    without the other would silently scramble the feature order."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path} -- run api/train.py first")
    ckpt = joblib.load(path)
    cfg = ckpt["config"]
    model = MLP(
        input_dim=ckpt["input_dim"],
        hidden_sizes=list(cfg["hidden_sizes"]),
        activation=cfg.get("activation", "relu"),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _state.update(model=model, pre=ckpt["preprocessor"], meta=ckpt)

    # Every logged prediction must name the run that served it, so the audit
    # can be reproduced against a specific model later.
    try:
        resp = (get_client().table("runs")
                .select("id").eq("config_name", name)
                .order("id", desc=True).limit(1).execute())
        _state["run_id"] = resp.data[0]["id"] if resp.data else None
    except Exception:
        _state["run_id"] = None


@app.on_event("startup")
def _startup() -> None:
    try:
        load_model()
    except Exception as exc:          # keep the service up so /healthz can report
        print(f"startup: model load failed: {exc}")


def _require_model():
    if _state["model"] is None:
        raise HTTPException(503, "model not loaded")
    if _state["run_id"] is None:
        raise HTTPException(503, "no runs row for this model; run api/train.py")


def _hash(record: Dict) -> str:
    """Log a hash of the inputs rather than raw PII, per the product brief."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:32]


def _log_predictions(rows: List[Dict]) -> None:
    try:
        get_client().table("predictions").insert(rows).execute()
    except Exception as exc:
        print(f"prediction logging failed: {exc}")   # never fail a prediction on logging


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    ok_model = _state["model"] is not None
    ok_db = True
    try:
        get_client().table("runs").select("id").limit(1).execute()
    except Exception:
        ok_db = False
    return {"status": "ok" if (ok_model and ok_db) else "degraded",
            "model_loaded": ok_model, "supabase": ok_db,
            "model_name": DEFAULT_MODEL, "run_id": _state["run_id"]}


@app.get("/version")
def version():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = os.environ.get("RENDER_GIT_COMMIT", "unknown")[:7]
    url = os.environ.get("SUPABASE_URL", "")
    return {"git_sha": sha, "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "supabase_project_ref": url.split("//")[-1].split(".")[0] if url else None,
            "model": DEFAULT_MODEL}


@app.get("/schema")
def schema():
    """Feature contract. The Streamlit form is generated from this, so the UI
    never hard-codes a column list that could drift from the model's."""
    return schema_payload()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _require_model()
    missing = [c for c in FEATURE_COLS if c not in req.record]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")

    proba = float(predict_proba(_state["model"], _state["pre"], [req.record])[0])
    label = int(proba >= 0.5)

    true_label = None
    if req.source_row_id is not None:
        got = (get_client().table("adult_income").select("label")
               .eq("id", req.source_row_id).limit(1).execute())
        if got.data:
            true_label = got.data[0]["label"]

    _log_predictions([{
        "run_id": _state["run_id"], "features": req.record,
        "proba": proba, "label": label,
        "request_hash": _hash(req.record),
        "source_row_id": req.source_row_id, "true_label": true_label,
    }])
    return PredictResponse(label=label, proba=proba,
                           predicted_class=">50K" if label else "<=50K",
                           run_id=_state["run_id"])


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Score an uploaded CSV. Every row is logged individually."""
    _require_model()
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, f"could not parse CSV: {exc}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(422, f"CSV is missing columns: {missing}")
    if len(df) > 5000:
        raise HTTPException(413, "batch limited to 5000 rows")

    records = df[FEATURE_COLS].to_dict(orient="records")
    probas = predict_proba(_state["model"], _state["pre"], records)

    # An uploaded file may carry the adult_income id and/or the true label;
    # if so the rows become auditable, otherwise they are prediction-only.
    ids = df["source_row_id"].tolist() if "source_row_id" in df.columns else [None] * len(df)
    truth = df["label"].tolist() if "label" in df.columns else [None] * len(df)

    out, log = [], []
    for rec, p, sid, tl in zip(records, probas, ids, truth):
        label = int(p >= 0.5)
        out.append({**rec, "proba": float(p), "label": label,
                    "predicted_class": ">50K" if label else "<=50K"})
        log.append({"run_id": _state["run_id"], "features": rec,
                    "proba": float(p), "label": label,
                    "request_hash": _hash(rec),
                    "source_row_id": int(sid) if pd.notna(sid) else None,
                    "true_label": int(tl) if pd.notna(tl) else None})

    for i in range(0, len(log), 500):
        _log_predictions(log[i:i + 500])
    return BatchResponse(n=len(out), run_id=_state["run_id"], predictions=out)


@app.post("/score_test_sample")
def score_test_sample(n: int = Query(500, ge=1, le=2000)):
    """Score n held-out test rows and log them WITH ground truth.

    The fairness audit needs predictions whose true label is known. Rows typed
    into the UI by hand never have one, so without this the audit dashboard
    would be permanently empty. This is the honest way to populate it: the
    rows come from the test split the model never trained on.
    """
    _require_model()
    cols = ",".join(FEATURE_COLS + ["id", "label"])
    resp = (get_client().table("adult_income").select(cols)
            .eq("split", "test").order("id").limit(n).execute())
    if not resp.data:
        raise HTTPException(404, "no test rows found")

    df = pd.DataFrame(resp.data)
    records = df[FEATURE_COLS].to_dict(orient="records")
    probas = predict_proba(_state["model"], _state["pre"], records)

    log = [{"run_id": _state["run_id"], "features": rec,
            "proba": float(p), "label": int(p >= 0.5),
            "request_hash": _hash(rec),
            "source_row_id": int(rid), "true_label": int(tl)}
           for rec, p, rid, tl in zip(records, probas, df["id"], df["label"])]
    for i in range(0, len(log), 500):
        _log_predictions(log[i:i + 500])
    return {"scored": len(log), "run_id": _state["run_id"]}


@app.get("/audit")
def audit(attribute: str = Query("sex")):
    """False-positive and false-negative rates by protected attribute,
    aggregated in Postgres by the audit_by_attribute() function."""
    if attribute not in PROTECTED_COLS + ["workclass", "marital_status", "relationship"]:
        raise HTTPException(422, f"attribute must be one of {PROTECTED_COLS}")
    try:
        resp = get_client().rpc("audit_by_attribute", {"attr": attribute}).execute()
    except Exception as exc:
        raise HTTPException(500, f"audit query failed: {exc}")
    return {"attribute": attribute, "groups": resp.data or [],
            "note": "Computed only over predictions with known ground truth."}


@app.get("/runs")
def runs(limit: int = Query(50, ge=1, le=200)):
    resp = (get_client().table("runs").select("*")
            .order("id", desc=True).limit(limit).execute())
    return {"runs": resp.data or []}


@app.get("/runs/{run_id}")
def run_detail(run_id: int):
    resp = get_client().table("runs").select("*").eq("id", run_id).limit(1).execute()
    if not resp.data:
        raise HTTPException(404, "run not found")
    return resp.data[0]


@app.get("/performance")
def performance(name: str = Query(DEFAULT_MODEL)):
    """Training curves, confusion matrix and calibration bins for a config.

    Read from the checkpoint rather than recomputed, so the UI plots exactly
    the numbers the model was evaluated on."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise HTTPException(404, f"no checkpoint for '{name}'")
    ckpt = joblib.load(path)
    return {"config": ckpt["config"], "metrics": ckpt["metrics"],
            "calibration": ckpt["calibration"], "history": ckpt["history"]}