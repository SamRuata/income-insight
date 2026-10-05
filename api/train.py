"""CLI training script.

    python -m api.train --config api/configs/baseline.yaml

Reads the training rows from Supabase, fits the pipeline + MLP, writes the
best checkpoint to models/, and records one row in `runs` so the three
configurations can be compared with a SQL query later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import joblib
import pandas as pd
import sklearn
import torch
import yaml
from supabase import create_client

from api.training import train_model
from shared.features import FEATURE_COLS

MODELS_DIR = Path("models")
PAGE_SIZE = 1000        # Supabase caps a single select; fetch in pages


def get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    return create_client(url, key)


def fetch_adult_income(client) -> pd.DataFrame:
    """Pull every row of adult_income, 1000 at a time.

    A plain .select() returns at most 1000 rows. Without paging the model
    would train on 1000 of 32,561 rows and still report plausible-looking
    metrics, which is the kind of bug that does not announce itself.
    """
    cols = ",".join(FEATURE_COLS + ["label", "split", "id"])
    frames, start = [], 0
    while True:
        resp = (client.table("adult_income")
                      .select(cols)
                      .order("id")
                      .range(start, start + PAGE_SIZE - 1)
                      .execute())
        if not resp.data:
            break
        frames.append(pd.DataFrame(resp.data))
        print(f"  fetched {start + len(resp.data):,} rows", end="\r")
        if len(resp.data) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not frames:
        sys.exit("adult_income is empty -- run `python -m db.load` first")
    df = pd.concat(frames, ignore_index=True)
    print(f"  fetched {len(df):,} rows total")
    return df


def ensure_dataset_row(client, df: pd.DataFrame) -> int:
    """`runs.dataset_id` is a NOT NULL foreign key into `datasets`, which the
    template built for synthetic datasets. The real data lives in its own
    table, so we keep one descriptive row in `datasets` for the UCI dataset
    and point every run at it rather than loosening the constraint.
    """
    existing = client.table("datasets").select("id").eq("name", "uci-adult").limit(1).execute()
    if existing.data:
        return existing.data[0]["id"]
    row = {
        "name": "uci-adult",
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLS),
        "positive_rate": float(df["label"].mean()),
    }
    created = client.table("datasets").insert(row).execute()
    return created.data[0]["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    print(f"Config: {config['name']}  {config['hidden_sizes']}  "
          f"{config['activation']}  dropout={config['dropout']}")

    client = get_client()
    print("Fetching training data from Supabase")
    df = fetch_adult_income(client)

    print("Training")
    model, pre, results = train_model(df, config)

    m = results["metrics"]
    print(f"\nHeld-out test metrics ({results['n_test']:,} rows)")
    print(f"  accuracy  {m['accuracy']:.4f}")
    print(f"  precision {m['precision']:.4f}")
    print(f"  recall    {m['recall']:.4f}")
    print(f"  f1        {m['f1']:.4f}")
    print(f"  roc_auc   {m['roc_auc']:.4f}")
    print(f"  confusion {m['confusion']}")

    # One checkpoint holds the weights AND the fitted preprocessor, so they
    # can never drift apart.
    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{config['name']}.joblib"
    joblib.dump({
        "state_dict": model.state_dict(),
        "preprocessor": pre,
        "config": config,
        "metrics": m,
        "calibration": results["calibration"],
        "history": results["history"],
        "input_dim": results["input_dim"],
        "feature_cols": FEATURE_COLS,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
    }, checkpoint_path)
    print(f"\nSaved checkpoint -> {checkpoint_path}")

    dataset_id = ensure_dataset_row(client, df)
    run_row = {
        "dataset_id":   dataset_id,
        "config_name":  config["name"],
        "hidden_sizes": ",".join(str(h) for h in config["hidden_sizes"]),
        "activation":   config["activation"],
        "dropout":      float(config["dropout"]),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "lr":           float(config["lr"]),
        "batch_size":   int(config["batch_size"]),
        "epochs":       int(config["epochs"]),
        "best_epoch":   results["best_epoch"],
        "n_train":      results["n_train"],
        "n_test":       results["n_test"],
        "accuracy":     m["accuracy"],
        "precision":    m["precision"],
        "recall":       m["recall"],
        "f1":           m["f1"],
        "roc_auc":      m["roc_auc"],
        "notes":        config.get("notes", ""),
    }
    created = client.table("runs").insert(run_row).execute()
    print(f"Wrote runs row id={created.data[0]['id']}")


if __name__ == "__main__":
    main()