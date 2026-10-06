"""Permutation importance: which features actually drive the predictions.

    python -m api.importance --config baseline

Shuffles one feature at a time in the held-out test split and measures how far
ROC-AUC falls. A feature the model leans on hard will cause a large drop when
scrambled; a feature it ignores will cause none. This is model-agnostic -- it
asks what the fitted model uses, not what the architecture could in principle
use -- which is why it works on an MLP where weights are uninterpretable.

Prints a markdown table for the README and writes docs/importance.json.
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
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from supabase import create_client

from api.training import MLP
from shared.features import FEATURE_COLS

PAGE_SIZE = 1000
N_REPEATS = 5          # average over several shuffles; one draw is noisy


def fetch_test_rows(client, limit: int = 6000) -> pd.DataFrame:
    cols = ",".join(FEATURE_COLS + ["label", "id"])
    frames, start = [], 0
    while start < limit:
        resp = (client.table("adult_income").select(cols)
                .eq("split", "test").order("id")
                .range(start, min(start + PAGE_SIZE, limit) - 1).execute())
        if not resp.data:
            break
        frames.append(pd.DataFrame(resp.data))
        if len(resp.data) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline")
    args = parser.parse_args()

    ckpt_path = Path("models") / f"{args.config}.joblib"
    if not ckpt_path.exists():
        sys.exit(f"no checkpoint at {ckpt_path}")
    ckpt = joblib.load(ckpt_path)

    cfg = ckpt["config"]
    model = MLP(input_dim=ckpt["input_dim"], hidden_sizes=list(cfg["hidden_sizes"]),
                activation=cfg.get("activation", "relu"),
                dropout=float(cfg.get("dropout", 0.0)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pre = ckpt["preprocessor"]

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY must be set")
    df = fetch_test_rows(create_client(url, key))
    print(f"Scoring {len(df):,} held-out test rows\n")

    y = df["label"].to_numpy()

    def auc_of(frame: pd.DataFrame) -> float:
        X = pre.transform(frame[FEATURE_COLS])
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
        return roc_auc_score(y, p)

    baseline_auc = auc_of(df)
    print(f"Baseline ROC-AUC: {baseline_auc:.4f}\n")

    rng = np.random.default_rng(42)
    rows = []
    for col in FEATURE_COLS:
        drops = []
        for _ in range(N_REPEATS):
            shuffled = df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(baseline_auc - auc_of(shuffled))
        rows.append({"feature": col,
                     "auc_drop": float(np.mean(drops)),
                     "std": float(np.std(drops))})
        print(f"  {col:<16} drop {np.mean(drops):+.4f}")

    rows.sort(key=lambda r: r["auc_drop"], reverse=True)

    print("\n| Feature | ROC-AUC drop when shuffled | sd |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['feature']} | {r['auc_drop']:.4f} | {r['std']:.4f} |")

    Path("docs").mkdir(exist_ok=True)
    out = {"config": args.config, "baseline_roc_auc": baseline_auc,
           "n_rows": int(len(df)), "n_repeats": N_REPEATS, "importances": rows}
    Path("docs/importance.json").write_text(json.dumps(out, indent=2))
    print("\nWrote docs/importance.json")


if __name__ == "__main__":
    main()