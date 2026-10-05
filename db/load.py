"""One-shot loader: UCI Adult Income -> Supabase `adult_income` table.

Run once after applying db/migrations/002_adult_income.sql:

    python -m db.load

Options:
    --force   wipe and reload even if the table already has rows
    --file    path to a local adult.data instead of downloading

Why this exists: the template's db/seed.py writes a *synthetic* dataset into
`datasets.records` as a JSONB blob. The assignment needs the real UCI Adult
data as one row per example so the fairness audit can GROUP BY protected
attributes in SQL. This script does that load; seed.py is left alone.
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from supabase import create_client

# The raw file has no header row; these are the column names from the UCI
# documentation, in file order.
RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# Columns we keep. fnlwgt is a census sampling weight (not a property of the
# person) and `education` is a text duplicate of education_num, so both are
# dropped rather than fed to the model as noise.
NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = [
    "workclass", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country",
]
KEEP_COLUMNS = NUMERIC_COLS + CATEGORICAL_COLS + ["income"]

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

BATCH_SIZE = 500          # Supabase rejects very large single inserts
TEST_FRACTION = 0.20
RANDOM_SEED = 42          # fixed so the train/test split is reproducible


def get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    return create_client(url, key)


def load_frame(path: str | None) -> pd.DataFrame:
    source = path or DATA_URL
    print(f"Reading {source}")
    df = pd.read_csv(
        source,
        header=None,
        names=RAW_COLUMNS,
        skipinitialspace=True,      # the raw file pads every field with a space
        na_values=["?"],            # UCI encodes missing categoricals as "?"
    )
    print(f"  {len(df):,} raw rows")

    df = df[KEEP_COLUMNS].copy()

    # Target: keep the raw string for readability, add the 0/1 the model trains on.
    df["income"] = df["income"].str.strip().str.rstrip(".")
    df["label"] = (df["income"] == ">50K").astype(int)

    # Deterministic train/test split, assigned once at load time so every
    # training run and every audit query sees the same held-out set.
    shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED).index
    n_test = int(len(df) * TEST_FRACTION)
    df["split"] = "train"
    df.loc[shuffled[:n_test], "split"] = "test"

    print(f"  kept {len(df.columns)} columns, "
          f"{(df['split'] == 'train').sum():,} train / {(df['split'] == 'test').sum():,} test")
    print(f"  positive rate: {df['label'].mean():.3f}")
    missing = df[CATEGORICAL_COLS].isna().sum()
    if missing.any():
        print("  missing values (left as NULL for the pipeline to impute):")
        for col, n in missing[missing > 0].items():
            print(f"    {col}: {n:,}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="wipe and reload")
    parser.add_argument("--file", default=None, help="local adult.data path")
    args = parser.parse_args()

    client = get_client()

    existing = client.table("adult_income").select("id", count="exact").limit(1).execute()
    if existing.count and not args.force:
        sys.exit(f"adult_income already has {existing.count:,} rows. Use --force to reload.")

    df = load_frame(args.file)

    if args.force and existing.count:
        print(f"Deleting {existing.count:,} existing rows")
        client.table("adult_income").delete().neq("id", 0).execute()

    # pandas uses NaN for missing; Postgres wants null.
    records = df.where(pd.notna(df), None).to_dict(orient="records")

    print(f"Inserting {len(records):,} rows in batches of {BATCH_SIZE}")
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        client.table("adult_income").insert(batch).execute()
        print(f"  {min(start + BATCH_SIZE, len(records)):,} / {len(records):,}", end="\r")

    final = client.table("adult_income").select("id", count="exact").limit(1).execute()
    print(f"\nDone. adult_income now has {final.count:,} rows.")


if __name__ == "__main__":
    main()