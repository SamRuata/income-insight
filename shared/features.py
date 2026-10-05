"""Feature contract for the REAL UCI Adult Income data.

The template's ``shared/data.py`` describes a 7-column *synthetic* dataset and
is left untouched so the existing seed script and tests keep working. This
module is the contract for the real data loaded into the ``adult_income``
table, and it is the single source of truth shared by:

  * the sklearn ColumnTransformer in api/training.py
  * the ``/schema`` endpoint that drives the Streamlit form
  * the bias audit, which needs to know which columns are protected

Changing a column here changes it everywhere, which is the point.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
NUMERIC_COLS: List[str] = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLS: List[str] = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS

TARGET_NAME = "income"
TARGET_CLASSES = ["<=50K", ">50K"]      # index 0 / 1

# Attributes the fairness audit groups by. `sex` and `race` are also model
# features: dropping them would not remove the bias (relationship, occupation
# and marital_status all proxy for sex), but it would hide the model's use of
# them from exactly the audit meant to catch it.
PROTECTED_COLS: List[str] = ["sex", "race"]

# ---------------------------------------------------------------------------
# Allowed values, used to build the UI form and validate incoming records.
# Taken from the UCI Adult documentation. "?" in the raw file is loaded as
# NULL and imputed by the pipeline, so it is not a valid input value.
# ---------------------------------------------------------------------------
CATEGORIES: Dict[str, List[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    ],
    "marital_status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried",
    ],
    "race": [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black",
    ],
    "sex": ["Female", "Male"],
    "native_country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
        "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
        "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
        "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
        "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru",
        "Hong", "Holand-Netherlands",
    ],
}

# Sensible defaults + bounds for the auto-generated UI form.
NUMERIC_BOUNDS: Dict[str, Dict[str, float]] = {
    "age":            {"min": 17,  "max": 90,     "default": 38},
    "education_num":  {"min": 1,   "max": 16,     "default": 10},
    "capital_gain":   {"min": 0,   "max": 99999,  "default": 0},
    "capital_loss":   {"min": 0,   "max": 4356,   "default": 0},
    "hours_per_week": {"min": 1,   "max": 99,     "default": 40},
}


def schema_payload() -> dict:
    """What GET /schema returns; the Streamlit form is built from this."""
    return {
        "numeric": [
            {"name": c, "dtype": "int", **NUMERIC_BOUNDS[c]} for c in NUMERIC_COLS
        ],
        "categorical": [
            {"name": c, "dtype": "str", "choices": CATEGORIES[c]} for c in CATEGORICAL_COLS
        ],
        "target": {"name": TARGET_NAME, "classes": TARGET_CLASSES},
        "protected": PROTECTED_COLS,
    }