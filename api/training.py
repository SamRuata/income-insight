"""Preprocessing pipeline + PyTorch MLP for the Adult-Income classifier.

Two objects are produced by a training run and must travel together forever
after: the fitted sklearn ColumnTransformer and the trained MLP weights. If
they are separated, the model receives columns in a different order or a
different one-hot width and its output becomes silently meaningless -- so both
are serialized into a single checkpoint file by api/train.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute then standardize. Categorical: impute the most
    frequent value then one-hot encode.

    Imputation is inside the pipeline rather than done to the dataframe
    beforehand so the median and the modal category are learned from the
    TRAINING rows only and then reapplied at inference. Computing them over
    the whole dataset would leak test information into training.

    handle_unknown="ignore" means a category never seen in training (a rare
    native_country, say) encodes as all-zeros instead of raising at serve time.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Feed-forward net with a configurable stack of hidden layers.

    Outputs a single raw logit; the sigmoid lives in BCEWithLogitsLoss during
    training and is applied explicitly at inference. Doing it this way is
    numerically stabler than a sigmoid layer plus plain BCE.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("the assignment requires at least two hidden layers")
        act = ACTIVATIONS[activation.lower()]

        layers: List[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy":  float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, proba)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """Mean predicted probability vs observed frequency, per bin.

    A well-calibrated model sits on the diagonal: among rows it scores 0.7,
    about 70% should actually earn >50K. Accuracy says nothing about this,
    which is why the client gets both.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1.0 else proba <= hi)
        if mask.sum() == 0:
            continue
        out.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "mean_predicted": float(proba[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame, config: Dict) -> Tuple[MLP, ColumnTransformer, Dict]:
    """Fit the pipeline and the MLP. Returns (model, fitted preprocessor, results).

    The dataframe must carry the feature columns, `label`, and `split`
    ('train' or 'test'). The test rows are never touched until the final
    evaluation; a slice of the training rows is held out as a validation set
    so early stopping has something honest to watch.
    """
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Carve a validation slice out of train (not out of test).
    rng = np.random.default_rng(config.get("seed", 42))
    val_mask = rng.random(len(train_df)) < 0.10
    val_df = train_df[val_mask]
    fit_df = train_df[~val_mask]

    pre = build_preprocessor()
    X_fit = pre.fit_transform(fit_df[FEATURE_COLS])       # fit on training rows ONLY
    X_val = pre.transform(val_df[FEATURE_COLS])
    X_test = pre.transform(test_df[FEATURE_COLS])

    y_fit = fit_df["label"].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    device = torch.device("cpu")
    model = MLP(
        input_dim=X_fit.shape[1],
        hidden_sizes=list(config["hidden_sizes"]),
        activation=config.get("activation", "relu"),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)

    # The target is ~24% positive. pos_weight scales the loss on the minority
    # class so the model does not collapse to "always predict <=50K", which
    # would score 76% accuracy while being useless.
    pos_weight = torch.tensor([(y_fit == 0).sum() / max((y_fit == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    Xt = torch.tensor(X_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    history: List[Dict] = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience = int(config.get("patience", 5))
    since_improved = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        running = 0.0
        correct = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = criterion(val_logits, yv).item()
            val_acc = ((torch.sigmoid(val_logits) >= 0.5).float() == yv).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": running / len(Xt),
            "train_acc": correct / len(Xt),
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"  epoch {epoch:3d}  train_loss {running/len(Xt):.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                print(f"  early stop at epoch {epoch} (best was {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on the untouched test split.
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy()

    results = {
        "metrics": compute_metrics(y_test, proba),
        "calibration": calibration_curve(y_test, proba),
        "history": history,
        "best_epoch": best_epoch,
        "n_train": int(len(fit_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "input_dim": int(X_fit.shape[1]),
    }
    return model, pre, results


def predict_proba(model: MLP, pre: ColumnTransformer, records: List[Dict]) -> np.ndarray:
    """Score raw feature dicts. Used by /predict and /predict_batch."""
    df = pd.DataFrame(records)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = None
    X = pre.transform(df[FEATURE_COLS])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()