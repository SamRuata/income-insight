"""Streamlit UI -- Cloud #1. A thin client: no torch, no sklearn, no model code.

Everything model-shaped goes through the FastAPI service over HTTPS. The only
direct database access is a read-only Supabase query with the ANON key, used
for the run history and the bias-audit breakdown.
"""
from __future__ import annotations

import io
import json

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")
SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/")
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

st.set_page_config(page_title="Income-Insight", page_icon="📊", layout="wide")
st.title("📊 Income-Insight — Adult-Income Classifier")
st.caption("Streamlit (this UI) → FastAPI (model) → Supabase (data). Three clouds.")


@st.cache_resource
def sb():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload=None, files=None, **params):
    r = requests.post(f"{API_URL}{path}", json=payload, files=files, params=params, timeout=300)
    r.raise_for_status()
    return r.json()


tabs = st.tabs([
    "Concepts", "Score a Row", "Score a CSV",
    "Model Performance", "Bias Audit", "Model Card",
])

# ===========================================================================
# 1. Concepts
# ===========================================================================
with tabs[0]:
    st.header("Forward and Backward Propagation")

    st.subheader("Forward pass, in matrix form")
    st.markdown(
        "For a batch of $m$ rows with $d$ features, the input is $X \\in \\mathbb{R}^{m \\times d}$. "
        "Each hidden layer $\\ell$ applies an affine map then a nonlinearity:"
    )
    st.latex(r"Z^{[\ell]} = A^{[\ell-1]} W^{[\ell]} + b^{[\ell]}, \qquad A^{[\ell]} = g\left(Z^{[\ell]}\right)")
    st.markdown(
        "with $A^{[0]} = X$. The final layer emits one logit per row, "
        "turned into a probability by the sigmoid:"
    )
    st.latex(r"\hat{y} = \sigma\left(Z^{[L]}\right) = \frac{1}{1 + e^{-Z^{[L]}}}")
    st.markdown("Loss is binary cross-entropy, averaged over the batch:")
    st.latex(r"\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]")

    st.subheader("Backward pass")
    st.markdown(
        "Backpropagation is the chain rule applied layer by layer, right to left. "
        "For binary cross-entropy composed with a sigmoid, the output error "
        "collapses to a remarkably simple form:"
    )
    st.latex(r"\delta^{[L]} = \hat{y} - y")
    st.markdown("Each earlier layer's error is the next layer's error pushed back through the weights:")
    st.latex(r"\delta^{[\ell]} = \left(\delta^{[\ell+1]} {W^{[\ell+1]}}^{\top}\right) \odot g'\left(Z^{[\ell]}\right)")
    st.markdown("and the gradients for each parameter fall out of those errors:")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial W^{[\ell]}} = \frac{1}{m}{A^{[\ell-1]}}^{\top}\delta^{[\ell]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[\ell]}} = \frac{1}{m}\sum_{i=1}^{m}\delta^{[\ell]}_i")
    st.markdown(
        "$\\odot$ is elementwise multiplication. Every gradient is a matrix product of "
        "things already computed in the forward pass, which is why training a deep "
        "network costs only a constant factor more than running one."
    )

    st.subheader("Worked example: XOR")
    st.markdown(
        "XOR is the classic proof that hidden layers are necessary. The four points "
        "$(0,0)\\to0$, $(0,1)\\to1$, $(1,0)\\to1$, $(1,1)\\to0$ are not linearly "
        "separable: no single straight line puts both 1s on one side. A network with "
        "no hidden layer cannot exceed 75% accuracy on it, no matter how long it trains."
    )
    st.markdown("Two hidden units solve it. Take these weights:")
    st.latex(r"W^{[1]} = \begin{bmatrix} 20 & -20 \\ 20 & -20 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} -10 \\ 30 \end{bmatrix}, \quad W^{[2]} = \begin{bmatrix} 20 \\ 20 \end{bmatrix}, \quad b^{[2]} = -30")
    st.markdown("Trace $x = (0, 1)$, expected output 1:")
    st.latex(r"Z^{[1]} = \begin{bmatrix}0 & 1\end{bmatrix}\begin{bmatrix} 20 & -20 \\ 20 & -20 \end{bmatrix} + \begin{bmatrix}-10 & 30\end{bmatrix} = \begin{bmatrix}10 & 10\end{bmatrix}")
    st.latex(r"A^{[1]} = \sigma\left(\begin{bmatrix}10 & 10\end{bmatrix}\right) \approx \begin{bmatrix}1 & 1\end{bmatrix}")
    st.latex(r"Z^{[2]} = 1(20) + 1(20) - 30 = 10 \quad\Rightarrow\quad \hat{y} = \sigma(10) \approx 0.9999")
    st.markdown(
        "The first hidden unit fires on *at least one* input (an OR gate), the second "
        "on *not both* (a NAND gate), and the output unit ANDs them. "
        "OR AND NAND is exactly XOR. The hidden layer builds features the input "
        "didn't have — the same thing happening, less legibly, in the income model's "
        "64 and 32 hidden units."
    )
    st.markdown(
        "Backprop for this example: the output error is $\\hat{y} - y = -0.0001$, "
        "tiny because the prediction is nearly right. That error flows back through "
        "$W^{[2]}$, is scaled by $\\sigma'(Z^{[1]}) = A^{[1]}(1 - A^{[1]}) \\approx 0$, "
        "and the gradients vanish — the saturated sigmoids have stopped learning. "
        "That saturation is why ReLU and GELU replaced sigmoid in hidden layers."
    )

# ===========================================================================
# 2. Score a Row -- form generated from /schema
# ===========================================================================
with tabs[1]:
    st.header("Score a single record")
    try:
        schema = api_get("/schema")
    except Exception as exc:
        st.error(f"Could not reach the API: {exc}")
        schema = None

    if schema:
        st.caption("This form is generated from the API's /schema endpoint, so it can never "
                   "drift out of sync with the columns the model expects.")
        record = {}
        cols = st.columns(3)
        for i, f in enumerate(schema["numeric"]):
            with cols[i % 3]:
                record[f["name"]] = st.number_input(
                    f["name"], min_value=int(f["min"]), max_value=int(f["max"]),
                    value=int(f["default"]), step=1,
                )
        cols = st.columns(2)
        for i, f in enumerate(schema["categorical"]):
            with cols[i % 2]:
                record[f["name"]] = st.selectbox(f["name"], f["choices"])

        if st.button("Predict", type="primary"):
            try:
                res = api_post("/predict", {"record": record})
                c1, c2 = st.columns(2)
                c1.metric("Predicted class", res["predicted_class"])
                c2.metric("P(income > $50K)", f"{res['proba']:.3f}")
                st.progress(min(max(res["proba"], 0.0), 1.0))
                st.caption(f"Served by run {res['run_id']}. This prediction was logged to Supabase.")
                if 0.4 < res["proba"] < 0.6:
                    st.warning("This probability is close to the 0.5 threshold — the model is "
                               "near-indifferent here, and a small change in inputs would flip it.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ===========================================================================
# 3. Score a CSV
# ===========================================================================
with tabs[2]:
    st.header("Score a CSV")
    st.caption("The file needs one column per feature. Optional `source_row_id` and `label` "
               "columns make the rows auditable; without them the predictions are logged "
               "but cannot contribute to error-rate fairness metrics.")
    try:
        schema = api_get("/schema")
        needed = [f["name"] for f in schema["numeric"]] + [f["name"] for f in schema["categorical"]]
        st.code(",".join(needed), language="text")
    except Exception:
        needed = []

    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None and st.button("Score file", type="primary"):
        try:
            res = api_post("/predict_batch", files={"file": (upload.name, upload.getvalue(), "text/csv")})
            out = pd.DataFrame(res["predictions"])
            st.success(f"Scored {res['n']} rows with run {res['run_id']}.")
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                out.to_csv(index=False).encode(),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

# ===========================================================================
# 4. Model Performance
# ===========================================================================
with tabs[3]:
    st.header("Model performance")
    try:
        runs = api_get("/runs")["runs"]
    except Exception as exc:
        runs = []
        st.error(f"Could not load runs: {exc}")

    if runs:
        st.subheader("Configuration comparison")
        comp = pd.DataFrame(runs)[[
            "config_name", "hidden_sizes", "activation", "dropout",
            "accuracy", "precision", "recall", "f1", "roc_auc", "best_epoch",
        ]]
        st.dataframe(comp, use_container_width=True)
        st.caption("Every row is a training run persisted to Supabase by api/train.py.")

    names = sorted({r["config_name"] for r in runs if r.get("config_name")}) or ["baseline"]
    chosen = st.selectbox("Inspect a configuration", names)

    try:
        perf = api_get("/performance", name=chosen)
    except Exception as exc:
        perf = None
        st.error(f"Could not load performance data: {exc}")

    if perf:
        m = perf["metrics"]
        c = st.columns(5)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Precision", f"{m['precision']:.3f}")
        c[2].metric("Recall", f"{m['recall']:.3f}")
        c[3].metric("F1", f"{m['f1']:.3f}")
        c[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")

        hist = pd.DataFrame(perf["history"]).set_index("epoch")
        left, right = st.columns(2)
        with left:
            st.subheader("Loss")
            st.line_chart(hist[["train_loss", "val_loss"]])
        with right:
            st.subheader("Accuracy")
            st.line_chart(hist[["train_acc", "val_acc"]])

        st.subheader("Confusion matrix (held-out test split)")
        cm = m["confusion"]
        st.table(pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actual ≤50K", "Actual >50K"],
            columns=["Predicted ≤50K", "Predicted >50K"],
        ))

        st.subheader("Calibration")
        cal = pd.DataFrame(perf["calibration"])
        if not cal.empty:
            plot = cal[["mean_predicted", "observed_rate"]].copy()
            plot["perfect_calibration"] = plot["mean_predicted"]
            st.line_chart(plot.set_index("mean_predicted"))
            st.caption("A perfectly calibrated model follows the diagonal: among rows scored "
                       "0.7, about 70% should truly earn >$50K. Gaps mean the probability "
                       "is not trustworthy as a probability, even when the label is right.")

# ===========================================================================
# 5. Bias Audit
# ===========================================================================
with tabs[4]:
    st.header("Bias audit")
    attribute = st.selectbox("Protected attribute", ["sex", "race"])

    try:
        audit = api_get("/audit", attribute=attribute)
        groups = pd.DataFrame(audit["groups"])
    except Exception as exc:
        groups = pd.DataFrame()
        st.error(f"Audit failed: {exc}")

    if groups.empty:
        st.info("No audited predictions yet. Predictions only count toward error rates when "
                "the true label is known — call POST /score_test_sample on the API to score "
                "held-out rows with ground truth.")
    else:
        st.subheader(f"Error rates by {attribute}")
        show = groups[["group_value", "n", "fpr", "fnr", "positive_rate", "accuracy"]].copy()
        show.columns = ["Group", "N", "False positive rate", "False negative rate",
                        "Predicted >50K rate", "Accuracy"]
        st.dataframe(show.style.format({
            "False positive rate": "{:.1%}", "False negative rate": "{:.1%}",
            "Predicted >50K rate": "{:.1%}", "Accuracy": "{:.1%}",
        }), use_container_width=True)

        st.bar_chart(groups.set_index("group_value")[["fpr", "fnr"]])
        st.caption("FPR: of people who truly earn ≤$50K, the share wrongly flagged as high "
                   "earners. FNR: of people who truly earn >$50K, the share the model missed. "
                   "Computed in Postgres by audit_by_attribute(), joining predictions to "
                   "ground truth.")

        if len(groups) > 1 and groups["fnr"].notna().all():
            gap = groups["fnr"].max() - groups["fnr"].min()
            worst = groups.loc[groups["fnr"].idxmax(), "group_value"]
            st.warning(
                f"False-negative rates differ by {gap:.1%} across groups, with '{worst}' "
                f"worst affected. Equal overall accuracy does not imply equal treatment: a "
                f"group the model rarely predicts positive for can score high accuracy while "
                f"having most of its true high earners missed."
            )

        small = groups[groups["n"] < 100]
        if not small.empty:
            st.info(f"Groups with fewer than 100 audited rows "
                    f"({', '.join(small['group_value'].astype(str))}) have noisy rates — "
                    f"score a larger sample before drawing conclusions about them.")

    st.divider()
    st.subheader("Prediction log (read directly from Supabase with the anon key)")
    try:
        rows = (sb().table("predictions")
                .select("id,label,proba,true_label,created_at")
                .order("id", desc=True).limit(200).execute())
        log = pd.DataFrame(rows.data)
        if log.empty:
            st.caption("No predictions logged yet.")
        else:
            st.dataframe(log, use_container_width=True, height=260)
            st.caption(f"{len(log)} most recent predictions. No API call — this tab queries "
                       "Supabase directly with the read-only anon key.")
    except Exception as exc:
        st.error(f"Supabase read failed: {exc}")

# ===========================================================================
# 6. Model Card
# ===========================================================================
with tabs[5]:
    st.header("Model Card")
    try:
        st.markdown(open("MODEL_CARD.md", encoding="utf-8").read())
    except FileNotFoundError:
        st.error("MODEL_CARD.md not found in the repository root.")