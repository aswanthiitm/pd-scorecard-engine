"""Interactive cut-off explorer: move the approval threshold, watch the portfolio change.

    streamlit run app.py

The scoring model is the same one `scripts/run_pipeline.py` builds. This app exists to make
the *decision* explorable — drag the cut-off and see approval rate, bad rate, expected loss,
profit and capital move together, because they never move in the same direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pdse.binning import BinningSet  # noqa: E402
from pdse.config import EAD, LGD, RAW, TARGET  # noqa: E402
from pdse.data import ARFF, SOURCE, clean, read_arff, split  # noqa: E402
from pdse.policy import constrained_cutoff, cutoff_curve, optimal_cutoff  # noqa: E402
from pdse.scorecard import fit_scorecard  # noqa: E402
from pdse.validation import decile_table, summarise  # noqa: E402

st.set_page_config(page_title="Customer Risk Scoring & Approval Strategy", layout="wide")


@st.cache_data(show_spinner="Fetching the portfolio (first run only)…")
def _ensure_data() -> Path:
    """The 7 MB source file is not in the repo — fetch it once into the app's storage."""
    if ARFF.exists():
        return ARFF
    RAW.mkdir(parents=True, exist_ok=True)
    response = requests.get(SOURCE, timeout=180)
    response.raise_for_status()
    ARFF.write_bytes(response.content)
    return ARFF


@st.cache_data(show_spinner="Cleaning data and fitting the scoring model…")
def _build():
    frame, audit = clean(read_arff(_ensure_data()))
    train, test = split(frame)
    features_all = [c for c in frame.columns if c != TARGET]
    binning = BinningSet().fit(train, TARGET, features_all)
    iv = binning.iv_summary()
    selected = iv[iv["iv"] >= 0.02]["feature"].tolist()
    card = fit_scorecard(train, TARGET, selected, binning)
    wrong_sign = [f for f, b in zip(selected, card.model.coef_[0]) if b > 0]
    if wrong_sign:  # a characteristic fighting the others fails review — drop it
        selected = [f for f in selected if f not in wrong_sign]
        card = fit_scorecard(train, TARGET, selected, binning)
    prob = card.probability(test)
    score = card.score(test)
    curve = cutoff_curve(score, prob, test[TARGET], grid=80)
    return {
        "audit": audit, "iv": iv, "selected": selected, "wrong_sign": wrong_sign,
        "test": test, "prob": prob, "score": score, "curve": curve,
        "points": card.points, "metrics": summarise(test[TARGET], prob, score),
        "deciles": decile_table(test[TARGET], score),
    }


built = _build()
curve, score, prob, test = built["curve"], built["score"], built["prob"], built["test"]

st.title("Customer Risk Scoring & Approval Strategy")
st.caption(
    f"{len(test):,} held-out customers from a 150,000-record retail portfolio. "
    "The model ranks risk; the cut-off turns that ranking into a lending policy. "
    "Everything below is recomputed from the data at load time."
)

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Policy levers")
approval = st.sidebar.slider("Approval rate (%)", 10, 99, 85, 1) / 100
st.sidebar.caption("Approve the best-scoring share of applicants.")
appetite = st.sidebar.slider("Risk appetite — maximum bad rate (%)", 1.0, 7.0, 3.0, 0.1) / 100
st.sidebar.divider()
st.sidebar.caption(
    f"Portfolio assumptions: ₹{EAD:,.0f} exposure per account, {LGD:.0%} loss given default, "
    "over a two-year outcome window. Edit them in `src/pdse/config.py`."
)

row = curve.iloc[(curve["approval_rate"] - approval).abs().argmin()]
best_profit = optimal_cutoff(curve)
try:
    best_appetite = constrained_cutoff(curve, max_bad_rate=appetite)
except ValueError:
    best_appetite = None

# ---------------------------------------------------------------- headline
c1, c2, c3, c4 = st.columns(4)
c1.metric("Approved", f"{row['approval_rate']:.0%}", f"{int(row['approved_accounts']):,} accounts")
c2.metric("Bad rate", f"{row['observed_bad_rate']:.2%}",
          f"{row['observed_bad_rate'] - test[TARGET].mean():+.2%} vs approve-all",
          delta_color="inverse")
c3.metric("Expected loss", f"₹{row['expected_loss_cr']:.1f} cr",
          f"RWA ₹{row['rwa_cr']:.0f} cr", delta_color="off")
c4.metric("Profit, 2 years", f"₹{row['profit_cr']:.1f} cr",
          f"{row['profit_cr'] - best_profit['profit_cr']:+.1f} cr vs best")

st.divider()

tab1, tab2, tab3 = st.tabs(["The cut-off decision", "Does the model rank?", "The scorecard"])

with tab1:
    left, right = st.columns([3, 2])
    with left:
        chart = curve.set_index(curve["approval_rate"] * 100)[
            ["profit_cr", "revenue_cr", "expected_loss_cr"]
        ].rename(columns={"profit_cr": "Profit", "revenue_cr": "Revenue",
                          "expected_loss_cr": "Expected credit loss"})
        st.line_chart(chart, height=320, y_label="₹ crore", x_label="approval rate (%)")
        st.caption(
            "Revenue rises with every account approved. Losses rise faster at the bottom of "
            "the ranking, so profit turns over before volume does."
        )
    with right:
        st.markdown("**Where the optima sit**")
        rows = [{
            "policy": "Profit-maximising",
            "approval": f"{best_profit['approval_rate']:.0%}",
            "bad rate": f"{best_profit['observed_bad_rate']:.2%}",
            "profit (₹ cr)": round(best_profit["profit_cr"], 1),
            "return on capital": f"{best_profit['roc_pct']:.0f}%",
        }]
        if best_appetite is not None:
            rows.append({
                "policy": f"Best within ≤{appetite:.1%} bad rate",
                "approval": f"{best_appetite['approval_rate']:.0%}",
                "bad rate": f"{best_appetite['observed_bad_rate']:.2%}",
                "profit (₹ cr)": round(best_appetite["profit_cr"], 1),
                "return on capital": f"{best_appetite['roc_pct']:.0f}%",
            })
        rows.append({
            "policy": "Your setting",
            "approval": f"{row['approval_rate']:.0%}",
            "bad rate": f"{row['observed_bad_rate']:.2%}",
            "profit (₹ cr)": round(row["profit_cr"], 1),
            "return on capital": f"{row['roc_pct']:.0f}%",
        })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info(
            "The two optima are different points. Maximising profit approves more than the "
            "capital-efficient policy does, because the marginal account at the cut-off "
            "consumes far more capital than the portfolio average."
        )

with tab2:
    m = built["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("AUC", f"{m['auc']:.4f}")
    b.metric("Gini", f"{m['gini']:.3f}")
    c.metric("KS", f"{m['ks']:.3f}")
    d.metric("Rank-ordering breaks", int(m["rank_ordering_breaks"]))
    deciles = built["deciles"]
    st.bar_chart(deciles.set_index("decile")["bad_rate"] * 100, height=280,
                 y_label="default rate (%)", x_label="score decile (1 = riskiest)")
    st.caption(
        f"The riskiest 30% of customers hold {deciles.iloc[2]['cum_bads_pct']:.0%} of all "
        f"defaults; the top decile carries {deciles.iloc[0]['lift']:.1f}x the portfolio average. "
        "Zero breaks means risk falls at every step up the score — without that, no cut-off is safe."
    )
    st.dataframe(
        deciles.assign(bad_rate=lambda d: (d["bad_rate"] * 100).round(2),
                       lift=lambda d: d["lift"].round(2),
                       cum_bads_pct=lambda d: (d["cum_bads_pct"] * 100).round(1))
        [["decile", "n", "bads", "bad_rate", "lift", "cum_bads_pct"]],
        hide_index=True, width="stretch",
    )

with tab3:
    st.markdown(
        "Each characteristic contributes points; a customer's score is their sum. That "
        "additivity is what lets a branch explain a decline and an auditor re-derive a score by hand."
    )
    st.dataframe(built["points"], hide_index=True, width="stretch", height=380)
    if built["wrong_sign"]:
        st.warning(
            f"Dropped during fitting: **{', '.join(built['wrong_sign'])}** — it came out with "
            "the wrong sign, meaning it pushed scores the opposite way to the other "
            "characteristics. That is a decline reason pointing backwards, and it fails review."
        )
    with st.expander("Data quality repairs made before any modelling"):
        st.dataframe(built["audit"], hide_index=True, width="stretch")
