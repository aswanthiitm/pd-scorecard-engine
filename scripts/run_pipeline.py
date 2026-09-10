#!/usr/bin/env python
"""End-to-end scorecard build: clean, bin, fit, validate, set a cut-off, price capital.

    python scripts/run_pipeline.py

Writes figures to ``reports/figures/``, the scorecard points table to
``reports/scorecard.csv``, and every headline number to ``reports/results.json``.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdse.binning import BinningSet  # noqa: E402
from pdse.capital import portfolio_capital  # noqa: E402
from pdse.config import BASE_ODDS, BASE_SCORE, EAD, FIGURES, LGD, PDO, REPORTS, TARGET  # noqa: E402
from pdse.data import load, split  # noqa: E402
from pdse.plotting import ACCENT, ACCENT2, ACCENT3, ACCENT4, MUTED, annotate, use_style  # noqa: E402
from pdse.policy import (  # noqa: E402
    account_economics, constrained_cutoff, cutoff_curve, optimal_cutoff, swap_set,
)
from pdse.scorecard import fit_challenger, fit_scorecard  # noqa: E402
from pdse.validation import (  # noqa: E402
    calibration_table, decile_table, ks_statistic, psi, psi_verdict, summarise,
)

use_style()
R: dict = {"generated_on": str(date.today())}


def log(msg: str) -> None:
    print(f"[pdse] {msg}", flush=True)


# ------------------------------------------------------------------- 1. data
log("loading and cleaning")
frame, audit = load()
train, test = split(frame)
features_all = [c for c in frame.columns if c != TARGET]
audit.to_csv(REPORTS / "data_quality_audit.csv", index=False)
R["data"] = {
    "rows": int(len(frame)), "train": int(len(train)), "test": int(len(test)),
    "bad_rate": round(float(frame[TARGET].mean()), 5),
    "audit": json.loads(audit.to_json(orient="records")),
}

# ---------------------------------------------------------------- 2. binning
log("binning and information value")
binning = BinningSet().fit(train, TARGET, features_all)
iv_summary = binning.iv_summary()
iv_summary.to_csv(REPORTS / "information_value.csv", index=False)
selected = iv_summary[iv_summary["iv"] >= 0.02]["feature"].tolist()
R["binning"] = {
    "iv": json.loads(iv_summary.round(4).to_json(orient="records")),
    "selected_by_iv": selected,
    "dropped_low_iv": [f for f in features_all if f not in selected],
}

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
ax = axes[0]
top = iv_summary.head(9).iloc[::-1]
ax.barh(top["feature"], top["iv"], color=[ACCENT if v >= 0.02 else MUTED for v in top["iv"]],
        height=0.6)
ax.axvline(0.02, color=ACCENT2, ls="--", lw=1)
ax.set_xlabel("information value"); ax.set_title("What actually predicts default")
ax.grid(axis="y", visible=False)
annotate(ax, "dashed line is the 0.02 cut below which a\ncharacteristic is conventionally dropped",
         loc="lower right")

ax = axes[1]
util = binning.binnings["revolving_utilisation"].table
ax.bar(range(len(util)), util["event_rate"] * 100, color=ACCENT, width=0.55)
ax.set_xticks(range(len(util)))
ax.set_xticklabels([r.replace("-inf", "0").replace("inf", "max") for r in util["range"]],
                   rotation=20, ha="right", fontsize=7.5)
ax_w = ax.twinx(); ax_w.grid(False)
ax_w.plot(range(len(util)), util["woe"], color=ACCENT2, marker="o", ms=4)
ax_w.set_ylabel("weight of evidence", color=ACCENT2)
ax.set_ylabel("default rate (%)")
ax.set_title("Revolving utilisation — bins are monotone by construction")
annotate(ax, "a scorecard bin that is not monotone in risk\ncannot be explained to a customer")
fig.tight_layout(); fig.savefig(FIGURES / "01_information_value.png"); plt.close(fig)

# -------------------------------------------------------------- 3. scorecard
log("fitting the scorecard")
card = fit_scorecard(train, TARGET, selected, binning)
wrong_sign = [f for f, b in zip(selected, card.model.coef_[0]) if b > 0]
if wrong_sign:
    log(f"dropping wrong-signed characteristics: {wrong_sign}")
    before = roc_auc_score(test[TARGET], card.probability(test))
    selected = [f for f in selected if f not in wrong_sign]
    card = fit_scorecard(train, TARGET, selected, binning)
    after = roc_auc_score(test[TARGET], card.probability(test))
else:
    before = after = roc_auc_score(test[TARGET], card.probability(test))

card.points.to_csv(REPORTS / "scorecard.csv", index=False)
R["scorecard"] = {
    "features": selected,
    "coefficients": {f: round(float(b), 4) for f, b in zip(selected, card.model.coef_[0])},
    "wrong_signed_dropped": wrong_sign,
    "auc_with_wrong_signed": round(float(before), 4),
    "auc_after_dropping": round(float(after), 4),
    "scaling": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
    "points_range": [float(card.points["points"].min()), float(card.points["points"].max())],
}

prob_train, prob_test = card.probability(train), card.probability(test)
score_train, score_test = card.score(train), card.score(test)
R["performance"] = {
    "train": {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in summarise(train[TARGET], prob_train, score_train).items()},
    "test": {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in summarise(test[TARGET], prob_test, score_test).items()},
}

log("fitting the gradient-boosting challenger")
challenger = fit_challenger(train, TARGET, features_all)
prob_challenger = challenger.predict_proba(test[features_all].to_numpy(float))[:, 1]
R["challenger"] = {
    "auc": round(float(roc_auc_score(test[TARGET], prob_challenger)), 4),
    "gini": round(float(2 * roc_auc_score(test[TARGET], prob_challenger) - 1), 4),
    "ks": round(float(ks_statistic(test[TARGET], prob_challenger)), 4),
    "auc_premium_over_scorecard": round(
        float(roc_auc_score(test[TARGET], prob_challenger) - roc_auc_score(test[TARGET], prob_test)), 4
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
ax = axes[0]
for label, probs, colour in [("scorecard", prob_test, ACCENT),
                             ("GBM challenger", prob_challenger, ACCENT2)]:
    fpr, tpr, _ = roc_curve(test[TARGET], probs)
    ax.plot(fpr, tpr, color=colour, label=f"{label} — AUC {roc_auc_score(test[TARGET], probs):.4f}")
ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Out-of-sample ROC")
ax.legend(loc="lower right")
annotate(ax, f"the challenger wins {R['challenger']['auc_premium_over_scorecard']*100:.2f} AUC points —\n"
             "that gap is the price of an explainable model")

ax = axes[1]
fpr, tpr, thresholds = roc_curve(test[TARGET], prob_test)
ax.plot(np.linspace(0, 1, len(tpr)), tpr, color=ACCENT2, label="cumulative bads")
ax.plot(np.linspace(0, 1, len(fpr)), fpr, color=ACCENT3, label="cumulative goods")
k = int(np.argmax(tpr - fpr))
ax.vlines(k / len(tpr), fpr[k], tpr[k], color=ACCENT, lw=2)
ax.set_xlabel("population ordered by score, riskiest first"); ax.set_ylabel("cumulative share")
ax.set_title(f"KS = {ks_statistic(test[TARGET], prob_test):.3f}")
ax.legend(loc="lower right")
annotate(ax, "KS is the widest separation between the two\ncurves — above 0.40 is a strong retail scorecard")
fig.tight_layout(); fig.savefig(FIGURES / "02_discrimination.png"); plt.close(fig)

# ------------------------------------------------------- 4. rank order + fit
deciles = decile_table(test[TARGET], score_test)
deciles.to_csv(REPORTS / "decile_table.csv", index=False)
calib = calibration_table(test[TARGET], prob_test)
psi_value, psi_detail = psi(score_train, score_test)
low_income = test[test["monthly_income"].fillna(0) < test["monthly_income"].median()]
psi_stress, _ = psi(score_train, card.score(low_income))
R["stability"] = {
    "psi_train_vs_test": round(psi_value, 4), "verdict": psi_verdict(psi_value),
    "psi_vs_below_median_income_segment": round(psi_stress, 4),
    "verdict_stress": psi_verdict(psi_stress),
    "max_calibration_gap_bp": round(float(calib["gap_bp"].abs().max()), 1),
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
ax = axes[0]
ax.bar(deciles["decile"], deciles["bad_rate"] * 100, color=ACCENT, width=0.6)
ax.set_xticks(deciles["decile"])
ax.set_xlabel("score decile (1 = riskiest)"); ax.set_ylabel("observed default rate (%)")
ax.set_title("Rank ordering holds in every decile")
ax_c = ax.twinx(); ax_c.grid(False)
ax_c.plot(deciles["decile"], deciles["cum_bads_pct"] * 100, color=ACCENT2, marker="o", ms=4)
ax_c.set_ylabel("cumulative bads captured (%)", color=ACCENT2)
annotate(ax, f"the riskiest 30% of accounts contain "
             f"{deciles.iloc[2]['cum_bads_pct']*100:.0f}%\nof all defaults; top-decile lift "
             f"{deciles.iloc[0]['lift']:.1f}x", loc="upper right")

ax = axes[1]
ax.plot([0, calib["observed"].max() * 100], [0, calib["observed"].max() * 100],
        color=MUTED, ls=":", lw=1)
ax.plot(calib["predicted"] * 100, calib["observed"] * 100, color=ACCENT, marker="o", ms=5)
ax.set_xlabel("predicted default rate (%)"); ax.set_ylabel("observed default rate (%)")
ax.set_title("Calibration — predicted against actual")
annotate(ax, f"worst bucket gap {R['stability']['max_calibration_gap_bp']:.0f} bp.\n"
             f"Hosmer-Lemeshow still rejects (p<0.01) because at\n"
             f"{len(test):,} accounts it detects gaps this small.")
fig.tight_layout(); fig.savefig(FIGURES / "03_rank_order_calibration.png"); plt.close(fig)

# ------------------------------------------------------------ 5. cut-off
log("running the cut-off economics")
curve = cutoff_curve(score_test, prob_test, test[TARGET])
curve.to_csv(REPORTS / "cutoff_curve.csv", index=False)
best_profit = optimal_cutoff(curve)
best_constrained = constrained_cutoff(curve, max_bad_rate=0.03)
base_all = portfolio_capital(prob_test)
R["policy"] = {
    "profit_maximising": json.loads(best_profit.round(4).to_json()),
    "risk_appetite_3pct_bad_rate": json.loads(best_constrained.round(4).to_json()),
    "approve_everyone": {
        "approval_rate": 1.0,
        "observed_bad_rate": round(float(test[TARGET].mean()), 4),
        "expected_loss_cr": round(float(account_economics(prob_test)["expected_loss"].sum()) / 1e7, 3),
        "revenue_cr": round(float(account_economics(prob_test)["revenue"].sum()) / 1e7, 3),
        "profit_cr": round(float(account_economics(prob_test)["profit"].sum()) / 1e7, 3),
        "rwa_cr": round(base_all["rwa_cr"], 2),
        "capital_cr": round(base_all["regulatory_capital_cr"], 2),
    },
    "assumptions": {"ead_inr": EAD, "lgd": LGD, "test_book_cr": round(len(test) * EAD / 1e7, 1)},
}

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
ax = axes[0]
ax.plot(curve["approval_rate"] * 100, curve["profit_cr"], color=ACCENT, label="profit")
ax.plot(curve["approval_rate"] * 100, curve["revenue_cr"], color=ACCENT3, ls="--", label="revenue")
ax.plot(curve["approval_rate"] * 100, curve["expected_loss_cr"], color=ACCENT2, ls="--",
        label="expected credit loss")
ax.axvline(best_profit["approval_rate"] * 100, color=MUTED, ls=":", lw=1.2)
ax.scatter([best_profit["approval_rate"] * 100], [best_profit["profit_cr"]], color=ACCENT,
           s=45, zorder=5)
ax.set_xlabel("approval rate (%)"); ax.set_ylabel("₹ crore over a 2-year book")
ax.set_title("Where the money is: revenue, loss and profit by cut-off")
ax.legend(loc="upper left")
annotate(ax, f"profit peaks at {best_profit['approval_rate']*100:.0f}% approval\n"
             f"(score {best_profit['cutoff']:.0f}, bad rate "
             f"{best_profit['observed_bad_rate']*100:.1f}%)", loc="lower right")

ax = axes[1]
ax.plot(curve["approval_rate"] * 100, curve["roc_pct"], color=ACCENT4, label="return on capital")
ax.set_xlabel("approval rate (%)"); ax.set_ylabel("annualised return on regulatory capital (%)")
ax.set_title("Capital tells a different story from profit")
ax_r = ax.twinx(); ax_r.grid(False)
ax_r.plot(curve["approval_rate"] * 100, curve["rwa_cr"], color=MUTED, ls="--")
ax_r.set_ylabel("risk-weighted assets (₹ cr)", color=MUTED)
ax.axvline(best_constrained["approval_rate"] * 100, color=ACCENT2, ls=":", lw=1.2)
annotate(ax, f"the profit-maximising cut-off is not the capital-efficient one:\n"
             f"{best_profit['approval_rate']*100:.0f}% approval returns "
             f"{best_profit['roc_pct']:.0f}% on capital, "
             f"{best_constrained['approval_rate']*100:.0f}% returns "
             f"{best_constrained['roc_pct']:.0f}%", loc="lower left")
fig.tight_layout(); fig.savefig(FIGURES / "04_cutoff_economics.png"); plt.close(fig)

# -------------------------------------------------------------- 6. swap set
log("swap-set analysis against the challenger")
swaps = swap_set(score_test, -prob_challenger, test[TARGET],
                 approval_rate=float(best_constrained["approval_rate"]))
swaps.to_csv(REPORTS / "swap_set.csv", index=False)
swap_in = swaps.loc[swaps["segment"].str.startswith("swap-in")].iloc[0]
swap_out = swaps.loc[swaps["segment"].str.startswith("swap-out")].iloc[0]
loss_saved = (swap_out["bad_rate"] - swap_in["bad_rate"]) * swap_in["accounts"] * EAD * LGD
R["swap_set"] = {
    "at_approval_rate": round(float(best_constrained["approval_rate"]), 4),
    "table": json.loads(swaps.round(4).to_json(orient="records")),
    "credit_loss_saved_cr": round(float(loss_saved) / 1e7, 2),
    "note": "value of the challenger's extra ranking power, at an identical approval rate",
}

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
ax = axes[0]
ax.bar(range(len(swaps)), swaps["bad_rate"] * 100, color=[ACCENT, ACCENT3, ACCENT2, MUTED],
       width=0.55)
ax.set_xticks(range(len(swaps)))
ax.set_xticklabels(["both\napprove", "swap-in\n(GBM only)", "swap-out\n(card only)",
                    "both\ndecline"], fontsize=8)
ax.set_ylabel("default rate (%)")
ax.set_title(f"Swap set at an identical {best_constrained['approval_rate']*100:.0f}% approval rate")
ax.grid(axis="x", visible=False)
annotate(ax, f"the challenger swaps in accounts defaulting at "
             f"{swap_in['bad_rate']*100:.1f}%\nand swaps out accounts defaulting at "
             f"{swap_out['bad_rate']*100:.1f}% —\n₹{loss_saved/1e7:.2f} cr of credit loss on this test book")

ax = axes[1]
top_features = card.points["feature"].unique()[:4]
offset = 0
labels, values, colours = [], [], []
palette = [ACCENT, ACCENT2, ACCENT3, ACCENT4]
for i, feature in enumerate(top_features):
    rows = card.points[card.points["feature"] == feature]
    for _, row in rows.iterrows():
        labels.append(f"{feature[:14]} {row['bin'][:16]}")
        values.append(row["points"])
        colours.append(palette[i % 4])
ax.barh(range(len(values)), values, color=colours, height=0.65)
ax.set_yticks(range(len(values))); ax.set_yticklabels(labels, fontsize=6.6)
ax.invert_yaxis()
ax.set_xlabel("points"); ax.set_title("The scorecard a credit committee signs")
ax.grid(axis="y", visible=False)
fig.text(0.53, -0.02, "A customer's score is the plain sum of one number per characteristic — "
                      "which is what makes a decline explainable.", fontsize=8.2, color=MUTED)
fig.tight_layout(); fig.savefig(FIGURES / "05_swapset_scorecard.png"); plt.close(fig)

(REPORTS / "results.json").write_text(json.dumps(R, indent=2, default=str))
log(f"wrote {REPORTS/'results.json'}, scorecard.csv and 5 figures")
print(json.dumps({k: R[k] for k in ("performance", "challenger", "stability", "swap_set")},
                 indent=2, default=str)[:2500])
