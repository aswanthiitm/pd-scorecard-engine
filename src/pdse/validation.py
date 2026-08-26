"""Validation: discrimination, calibration, rank-ordering and stability.

Four different questions, and a model can pass any one of them while failing the others:

* **Discrimination** — can it tell a good from a bad at all? (AUC, Gini, KS)
* **Rank-ordering** — does risk fall monotonically as the score rises, in every decile?
  A model with a fine AUC that inverts in the middle deciles cannot carry a cut-off.
* **Calibration** — is a predicted 3% actually 3%? Discrimination alone is invariant to
  any monotone transform of the score, so a perfectly-ranking model can still be
  uniformly wrong about the level — which is what pricing and provisioning depend on.
* **Stability** — does the score distribution still look like development? (PSI)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y_true, y_score) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the widest gap between the good and bad cumulative curves.

    Still the number an Indian credit committee asks for first. Above 0.40 is a strong
    retail scorecard; below 0.30 rarely supports a cut-off worth having.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def decile_table(y_true: pd.Series, score: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Bad rate, lift and cumulative capture by score decile (decile 1 = riskiest)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(score)})
    frame["decile"] = pd.qcut(frame["score"].rank(method="first"), bins,
                              labels=range(1, bins + 1)).astype(int)
    grouped = frame.groupby("decile").agg(n=("y", "size"), bads=("y", "sum"),
                                          min_score=("score", "min"),
                                          max_score=("score", "max"))
    grouped["bad_rate"] = grouped["bads"] / grouped["n"]
    grouped["lift"] = grouped["bad_rate"] / frame["y"].mean()
    grouped = grouped.sort_index()
    grouped["cum_bads_pct"] = grouped["bads"].cumsum() / grouped["bads"].sum()
    grouped["cum_accounts_pct"] = grouped["n"].cumsum() / grouped["n"].sum()
    return grouped.reset_index()


def rank_ordering_breaks(deciles: pd.DataFrame) -> int:
    """Count deciles where the bad rate rises as the score rises — it never should."""
    rates = deciles.sort_values("decile")["bad_rate"].to_numpy()
    return int(np.sum(np.diff(rates) > 0))


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed default rate, by predicted-probability bucket."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["bucket"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=range(bins))
    out = frame.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    ).reset_index()
    out["gap_bp"] = (out["observed"] - out["predicted"]) * 10_000
    return out


def hosmer_lemeshow(y_true, y_prob, bins: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit. A *small* chi-square is the good outcome."""
    from scipy import stats

    table = calibration_table(y_true, y_prob, bins)
    expected = table["n"] * table["predicted"]
    observed = table["n"] * table["observed"]
    chi2 = float(np.sum((observed - expected) ** 2 / (expected * (1 - table["predicted"]))))
    return {"chi2": chi2, "dof": bins - 2,
            "p_value": float(1 - stats.chi2.cdf(chi2, bins - 2)),
            "well_calibrated_5pct": bool(1 - stats.chi2.cdf(chi2, bins - 2) > 0.05)}


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Population Stability Index between a development and a later population.

    Bucket the development scores into deciles, then ask what share of the new population
    lands in each. The convention: below 0.10 stable, 0.10-0.25 investigate, above 0.25
    the population has shifted enough that the model should be re-developed. PSI says
    nothing about whether the model is still *accurate* — only that the input has moved.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_share = np.histogram(expected, edges)[0] / len(expected)
    act_share = np.histogram(actual, edges)[0] / len(actual)
    exp_share = np.clip(exp_share, 1e-6, None)
    act_share = np.clip(act_share, 1e-6, None)
    contribution = (act_share - exp_share) * np.log(act_share / exp_share)
    detail = pd.DataFrame({
        "bucket": range(len(exp_share)),
        "lower": edges[:-1], "upper": edges[1:],
        "expected_share": exp_share, "actual_share": act_share,
        "psi_contribution": contribution,
    })
    return float(contribution.sum()), detail


def psi_verdict(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "shifted — investigate"
    return "materially shifted — redevelop"


def summarise(y_true, y_prob, score) -> dict:
    deciles = decile_table(y_true, score)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "gini": float(gini(y_true, y_prob)),
        "ks": float(ks_statistic(y_true, y_prob)),
        "bad_rate": float(np.mean(y_true)),
        "rank_ordering_breaks": rank_ordering_breaks(deciles),
        "top_decile_lift": float(deciles.iloc[0]["lift"]),
        "bads_caught_in_riskiest_30pct": float(deciles.iloc[2]["cum_bads_pct"]),
        **{f"hl_{k}": v for k, v in hosmer_lemeshow(y_true, y_prob).items()},
    }
