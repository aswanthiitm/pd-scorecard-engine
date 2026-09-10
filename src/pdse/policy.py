"""The lending policy engine: turning a score into a cut-off.

A scorecard does not make the bank any money. A **cut-off** does, and choosing one is an
economics problem, not a statistics problem:

* approve too much and expected credit loss eats the margin,
* approve too little and the fixed cost base is spread over too few accounts, and every
  declined good customer is revenue handed to a competitor.

For every candidate cut-off this module computes approval rate, the bad rate of what gets
approved, expected credit loss, interest income net of funding and servicing cost, profit,
and Basel IRB capital — then reports the profit-maximising cut-off and the return on the
capital it consumes. Two policies are also compared head to head through a **swap-set**
analysis, which is the only view that shows *who* changes hands rather than how the
averages move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .capital import portfolio_capital
from .config import (
    ANNUAL_YIELD, BEHAVIOURAL_LIFE, CAPITAL_RATIO, EAD, FUNDING_COST, LGD, OPEX_RATE,
)


def account_economics(pd_values: np.ndarray, ead: float = EAD, lgd: float = LGD) -> pd.DataFrame:
    """Per-account revenue, loss and profit over the behavioural life.

    Revenue is the net interest margin plus fee income less servicing cost, accrued over
    ``BEHAVIOURAL_LIFE`` years. Expected credit loss is ``PD x LGD x EAD``, taken over the
    same two-year window the target is defined on — matching the loss horizon to the
    outcome window is the part most often got wrong.
    """
    pd_values = np.asarray(pd_values, float)
    margin = (ANNUAL_YIELD - FUNDING_COST - OPEX_RATE) * ead * BEHAVIOURAL_LIFE
    expected_loss = pd_values * lgd * ead
    # A defaulting account stops paying part-way through, so it earns roughly half the
    # margin before it charges off. Ignoring this flatters every aggressive cut-off.
    revenue = margin * (1 - 0.5 * pd_values)
    return pd.DataFrame({"pd": pd_values, "revenue": revenue,
                         "expected_loss": expected_loss, "profit": revenue - expected_loss})


def cutoff_curve(score: pd.Series, pd_values: np.ndarray, y_true: pd.Series,
                 grid: int = 60) -> pd.DataFrame:
    """Scan every candidate cut-off and report the portfolio it would produce."""
    score = pd.Series(np.asarray(score, float))
    y_true = pd.Series(np.asarray(y_true, int))
    pd_values = np.asarray(pd_values, float)
    economics = account_economics(pd_values)

    thresholds = np.quantile(score, np.linspace(0.01, 0.95, grid))
    rows = []
    for threshold in thresholds:
        approved = score >= threshold
        n = int(approved.sum())
        if n < 200:
            continue
        capital = portfolio_capital(pd_values[approved.to_numpy()])
        profit_cr = float(economics.loc[approved.to_numpy(), "profit"].sum()) / 1e7
        rows.append({
            "cutoff": float(threshold),
            "approval_rate": n / len(score),
            "approved_accounts": n,
            "observed_bad_rate": float(y_true[approved.to_numpy()].mean()),
            "predicted_pd": float(pd_values[approved.to_numpy()].mean()),
            "expected_loss_cr": float(economics.loc[approved.to_numpy(), "expected_loss"].sum()) / 1e7,
            "revenue_cr": float(economics.loc[approved.to_numpy(), "revenue"].sum()) / 1e7,
            "profit_cr": profit_cr,
            "rwa_cr": capital["rwa_cr"],
            "capital_cr": capital["regulatory_capital_cr"],
            "rorwa_pct": profit_cr / max(capital["rwa_cr"], 1e-9) * 100 / BEHAVIOURAL_LIFE,
            "roc_pct": profit_cr / max(capital["regulatory_capital_cr"], 1e-9) * 100 / BEHAVIOURAL_LIFE,
        })
    return pd.DataFrame(rows)


def optimal_cutoff(curve: pd.DataFrame, objective: str = "profit_cr") -> pd.Series:
    """The row of the cut-off curve that maximises `objective`."""
    return curve.loc[curve[objective].idxmax()]


def constrained_cutoff(curve: pd.DataFrame, max_bad_rate: float) -> pd.Series:
    """Most profitable cut-off that still respects a bad-rate ceiling.

    Real credit policy is written this way — a risk-appetite statement caps the portfolio
    bad rate, and the business optimises underneath it.
    """
    feasible = curve[curve["observed_bad_rate"] <= max_bad_rate]
    if feasible.empty:
        raise ValueError(f"no cut-off achieves a bad rate at or below {max_bad_rate:.2%}")
    return feasible.loc[feasible["profit_cr"].idxmax()]


def swap_set(score_a: pd.Series, score_b: pd.Series, y_true: pd.Series,
             approval_rate: float) -> pd.DataFrame:
    """Compare two models at the *same* approval rate — who actually changes hands.

    Both scores are cut at their own quantile so the two policies approve identical
    volumes, which is the only fair comparison. The interesting rows are the swap sets:
    the accounts one model takes and the other declines. A better model is one whose
    "swap-in" population defaults less than its "swap-out" population — and that
    difference, not the AUC gap, is the money.
    """
    a = pd.Series(np.asarray(score_a, float))
    b = pd.Series(np.asarray(score_b, float))
    y = pd.Series(np.asarray(y_true, int))
    approve_a = a >= a.quantile(1 - approval_rate)
    approve_b = b >= b.quantile(1 - approval_rate)

    groups = {
        "approved by both": approve_a & approve_b,
        "swap-in (B approves, A declines)": (~approve_a) & approve_b,
        "swap-out (A approves, B declines)": approve_a & (~approve_b),
        "declined by both": (~approve_a) & (~approve_b),
    }
    rows = [{"segment": name, "accounts": int(mask.sum()),
             "share": float(mask.mean()),
             "bad_rate": float(y[mask].mean()) if mask.any() else np.nan}
            for name, mask in groups.items()]
    return pd.DataFrame(rows)
