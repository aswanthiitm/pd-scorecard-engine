"""Basel IRB capital for retail exposures.

A better model does not only reduce losses — it reduces the *capital* the bank has to
hold, because under the internal-ratings-based approach the risk weight is a function of
each account's own PD. Two portfolios with the same average PD but different dispersion
attract different capital, and that difference is invisible to any accuracy metric.

For other retail exposures the correlation is

.. math:: R = 0.03\\frac{1-e^{-35\\,PD}}{1-e^{-35}} + 0.16\\left(1 - \\frac{1-e^{-35\\,PD}}{1-e^{-35}}\\right)

and the capital requirement per unit of exposure is the difference between the loss at a
99.9% one-year systematic shock and the expected loss already provisioned for:

.. math:: K = LGD\\left[\\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) + \\sqrt{R}\\,\\Phi^{-1}(0.999)}{\\sqrt{1-R}}\\right) - PD\\right]

Note the shape: R *falls* as PD rises, so a high-PD account is treated as more
idiosyncratic and less correlated with the cycle. That is why concentrating risk in a few
clearly-bad accounts is cheaper in capital than spreading the same expected loss thinly
across the book — and why ranking, not just averaging, is what the capital rule rewards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    CAPITAL_RATIO, EAD, IRB_CORRELATION_MAX, IRB_CORRELATION_MIN, IRB_K_FACTOR, LGD,
)


def irb_correlation(pd_values: np.ndarray) -> np.ndarray:
    """Asset correlation R for other retail exposures."""
    pd_values = np.clip(np.asarray(pd_values, float), 1e-6, 0.9999)
    weight = (1 - np.exp(-35 * pd_values)) / (1 - np.exp(-35.0))
    return IRB_CORRELATION_MIN * weight + IRB_CORRELATION_MAX * (1 - weight)


def irb_capital_requirement(pd_values, lgd: float = LGD) -> np.ndarray:
    """K — capital required per unit of exposure, unexpected loss only."""
    pd_values = np.clip(np.asarray(pd_values, float), 1e-6, 0.9999)
    r = irb_correlation(pd_values)
    conditional = norm.cdf(
        (norm.ppf(pd_values) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)
    )
    return lgd * (conditional - pd_values)


def portfolio_capital(pd_values, ead: float = EAD, lgd: float = LGD) -> dict:
    """Expected loss, risk-weighted assets and regulatory capital for a book."""
    pd_values = np.asarray(pd_values, float)
    n = pd_values.size
    k = irb_capital_requirement(pd_values, lgd)
    rwa = k * IRB_K_FACTOR * ead
    expected_loss = pd_values * lgd * ead
    return {
        "accounts": int(n),
        "exposure_cr": n * ead / 1e7,
        "average_pd": float(pd_values.mean()),
        "expected_loss_cr": float(expected_loss.sum()) / 1e7,
        "rwa_cr": float(rwa.sum()) / 1e7,
        "average_risk_weight": float(rwa.sum() / (n * ead)),
        "regulatory_capital_cr": float(rwa.sum() * CAPITAL_RATIO) / 1e7,
    }
