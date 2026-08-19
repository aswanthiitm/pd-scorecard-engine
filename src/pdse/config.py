"""Paths, modelling constants and the portfolio economics used by the policy engine."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
for _p in (RAW, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

TARGET = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.30

# --- scorecard scaling -------------------------------------------------------
# The industry convention: a score of BASE_SCORE means BASE_ODDS good-to-bad, and
# every PDO points of score doubles the odds. Nothing about these three numbers
# changes the model's ranking - they only fix the units the credit committee reads.
BASE_SCORE = 600.0
BASE_ODDS = 50.0
PDO = 20.0

# --- portfolio economics (unsecured retail, indicative) ----------------------
EAD = 200_000.0          # rupee exposure at default per account
LGD = 0.65               # loss given default, unsecured retail
ANNUAL_YIELD = 0.16      # portfolio APR
FUNDING_COST = 0.070     # cost of funds
OPEX_RATE = 0.025        # servicing cost as a share of exposure
BEHAVIOURAL_LIFE = 2.0   # years, matching the target's two-year outcome window

# --- Basel IRB (other retail exposures) --------------------------------------
IRB_CORRELATION_MIN = 0.03
IRB_CORRELATION_MAX = 0.16
IRB_K_FACTOR = 12.5      # RWA = K * 12.5 * EAD
CAPITAL_RATIO = 0.115    # RBI minimum CRAR including CCB
