"""Retail credit PD scorecard and lending-policy engine.

modules
-------
data        : loading, the data-quality repairs the raw file needs, and splitting
binning     : monotonic weight-of-evidence binning and information value
scorecard   : logistic regression turned into an auditable points table
validation  : discrimination, calibration, rank-ordering and population stability
policy      : cut-off economics — approval rate against expected loss and profit
capital     : Basel IRB retail risk-weighted assets and regulatory capital
"""

__version__ = "1.0.0"
