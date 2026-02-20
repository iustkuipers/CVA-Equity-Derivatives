"""
cva/hazard_sensitivity.py

Q5(a): CVA SENSITIVITY COMPUTATION (Exposure-Weighted Credit Delta)

This module computes ΔCVA under +10bps bumps to forward hazard buckets.
CRITICAL: This is NOT CDS pricing (see pricing/cds_pricing.py for Q5(b)).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS COMPUTES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ΔCVA_bucket = CVA(λ + 10bps) - CVA(λ)
    
    where CVA = LGD · ∑_k DF(t_k) · EPE(t_k) · ΔPD_k

Key Properties:
  ✓ EPE stays FIXED (from Q2, no resimulation)
  ✓ Only default probabilities change
  ✓ Expected: all ΔCVA > 0 (more hazard → more CVA cost)
  ✓ Portfolio-specific (depends on your exposure profile)
  ✓ This is what a CVA desk actually hedges

Mechanical Flow:
  1. Deep-copy data
  2. Bump one forward hazard bucket by +10bps
  3. Recompute survival/default probabilities
  4. Recompute CVA using SAME EPE
  5. Store ΔCVA = CVA_bumped - CVA_base

Input Data Structure:
  data = data.orchestrator.initialize()
  
Required Keys:
  - forward_hazard_rates: dict[bucket_key] → float (e.g., {"0_1": 0.005, "1_3": 0.007})
  - lgd: float (loss-given-default)
  - risk_free_rate: float

Financial Interpretation:
  This measures: ∂CVA/∂λ_bucket × 10bps
  
  It is the first-order credit delta of the portfolio 
  with respect to credit curve shifts. Base for hedging.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT CONFUSE WITH Q5(b): CDS SENSITIVITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CDS sensitivities (pricing/cds_pricing.py) price a TRADABLE instrument.
Different formula, different mechanics, different interpretation.

This module is EXPOSURE WEIGHTED.
CDS module is INSTRUMENT VALUATION.

Author intent: deterministic, transparent, no Monte Carlo.
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np

from cva.cva_calculator import compute_survival_and_default


# ---------------------------------------------------------------------
# Helpers: locate keys in your 'data' dict safely
# ---------------------------------------------------------------------

def _get_forward_hazard_rates(data: dict) -> Dict[str, float]:
    """
    Returns the forward hazard rates dict from data, supporting both:
      data["forward_hazard_rates"]
      data["credit"]["forward_hazard_rates"]
    """
    if "forward_hazard_rates" in data and isinstance(data["forward_hazard_rates"], dict):
        return data["forward_hazard_rates"]

    credit = data.get("credit", None)
    if isinstance(credit, dict) and "forward_hazard_rates" in credit:
        return credit["forward_hazard_rates"]

    raise KeyError(
        "Could not find forward hazard rates in data. "
        "Expected data['forward_hazard_rates'] or data['credit']['forward_hazard_rates']."
    )


def _set_forward_hazard_rates(data: dict, new_rates: Dict[str, float]) -> None:
    """Writes hazard rates back into the same location they came from."""
    if "forward_hazard_rates" in data and isinstance(data["forward_hazard_rates"], dict):
        data["forward_hazard_rates"] = new_rates
        return

    credit = data.get("credit", None)
    if isinstance(credit, dict) and "forward_hazard_rates" in credit:
        data["credit"]["forward_hazard_rates"] = new_rates
        return

    # Shouldn't happen if _get_forward_hazard_rates worked.
    raise KeyError("Internal error: cannot set forward hazard rates.")


def _get_lgd(data: dict) -> float:
    for k in ["lgd", "LGD"]:
        if k in data:
            return float(data[k])
    credit = data.get("credit", None)
    if isinstance(credit, dict):
        for k in ["lgd", "LGD"]:
            if k in credit:
                return float(credit[k])
    raise KeyError("Could not find LGD in data. Expected data['lgd'] or data['credit']['lgd'].")


def _get_risk_free_rate(data: dict) -> float:
    for k in ["risk_free_rate", "RISK_FREE_RATE", "r"]:
        if k in data:
            return float(data[k])
    rates = data.get("rates", None)
    if isinstance(rates, dict):
        for k in ["risk_free_rate", "RISK_FREE_RATE", "r"]:
            if k in rates:
                return float(rates[k])
    raise KeyError(
        "Could not find risk-free rate in data. "
        "Expected data['risk_free_rate'] or data['rates']['risk_free_rate']."
    )


# ---------------------------------------------------------------------
# 1) Bump utility
# ---------------------------------------------------------------------

def bump_forward_hazard(data: dict, bucket_key: str, bump: float = 0.0010) -> dict:
    """
    Returns a deep-copied 'data' dict with one forward hazard bucket bumped.

    Parameters
    ----------
    data : dict
        Output of initialize()
    bucket_key : str
        e.g. "0_1", "1_3", "3_5"
    bump : float
        +10bps = 0.0010 by default

    Returns
    -------
    bumped_data : dict
    """
    bumped_data = copy.deepcopy(data)

    base_rates = _get_forward_hazard_rates(bumped_data)
    if bucket_key not in base_rates:
        raise KeyError(f"Bucket '{bucket_key}' not found in forward hazard rates: {list(base_rates.keys())}")

    new_rates = dict(base_rates)
    new_rates[bucket_key] = float(new_rates[bucket_key]) + float(bump)

    _set_forward_hazard_rates(bumped_data, new_rates)
    return bumped_data


# ---------------------------------------------------------------------
# 2) Q5(a): CVA sensitivities via bump-and-reprice
# ---------------------------------------------------------------------

def compute_cva_sensitivities(
    data: dict,
    epe: np.ndarray,
    times: np.ndarray,
    bump: float = 0.0010
) -> Dict[str, float]:
    """
    Computes ΔCVA for +bump applied to each forward hazard bucket.

    Returns dict mapping bucket_key -> (CVA_bumped - CVA_base)

    Notes
    -----
    - epe and times must align (same length).
    - No Monte Carlo is rerun; EPE is reused (standard independence assumption).
    """
    epe = np.asarray(epe, dtype=float)
    times = np.asarray(times, dtype=float)

    if epe.shape[0] != times.shape[0]:
        raise ValueError(f"epe and times length mismatch: {epe.shape[0]} vs {times.shape[0]}")

    lgd = _get_lgd(data)
    r = _get_risk_free_rate(data)
    df = np.exp(-r * times)

    # Base CVA
    base_rates = _get_forward_hazard_rates(data)
    _, default_probs_base, _ = compute_survival_and_default(times, base_rates)
    # CVA = LGD · ∑_k DF(t_k) · EPE(t_k) · ΔPD_k
    cva_base = lgd * np.sum(df * epe * default_probs_base)

    # Bumps
    buckets = list(base_rates.keys())

    out: Dict[str, float] = {}
    for b in buckets:
        bumped_data = bump_forward_hazard(data, b, bump=bump)
        bumped_rates = _get_forward_hazard_rates(bumped_data)
        _, default_probs_bumped, _ = compute_survival_and_default(times, bumped_rates)

        cva_bumped = lgd * np.sum(df * epe * default_probs_bumped)

        out[b] = float(cva_bumped - cva_base)

    return out