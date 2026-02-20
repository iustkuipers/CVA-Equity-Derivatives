"""
pricing/cds_pricing.py

Q5(b): CDS SENSITIVITY COMPUTATION (Market Instrument Valuation)

This module prices CDS contracts and computes ΔV under +10bps hazard bumps.
CRITICAL: This is NOT exposure-weighted CVA (see cva/hazard_sensitivity.py for Q5(a)).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS COMPUTES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ΔV_bucket,T_CDS = V(λ + 10bps, s_base) - V(λ, s_base)
    
    where:
      V = PV_protection(λ, spread) - PV_premium(λ, spread)
      s_base = par spread computed from BASE curve (fixed on bumps)

Critical Subtlety (DO NOT MISS):
  ✓ Compute par spread s_base under BASE hazard curve
  ✓ Fix s_base as contractual rate (this is the CDS contract)
  ✓ Bump hazard by +10bps
  ✓ Reprice with SAME s_base (not new par spread)
  ✓ Measure PV change = ΔV
  
  This is exactly how real CDS MTM works (mark-to-market a fixed-spread contract).

Expected Behavior:
  ✓ Base PV ≈ 0 (by definition of par spread)
  ✓ Bump in 0–1 affects all maturities (1Y, 3Y, 5Y)
  ✓ Bump in 3–5 affects only 5Y (depends on forward curve structure)
  ✓ All ΔV signs should be consistent (typically positive: hazard ↑ → buyer gains)

Key Properties:
  ✓ Market instrument (tradable CDS, not portfolio-specific)
  ✓ Pure credit instrument pricing
  ✓ Independent of portfolio exposure
  ✓ Shows how hedge instrument reacts to credit curve shifts

Mechanical Flow:
  1. Compute par spreads at BASE curve for 1Y, 3Y, 5Y
  2. For each hazard bucket:
     - Deep-copy data
     - Bump hazard by +10bps
     - For each maturity:
       * Price CDS with same par spread (s_base)
       * Store ΔV = PV_bumped - PV_base
  3. Return: deltas[bucket][maturity] = ΔV

Input Data Structure:
  data = data.orchestrator.initialize()
  
Required Keys:
  - forward_hazard_rates: dict[bucket_key] → float
  - lgd: float (loss-given-default)
  - risk_free_rate: float

CDS Pricing Conventions:
  - Protection buyer PV returned (long protection)
  - Quarterly premium payments by default (pay_freq=4)
  - Accrued premium approximated (half-period on default)
  - Time-dependent hazard rates from your curve function

Financial Interpretation:
  This measures: ∂V_CDS(bucket)/∂λ_bucket × 10bps
  
  It is the credit delta of a tradable CDS instrument.
  Used to hedge the portfolio's CVA exposure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT CONFUSE WITH Q5(a): CVA SENSITIVITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CVA sensitivities (cva/hazard_sensitivity.py) weight exposure by EPE.
CDS sensitivities price a STANDALONE tradable instrument.
Fundamentally different mechanics and use cases.

This module is INSTRUMENT VALUATION.
Hazard sensitivity module is EXPOSURE WEIGHTED.

Author intent: deterministic, transparent, no Monte Carlo.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from cva.cva_calculator import compute_survival_and_default


# ---------------------------------------------------------------------
# Helpers to access parameters from your 'data' dict (same philosophy as hazard_sensitivity.py)
# ---------------------------------------------------------------------

def _get_forward_hazard_rates(data: dict) -> Dict[str, float]:
    if "forward_hazard_rates" in data and isinstance(data["forward_hazard_rates"], dict):
        return data["forward_hazard_rates"]
    credit = data.get("credit", None)
    if isinstance(credit, dict) and "forward_hazard_rates" in credit:
        return credit["forward_hazard_rates"]
    raise KeyError("Missing forward hazard rates in data.")


def _set_forward_hazard_rates(data: dict, new_rates: Dict[str, float]) -> None:
    if "forward_hazard_rates" in data and isinstance(data["forward_hazard_rates"], dict):
        data["forward_hazard_rates"] = new_rates
        return
    credit = data.get("credit", None)
    if isinstance(credit, dict) and "forward_hazard_rates" in credit:
        data["credit"]["forward_hazard_rates"] = new_rates
        return
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
    raise KeyError("Missing LGD in data.")


def _get_risk_free_rate(data: dict) -> float:
    for k in ["risk_free_rate", "RISK_FREE_RATE", "r"]:
        if k in data:
            return float(data[k])
    rates = data.get("rates", None)
    if isinstance(rates, dict):
        for k in ["risk_free_rate", "RISK_FREE_RATE", "r"]:
            if k in rates:
                return float(rates[k])
    raise KeyError("Missing risk-free rate in data.")


def bump_forward_hazard(data: dict, bucket_key: str, bump: float = 0.0010) -> dict:
    """Deep-copy and bump one hazard bucket."""
    bumped = copy.deepcopy(data)
    rates = _get_forward_hazard_rates(bumped)
    if bucket_key not in rates:
        raise KeyError(f"Bucket '{bucket_key}' not found. Available: {list(rates.keys())}")
    new_rates = dict(rates)
    new_rates[bucket_key] = float(new_rates[bucket_key]) + float(bump)
    _set_forward_hazard_rates(bumped, new_rates)
    return bumped


# ---------------------------------------------------------------------
# CDS schedule and pricing core
# ---------------------------------------------------------------------

def _premium_payment_times(maturity: float, pay_freq: int = 4) -> np.ndarray:
    """
    Premium payment times: t_i = i/pay_freq for i=1..maturity*pay_freq
    """
    n = int(round(maturity * pay_freq))
    if n <= 0:
        raise ValueError("Maturity must be positive.")
    return np.array([(i / pay_freq) for i in range(1, n + 1)], dtype=float)


def _discount_factors(r: float, times: np.ndarray) -> np.ndarray:
    return np.exp(-r * times)


def price_cds(
    data: dict,
    maturity: float,
    spread: float,
    notional: float = 1.0,
    pay_freq: int = 4,
    include_accrual: bool = True,
    protection_buyer: bool = True
) -> float:
    """
    Prices a CDS contract using discrete premium payments and default leg consistent
    with the hazard curve used in compute_survival_and_default().

    Parameters
    ----------
    data : dict
        Model parameters dict (from initialize()).
    maturity : float
        CDS maturity in years (e.g. 1, 3, 5).
    spread : float
        Contractual premium rate (e.g. 0.01 = 100 bps).
    notional : float
        CDS notional.
    pay_freq : int
        Payments per year (4 = quarterly).
    include_accrual : bool
        If True, include an approximation for accrued premium on default:
        half-period accrual * spread * default probability in each interval.
    protection_buyer : bool
        If True, returns PV to protection buyer (long protection).
        If False, returns PV to protection seller (short protection).

    Returns
    -------
    pv : float
    """

    lgd = _get_lgd(data)
    r = _get_risk_free_rate(data)

    pay_times = _premium_payment_times(maturity, pay_freq=pay_freq)
    dt = 1.0 / pay_freq

    # Survival at payment times (and default increments aligned to the same grid)
    # We rely on your engine for survival/default with time-dependent hazards.
    # Note: compute_survival_and_default expects times to include t=0, so prepend it
    forward_hazard_rates = _get_forward_hazard_rates(data)
    times_with_zero = np.concatenate(([0.0], pay_times))
    survival, default_probs, _ = compute_survival_and_default(times_with_zero, forward_hazard_rates)
    # Now survival[i] corresponds to S at times_with_zero[i], and dPD[i] is default in interval i
    # We want survival and dPD aligned to pay_times; skip the t=0 state
    survival = survival[1:]  # S(t_1), S(t_2), ..., S(t_n)
    default_probs = default_probs[1:]  # ΔPD over (0,t_1], (t_1,t_2], ..., (t_{n-1},t_n]

    # Discount factors at payment times
    df = _discount_factors(r, pay_times)

    # NOW: survival and default_probs are aligned with pay_times (t=0 state removed)
    # survival[i] = S(pay_times[i])
    # default_probs[i] = default prob in interval ending at pay_times[i]
    
    # --- Premium leg (paid while alive) ---
    # PV_prem ≈ spread * sum_i DF(t_i) * dt * S(t_i)
    pv_prem = spread * np.sum(df * dt * survival)

    # --- Protection leg ---
    # PV_prot ≈ LGD * sum_i DF(t_i) * ΔPD_i
    pv_prot = lgd * np.sum(df * default_probs)

    # --- Accrued premium on default (approx) ---
    if include_accrual:
        # Typical approximation: expected accrual ~ 0.5*dt premium over each interval
        pv_accr = spread * np.sum(df * (0.5 * dt) * default_probs)
    else:
        pv_accr = 0.0

    # Buyer pays premium (incl accrual), receives protection
    pv_buyer = notional * (pv_prot - (pv_prem + pv_accr))

    if protection_buyer:
        return float(pv_buyer)
    return float(-pv_buyer)


def par_spread(
    data: dict,
    maturity: float,
    pay_freq: int = 4,
    include_accrual: bool = True
) -> float:
    """
    Compute the par spread such that PV (buyer) = 0 under the given curve.

    With accrual approximation:
      s * A = LGD * B
      where A = sum DF * (dt*S + 0.5*dt*ΔPD)  (if include_accrual)
            B = sum DF * ΔPD
    """
    lgd = _get_lgd(data)
    r = _get_risk_free_rate(data)

    pay_times = _premium_payment_times(maturity, pay_freq=pay_freq)
    dt = 1.0 / pay_freq

    forward_hazard_rates = _get_forward_hazard_rates(data)
    # Prepend t=0 to get full time grid (function expects times with t=0 included)
    times_with_zero = np.concatenate(([0.0], pay_times))
    survival, default_probs, _ = compute_survival_and_default(times_with_zero, forward_hazard_rates)
    # Skip t=0 state to align with pay_times
    survival = survival[1:]
    default_probs = default_probs[1:]
    df = _discount_factors(r, pay_times)

    B = np.sum(df * default_probs)

    if include_accrual:
        A = np.sum(df * (dt * survival + 0.5 * dt * default_probs))
    else:
        A = np.sum(df * dt * survival)

    if A <= 0:
        raise ValueError("Premium annuity is non-positive; check inputs/grid.")
    return float(lgd * B / A)


# ---------------------------------------------------------------------
# Q5(b): CDS sensitivities (bump-and-reprice)
# ---------------------------------------------------------------------

def compute_cds_sensitivities(
    data: dict,
    bump: float = 0.0010,
    maturities: Tuple[float, ...] = (1.0, 3.0, 5.0),
    pay_freq: int = 4,
    include_accrual: bool = True,
    protection_buyer: bool = True
) -> Tuple[Dict[float, float], Dict[str, Dict[float, float]]]:
    """
    Computes ΔMTM for CDS maturities under +bump to each hazard bucket.

    Returns
    -------
    base_spreads : dict[maturity] -> par spread (contractual spread)
    deltas : dict[bucket_key] -> dict[maturity] -> (PV_bumped - PV_base)

    Notes
    -----
    - Contractual spreads are fixed at BASE par spreads.
    - PV_base should be ~0 (numerical tolerance).
    """
    # contractual spreads under base curve
    base_spreads: Dict[float, float] = {}
    base_pv: Dict[float, float] = {}

    for T in maturities:
        s = par_spread(data, T, pay_freq=pay_freq, include_accrual=include_accrual)
        base_spreads[T] = s
        base_pv[T] = price_cds(
            data, T, spread=s,
            notional=1.0,
            pay_freq=pay_freq,
            include_accrual=include_accrual,
            protection_buyer=protection_buyer
        )

    buckets = list(_get_forward_hazard_rates(data).keys())
    deltas: Dict[str, Dict[float, float]] = {b: {} for b in buckets}

    for b in buckets:
        bumped_data = bump_forward_hazard(data, b, bump=bump)

        for T in maturities:
            pv_bumped = price_cds(
                bumped_data, T, spread=base_spreads[T],
                notional=1.0,
                pay_freq=pay_freq,
                include_accrual=include_accrual,
                protection_buyer=protection_buyer
            )
            deltas[b][T] = float(pv_bumped - base_pv[T])

    return base_spreads, deltas