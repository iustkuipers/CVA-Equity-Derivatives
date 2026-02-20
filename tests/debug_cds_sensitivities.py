"""
Diagnostic script to identify the CDS sensitivity bug.
Prints key quantities to verify correct implementation.
"""

import numpy as np
from data.orchestrator import initialize
from pricing.cds_pricing import (
    _premium_payment_times,
    _get_forward_hazard_rates,
    _get_lgd,
    _get_risk_free_rate,
    _discount_factors,
    par_spread,
    price_cds
)
from cva.cva_calculator import compute_survival_and_default

# Initialize
data = initialize()
lgd = _get_lgd(data)
r = _get_risk_free_rate(data)
print(f"LGD: {lgd:.4f}")
print(f"Risk-free rate: {r:.4f}\n")

# Test for each maturity
maturities = (1.0, 3.0, 5.0)

print("=" * 80)
print("PAYMENT SCHEDULE AND DEFAULT PROBABILITY ANALYSIS")
print("=" * 80)

for T in maturities:
    pay_times = _premium_payment_times(T, pay_freq=4)
    print(f"\nMaturity T = {T}Y:")
    print(f"  Payment times length: {len(pay_times)}")
    print(f"  First 3 times: {pay_times[:3]}")
    print(f"  Last 3 times: {pay_times[-3:]}")

    # Compute survival and default on BASE curve
    forward_hazard_rates = _get_forward_hazard_rates(data)
    survival, dPD, lambdas = compute_survival_and_default(pay_times, forward_hazard_rates)

    print(f"  Survival[0] (should be 1.0): {survival[0]:.6f}")
    print(f"  Survival[-1] (at maturity): {survival[-1]:.6f}")
    print(f"  Sum of dPD (cumulative default prob): {np.sum(dPD):.6f}")
    print(f"  dPD[1] (first interval default): {dPD[1]:.8f}")
    print(f"  dPD[-1] (last interval default): {dPD[-1]:.8f}")

    # Compute protection and premium legs
    df = _discount_factors(r, pay_times)
    pv_prot = lgd * np.sum(df * dPD)
    pv_prem_base = par_spread(data, T, pay_freq=4, include_accrual=True)
    dt = 1.0 / 4.0
    pv_prem = pv_prem_base * np.sum(df * dt * survival)

    print(f"  Protection leg annuity (LGD * sum(df*dPD)): {pv_prot:.6f}")
    print(f"  Premium annuity factor (sum(df*dt*S)): {np.sum(df * dt * survival):.6f}")
    print(f"  Par spread: {pv_prem_base:.6f}")

print("\n" + "=" * 80)
print("CDS SENSITIVITY TO [0,1] BUCKET BUMP")
print("=" * 80)

# Now compute sensitivities to [0,1] bump
from pricing.cds_pricing import bump_forward_hazard

print(f"\nBumping [0,1] by +10 bps...")

for T in maturities:
    pay_times = _premium_payment_times(T, pay_freq=4)
    
    # Base
    forward_hazard_rates = _get_forward_hazard_rates(data)
    s_base = par_spread(data, T, pay_freq=4, include_accrual=True)
    pv_base = price_cds(data, T, spread=s_base, notional=1.0, pay_freq=4, include_accrual=True)
    
    # Bumped
    bumped_data = bump_forward_hazard(data, "0_1", bump=0.001)
    pv_bumped = price_cds(bumped_data, T, spread=s_base, notional=1.0, pay_freq=4, include_accrual=True)
    
    delta = float(pv_bumped - pv_base)
    
    print(f"\n  T = {T}Y:")
    print(f"    Base PV: {pv_base:.8f}")
    print(f"    Bumped PV: {pv_bumped:.8f}")
    print(f"    Δ (bumped - base): {delta:.8f}")
    
print("\n" + "=" * 80)
print("DETAILED SURVIVAL/DEFAULT COMPARISON")
print("=" * 80)

# Show survival at key times for all three maturities
key_times = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
forward_hazard_rates = _get_forward_hazard_rates(data)

print("\nBase curve survival at key times:")
print("Time | 1Y contract | 3Y contract | 5Y contract")
print("-" * 60)

for t in key_times:
    survivals = {}
    for T in maturities:
        pay_times = _premium_payment_times(T, pay_freq=4)
        if t <= T:
            # Find or interpolate
            survival, _, _ = compute_survival_and_default(pay_times, forward_hazard_rates)
            # Find closest time
            idx = np.argmin(np.abs(pay_times - t))
            survivals[T] = survival[idx]
        else:
            survivals[T] = np.nan
    
    s1 = survivals.get(1.0, np.nan)
    s3 = survivals.get(3.0, np.nan)
    s5 = survivals.get(5.0, np.nan)
    print(f"{t:4.2f} | {s1:11.6f} | {s3:11.6f} | {s5:11.6f}")

print("\n" + "=" * 80)
print("HYPOTHESIS CHECK")
print("=" * 80)
print("\nIf sum(dPD) is identical across maturities, the bug is in")
print("how default_probs are computed or accumulated.")
print("\nIf survival differs correctly but sensitivities are identical,")
print("the bug is in how the legs are being priced (e.g., using wrong")
print("part of the array or reusing values).")
