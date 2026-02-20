"""
Diagnostic: Decompose CDS delta into protection leg, premium leg, accrual
to check if cancellation explains the "flat [0,1] deltas".
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
    bump_forward_hazard,
)
from cva.cva_calculator import compute_survival_and_default


def decompose_cds_delta(data, maturity, spread, base=True):
    """
    Compute protection leg, premium leg, and accrual separately.
    Returns dict with absolute values and delta on bump.
    """
    lgd = _get_lgd(data)
    r = _get_risk_free_rate(data)
    pay_times = _premium_payment_times(maturity, pay_freq=4)
    dt = 1.0 / 4.0
    
    times_with_zero = np.concatenate(([0.0], pay_times))
    forward_hazard_rates = _get_forward_hazard_rates(data)
    survival, default_probs, _ = compute_survival_and_default(times_with_zero, forward_hazard_rates)
    survival = survival[1:]
    default_probs = default_probs[1:]
    df = _discount_factors(r, pay_times)
    
    pv_prem = spread * np.sum(df * dt * survival)
    pv_prot = lgd * np.sum(df * default_probs)
    pv_accr = spread * np.sum(df * (0.5 * dt) * default_probs)
    pv_net = pv_prot - (pv_prem + pv_accr)
    
    return {
        "pv_prot": pv_prot,
        "pv_prem": pv_prem,
        "pv_accr": pv_accr,
        "pv_net": pv_net,
    }


print("=" * 80)
print("LEG DECOMPOSITION: [0,1] BUCKET BUMP")
print("=" * 80)

data = initialize()
maturities = (1.0, 3.0, 5.0)

print("\n1. BASE CURVE (no bump):")
for T in maturities:
    s_par = par_spread(data, T, pay_freq=4, include_accrual=True)
    base_legs = decompose_cds_delta(data, T, s_par, base=True)
    print(f"\n  T = {T}Y (par spread = {s_par*10000:.2f} bps):")
    print(f"    Protection leg: {base_legs['pv_prot']:.8f}")
    print(f"    Premium leg:    {base_legs['pv_prem']:.8f}")
    print(f"    Accrual:        {base_legs['pv_accr']:.8f}")
    print(f"    Net (should ~0): {base_legs['pv_net']:.8f}")

print("\n" + "=" * 80)
print("2. BUMPED [0,1] (+10 bps):")

bumped_data = bump_forward_hazard(data, "0_1", bump=0.001)

print("\nDELTA DECOMPOSITION:")
print("Maturity | Δ Prot Leg | Δ Prem Leg | Δ Accrual | Δ Net PV")
print("-" * 70)

for T in maturities:
    # Base
    s_par = par_spread(data, T, pay_freq=4, include_accrual=True)
    base = decompose_cds_delta(data, T, s_par, base=True)
    
    # Bumped (same par spread)
    bumped = decompose_cds_delta(bumped_data, T, s_par, base=False)
    
    # Deltas
    d_prot = bumped["pv_prot"] - base["pv_prot"]
    d_prem = bumped["pv_prem"] - base["pv_prem"]
    d_accr = bumped["pv_accr"] - base["pv_accr"]
    d_net = bumped["pv_net"] - base["pv_net"]
    
    print(f"{T:8.1f}Y | {d_prot:10.8f} | {d_prem:10.8f} | {d_accr:9.8f} | {d_net:9.8f}")

print("\n" + "=" * 80)
print("3. FIXED SPREAD TEST (100 bps for all maturities):")
print("=" * 80)

spread_fixed = 0.01  # 100 bps
print("\nUsing fixed spread = 100 bps for 1Y, 3Y, 5Y:")
print("Maturity | Base PV | Bumped PV | Δ PV")
print("-" * 50)

for T in maturities:
    base = decompose_cds_delta(data, T, spread_fixed, base=True)
    bumped = decompose_cds_delta(bumped_data, T, spread_fixed, base=False)
    d_net = bumped["pv_net"] - base["pv_net"]
    
    print(f"{T:8.1f}Y | {base['pv_net']:7.8f} | {bumped['pv_net']:9.8f} | {d_net:8.8f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
If deltas INCREASE with maturity in both protection and premium legs,
but the NET deltas are nearly identical, then:
  → Protection leg effect is being offset by premium leg effect
  → This is a par-spread recalibration artifact
  → The "flat deltas" are defensible as a cancellation effect

If fixed-spread deltas are significantly different across maturities,
then:
  → Your model DOES have maturity dependence
  → Par spread recalibration is washing it out
  → Document this in your report
""")
