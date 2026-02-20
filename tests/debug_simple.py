"""
Simpler diagnostic: just show survival and default prob arrays.
No par_spread computation.
"""

import numpy as np
from data.orchestrator import initialize
from pricing.cds_pricing import _premium_payment_times, _get_forward_hazard_rates
from cva.cva_calculator import compute_survival_and_default

# Initialize
data = initialize()
forward_hazard_rates = _get_forward_hazard_rates(data)

print("Hazard rate buckets:", list(forward_hazard_rates.keys()))
print("Hazard rate values:", {k: f"{v:.6f}" for k, v in forward_hazard_rates.items()})
print()

# Check payment schedules
maturities = (1.0, 3.0, 5.0)

print("=" * 80)
print("PAYMENT SCHEDULE AND DEFAULT PROBABILITY")
print("=" * 80)

for T in maturities:
    pay_times = _premium_payment_times(T, pay_freq=4)
    print(f"\nMaturity T = {T}Y:")
    print(f"  Number of payment times: {len(pay_times)}")
    print(f"  Payment times shape: {pay_times.shape}")
    
    # Compute survival and default
    survival, dPD, lambdas = compute_survival_and_default(pay_times, forward_hazard_rates)
    
    print(f"  Survival array shape: {survival.shape}")
    print(f"  dPD array shape: {dPD.shape}")
    print(f"  Lambdas array shape: {lambdas.shape}")
    
    print(f"\n  Survival values:")
    print(f"    S[0] = {survival[0]:.10f} (should be ~1.0)")
    print(f"    S[-1] = {survival[-1]:.10f} (at maturity)")
    print(f"\n  Default prob values:")
    print(f"    dPD[0] = {dPD[0]:.10f} (should be 0)")
    print(f"    dPD[1] = {dPD[1]:.10f} (first increment)")
    print(f"    dPD[2] = {dPD[2]:.10f}")
    print(f"    dPD[-1] = {dPD[-1]:.10f} (last increment)")
    
    print(f"\n  Cumulative default probability = sum(dPD) = {np.sum(dPD):.10f}")
    print(f"                                           = 1 - S[-1] = {1 - survival[-1]:.10f}")
    
    # Show growth of cumulative default prob
    cumulative_dpd = np.cumsum(dPD)
    print(f"\n  Cumulative DP (last 5 points):")
    for i in range(max(0, len(cumulative_dpd)-5), len(cumulative_dpd)):
        print(f"    Index {i}: {cumulative_dpd[i]:.10f}")

print("\n" + "=" * 80)
print("COMPARISON: Are cumulative defaults different?")
print("=" * 80)

totals = {}
for T in maturities:
    pay_times = _premium_payment_times(T, pay_freq=4)
    survival, dPD, _ = compute_survival_and_default(pay_times, forward_hazard_rates)
    totals[T] = np.sum(dPD)

print(f"Sum(dPD) for 1Y: {totals[1.0]:.10e}")
print(f"Sum(dPD) for 3Y: {totals[3.0]:.10e}")
print(f"Sum(dPD) for 5Y: {totals[5.0]:.10e}")

if np.allclose(totals[1.0], totals[3.0]) and np.allclose(totals[3.0], totals[5.0]):
    print("\n⚠️ ERROR: All three are IDENTICAL! This is definitely wrong.")
else:
    print("\n✓ OK: They differ as expected (3Y > 1Y, 5Y > 3Y).")
