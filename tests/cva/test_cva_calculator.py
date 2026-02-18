import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cva.cva_calculator import (
    _build_piecewise_hazard_vector,
    compute_survival_and_default,
    compute_cva_from_epe
)
from cva.exposure import (
    compute_netted_exposure,
    compute_epe
)
from pricing.equity_instruments import value_portfolio_over_time
from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator


def test_cva_calculator():
    """
    Test CVA calculator components and full CVA computation.
    
    Tests:
    1. Hazard rate vector builder
    2. Survival and default probability calculations
    3. CVA calculation from EPE
    4. Mathematical properties validation
    """
    
    print("=" * 80)
    print("CVA CALCULATOR TEST")
    print("=" * 80)
    print()
    
    # Test 1: Hazard rate vector builder
    print("TEST 1: Piecewise Hazard Rate Builder")
    print("-" * 40)
    
    times = np.array([0.0, 1.0, 3.0, 5.0])
    
    # Test with dict input (as in config)
    forward_hazard_rates = {
        1.0: 0.0200,    # 0 ≤ t < 1: 2.00%
        3.0: 0.0215,    # 1 ≤ t < 3: 2.15%
        5.0: 0.0220,    # 3 ≤ t ≤ 5: 2.20%
    }
    
    lambdas_dict = _build_piecewise_hazard_vector(times, forward_hazard_rates)
    print(f"  Times: {times}")
    print(f"  Hazard rates (dict input):")
    for i, lam in enumerate(lambdas_dict):
        print(f"    Interval [{times[i]:.1f}, {times[i+1]:.1f}]: λ = {lam:.4f}")
    
    # Test with array input
    lambdas_array = np.array([0.0200, 0.0215, 0.0220])
    lambdas_arr = _build_piecewise_hazard_vector(times, lambdas_array)
    assert np.allclose(lambdas_dict, lambdas_arr), "Dict and array inputs should match"
    print(f"✓ Dict and array inputs produce same hazard vector")
    print()
    
    # Test 2: Survival and default probabilities
    print("TEST 2: Survival and Default Probabilities")
    print("-" * 40)
    
    survival, dPD, lambdas = compute_survival_and_default(times, forward_hazard_rates)
    
    # Check properties
    print(f"  Survival probabilities S(t):")
    for i, (t, s) in enumerate(zip(times, survival)):
        print(f"    t={t:.1f}y: S(t) = {s:.6f}")
    
    # Survival should be decreasing
    assert np.all(np.diff(survival) <= 0), "Survival should be non-increasing"
    print(f"✓ Survival is non-increasing")
    
    # Survival should be in [0, 1]
    assert np.all(survival >= 0) and np.all(survival <= 1), "Survival should be in [0, 1]"
    print(f"✓ Survival in [0, 1]")
    
    # Default probabilities should be non-negative and sum to 1
    assert np.all(dPD >= 0), "Default probabilities should be non-negative"
    print(f"✓ Default probabilities non-negative")
    
    print(f"  Incremental default probabilities ΔPD:")
    for i, (t, dpd) in enumerate(zip(times, dPD)):
        print(f"    Δt={t:.1f}y: ΔPD = {dpd:.6f}")
    
    # Sum of survival changes should equal 1 - final survival
    survival_drop = 1.0 - survival[-1]
    print(f"  Total default probability (1 - S(T)): {survival_drop:.6f}")
    print(f"  Sum of incremental PDs: {np.sum(dPD):.6f}")
    assert np.isclose(survival_drop, np.sum(dPD)), "Survival drop should equal sum of dPDs"
    print(f"✓ Probability conservation verified")
    print()
    
    # Test 3: CVA calculation from simple EPE
    print("TEST 3: CVA Calculation - Simple EPE Profile")
    print("-" * 40)
    
    # Create simple EPE: increases from 100 to 200
    simple_epe = np.linspace(100, 200, len(times))
    
    print(f"  Simple EPE profile: {simple_epe}")
    
    config = {
        'risk_free_rate': 0.03,
        'lgd': 0.40,
        'forward_hazard_rates': forward_hazard_rates
    }
    
    cva_simple, breakdown = compute_cva_from_epe(times, simple_epe, config, return_breakdown=True)
    
    print(f"  CVA Breakdown:")
    print(breakdown.to_string(index=False))
    print()
    print(f"  Total CVA: ${cva_simple:.2f}")
    
    # CVA should be positive (EPE is positive, hazard rates positive, LGD positive)
    assert cva_simple > 0, "CVA should be positive for positive EPE"
    print(f"✓ CVA is positive")
    print()
    
    # Test 4: Full pipeline - Simulation to CVA
    print("TEST 4: Full Pipeline - Simulation to CVA")
    print("-" * 40)
    
    print("  Initializing and simulating...")
    data = initialize()
    portfolio = data['portfolio']
    
    simulator = EquityProcessSimulator(data, portfolio)
    paths = simulator.simulate_paths()
    print(f"  ✓ Simulation complete")
    
    print("  Valuing portfolio...")
    V = value_portfolio_over_time(paths, portfolio, data)
    print(f"  ✓ Valuation complete")
    
    print("  Computing exposures...")
    E_netted = compute_netted_exposure(V)
    epe_netted = compute_epe(E_netted)
    print(f"  ✓ Exposures computed")
    
    times_sim = paths['times']
    print(f"  EPE profile shape: {epe_netted.shape}")
    print(f"  Mean EPE: {np.mean(epe_netted):.2f}")
    print(f"  Final EPE (T=5y): {epe_netted[-1]:.2f}")
    print()
    
    # Compute CVA with actual forward hazard rates from config
    forward_hz = data['forward_hazard_rates']
    config_full = {
        'risk_free_rate': data['risk_free_rate'],
        'lgd': data['lgd'],
        'forward_hazard_rates': forward_hz
    }
    
    print("  Computing CVA from simulated exposures...")
    cva_full, breakdown_full = compute_cva_from_epe(times_sim, epe_netted, config_full, return_breakdown=True)
    print(f"  ✓ CVA computed")
    print()
    
    print(f"  CVA Summary:")
    print(f"    LGD: {config_full['lgd']:.1%}")
    print(f"    Risk-Free Rate: {config_full['risk_free_rate']:.2%}")
    print(f"    Total CVA: ${cva_full:.2f}")
    print(f"    CVA as % of mean exposure: {100.0 * cva_full / np.mean(epe_netted):.1f}%")
    print()
    
    # Show major contributors
    contrib_sorted = breakdown_full.nlargest(5, 'cva_contrib')
    print(f"  Top 5 CVA Contributors:")
    print(contrib_sorted[['t', 'epe', 'dPD', 'cva_contrib']].to_string(index=False))
    print()
    
    # Verify mathematical properties
    assert cva_full >= 0, "CVA should be non-negative"
    print(f"✓ CVA non-negative")
    
    assert cva_full <= data['lgd'] * np.sum(epe_netted * np.diff(times_sim, prepend=0)), \
        "CVA should not exceed LGD × sum of exposures"
    print(f"✓ CVA upper bound satisfied")
    
    print()
    print("✓ All CVA Calculator Tests Passed")


if __name__ == "__main__":
    test_cva_calculator()
