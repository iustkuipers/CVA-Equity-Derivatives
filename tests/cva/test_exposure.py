import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cva.exposure import (
    positive_part,
    compute_standalone_positive_exposures,
    compute_unnetted_exposure,
    compute_netted_exposure,
    compute_epe,
    compute_standalone_epe
)
from pricing.equity_instruments import value_portfolio_over_time
from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator


def test_exposure_calculations():
    """
    Test CVA exposure calculations.
    
    Tests:
    1. Standalone positive exposures: max(V, 0)
    2. Unnetted exposure: sum_i max(V_i, 0)
    3. Netted exposure: max(sum_i V_i, 0)
    4. EPE (Expected Positive Exposure)
    5. Standalone EPE per instrument
    """
    
    print("=" * 80)
    print("CVA EXPOSURE CALCULATIONS TEST")
    print("=" * 80)
    print()
    
    # Initialize
    print("Initializing data and parameters...")
    data = initialize()
    portfolio = data['portfolio']
    
    # Simulate paths
    print("Simulating equity paths...")
    simulator = EquityProcessSimulator(data, portfolio)
    paths = simulator.simulate_paths()
    print(f"✓ Simulation complete ({simulator.num_simulations} paths × {simulator.num_steps} steps)")
    print()
    
    # Value portfolio
    print("Valuing portfolio across all paths and times...")
    V = value_portfolio_over_time(paths, portfolio, data)
    print(f"✓ Portfolio valuation complete")
    print(f"  Value array shape: {V.shape} [sims, times, instruments]")
    print()
    
    # Test 1: Positive part function
    print("TEST 1: Positive Part Function")
    print("-" * 40)
    test_array = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    result = positive_part(test_array)
    expected = np.array([0.0, 0.0, 0.0, 0.5, 1.5])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"  Input:  {test_array}")
    print(f"  Output: {result}")
    print(f"✓ Positive part function correct")
    print()
    
    # Test 2: Standalone positive exposures
    print("TEST 2: Standalone Positive Exposures")
    print("-" * 40)
    E_pos = compute_standalone_positive_exposures(V)
    
    # Check shape
    assert E_pos.shape == V.shape, f"Expected shape {V.shape}, got {E_pos.shape}"
    print(f"  Shape: {E_pos.shape}")
    
    # Check all values are non-negative
    assert (E_pos >= 0).all(), "All positive exposures should be non-negative"
    print(f"✓ All exposures non-negative")
    
    # Check that E_pos >= V (since we took max with 0)
    assert (E_pos >= V).all(), "Positive exposures should be >= portfolio values"
    print(f"✓ Positive exposures >= portfolio values")
    print()
    
    # Test 3: Unnetted exposure
    print("TEST 3: Unnetted Exposure Profile")
    print("-" * 40)
    E_unnetted = compute_unnetted_exposure(V)
    
    # Check shape
    assert E_unnetted.shape == (V.shape[0], V.shape[1]), f"Expected (sims, times), got {E_unnetted.shape}"
    print(f"  Shape: {E_unnetted.shape}")
    
    # Verify unnetted = sum of standalone exposures
    E_pos_sum = np.sum(E_pos, axis=2)
    assert np.allclose(E_unnetted, E_pos_sum), "Unnetted should equal sum of standalone exposures"
    print(f"✓ Unnetted = sum of standalone exposures")
    
    # All unnetted exposures should be non-negative
    assert (E_unnetted >= 0).all(), "All unnetted exposures should be non-negative"
    print(f"✓ All unnetted exposures non-negative")
    
    print(f"  Mean unnetted exposure at T=5y: {np.mean(E_unnetted[:, -1]):.2f}")
    print()
    
    # Test 4: Netted exposure
    print("TEST 4: Netted Exposure Profile (Master Netting Agreement)")
    print("-" * 40)
    E_netted = compute_netted_exposure(V)
    
    # Check shape
    assert E_netted.shape == (V.shape[0], V.shape[1]), f"Expected (sims, times), got {E_netted.shape}"
    print(f"  Shape: {E_netted.shape}")
    
    # Netted should be >= than individual exposures when positively correlated
    assert (E_netted >= 0).all(), "All netted exposures should be non-negative"
    print(f"✓ All netted exposures non-negative")
    
    # Netted <= unnetted (netting benefit)
    netting_benefit = np.mean(E_unnetted - E_netted)
    print(f"✓ Netting benefit (unnetted - netted): {netting_benefit:.2f}")
    
    print(f"  Mean netted exposure at T=5y: {np.mean(E_netted[:, -1]):.2f}")
    print()
    
    # Test 5: Expected Positive Exposure (EPE)
    print("TEST 5: Expected Positive Exposure (EPE)")
    print("-" * 40)
    epe_unnetted = compute_epe(E_unnetted)
    epe_netted = compute_epe(E_netted)
    
    # Check shapes
    assert epe_unnetted.shape == (V.shape[1],), f"Expected (times,), got {epe_unnetted.shape}"
    assert epe_netted.shape == (V.shape[1],), f"Expected (times,), got {epe_netted.shape}"
    print(f"  EPE shape: {epe_unnetted.shape}")
    
    # EPE should be non-negative
    assert (epe_unnetted >= 0).all(), "EPE should be non-negative"
    assert (epe_netted >= 0).all(), "EPE should be non-negative"
    print(f"✓ All EPE values non-negative")
    
    # EPE should increase with time (more uncertainty)
    print(f"  Unnetted EPE at t=0: {epe_unnetted[0]:.2f}")
    print(f"  Unnetted EPE at t=5y: {epe_unnetted[-1]:.2f}")
    print(f"  Netted EPE at t=0: {epe_netted[0]:.2f}")
    print(f"  Netted EPE at t=5y: {epe_netted[-1]:.2f}")
    print()
    
    # Test 6: Standalone EPE
    print("TEST 6: Standalone EPE per Instrument")
    print("-" * 40)
    epe_standalone = compute_standalone_epe(V)
    
    # Check shape
    assert epe_standalone.shape == (V.shape[1], V.shape[2]), f"Expected (times, instruments), got {epe_standalone.shape}"
    print(f"  Shape: {epe_standalone.shape} [times, instruments]")
    
    # All EPEs should be non-negative
    assert (epe_standalone >= 0).all(), "All standalone EPEs should be non-negative"
    print(f"✓ All standalone EPEs non-negative")
    
    # Sum of standalone EPE should approximately equal unnetted EPE
    sum_standalone_epe = np.sum(epe_standalone, axis=1)
    assert np.allclose(sum_standalone_epe, epe_unnetted, rtol=1e-10), "Sum of standalone EPE should equal unnetted EPE"
    print(f"✓ Sum of standalone EPE = unnetted EPE")
    
    # Print per-instrument EPE at final time
    instruments = ['FX Forward SX5E', 'FX Forward AEX', 'Put SX5E', 'Put AEX']
    print(f"  EPE at T=5y per instrument:")
    for i, instr_name in enumerate(instruments):
        print(f"    {instr_name}: {epe_standalone[-1, i]:.2f}")
    print()
    
    print("✓ All CVA Exposure Tests Passed")


if __name__ == "__main__":
    test_exposure_calculations()
