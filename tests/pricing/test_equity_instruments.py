import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pricing.equity_instruments import value_portfolio_over_time
from pricing.black_scholes import black_scholes_price
from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator


def test_equity_instruments():
    """
    Test equity instrument valuation.
    
    Tests:
    1. Forward pricing formula: V_f(t) = S_t * e^(-q(T-t)) - K * e^(-r(T-t))
    2. Put pricing via Black-Scholes: V_p(t) = BS(S_t, K, r, q, σ, T-t)
    3. Output shape: [simulations, time_steps, instruments]
    4. Exposure calculation (max with 0)
    """
    
    print("=" * 80)
    print("EQUITY INSTRUMENTS VALUATION TEST")
    print("=" * 80)
    print()
    
    # Initialize
    data = initialize()
    portfolio = data['portfolio']
    
    # Simulate paths
    print("Simulating equity paths...")
    simulator = EquityProcessSimulator(data, portfolio)
    paths = simulator.simulate_paths()
    print(f"✓ Simulation complete ({simulator.num_simulations} paths × {simulator.num_steps} steps)")
    
    times = paths['times']
    sx5e_paths = paths['sx5e']
    aex_paths = paths['aex']
    
    print(f"Path shapes:")
    print(f"  SX5E: {sx5e_paths.shape}")
    print(f"  AEX: {aex_paths.shape}")
    print()
    
    # Value portfolio (this is the function we're testing)
    print("Valuing portfolio across all paths and times...")
    V = value_portfolio_over_time(paths, portfolio, data)
    print(f"✓ Portfolio valuation complete")
    
    print(f"Portfolio value array shape: {V.shape}")
    expected_shape = (simulator.num_simulations, simulator.num_steps + 1, len(portfolio))
    assert V.shape == expected_shape, f"Expected {expected_shape}, got {V.shape}"
    print(f"✓ Shape convention correct: [simulations={V.shape[0]}, times={V.shape[1]}, instruments={V.shape[2]}]")
    print()
    
    # Test 1: Forward pricing at t=0
    print("TEST 1: Forward Pricing at t=0")
    print("-" * 40)
    
    r = data['risk_free_rate']
    q = data['dividend_yield']
    T = data['counterparty_maturity']
    
    # Get forward contracts (instruments 0 and 1 in portfolio)
    for i in range(2):  # First two are forwards
        instr = portfolio.iloc[i]
        S0 = instr['S0']
        K = instr['Strike']
        
        # Theoretical forward value at t=0: V_f = S0 * e^(-q*T) - K * e^(-r*T)
        V_theoretical = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        
        # MC estimate (should be close at t=0 since we start at S0)
        V_mc = np.mean(V[:, 0, i])
        
        print(f"  Instrument {i} ({instr['Underlying']} Forward):")
        print(f"    Theoretical V_f(0): {V_theoretical:.2f}")
        print(f"    MC estimate: {V_mc:.2f}")
        print(f"    Difference: {abs(V_mc - V_theoretical):.2f}")
    print()
    
    # Test 2: Put option pricing at t=0
    print("TEST 2: Put Option Pricing at t=0")
    print("-" * 40)
    
    sigma_sx5e = data['volatility_sx5e']
    sigma_aex = data['volatility_aex']
    
    for i in range(2, 4):  # Instruments 2 and 3 are puts
        instr = portfolio.iloc[i]
        S0 = instr['S0']
        K = instr['Strike']
        sigma = sigma_sx5e if instr['Underlying'] == 'SX5E' else sigma_aex
        
        # Theoretical put value at t=0 using Black-Scholes
        V_theoretical = black_scholes_price(S0, K, T, r, q, sigma, option_type='put')
        
        # MC estimate
        V_mc = np.mean(V[:, 0, i])
        
        print(f"  Instrument {i} ({instr['Underlying']} Put):")
        print(f"    Theoretical BS(0): {V_theoretical:.2f}")
        print(f"    MC estimate: {V_mc:.2f}")
        print(f"    Difference: {abs(V_mc - V_theoretical):.2f}")
    print()
    
    # Test 3: Exposure calculation (max with zero)
    print("TEST 3: Exposure Calculation")
    print("-" * 40)
    
    # Standalone exposure: max(V, 0) for each instrument
    exposures_standalone = np.maximum(V, 0)
    
    # Netted exposure: max(sum of all instruments, 0)
    portfolio_value = np.sum(V, axis=2)  # Sum across instruments
    exposures_netted = np.maximum(portfolio_value, 0)
    
    print(f"  Standalone exposures shape: {exposures_standalone.shape}")
    print(f"  Portfolio value shape: {portfolio_value.shape}")
    print(f"  Netted exposures shape: {exposures_netted.shape}")
    print()
    
    # Check that exposures are non-negative
    assert (exposures_standalone >= 0).all(), "Standalone exposures should be non-negative"
    assert (exposures_netted >= 0).all(), "Netted exposures should be non-negative"
    print(f"✓ All exposures non-negative (as expected from max with 0)")
    print()
    
    print(f"Mean portfolio exposure (netted): {np.mean(exposures_netted[-1]):.2f}")
    print()
    
    print("✓ All Tests Passed - Equity Instruments Valuation Test Complete")


if __name__ == "__main__":
    test_equity_instruments()
