import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator


def test_validation_orchestrator():
    """Test the validation orchestrator setup"""
    
    print("=" * 80)
    print("VALIDATION ORCHESTRATOR TEST")
    print("=" * 80)
    print()
    
    # Initialize config and portfolio
    data = initialize()
    config = data
    portfolio = data['portfolio']
    
    print("Initializing Equity Process Simulator for Validation...")
    simulator = EquityProcessSimulator(config, portfolio)
    
    print(f"Simulator initialized with {simulator.num_steps} timesteps")
    print(f"Number of simulations configured: {simulator.num_simulations}")
    print(f"Random seed: {simulator.seed}")
    print()
    
    # Simulate paths
    print("Running equity process simulation...")
    paths = simulator.simulate_paths()
    
    print(f"✓ Simulation complete")
    print(f"  SX5E paths shape: {paths['sx5e'].shape}")
    print(f"  AEX paths shape: {paths['aex'].shape}")
    print()
    
    # Basic validation checks
    print("Running basic validation checks...")
    
    # Check 1: Paths start at correct initial prices
    assert abs(paths['sx5e'][0, 0] - simulator.S0_sx5e) < 1e-6, "SX5E initial price mismatch"
    assert abs(paths['aex'][0, 0] - simulator.S0_aex) < 1e-6, "AEX initial price mismatch"
    print("✓ Initial prices correct")
    
    # Check 2: No NaN or Inf values
    assert not np.isnan(paths['sx5e']).any(), "NaN found in SX5E paths"
    assert not np.isnan(paths['aex']).any(), "NaN found in AEX paths"
    assert not np.isinf(paths['sx5e']).any(), "Inf found in SX5E paths"
    assert not np.isinf(paths['aex']).any(), "Inf found in AEX paths"
    print("✓ No NaN or Inf values")
    
    # Check 3: All prices are positive
    assert (paths['sx5e'] > 0).all(), "Non-positive prices in SX5E"
    assert (paths['aex'] > 0).all(), "Non-positive prices in AEX"
    print("✓ All prices positive")
    
    print()
    print("✓ Validation Orchestrator Test Complete")


if __name__ == "__main__":
    import numpy as np
    test_validation_orchestrator()

