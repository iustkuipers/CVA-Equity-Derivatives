"""Test equity process simulator"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator
import numpy as np

print("=" * 80)
print("EQUITY PROCESS SIMULATOR TEST")
print("=" * 80)
print()

# Initialize parameters and portfolio
data = initialize()

print("Initializing Equity Process Simulator with:")
print(f"  SX5E Initial Price (S0): {data['portfolio'][data['portfolio']['Underlying'] == 'SX5E']['S0'].iloc[0]}")
print(f"  AEX Initial Price (S0): {data['portfolio'][data['portfolio']['Underlying'] == 'AEX']['S0'].iloc[0]}")
print(f"  Risk-Free Rate (r): {data['risk_free_rate']:.2%}")
print(f"  Dividend Yield (q): {data['dividend_yield']:.2%}")
print(f"  Volatility SX5E (σ): {data['volatility_sx5e']:.2%}")
print(f"  Volatility AEX (σ): {data['volatility_aex']:.2%}")
print(f"  Correlation (ρ): {data['correlation']:.1%}")
print(f"  Time Step (Δ): {data['time_step_years']:.4f} years")
print(f"  Total Maturity (T): {data['counterparty_maturity']} years")
print()

# Create simulator
simulator = EquityProcessSimulator(data, data['portfolio'])

print(f"Simulator initialized with {simulator.num_steps} timesteps")
print()

# Run simulation and measure performance
import time
start = time.time()
paths = simulator.simulate_paths()
end = time.time()

print("Simulation completed successfully!")
print(f"Elapsed time: {end - start:.2f} seconds")
print()

print("Path Data Shapes:")
print(f"  Times: {paths['times'].shape}")
print(f"  SX5E Paths: {paths['sx5e'].shape}")
print(f"  AEX Paths: {paths['aex'].shape}")
print()

print("SX5E Statistics (at final time T=5 years):")
final_sx5e = paths['sx5e'][:, -1]
print(f"  Initial: {simulator.S0_sx5e:.2f}")
print(f"  Final Mean: {final_sx5e.mean():.2f}")
print(f"  Final Std Dev: {final_sx5e.std():.2f}")
print(f"  Final Min: {final_sx5e.min():.2f}")
print(f"  Final Max: {final_sx5e.max():.2f}")
print()

print("AEX Statistics (at final time T=5 years):")
final_aex = paths['aex'][:, -1]
print(f"  Initial: {simulator.S0_aex:.2f}")
print(f"  Final Mean: {final_aex.mean():.2f}")
print(f"  Final Std Dev: {final_aex.std():.2f}")
print(f"  Final Min: {final_aex.min():.2f}")
print(f"  Final Max: {final_aex.max():.2f}")
print()

# Check correlation of final returns
sx5e_returns = (paths['sx5e'][:, -1] - simulator.S0_sx5e) / simulator.S0_sx5e
aex_returns = (paths['aex'][:, -1] - simulator.S0_aex) / simulator.S0_aex
correlation = np.corrcoef(sx5e_returns, aex_returns)[0, 1]

print(f"Simulated Correlation of Final Returns: {correlation:.4f}")
print(f"Expected Correlation: {data['correlation']:.4f}")
print()
