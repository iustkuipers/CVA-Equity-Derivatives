"""Test orchestrator initialization"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.orchestrator import initialize

print("=" * 80)
print("ORCHESTRATOR INITIALIZATION TEST")
print("=" * 80)
print()

data = initialize()

print("Returned Keys:")
for key in sorted(data.keys()):
    if key == 'portfolio':
        print(f"  {key}: DataFrame shape {data[key].shape}")
    else:
        print(f"  {key}: {type(data[key]).__name__}")
print()

print("Portfolio Data:")
print(data['portfolio'])
print()

print("Sample Config Values from Initialize:")
print(f"  Counterparty: {data['counterparty']}")
print(f"  LGD: {data['lgd']:.1%}")
print(f"  Risk-Free Rate: {data['risk_free_rate']:.2%}")
print(f"  Volatility SX5E: {data['volatility_sx5e']:.2%}")
print(f"  Volatility AEX: {data['volatility_aex']:.2%}")
print()
