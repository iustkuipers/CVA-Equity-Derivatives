"""Test configuration parameters loading"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.config import (
    COUNTERPARTY, LGD, FORWARD_HAZARD_RATES, RISK_FREE_RATE,
    DIVIDEND_YIELD, VOLATILITY_SX5E, VOLATILITY_AEX, CORRELATION,
    TIME_STEP_YEARS, UNDERLYINGS, OPTION_TYPE, COUNTERPARTY_MATURITY
)

print("=" * 80)
print("CONFIG PARAMETERS TEST")
print("=" * 80)
print()

print("Credit Data:")
print(f"  Counterparty: {COUNTERPARTY}")
print(f"  LGD: {LGD:.1%}")
print(f"  Forward Hazard Rates: {FORWARD_HAZARD_RATES}")
print()

print("Interest Rates:")
print(f"  Risk-Free Rate: {RISK_FREE_RATE:.2%}")
print()

print("Equity Parameters:")
print(f"  Dividend Yield: {DIVIDEND_YIELD:.2%}")
print(f"  Volatility SX5E: {VOLATILITY_SX5E:.2%}")
print(f"  Volatility AEX: {VOLATILITY_AEX:.2%}")
print(f"  Correlation: {CORRELATION:.1%}")
print(f"  Time Step: {TIME_STEP_YEARS:.4f} years ({int(TIME_STEP_YEARS*12)} month(s))")
print()

print("Underlyings:")
for underlying, params in UNDERLYINGS.items():
    print(f"  {underlying}: {params['name']} (σ={params['volatility']:.2%})")
print()

print("Valuation Parameters:")
print(f"  Option Type: {OPTION_TYPE}")
print(f"  Counterparty Maturity: {COUNTERPARTY_MATURITY} years")
print()
