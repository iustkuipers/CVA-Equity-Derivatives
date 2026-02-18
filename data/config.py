"""
Configuration parameters for CVA equity derivatives modeling
Case 3: CVA Equity Derivatives - 2026
"""

# ============================================================================
# 1. CREDIT DATA (Counterparty "C")
# ============================================================================
COUNTERPARTY = "C"
LGD = 0.40  # Loss Given Default: 40%

# Forward hazard rates by period
FORWARD_HAZARD_RATES = {
    "0_1": 0.0200,      # 0 ≤ t < 1 years: 2.00%
    "1_3": 0.0215,      # 1 ≤ t < 3 years: 2.15%
    "3_5": 0.0220,      # 3 ≤ t ≤ 5 years: 2.20%
}

# ============================================================================
# 2. INTEREST RATES
# ============================================================================
RISK_FREE_RATE = 0.03  # Flat term structure at 3% (continuously compounded)

# ============================================================================
# 3. EQUITY PARAMETERS
# ============================================================================
DIVIDEND_YIELD = 0.02  # q = 2% for both assets

# Volatilities (to be used in equity processes defined elsewhere)
VOLATILITY_SX5E = 0.15  # σ_SX5E = 15%
VOLATILITY_AEX = 0.15   # σ_AEX = 15%

# Correlation between assets
CORRELATION = 0.80  # ρ = 80%

# Time step (in years)
TIME_STEP_MONTHS = 1
TIME_STEP_YEARS = TIME_STEP_MONTHS / 12

# ============================================================================
# UNDERLYINGS
# ============================================================================
UNDERLYINGS = {
    'SX5E': {
        'name': 'STOXX 50',
        'volatility': VOLATILITY_SX5E,
    },
    'AEX': {
        'name': 'Amsterdam Exchange Index',
        'volatility': VOLATILITY_AEX,
    }
}

# ============================================================================
# PORTFOLIO & VALUATION PARAMETERS
# ============================================================================
OPTION_TYPE = 'European'
COUNTERPARTY_MATURITY = 5  # Years

# ============================================================================
# SIMULATION & VALIDATION PARAMETERS
# ============================================================================
NUM_SIMULATIONS = 10000  # Number of Monte Carlo paths
RANDOM_SEED = 42  # For reproducibility
