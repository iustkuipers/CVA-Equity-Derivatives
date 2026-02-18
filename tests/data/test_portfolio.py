import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.portfolio import create_portfolio
from data.config import VOLATILITY_SX5E, VOLATILITY_AEX

# Test what main would see
volatilities = {
    'SX5E': VOLATILITY_SX5E,
    'AEX': VOLATILITY_AEX,
}
portfolio = create_portfolio(volatilities)
print(portfolio)

