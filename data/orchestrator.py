"""
Orchestrator for CVA equity derivatives modeling
Coordinates data flow from configuration to portfolio composition
"""

from data import config
from data.portfolio import create_portfolio


def initialize():
    """
    Initialize and return all configuration parameters and portfolio data.
    
    Returns:
        dict: Dictionary containing all config parameters and portfolio DataFrame
    """
    volatilities = {
        'SX5E': config.VOLATILITY_SX5E,
        'AEX': config.VOLATILITY_AEX,
    }
    
    portfolio = create_portfolio(volatilities)
    
    # Aggregate all configuration and portfolio data
    data = {
        # Credit Data
        'counterparty': config.COUNTERPARTY,
        'lgd': config.LGD,
        'forward_hazard_rates': config.FORWARD_HAZARD_RATES,
        
        # Interest Rates
        'risk_free_rate': config.RISK_FREE_RATE,
        
        # Equity Parameters
        'dividend_yield': config.DIVIDEND_YIELD,
        'volatility_sx5e': config.VOLATILITY_SX5E,
        'volatility_aex': config.VOLATILITY_AEX,
        'correlation': config.CORRELATION,
        'time_step_months': config.TIME_STEP_MONTHS,
        'time_step_years': config.TIME_STEP_YEARS,
        
        # Underlyings
        'underlyings': config.UNDERLYINGS,
        
        # Valuation Parameters
        'option_type': config.OPTION_TYPE,
        'counterparty_maturity': config.COUNTERPARTY_MATURITY,
        
        # Simulation & Validation Parameters
        'num_simulations': config.NUM_SIMULATIONS,
        'random_seed': config.RANDOM_SEED,
        
        # Portfolio
        'portfolio': portfolio,
    }
    
    return data
