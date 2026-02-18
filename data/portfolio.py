import pandas as pd
import numpy as np


def create_portfolio(volatilities):
    """
    Create the portfolio composition with equity derivatives exposures.
    
    Args:
        volatilities (dict): Dictionary with 'SX5E' and 'AEX' volatility values
    
    Returns:
        DataFrame: Portfolio composition with all instrument details
    """
    
    portfolio_data = {
        'Instrument': [1, 2, 3, 4],
        'Type': ['Forward', 'Forward', 'Option', 'Option'],
        'OptionType': [np.nan, np.nan, 'Put', 'Put'],
        'Direction': ['Long', 'Long', 'Long', 'Long'],
        'Underlying': ['SX5E', 'AEX', 'SX5E', 'AEX'],
        'Contracts': [10_000, 55_000, 10_000, 55_000],
        'S0': [6_000, 1_000, 6_000, 1_000],
        'Strike': [6_000, 1_000, 4_800, 800],
        'Vol': [np.nan, np.nan, volatilities['SX5E'], volatilities['AEX']],
        'Maturity_Years': [5, 5, 5, 5],
    }
    
    portfolio_df = pd.DataFrame(portfolio_data)
    return portfolio_df
