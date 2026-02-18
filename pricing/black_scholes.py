import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, q, sigma, option_type='call'):
    """
    Black-Scholes-Merton price for European call or put option.
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: 'call' or 'put'
    Returns:
        Option price (float)
    """
    if T == 0:
        if option_type == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


def black_scholes_put(S0, K, r, q, sigma, T):
    """
    Black-Scholes-Merton European Put Price
    """
    if T == 0:
        return max(K - S0, 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S0 * np.exp(-q * T) * norm.cdf(-d1)
    )
    return put_price
