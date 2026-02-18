import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )

from pricing.black_scholes import black_scholes_price


def test_black_scholes():
    print("Testing Black-Scholes Put Pricing:")
    
    # Test case 1: At-the-money put
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.02
    sigma = 0.2
    option_type = 'put'

    price = black_scholes_price(S, K, T, r, q, sigma, option_type)
    print(f"  ATM Put (S=K=100, T=1y, r=5%, q=2%, σ=20%): {price:.4f}")
    # Verified against independent calculation: put price ≈ 6.33
    assert abs(price - 6.3301) < 0.01, f"Expected ~6.3301, got {price}"
    
    # Test case 2: In-the-money put
    price_itm = black_scholes_price(S=80, K=100, T=1, r=0.05, q=0.02, sigma=0.2, option_type='put')
    print(f"  ITM Put (S=80, K=100): {price_itm:.4f}")
    assert price_itm > price, "ITM put should be more valuable than ATM put"
    
    # Test case 3: Call option (sanity check)
    price_call = black_scholes_price(S=100, K=100, T=1, r=0.05, q=0.02, sigma=0.2, option_type='call')
    print(f"  ATM Call (S=K=100, T=1y, r=5%, q=2%, σ=20%): {price_call:.4f}")
    # Call-put parity: C - P = S*e^(-qT) - K*e^(-rT)
    parity_lhs = price_call - price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    print(f"  Call-Put Parity Check: {parity_lhs:.4f} ≈ {parity_rhs:.4f}")
    assert abs(parity_lhs - parity_rhs) < 1e-6, "Call-Put parity violated"
    
    print("✓ All Black-Scholes tests passed!")

if __name__ == "__main__":
    test_black_scholes()
