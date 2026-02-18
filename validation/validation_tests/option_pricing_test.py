import numpy as np
from pricing.black_scholes import black_scholes_put


def run_option_pricing_test(simulator, portfolio, config):
    """
    Option Pricing Test:
    Compare Monte Carlo discounted payoff of Put options
    with analytical Black-Scholes-Merton price.

    Returns:
        list of dict results
    """

    sim_data = simulator.simulate_paths()

    sx5e_paths = sim_data["sx5e"]
    aex_paths = sim_data["aex"]

    T = config["counterparty_maturity"]
    r = config["risk_free_rate"]
    q = config["dividend_yield"]
    N = config["num_simulations"]

    puts = portfolio[
        (portfolio["Type"] == "Option") &
        (portfolio["OptionType"] == "Put")
    ]

    results = []

    for _, row in puts.iterrows():

        underlying = row["Underlying"]
        K = row["Strike"]
        S0 = row["S0"]
        sigma = row["Vol"]

        if underlying == "SX5E":
            ST = sx5e_paths[:, -1]
        elif underlying == "AEX":
            ST = aex_paths[:, -1]
        else:
            continue

        # --- Monte Carlo discounted payoff ---
        payoff = np.maximum(K - ST, 0.0)
        discounted_payoff = np.exp(-r * T) * payoff

        mc_estimate = np.mean(discounted_payoff)
        std_dev = np.std(discounted_payoff, ddof=1)
        std_error = std_dev / np.sqrt(N)

        ci_lower = mc_estimate - 1.96 * std_error
        ci_upper = mc_estimate + 1.96 * std_error

        # --- Analytical price ---
        theoretical_value = black_scholes_put(
            S0=S0,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            T=T
        )

        passes = ci_lower <= theoretical_value <= ci_upper

        results.append({
            "test": "Option Pricing Test",
            "instrument": f"{underlying} Put",
            "mc_estimate": mc_estimate,
            "theoretical_value": theoretical_value,
            "std_error": std_error,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "passes_test": passes
        })

    return results
