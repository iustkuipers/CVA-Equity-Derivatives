import numpy as np


def run_martingale_test(simulator, portfolio, config):
    """
    Martingale Test:
    Under the risk-neutral measure, discounted asset prices must be martingales.

    We test this by comparing:
        - Monte Carlo discounted expected payoff of forwards
        - Analytical t=0 forward valuation

    Returns:
        dict with test results
    """

    # --- Simulate paths ---
    sim_data = simulator.simulate_paths()

    sx5e_paths = sim_data["sx5e"]
    aex_paths = sim_data["aex"]

    T = config["counterparty_maturity"]
    r = config["risk_free_rate"]
    q = config["dividend_yield"]
    N = config["num_simulations"]

    # --- Extract forward positions from portfolio ---
    forwards = portfolio[portfolio["Type"] == "Forward"]

    results = []

    for _, row in forwards.iterrows():

        underlying = row["Underlying"]
        K = row["Strike"]
        S0 = row["S0"]

        if underlying == "SX5E":
            ST = sx5e_paths[:, -1]
        elif underlying == "AEX":
            ST = aex_paths[:, -1]
        else:
            continue

        # --- Monte Carlo discounted payoff ---
        discounted_payoff = np.exp(-r * T) * (ST - K)

        mc_estimate = np.mean(discounted_payoff)
        std_dev = np.std(discounted_payoff, ddof=1)
        std_error = std_dev / np.sqrt(N)

        ci_lower = mc_estimate - 1.96 * std_error
        ci_upper = mc_estimate + 1.96 * std_error

        # --- Analytical forward value at t=0 ---
        theoretical_value = S0 * np.exp(-q * T) - K * np.exp(-r * T)

        passes = ci_lower <= theoretical_value <= ci_upper

        results.append({
            "test": "Martingale Test",
            "instrument": f"{underlying} Forward",
            "mc_estimate": mc_estimate,
            "theoretical_value": theoretical_value,
            "std_error": std_error,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "passes_test": passes
        })

    return results
