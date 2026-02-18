import numpy as np
from pricing.black_scholes import black_scholes_put


def value_portfolio_over_time(sim_data, portfolio, config):
    """
    Computes mark-to-market values of each instrument
    at each simulation path and time step.

    Returns:
        V: numpy array with shape
           (num_simulations, num_steps + 1, num_instruments)
    """

    sx5e_paths = sim_data["sx5e"]
    aex_paths = sim_data["aex"]
    times = sim_data["times"]

    r = config["risk_free_rate"]
    q = config["dividend_yield"]
    T = config["counterparty_maturity"]

    num_simulations, num_steps_plus_one = sx5e_paths.shape
    num_instruments = len(portfolio)

    # Initialize value tensor
    V = np.zeros((num_simulations, num_steps_plus_one, num_instruments))

    for idx, row in portfolio.iterrows():
        print(f"  Valuing instrument {idx + 1}/{num_instruments}: {row['Type']} on {row['Underlying']}...", end=" ", flush=True)

        instrument_type = row["Type"]
        underlying = row["Underlying"]
        K = row["Strike"]
        sigma = row.get("Vol", None)

        if underlying == "SX5E":
            S_paths = sx5e_paths
        elif underlying == "AEX":
            S_paths = aex_paths
        else:
            raise ValueError(f"Unknown underlying: {underlying}")

        for t_index, t in enumerate(times):

            tau = max(T - t, 0.0)

            S_t = S_paths[:, t_index]

            # --- Forward valuation ---
            if instrument_type == "Forward":

                # V(t) = S_t e^{-q tau} - K e^{-r tau}
                V[:, t_index, idx] = (
                    S_t * np.exp(-q * tau)
                    - K * np.exp(-r * tau)
                )

            # --- Put option valuation ---
            elif instrument_type == "Option":

                if tau == 0:
                    # At maturity: intrinsic value
                    V[:, t_index, idx] = np.maximum(K - S_t, 0.0)
                else:
                    # Vectorized BSM valuation
                    V[:, t_index, idx] = np.array([
                        black_scholes_put(
                            S0=S_val,
                            K=K,
                            r=r,
                            q=q,
                            sigma=sigma,
                            T=tau
                        )
                        for S_val in S_t
                    ])

            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")
        
        print("✓")

    return V
