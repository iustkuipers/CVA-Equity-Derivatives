import numpy as np


def run_correlation_test(simulator, config):
    """
    Correlation Structure Check:

    Verifies that the empirical correlation between
    log-returns of SX5E and AEX equals the target correlation.

    Confidence Interval:
        Uses Fisher z-transformation.

    Returns:
        dict result
    """

    sim_data = simulator.simulate_paths()

    sx5e_paths = sim_data["sx5e"]
    aex_paths = sim_data["aex"]

    rho_theoretical = config["correlation"]

    # --- Compute log-returns ---
    # Use full path increments
    log_returns_sx5e = np.diff(np.log(sx5e_paths), axis=1)
    log_returns_aex = np.diff(np.log(aex_paths), axis=1)

    # Flatten all increments (all paths, all timesteps)
    r1 = log_returns_sx5e.flatten()
    r2 = log_returns_aex.flatten()

    N = len(r1)

    # --- Sample correlation ---
    rho_hat = np.corrcoef(r1, r2)[0, 1]

    # --- Fisher z-transformation ---
    z_hat = 0.5 * np.log((1 + rho_hat) / (1 - rho_hat))
    se_z = 1 / np.sqrt(N - 3)

    z_lower = z_hat - 1.96 * se_z
    z_upper = z_hat + 1.96 * se_z

    # Transform back
    rho_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    rho_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    passes = rho_lower <= rho_theoretical <= rho_upper

    return {
        "test": "Correlation Structure Check",
        "instrument": "SX5E vs AEX",
        "mc_estimate": rho_hat,
        "theoretical_value": rho_theoretical,
        "std_error": se_z,
        "ci_lower_95": rho_lower,
        "ci_upper_95": rho_upper,
        "ci_method": "Fisher Z-transformation",
        "passes_test": passes
    }
