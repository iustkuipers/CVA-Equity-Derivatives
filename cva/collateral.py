import numpy as np


def apply_variation_margin(V_portfolio: np.ndarray, M: int) -> np.ndarray:
    """
    Apply periodic Variation Margin with update frequency M (months).

    Parameters
    ----------
    V_portfolio : np.ndarray
        Portfolio values with shape (num_sims, num_times)

    M : int
        Margin update frequency in months (1 to 60)

    Returns
    -------
    exposure : np.ndarray
        Collateralized exposure with shape (num_sims, num_times)
    """

    num_sims, num_times = V_portfolio.shape

    exposure = np.zeros_like(V_portfolio)

    for t in range(num_times):

        # Determine last margin update time index
        last_update = (t // M) * M

        collateral_level = V_portfolio[:, last_update]

        # Exposure = positive change since last update
        exposure[:, t] = np.maximum(
            V_portfolio[:, t] - collateral_level,
            0.0
        )

    return exposure


def apply_initial_margin(V_portfolio: np.ndarray, IM: float) -> np.ndarray:
    """
    Apply fixed Initial Margin (Independent Amount).

    Parameters
    ----------
    V_portfolio : np.ndarray
        Portfolio values with shape (num_sims, num_times)

    IM : float
        Initial Margin amount in EUR

    Returns
    -------
    exposure : np.ndarray
        Collateral-adjusted exposure
    """

    return np.maximum(V_portfolio - IM, 0.0)
