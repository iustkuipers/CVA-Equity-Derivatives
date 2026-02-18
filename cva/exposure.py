import numpy as np


def positive_part(x: np.ndarray) -> np.ndarray:
    """Elementwise max(x, 0)."""
    return np.maximum(x, 0.0)


def compute_standalone_positive_exposures(V: np.ndarray) -> np.ndarray:
    """
    Standalone positive exposures per instrument.

    Args:
        V: portfolio values with shape (num_sims, num_times, num_instruments)

    Returns:
        E_pos: shape (num_sims, num_times, num_instruments)
               where E_pos[:,:,i] = max(V[:,:,i], 0)
    """
    if V.ndim != 3:
        raise ValueError(f"Expected V to be 3D (sims, times, instruments). Got shape {V.shape}")

    return positive_part(V)


def compute_unnetted_exposure(V: np.ndarray) -> np.ndarray:
    """
    Unnetted exposure profile:
        E(t) = sum_i max(V_i(t), 0)

    Args:
        V: (num_sims, num_times, num_instruments)

    Returns:
        E_unnetted: (num_sims, num_times)
    """
    E_pos = compute_standalone_positive_exposures(V)
    return np.sum(E_pos, axis=2)


def compute_netted_exposure(V: np.ndarray) -> np.ndarray:
    """
    Netted exposure profile (Master Netting Agreement):
        E(t) = max(sum_i V_i(t), 0)

    Args:
        V: (num_sims, num_times, num_instruments)

    Returns:
        E_netted: (num_sims, num_times)
    """
    V_portfolio = np.sum(V, axis=2)
    return positive_part(V_portfolio)


def compute_epe(exposure: np.ndarray) -> np.ndarray:
    """
    Expected Positive Exposure (EPE) at each time:
        EPE(t) = E[ exposure(t) ] across simulations

    Args:
        exposure: (num_sims, num_times) or (num_sims, num_times, ...)

    Returns:
        epe: (num_times,) if exposure is 2D
             otherwise averages over axis=0 leaving remaining dims
    """
    if exposure.ndim < 2:
        raise ValueError(f"Exposure must have at least 2 dimensions. Got shape {exposure.shape}")

    # Average across simulations axis=0
    return np.mean(exposure, axis=0)


def compute_standalone_epe(V: np.ndarray) -> np.ndarray:
    """
    Standalone EPE per instrument:
        EPE_i(t) = E[ max(V_i(t), 0) ]

    Args:
        V: (num_sims, num_times, num_instruments)

    Returns:
        epe_i: (num_times, num_instruments)
    """
    E_pos = compute_standalone_positive_exposures(V)  # (sims, times, inst)
    return np.mean(E_pos, axis=0)  # (times, inst)
