import numpy as np
import pandas as pd

'''
def _build_piecewise_hazard_vector(times, forward_hazard_rates):
    """
    Build a hazard rate λ_k for each interval (t_{k-1}, t_k].

    Supports forward_hazard_rates provided as:
      - dict keyed by end time (e.g., {1: λ1, 3: λ2, 5: λ3, ...})
        where λ applies on (prev_key, key]
      - dict keyed by interval string (e.g., {"0_1": λ1, "1_3": λ2, "3_5": λ3})
      - list/np.array already aligned to intervals length (len(times)-1)

    Returns:
        lambdas: np.ndarray length (len(times)-1)
    """
    times = np.asarray(times)
    n_intervals = len(times) - 1

    # Case 1: already vector
    if isinstance(forward_hazard_rates, (list, tuple, np.ndarray)):
        lambdas = np.asarray(forward_hazard_rates, dtype=float)
        if lambdas.shape[0] != n_intervals:
            raise ValueError(
                f"forward_hazard_rates length {lambdas.shape[0]} "
                f"does not match number of intervals {n_intervals}."
            )
        return lambdas

    # Case 2: dict keyed by maturities or interval strings
    if isinstance(forward_hazard_rates, dict):
        # Convert string keys like "0_1", "1_3", "3_5" to numeric endpoints
        converted_rates = {}
        for key, val in forward_hazard_rates.items():
            if isinstance(key, str) and "_" in key:
                # Extract endpoint from "start_end" format
                parts = key.split("_")
                end_time = float(parts[-1])
                converted_rates[end_time] = val
            else:
                converted_rates[float(key)] = val
        
        segment_ends = sorted(converted_rates.keys())

        lambdas = np.zeros(n_intervals, dtype=float)

        for k in range(1, len(times)):
            t_prev = float(times[k - 1])
            t_curr = float(times[k])

            # find smallest segment_end >= t_curr
            seg_end = None
            for se in segment_ends:
                if t_curr <= se + 1e-12:
                    seg_end = se
                    break
            if seg_end is None:
                # if beyond last key, assume last hazard continues
                seg_end = segment_ends[-1]

            lambdas[k - 1] = float(converted_rates[seg_end])

        return lambdas

    raise TypeError("forward_hazard_rates must be a dict or a list/array.")
'''
def _build_piecewise_hazard_vector(times, forward_hazard_rates):
    times = np.asarray(times)
    n = len(times) - 1
    lambdas = np.zeros(n)
    
    # 1. Normalize the keys (handle both 0_1 and 0-1 and numeric 1.0)
    clean_rates = {}
    for k, v in forward_hazard_rates.items():
        if isinstance(k, str):
            # Replace common separators
            clean_key = k.replace("-", "_")
            end_t = float(clean_key.split("_")[-1])
            clean_rates[end_t] = v
        else:
            clean_rates[float(k)] = v
    
    sorted_ends = sorted(clean_rates.keys())

    # 2. Map every interval to the correct bucket
    for i in range(n):
        t_mid = (times[i] + times[i+1]) / 2.0
        # Find the first segment end that is >= our current time
        found = False
        for end_t in sorted_ends:
            if t_mid <= end_t + 1e-9:
                lambdas[i] = clean_rates[end_t]
                found = True
                break
        if not found:
            lambdas[i] = clean_rates[sorted_ends[-1]]
            
    return lambdas

def compute_survival_and_default(times, forward_hazard_rates):
    """
    Compute survival S(t_k) and incremental default probabilities ΔPD_k.

    S(t_k) = exp(-∑_{j=1..k} λ_j Δt_j)
    ΔPD_k  = S(t_{k-1}) - S(t_k)

    Args:
        times: array-like length (n+1), increasing time grid
        forward_hazard_rates: dict or vector (see helper)

    Returns:
        survival: np.ndarray length (n+1)
        dPD: np.ndarray length (n+1) with dPD[0]=0 and dPD[k]=ΔPD_k for k>=1
        lambdas: np.ndarray length n (interval hazards)
    """
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing.")

    dt = np.diff(times)
    lambdas = _build_piecewise_hazard_vector(times, forward_hazard_rates)

    # cumulative hazard at each time point
    cum_hazard = np.zeros(len(times), dtype=float)
    cum_hazard[1:] = np.cumsum(lambdas * dt)

    survival = np.exp(-cum_hazard)

    dPD = np.zeros(len(times), dtype=float)
    dPD[1:] = survival[:-1] - survival[1:]

    return survival, dPD, lambdas


def compute_cva_from_epe(times, epe, config, return_breakdown=True):
    """
    Compute CVA using discretized sum:
        CVA = LGD * Σ DF(t_k) * EPE(t_k) * ΔPD_k , k=1..n

    Args:
        times: (n+1,) time grid
        epe:   (n+1,) expected positive exposure at those times
        config: dict containing:
            - risk_free_rate (r)
            - lgd (decimal, e.g., 0.4)
            - forward_hazard_rates (dict or vector)
        return_breakdown: if True, return a DataFrame with per-step contributions

    Returns:
        cva: float
        breakdown_df (optional): DataFrame with columns
            t, dt, lambda, survival, dPD, df, epe, contrib
    """
    times = np.asarray(times, dtype=float)
    epe = np.asarray(epe, dtype=float)

    if times.shape[0] != epe.shape[0]:
        raise ValueError(f"times length {len(times)} must equal epe length {len(epe)}")

    r = float(config["risk_free_rate"])
    lgd = float(config["lgd"])
    forward_hazard_rates = config["forward_hazard_rates"]

    survival, dPD, lambdas = compute_survival_and_default(times, forward_hazard_rates)

    df = np.exp(-r * times)

    # Contribution starts from k=1 (no default increment at time 0)
    contrib = lgd * df[1:] * epe[1:] * dPD[1:]
    cva = float(np.sum(contrib))

    if not return_breakdown:
        return cva

    breakdown = pd.DataFrame({
        "t": times[1:],
        "dt": np.diff(times),
        "lambda": lambdas,
        "survival_t": survival[1:],
        "dPD": dPD[1:],
        "df": df[1:],
        "epe": epe[1:],
        "cva_contrib": contrib
    })

    return cva, breakdown
