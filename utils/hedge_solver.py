"""
utils/hedge_solver.py

Solve Q5 delta hedge notionals using CDS contracts (1Y, 3Y, 5Y) to immunize
CVA against forward-hazard-bucket shifts.

We solve for notionals N_1Y, N_3Y, N_5Y such that, for each hazard bucket k:

    ΔCVA_k + N_1Y * ΔV_1Y,k + N_3Y * ΔV_3Y,k + N_5Y * ΔV_5Y,k = 0

Conventions:
- delta_cva[bucket] is (CVA_bumped - CVA_base) in EUR
- delta_cds[bucket][maturity] is (PV_bumped - PV_base) for 1 unit notional CDS
  in EUR per 1 notional (so notionals come out in EUR notional)
- Positive notional means "buy protection" IF your CDS PV function returns PV to
  protection buyer. If you use seller PVs, flip sign or set protection_buyer=True
  consistently in cds_pricer.py.

This module supports both:
- Full 3x3 solve (robust), and
- Triangular solve (faster + aligns with the zero-pattern in the question).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

import numpy as np


_BUCKETS_DEFAULT = ("0_1", "1_3", "3_5")
_MATS_DEFAULT = (1.0, 3.0, 5.0)


@dataclass(frozen=True)
class HedgeResult:
    notionals: Dict[float, float]          # {1.0: N1, 3.0: N3, 5.0: N5}
    A: np.ndarray                          # sensitivity matrix
    b: np.ndarray                          # RHS vector (-ΔCVA)
    residual: np.ndarray                   # A*N - b (should be ~0)


def _as_float(x) -> float:
    return float(x)


def build_sensitivity_system(
    delta_cva: Dict[str, float],
    delta_cds: Dict[str, Dict[float, float]],
    buckets: Iterable[str] = _BUCKETS_DEFAULT,
    maturities: Iterable[float] = _MATS_DEFAULT,
    enforce_zero_pattern: bool = False
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[float, ...]]:
    """
    Build A and b for the linear system A N = b, where b = -ΔCVA.

    Parameters
    ----------
    delta_cva : dict[bucket] -> float
    delta_cds : dict[bucket] -> dict[maturity] -> float
    buckets : iterable[str]
        Order of hazard buckets in the system.
    maturities : iterable[float]
        Order of CDS maturities in the system.
    enforce_zero_pattern : bool
        If True, sets elements that should be structurally zero to exactly 0:
          bucket 1_3 has 1Y = 0
          bucket 3_5 has 1Y = 0 and 3Y = 0

    Returns
    -------
    A : (len(buckets), len(maturities)) ndarray
    b : (len(buckets),) ndarray
    buckets_tuple, maturities_tuple : tuple orderings used
    """
    buckets_t = tuple(buckets)
    mats_t = tuple(float(m) for m in maturities)

    A = np.zeros((len(buckets_t), len(mats_t)), dtype=float)
    b = np.zeros((len(buckets_t),), dtype=float)

    for i, buck in enumerate(buckets_t):
        if buck not in delta_cva:
            raise KeyError(f"delta_cva missing bucket '{buck}'")
        b[i] = -_as_float(delta_cva[buck])

        if buck not in delta_cds:
            raise KeyError(f"delta_cds missing bucket '{buck}'")

        for j, T in enumerate(mats_t):
            if T not in delta_cds[buck]:
                raise KeyError(f"delta_cds['{buck}'] missing maturity {T}")
            A[i, j] = _as_float(delta_cds[buck][T])

    if enforce_zero_pattern:
        # expected structure from the question (piecewise hazards)
        # bucket 1_3: 1Y unaffected
        if "1_3" in buckets_t and 1.0 in mats_t:
            A[buckets_t.index("1_3"), mats_t.index(1.0)] = 0.0
        # bucket 3_5: 1Y and 3Y unaffected
        if "3_5" in buckets_t and 1.0 in mats_t:
            A[buckets_t.index("3_5"), mats_t.index(1.0)] = 0.0
        if "3_5" in buckets_t and 3.0 in mats_t:
            A[buckets_t.index("3_5"), mats_t.index(3.0)] = 0.0

    return A, b, buckets_t, mats_t


def solve_delta_hedge(
    delta_cva: Dict[str, float],
    delta_cds: Dict[str, Dict[float, float]],
    buckets: Iterable[str] = _BUCKETS_DEFAULT,
    maturities: Iterable[float] = _MATS_DEFAULT,
    method: str = "auto",
    enforce_zero_pattern: bool = True,
    rcond: float | None = None
) -> HedgeResult:
    """
    Solve hedge notionals N for CDS maturities to offset ΔCVA bucket shocks.

    Parameters
    ----------
    delta_cva : dict[bucket] -> float
        ΔCVA under +10bps hazard bump (EUR)
    delta_cds : dict[bucket] -> dict[maturity] -> float
        ΔPV of CDS (per 1 notional) under the same bump (EUR per 1 notional)
    method : {"auto","triangular","solve","lstsq"}
        - "triangular": uses the known zero-pattern to solve sequentially
        - "solve": generic np.linalg.solve on square system
        - "lstsq": least squares (fallback if near-singular)
        - "auto": tries triangular if structure matches, else solve, else lstsq
    enforce_zero_pattern : bool
        If True, explicitly sets the known structural zeros in A to 0.
    rcond : float or None
        Passed to np.linalg.lstsq if used.

    Returns
    -------
    HedgeResult with notionals by maturity and diagnostics.
    """
    buckets_t = tuple(buckets)
    mats_t = tuple(maturities)
    A, b, buckets_t, mats_t = build_sensitivity_system(
        delta_cva, delta_cds, buckets=buckets_t, maturities=mats_t,
        enforce_zero_pattern=enforce_zero_pattern
    )

    # Ensure it's 3x3 for the default case; otherwise fall back to generic methods.
    is_square = A.shape[0] == A.shape[1]

    def _triangular_possible() -> bool:
        # Checks the intended upper-triangular pattern in bucket order (0_1,1_3,3_5) and maturities (1,3,5)
        try:
            i13 = buckets_t.index("1_3")
            i35 = buckets_t.index("3_5")
            j1 = mats_t.index(1.0)
            j3 = mats_t.index(3.0)
        except ValueError:
            return False
        # Allow tiny numerical noise
        eps = 1e-12
        return (abs(A[i13, j1]) < eps) and (abs(A[i35, j1]) < eps) and (abs(A[i35, j3]) < eps)

    if method == "auto":
        if is_square and _triangular_possible():
            method_use = "triangular"
        elif is_square:
            method_use = "solve"
        else:
            method_use = "lstsq"
    else:
        method_use = method

    if method_use == "triangular":
        # Sequential solve exploiting structure:
        # row for 3_5 uses only 5Y
        # row for 1_3 uses 3Y and 5Y
        # row for 0_1 uses 1Y, 3Y, 5Y
        try:
            i01 = buckets_t.index("0_1")
            i13 = buckets_t.index("1_3")
            i35 = buckets_t.index("3_5")
            j1 = mats_t.index(1.0)
            j3 = mats_t.index(3.0)
            j5 = mats_t.index(5.0)
        except ValueError as e:
            raise ValueError("Triangular method requires buckets {0_1,1_3,3_5} and maturities {1,3,5}.") from e

        # Solve N5
        if abs(A[i35, j5]) < 1e-18:
            raise np.linalg.LinAlgError("Cannot solve triangular system: Δ5Y CDS sensitivity to 3_5 bump is ~0.")
        N5 = b[i35] / A[i35, j5]

        # Solve N3
        if abs(A[i13, j3]) < 1e-18:
            raise np.linalg.LinAlgError("Cannot solve triangular system: Δ3Y CDS sensitivity to 1_3 bump is ~0.")
        N3 = (b[i13] - A[i13, j5] * N5) / A[i13, j3]

        # Solve N1
        if abs(A[i01, j1]) < 1e-18:
            raise np.linalg.LinAlgError("Cannot solve triangular system: Δ1Y CDS sensitivity to 0_1 bump is ~0.")
        N1 = (b[i01] - A[i01, j3] * N3 - A[i01, j5] * N5) / A[i01, j1]

        N_vec = np.zeros((len(mats_t),), dtype=float)
        N_vec[j1], N_vec[j3], N_vec[j5] = N1, N3, N5

    elif method_use == "solve":
        if not is_square:
            raise ValueError("method='solve' requires a square system.")
        N_vec = np.linalg.solve(A, b)

    elif method_use == "lstsq":
        N_vec, *_ = np.linalg.lstsq(A, b, rcond=rcond)

    else:
        raise ValueError("method must be one of {'auto','triangular','solve','lstsq'}")

    residual = A @ N_vec - b
    notionals = {float(mats_t[j]): float(N_vec[j]) for j in range(len(mats_t))}

    return HedgeResult(notionals=notionals, A=A, b=b, residual=residual)