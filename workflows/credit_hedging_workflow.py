"""
workflows/credit_hedging_workflow.py

Q5: Credit Risk Exposure Hedging

- Computes CVA sensitivities (ΔCVA) under +10bps hazard bumps
- Computes CDS sensitivities (ΔMTM) for 1Y, 3Y, 5Y CDS
- Solves first-order delta hedge notionals
- Writes results to output/q5/

Assumes:
- baseline_results from compute_cva_pipeline()
- data dict from initialize()
"""

import os
import pandas as pd

from cva.hazard_sensitivity import compute_cva_sensitivities
from pricing.cds_pricing import compute_cds_sensitivities
from utils.hedge_solver import solve_delta_hedge


OUTPUT_DIR = "output/q5"


def run_q5_credit_hedging(data, baseline_results):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣ Extract baseline EPE and time grid
    # ------------------------------------------------------------------

    epe = baseline_results["EPE_netted"]
    times = baseline_results["times"]

    # ------------------------------------------------------------------
    # 2️⃣ CVA Sensitivities
    # ------------------------------------------------------------------

    delta_cva = compute_cva_sensitivities(data, epe, times)

    df_cva = pd.DataFrame.from_dict(
        delta_cva,
        orient="index",
        columns=["Delta_CVA_EUR"]
    )

    df_cva.index.name = "Hazard_Bucket"
    df_cva.to_csv(f"{OUTPUT_DIR}/q5_cva_sensitivities.csv")

    # ------------------------------------------------------------------
    # 3️⃣ CDS Sensitivities
    # ------------------------------------------------------------------

    base_spreads, delta_cds = compute_cds_sensitivities(data)
    # Convert nested dict to DataFrame
    df_cds = pd.DataFrame(delta_cds).T
    df_cds.columns = [f"Delta_CDS_{int(m)}Y" for m in df_cds.columns]
    df_cds.index.name = "Hazard_Bucket"

    df_cds.to_csv(f"{OUTPUT_DIR}/q5_cds_sensitivities.csv")

    # ------------------------------------------------------------------
    # 4️⃣ Hedge Notionals
    # ------------------------------------------------------------------

    print("\n[Q5d] Constructing delta hedge (bootstrapping)...")

    hedge_result = solve_delta_hedge(
        delta_cva,
        delta_cds,
        method="triangular"  # Force backward solve
    )

    print("\nBootstrapping logic:")
    print("  Step 1: Use 5Y CDS to neutralize [3,5] bucket")
    print("  Step 2: Use 3Y CDS (+ 5Y position) to neutralize [1,3] bucket")
    print("  Step 3: Use 1Y CDS (+ 3Y & 5Y positions) to neutralize [0,1] bucket")
    flipped_notionals = {k: -v for k, v in hedge_result.notionals.items()}
    df_hedge = pd.DataFrame.from_dict(
        flipped_notionals,
        orient="index",
        columns=["Notional_EUR"]
    )

    df_hedge.index.name = "CDS_Maturity_Y"
    df_hedge.to_csv(f"{OUTPUT_DIR}/q5_hedge_notionals.csv")

    # ------------------------------------------------------------------
    # 5️⃣ Console Summary
    # ------------------------------------------------------------------

    print("\n" + "="*60)
    print("Q5 RESULTS — CREDIT HEDGING")
    print("="*60)

    print("\nΔCVA (EUR):")
    print(df_cva)

    print("\nΔCDS MTM (per 1 notional):")
    print(df_cds)

    print("\nHedge Notionals (EUR):")
    print(df_hedge)

    print("\nHedge residual (should be ~0):")
    print(hedge_result.residual)

    return {
        "delta_cva": delta_cva,
        "delta_cds": delta_cds,
        "hedge_notionals": hedge_result.notionals,
        "residual": hedge_result.residual
    }