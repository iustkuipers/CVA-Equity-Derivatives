import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cva.exposure import compute_epe
from cva.cva_calculator import compute_cva_from_epe
from cva.collateral import apply_variation_margin, apply_initial_margin
from utils.output_writer import OutputWriter


def run_q4_collateral_impact(config, portfolio, baseline_results):
    """
    Q4: Collateral Impact on CVA (Netted Portfolio)
    
    REUSES baseline_results from main — does NOT recompute.

    (a) Variation Margin update frequency M = 1..60 months
    (b) Initial Margin IM in {1e6, 1e7, 1e8}

    Args:
        config: configuration dict
        portfolio: portfolio dataframe
        baseline_results: pre-computed results from compute_cva_pipeline (baseline scenario)

    Saves outputs to output/q4/
    """

    print("===================================================")
    print("Running Q4 - Collateral Impact on CVA")
    print("===================================================")

    base_path = "output/q4"
    OutputWriter.ensure_directory(base_path)

    # ---------------------------------------------------
    # 1) Extract baseline (no collateral baseline = Q2d netted)
    # ---------------------------------------------------
    # Reuse pre-computed baseline — NO RECOMPUTATION
    times = baseline_results["times"]
    baseline_cva_netted = baseline_results["cva_netted"]

    # We need portfolio MtM per path/time (netted value, BEFORE positive part)
    # Already computed baseline in main — reuse it
    if "V" not in baseline_results:
        raise KeyError(
            "baseline_results must contain the valuation tensor 'V' "
            "(shape: sims x times x instruments) so Q4 can reuse it without resimulation."
        )

    V = baseline_results["V"]  # (num_sims, num_times, num_instruments)
    V_portfolio = np.sum(V, axis=2)  # (num_sims, num_times)

    # ---------------------------------------------------
    # 2) (a) Variation Margin frequency stress: M = 1..60
    # ---------------------------------------------------
    vm_rows = []
    print(f"Average V0 (Portfolio NPV at t=0): {np.mean(V_portfolio[:, 0]):.2f} EUR")
    for M in range(1, 61):
        E_vm = apply_variation_margin(V_portfolio, M=M)   # (sims, times)
        EPE_vm = compute_epe(E_vm)                        # (times,)
        cva_vm, _ = compute_cva_from_epe(times, EPE_vm, config)

        vm_rows.append({
            "M_months": M,
            "CVA_VM": cva_vm,
            "CVA_change_EUR": cva_vm - baseline_cva_netted,
            "CVA_change_percent": (cva_vm - baseline_cva_netted) / baseline_cva_netted * 100
        })

    vm_df = pd.DataFrame(vm_rows)
    vm_df.to_csv(f"{base_path}/q4a_variation_margin_cva_by_frequency.csv", index=False)

    # Plot: CVA vs M
    plt.figure(figsize=(8, 5))
    plt.plot(vm_df["M_months"], vm_df["CVA_VM"])
    plt.axhline(baseline_cva_netted, linestyle="--", label="No-collateral baseline (Q2d)")
    plt.xlabel("Update Frequency M (months)")
    plt.ylabel("CVA")
    plt.title("Q4a: CVA vs Variation Margin Update Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_path}/q4a_cva_vs_vm_frequency.png")
    plt.close()

    # ---------------------------------------------------
    # 3) (b) Initial Margin stress: IM in {1M, 10M, 100M}
    # ---------------------------------------------------
    im_levels = [1e6, 1e7, 1e8]
    im_rows = []

    for IM in im_levels:
        E_im = apply_initial_margin(V_portfolio, IM=IM)   # (sims, times)
        EPE_im = compute_epe(E_im)
        cva_im, breakdown_im = compute_cva_from_epe(times, EPE_im, config)

        im_rows.append({
            "IM_EUR": IM,
            "CVA_IM": cva_im,
            "CVA_change_EUR": cva_im - baseline_cva_netted,
            "CVA_change_percent": (cva_im - baseline_cva_netted) / baseline_cva_netted * 100
        })

        # Save breakdown for each IM
        breakdown_im.to_csv(f"{base_path}/q4b_cva_breakdown_im_{int(IM):d}.csv", index=False)

    im_df = pd.DataFrame(im_rows)
    im_df.to_csv(f"{base_path}/q4b_initial_margin_cva.csv", index=False)

    # Plot: CVA vs IM (log x makes sense here)
    plt.figure(figsize=(8, 5))
    plt.plot(im_df["IM_EUR"], im_df["CVA_IM"], marker="o")
    plt.axhline(baseline_cva_netted, linestyle="--", label="No-initial-margin baseline (Q2d)")
    plt.xscale("log")
    plt.xlabel("Initial Margin IM (EUR, log scale)")
    plt.ylabel("CVA")
    plt.title("Q4b: CVA vs Initial Margin")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_path}/q4b_cva_vs_initial_margin.png")
    plt.close()

    # ---------------------------------------------------
    # 4) Save baseline reference
    # ---------------------------------------------------
    baseline_df = pd.DataFrame({
        "Baseline_CVA_no_collateral_Q2d": [baseline_cva_netted]
    })
    baseline_df.to_csv(f"{base_path}/q4_baseline_reference.csv", index=False)

    print("\nQ4 outputs saved to output/q4/")
    print(f"Baseline (no collateral) CVA: {baseline_cva_netted:.6f}")

    return {
        "baseline_cva": baseline_cva_netted,
        "vm_results": vm_df,
        "im_results": im_df
    }
