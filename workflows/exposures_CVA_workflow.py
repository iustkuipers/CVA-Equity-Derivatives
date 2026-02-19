import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.equity_processes import EquityProcessSimulator
from pricing.equity_instruments import value_portfolio_over_time
from cva.exposure import (
    compute_unnetted_exposure,
    compute_netted_exposure,
    compute_epe,
    compute_standalone_epe,
)
from cva.cva_calculator import compute_cva_from_epe
from utils.output_writer import OutputWriter


def compute_cva_pipeline(config, portfolio):
    """
    Pure computation engine — no prints, no file I/O.
    
    Returns dictionary with all computed results:
    - times, EPE_unnetted, EPE_netted, EPE_standalone
    - cva_unnetted, cva_netted, standalone_cvas
    - breakdown_unnetted, breakdown_netted
    """
    # ---------------------------------------------------
    # 1️⃣ Simulation
    # ---------------------------------------------------
    simulator = EquityProcessSimulator(config, portfolio)
    sim_data = simulator.simulate_paths()
    times = sim_data["times"]

    # ---------------------------------------------------
    # 2️⃣ Portfolio Valuation
    # ---------------------------------------------------
    V = value_portfolio_over_time(sim_data, portfolio, config)

    # DEBUG: Verify scaling is correct
    V_portfolio = np.sum(V, axis=2)
    print("\n[DEBUG] Valuation Scaling Check:")
    print(f"  Mean portfolio value at t=0: {np.mean(V_portfolio[:, 0]):,.0f} EUR")
    print(f"  Max portfolio value: {np.max(V_portfolio):,.0f} EUR")
    print(f"  Min portfolio value: {np.min(V_portfolio):,.0f} EUR")

    # ---------------------------------------------------
    # 3️⃣ Exposure Profiles
    # ---------------------------------------------------
    E_unnetted = compute_unnetted_exposure(V)
    E_netted = compute_netted_exposure(V)
    EPE_unnetted = compute_epe(E_unnetted)
    EPE_netted = compute_epe(E_netted)
    EPE_standalone = compute_standalone_epe(V)

    # ---------------------------------------------------
    # 4️⃣ CVA Calculations
    # ---------------------------------------------------
    cva_unnetted, breakdown_unnetted = compute_cva_from_epe(
        times, EPE_unnetted, config
    )
    cva_netted, breakdown_netted = compute_cva_from_epe(
        times, EPE_netted, config
    )

    # Standalone CVAs per instrument
    standalone_cvas = []
    for i in range(EPE_standalone.shape[1]):
        cva_i, _ = compute_cva_from_epe(times, EPE_standalone[:, i], config)
        standalone_cvas.append(cva_i)

    return {
        "times": times,
        "V": V,
        "EPE_unnetted": EPE_unnetted,
        "EPE_netted": EPE_netted,
        "EPE_standalone": EPE_standalone,
        "cva_unnetted": cva_unnetted,
        "cva_netted": cva_netted,
        "standalone_cvas": standalone_cvas,
        "breakdown_unnetted": breakdown_unnetted,
        "breakdown_netted": breakdown_netted,
    }


def run_q2_reporting(baseline_results, portfolio):
    """
    Q2 Reporting: takes pre-computed results and saves/plots them.
    Does NOT recompute (use compute_cva_pipeline for that).
    """
    print("Running Q2 - Exposure and CVA Workflow")
    
    times = baseline_results["times"]
    EPE_unnetted = baseline_results["EPE_unnetted"]
    EPE_netted = baseline_results["EPE_netted"]
    cva_unnetted = baseline_results["cva_unnetted"]
    cva_netted = baseline_results["cva_netted"]
    standalone_cvas = baseline_results["standalone_cvas"]
    breakdown_unnetted = baseline_results["breakdown_unnetted"]
    breakdown_netted = baseline_results["breakdown_netted"]

    # Build standalone CVA dataframe
    standalone_cva_results = []
    for i, instrument in portfolio.iterrows():
        standalone_cva_results.append({
            "instrument": f"{instrument['Type']} {instrument['Underlying']}",
            "cva": standalone_cvas[i]
        })
    standalone_cva_df = pd.DataFrame(standalone_cva_results)

    # ---------------------------------------------------
    # Netting Benefit
    # ---------------------------------------------------
    netting_benefit = cva_unnetted - cva_netted
    netting_benefit_pct = netting_benefit / cva_unnetted * 100

    # ---------------------------------------------------
    # Save Outputs
    # ---------------------------------------------------
    base_path = "output/q2"
    OutputWriter.ensure_directory(base_path)

    # Save EPE profiles
    pd.DataFrame({
        "time": times,
        "EPE_unnetted": EPE_unnetted,
        "EPE_netted": EPE_netted
    }).to_csv(f"{base_path}/epe_profiles.csv", index=False)

    standalone_cva_df.to_csv(f"{base_path}/standalone_cva.csv", index=False)

    breakdown_unnetted.to_csv(f"{base_path}/cva_breakdown_unnetted.csv", index=False)
    breakdown_netted.to_csv(f"{base_path}/cva_breakdown_netted.csv", index=False)

    summary_df = pd.DataFrame({
        "CVA_unnetted": [cva_unnetted],
        "CVA_netted": [cva_netted],
        "Netting_benefit_EUR": [netting_benefit],
        "Netting_benefit_percent": [netting_benefit_pct]
    })

    summary_df.to_csv(f"{base_path}/cva_summary.csv", index=False)

    # ---------------------------------------------------
    # Plot EPE Profiles
    # ---------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(times, EPE_unnetted, label="Unnetted EPE")
    plt.plot(times, EPE_netted, label="Netted EPE")
    plt.xlabel("Time (Years)")
    plt.ylabel("Expected Positive Exposure")
    plt.title("EPE Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_path}/epe_plot.png")
    plt.close()

    print("Q2 Results Saved to output/q2/")

    return {
        "cva_unnetted": cva_unnetted,
        "cva_netted": cva_netted,
        "netting_benefit": netting_benefit,
        "netting_benefit_pct": netting_benefit_pct
    }


def run_q2_workflow(config, portfolio):
    """
    Q2 Workflow wrapper: calls engine, then handles reporting (prints, files, plots).
    Kept for backward compatibility. Use compute_cva_pipeline + run_q2_reporting for efficiency.
    """
    print("Running Q2 - Exposure and CVA Workflow")

    # Call pure compute engine
    results = compute_cva_pipeline(config, portfolio)
    
    times = results["times"]
    EPE_unnetted = results["EPE_unnetted"]
    EPE_netted = results["EPE_netted"]
    EPE_standalone = results["EPE_standalone"]
    cva_unnetted = results["cva_unnetted"]
    cva_netted = results["cva_netted"]
    standalone_cvas = results["standalone_cvas"]
    breakdown_unnetted = results["breakdown_unnetted"]
    breakdown_netted = results["breakdown_netted"]

    # Build standalone CVA dataframe
    standalone_cva_results = []
    for i, instrument in portfolio.iterrows():
        standalone_cva_results.append({
            "instrument": f"{instrument['Type']} {instrument['Underlying']}",
            "cva": standalone_cvas[i]
        })
    standalone_cva_df = pd.DataFrame(standalone_cva_results)

    # ---------------------------------------------------
    # 5️⃣ Netting Benefit
    # ---------------------------------------------------
    netting_benefit = cva_unnetted - cva_netted
    netting_benefit_pct = netting_benefit / cva_unnetted * 100

    # ---------------------------------------------------
    # 6️⃣ Save Outputs
    # ---------------------------------------------------
    base_path = "output/q2"
    OutputWriter.ensure_directory(base_path)

    # Save EPE profiles
    pd.DataFrame({
        "time": times,
        "EPE_unnetted": EPE_unnetted,
        "EPE_netted": EPE_netted
    }).to_csv(f"{base_path}/epe_profiles.csv", index=False)

    standalone_cva_df.to_csv(f"{base_path}/standalone_cva.csv", index=False)

    breakdown_unnetted.to_csv(f"{base_path}/cva_breakdown_unnetted.csv", index=False)
    breakdown_netted.to_csv(f"{base_path}/cva_breakdown_netted.csv", index=False)

    summary_df = pd.DataFrame({
        "CVA_unnetted": [cva_unnetted],
        "CVA_netted": [cva_netted],
        "Netting_benefit_EUR": [netting_benefit],
        "Netting_benefit_percent": [netting_benefit_pct]
    })

    summary_df.to_csv(f"{base_path}/cva_summary.csv", index=False)

    # ---------------------------------------------------
    # 7️⃣ Plot EPE Profiles
    # ---------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(times, EPE_unnetted, label="Unnetted EPE")
    plt.plot(times, EPE_netted, label="Netted EPE")
    plt.xlabel("Time (Years)")
    plt.ylabel("Expected Positive Exposure")
    plt.title("EPE Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_path}/epe_plot.png")
    plt.close()

    print("Q2 Results Saved to output/q2/")

    return {
        "cva_unnetted": cva_unnetted,
        "cva_netted": cva_netted,
        "netting_benefit": netting_benefit,
        "netting_benefit_pct": netting_benefit_pct
    }
