import copy
import pandas as pd
import matplotlib.pyplot as plt

from workflows.exposures_CVA_workflow import compute_cva_pipeline
from utils.output_writer import OutputWriter


def run_q3_sensitivity_analysis(config, portfolio, baseline_results):
    """
    Q3 Sensitivity Analysis: uses pre-computed baseline and only recomputes stress scenarios.
    
    Args:
        config: configuration dict
        portfolio: portfolio dataframe
        baseline_results: pre-computed results from compute_cva_pipeline (baseline scenario)
    """
    print("===================================================")
    print("Running Q3 - Sensitivity Analysis")
    print("===================================================")

    base_path = "output/q3"
    OutputWriter.ensure_directory(base_path)

    # ---------------------------------------------------
    # 1️⃣ Use Baseline (already computed once in main)
    # ---------------------------------------------------
    print("\nUsing Baseline Scenario (pre-computed)...")
    baseline_cva = baseline_results["cva_netted"]

    # ---------------------------------------------------
    # 2️⃣ Volatility Stress (30%)
    # ---------------------------------------------------
    print("\nRunning Volatility Stress (σ = 30%)...")

    vol_config = copy.deepcopy(config)
    vol_config["volatility_sx5e"] = 0.30
    vol_config["volatility_aex"] = 0.30

    vol_results = compute_cva_pipeline(vol_config, portfolio)
    vol_cva = vol_results["cva_netted"]

    # ---------------------------------------------------
    # 3️⃣ Correlation Stress (ρ = 40%)
    # ---------------------------------------------------
    print("\nRunning Correlation Stress (ρ = 40%)...")

    corr_config = copy.deepcopy(config)
    corr_config["correlation"] = 0.40

    corr_results = compute_cva_pipeline(corr_config, portfolio)
    corr_cva = corr_results["cva_netted"]

    # ---------------------------------------------------
    # 4️⃣ Build Summary Table
    # ---------------------------------------------------
    summary_df = pd.DataFrame({
        "Scenario": ["Baseline", "Volatility 30%", "Correlation 40%"],
        "CVA": [baseline_cva, vol_cva, corr_cva],
    })

    summary_df["Change_EUR"] = summary_df["CVA"] - baseline_cva
    summary_df["Change_percent"] = summary_df["Change_EUR"] / baseline_cva * 100

    summary_df.to_csv(f"{base_path}/q3_cva_sensitivity_summary.csv", index=False)

    # ---------------------------------------------------
    # 5️⃣ Plot CVA Comparison
    # ---------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["Scenario"], summary_df["CVA"])
    plt.ylabel("CVA")
    plt.title("CVA Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(f"{base_path}/q3_cva_comparison.png")
    plt.close()

    print("\nSensitivity Results:")
    print(summary_df)

    print("\nResults saved to output/q3/")

    return summary_df
