import time
from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator
from validation.orchestrator import run_question_1
from workflows.exposures_CVA_workflow import compute_cva_pipeline, run_q2_reporting
from workflows.sensitivity_analysis_workflow import run_q3_sensitivity_analysis
from workflows.collateral_impact_workflow import run_q4_collateral_impact
from utils.output_writer import OutputWriter


def main():
    start_time = time.time()
    
    # step 0: initializing parameters
    print("\n" + "="*60)
    print("[STEP 0] Initializing parameters...")
    step_start = time.time()
    data = initialize()
    elapsed = time.time() - step_start
    print(f"✓ Initialized in {elapsed:.2f}s")

    # step 1: simulate equity processes
    print("\n" + "="*60)
    print("[STEP 1] Simulating equity processes...")
    step_start = time.time()
    simulator = EquityProcessSimulator(data, data['portfolio'])
    equity_paths = simulator.simulate_paths()
    elapsed = time.time() - step_start
    print(f"✓ Simulation completed in {elapsed:.2f}s")

    # step 2: run validation tests
    print("\n" + "="*60)
    print("[STEP 2] Running validation tests...")
    step_start = time.time()
    validation_results = run_question_1(data, data['portfolio'])
    elapsed = time.time() - step_start
    print(f"✓ Validation completed in {elapsed:.2f}s")

    # step 3: run exposures and CVA workflow (BASELINE COMPUTATION)
    print("\n" + "="*60)
    print("[STEP 3] Computing baseline CVA...")
    step_start = time.time()
    baseline_results = compute_cva_pipeline(data, data['portfolio'])
    elapsed = time.time() - step_start
    print(f"✓ Baseline CVA computed in {elapsed:.2f}s")
    
    # step 3b: report Q2 outputs
    print("\n" + "="*60)
    print("[STEP 3b] Generating Q2 reports...")
    step_start = time.time()
    cva_results = run_q2_reporting(baseline_results, data['portfolio'])
    elapsed = time.time() - step_start
    print(f"✓ Q2 reports generated in {elapsed:.2f}s")
    
    # step 4: run sensitivity analysis (REUSES BASELINE)
    print("\n" + "="*60)
    print("[STEP 4] Running sensitivity analysis...")
    step_start = time.time()
    sensitivity_results = run_q3_sensitivity_analysis(data, data['portfolio'], baseline_results)
    elapsed = time.time() - step_start
    print(f"✓ Sensitivity analysis completed in {elapsed:.2f}s")
    
    # step 5: run collateral impact analysis (REUSES BASELINE)
    print("\n" + "="*60)
    print("[STEP 5] Running collateral impact analysis...")
    step_start = time.time()
    collateral_results = run_q4_collateral_impact(data, data['portfolio'], baseline_results)
    elapsed = time.time() - step_start
    print(f"✓ Collateral impact analysis completed in {elapsed:.2f}s")
    
    # Final summary
    total_elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"✓ ALL COMPLETE - Total execution time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("="*60)
    
    return data, equity_paths, validation_results, cva_results, sensitivity_results, collateral_results


if __name__ == "__main__":
    main()

