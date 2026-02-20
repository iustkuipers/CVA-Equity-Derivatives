"""
Run Q5 (credit hedging workflow) only to test the CDS sensitivity fix.
"""

import sys
import time
from data.orchestrator import initialize
from workflows.credit_hedging_workflow import run_q5_credit_hedging

print("=" * 80)
print("Q5 CREDIT HEDGING - CDS SENSITIVITY FIX TEST")
print("=" * 80)

# Setup
print("Initializing parameters...")
data = initialize()
print(f"  Hazard buckets: {list(data.get('forward_hazard_rates', {}).keys())}")

# Run baseline first (assumes Q1-Q4 results exist)
# The workflow expects baseline_results from Q1-Q4
print("\nLoading baseline results from main.py execution...")
import pickle
try:
    with open("output/baseline_results.pkl", "rb") as f:
        baseline_results = pickle.load(f)
    print(f"  Loaded baseline: keys = {list(baseline_results.keys())}")
except FileNotFoundError:
    print("  ERROR: baseline_results.pkl not found. Run main.py first.")
    sys.exit(1)

# Run Q5
print("\nRunning Q5 credit hedging workflow...")
start = time.time()
hedge_results = run_q5_credit_hedging(data, baseline_results)
elapsed = time.time() - start

print(f"\nQ5 completed in {elapsed:.2f} seconds")
print("\n" + "=" * 80)
print("Q5 RESULTS SUMMARY")
print("=" * 80)

# Show CDS sensitivities
print("\nCDS Sensitivities (ΔV per bucket):")
for bucket, maturities_dict in hedge_results['delta_cds'].items():
    print(f"\n  Bucket {bucket}:")
    for T, delta in maturities_dict.items():
        print(f"    {T}Y: {delta:.8f} EUR")

# Show hedge notionals
print("\n\nHedge Notionals (Q5d delta hedge):")
for T, notional in hedge_results['hedge_notionals'].items():
    print(f"  {T}Y: {notional:.2e} EUR")

print(f"\nResidual: {hedge_results.get('residual', 'N/A')}")

print("\n✓ Q5 computation complete with CDS sensitivity fix applied.")
