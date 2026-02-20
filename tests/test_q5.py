"""
Test script for Q5 workflow only
"""
import sys
sys.path.insert(0, '.')

from data.orchestrator import initialize
from workflows.exposures_CVA_workflow import compute_cva_pipeline
from workflows.credit_hedging_workflow import run_q5_credit_hedging

print("[TEST] Initializing data...")
data = initialize()

print("[TEST] Computing baseline CVA...")
baseline_results = compute_cva_pipeline(data, data['portfolio'])

print("[TEST] Running Q5 credit hedging...")
try:
    hedge_results = run_q5_credit_hedging(data, baseline_results)
    print("\n✓ Q5 PASSED!")
    print(f"  CVA sensitivities computed: {len(hedge_results['delta_cva'])} buckets")
    print(f"  CDS sensitivities computed: {len(hedge_results['delta_cds'])} buckets")
    print(f"  Hedge notionals: {hedge_results['hedge_notionals']}")
    print(f"  Residual: {hedge_results['residual']}")
except Exception as e:
    print(f"\n✗ Q5 FAILED: {e}")
    import traceback
    traceback.print_exc()
