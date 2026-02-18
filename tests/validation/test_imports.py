import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("Testing imports...")
try:
    from data.orchestrator import initialize
    print("✓ data.orchestrator imported")
    
    from models.equity_processes import EquityProcessSimulator
    print("✓ models.equity_processes imported")
    
    print("\nInitializing data...")
    data = initialize()
    print(f"✓ Data initialized with keys: {list(data.keys())}")
    
    print("\nTest passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
