"""
tests/conftest.py

Pytest configuration for setting up import paths and shared fixtures.
"""

import sys
import os

# Add the workspace root to sys.path so imports work correctly
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
