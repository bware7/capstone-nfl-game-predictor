"""
Quick test script to verify NFL predictor setup
Run this after installation to ensure everything works
"""

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        print("✓ pandas imported")

        import numpy as np
        print("✓ numpy imported")

        import sklearn
        print("✓ scikit-learn imported")

        import nfl_data_py as nfl
        print("✓ nfl_data_py imported")

        import matplotlib.pyplot as plt
        print("✓ matplotlib imported")

        import seaborn as sns
        print("✓ seaborn imported")

        print("\nAll dependencies installed correctly!")
        return True
    except ImportError as e:
        print(f"✗ Error importing: {e}")
        return False

def test_nfl_predictor():
    """Test the nfl_predictor package"""
    print("\nTesting nfl_predictor package...")
    try:
        import sys
        sys.path.append('src')
        from nfl_predictor import collect_nfl_data, clean_nfl_data, build_prediction_model
        print("✓ nfl_predictor package imported successfully")

        # Test data collection (small sample)
        print("\nTesting data collection (this may take a moment)...")
        games = collect_nfl_data([2024])
        print(f"✓ Collected {len(games)} games from 2024 season")

        return True
    except Exception as e:
        print(f"✗ Error with nfl_predictor: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("NFL GAME PREDICTOR - SYSTEM TEST")
    print("="*50)

    # Test dependencies
    deps_ok = test_imports()

    # Test package
    if deps_ok:
        package_ok = test_nfl_predictor()

        if package_ok:
            print("\n" + "="*50)
            print("✓ ALL TESTS PASSED - System ready!")
            print("="*50)
        else:
            print("\n⚠ Package tests failed - check installation")
    else:
        print("\n⚠ Dependency tests failed - run: pip install -r requires.txt")
