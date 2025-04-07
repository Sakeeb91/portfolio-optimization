import unittest
import os
import sys
from test_portfolio_optimizer import TestPortfolioOptimizer
from portfolio_optimizer import main

def run_tests():
    """Run all unit tests"""
    print("Running unit tests...")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPortfolioOptimizer)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    if not result.wasSuccessful():
        print("\nTests failed. Please fix the issues before proceeding.")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")

if __name__ == "__main__":
    # Run tests first
    run_tests()
    
    # If tests pass, run the main analysis
    print("\nStarting portfolio optimization analysis...")
    main() 