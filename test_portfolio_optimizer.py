import unittest
import numpy as np
import pandas as pd
from portfolio_optimizer import PortfolioOptimizer
import os
from datetime import datetime

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.tickers = ['AAPL', 'XOM']
        self.optimizer = PortfolioOptimizer(self.tickers)
        self.optimizer.download_data()
        
    def test_initialization(self):
        """Test if the optimizer initializes correctly"""
        self.assertEqual(self.optimizer.tickers, self.tickers)
        self.assertIsNotNone(self.optimizer.risk_free_rate)
        self.assertIsNotNone(self.optimizer.start_date)
        self.assertIsNotNone(self.optimizer.end_date)
        
    def test_data_download(self):
        """Test if data is downloaded correctly"""
        self.assertIsNotNone(self.optimizer.data)
        self.assertIsInstance(self.optimizer.data, pd.DataFrame)
        self.assertEqual(len(self.optimizer.data.columns), len(self.tickers))
        
    def test_returns_calculation(self):
        """Test if returns are calculated correctly"""
        self.assertIsNotNone(self.optimizer.returns)
        self.assertIsInstance(self.optimizer.returns, pd.DataFrame)
        self.assertFalse(self.optimizer.returns.isnull().any().any())
        
    def test_covariance_matrix(self):
        """Test if covariance matrix is calculated correctly"""
        self.assertIsNotNone(self.optimizer.cov_matrix)
        self.assertIsInstance(self.optimizer.cov_matrix, pd.DataFrame)
        self.assertEqual(self.optimizer.cov_matrix.shape, (len(self.tickers), len(self.tickers)))
        
    def test_correlation_matrix(self):
        """Test if correlation matrix is calculated correctly"""
        self.assertIsNotNone(self.optimizer.cor_matrix)
        self.assertIsInstance(self.optimizer.cor_matrix, pd.DataFrame)
        self.assertEqual(self.optimizer.cor_matrix.shape, (len(self.tickers), len(self.tickers)))
        # Check if correlation values are between -1 and 1
        self.assertTrue((self.optimizer.cor_matrix.values >= -1).all())
        self.assertTrue((self.optimizer.cor_matrix.values <= 1).all())
        
    def test_portfolio_stats(self):
        """Test if portfolio statistics are calculated correctly"""
        weights = np.array([0.5, 0.5])
        return_, volatility, sharpe = self.optimizer.calculate_portfolio_stats(weights)
        
        self.assertIsInstance(return_, float)
        self.assertIsInstance(volatility, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreaterEqual(volatility, 0)  # Volatility should be non-negative
        
    def test_monte_carlo_simulation(self):
        """Test if Monte Carlo simulation works correctly"""
        num_portfolios = 100
        results_df = self.optimizer.monte_carlo_simulation(num_portfolios)
        
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), num_portfolios)
        self.assertTrue(all(col in results_df.columns for col in ['Return', 'Volatility', 'Sharpe', 'Weights']))
        
    def test_optimization(self):
        """Test if portfolio optimization works correctly"""
        optimal = self.optimizer.optimize_portfolio()
        
        self.assertIsInstance(optimal, dict)
        self.assertTrue(all(key in optimal for key in ['weights', 'return', 'volatility', 'sharpe']))
        self.assertAlmostEqual(sum(optimal['weights']), 1.0)  # Weights should sum to 1
        self.assertTrue(all(0 <= w <= 1 for w in optimal['weights']))  # Weights should be between 0 and 1

if __name__ == '__main__':
    unittest.main() 