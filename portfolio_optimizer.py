import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime
import os
import json

class PortfolioOptimizer:
    def __init__(self, tickers, start_date='2018-01-01', end_date='2023-12-31', risk_free_rate=0.05):
        """
        Initialize the PortfolioOptimizer with stock tickers and date range.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers (e.g., ['AAPL', 'XOM'])
        start_date : str
            Start date for historical data
        end_date : str
            End date for historical data
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.cor_matrix = None
        self.mean_returns = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_{self.run_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def download_data(self):
        """Download historical price data from Yahoo Finance."""
        print(f"Downloading data for {', '.join(self.tickers)}...")
        # Download data for each ticker separately and combine
        data_list = []
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=self.start_date, end=self.end_date)
                data_list.append(stock_data['Close'].rename(ticker))
            except Exception as e:
                print(f"Error downloading data for {ticker}: {str(e)}")
                raise
        
        self.data = pd.concat(data_list, axis=1)
        self.returns = self.data.pct_change().dropna()
        self.cov_matrix = self.returns.cov() * 252  # Annualized covariance
        self.cor_matrix = self.returns.corr()
        self.mean_returns = self.returns.mean() * 252  # Annualized returns
        
    def calculate_portfolio_stats(self, weights):
        """
        Calculate portfolio statistics for given weights.
        
        Parameters:
        -----------
        weights : np.array
            Portfolio weights
            
        Returns:
        --------
        tuple : (portfolio_return, portfolio_volatility, sharpe_ratio)
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def monte_carlo_simulation(self, num_portfolios=10000):
        """
        Perform Monte Carlo simulation to generate random portfolios.
        
        Parameters:
        -----------
        num_portfolios : int
            Number of random portfolios to generate
            
        Returns:
        --------
        DataFrame : Results of Monte Carlo simulation
        """
        results = np.zeros((num_portfolios, 3))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_stats(weights)
            results[i, :] = [portfolio_return, portfolio_volatility, sharpe_ratio]
        
        results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])
        results_df['Weights'] = weights_record
        return results_df
    
    def optimize_portfolio(self, min_weight=0.2):
        """
        Optimize portfolio for maximum Sharpe ratio with minimum weight constraints.
        
        Parameters:
        -----------
        min_weight : float
            Minimum weight for each asset (for diversification)
            
        Returns:
        --------
        dict : Optimization results
        """
        def neg_sharpe(weights):
            portfolio_return, portfolio_volatility, _ = self.calculate_portfolio_stats(weights)
            return -((portfolio_return - self.risk_free_rate) / portfolio_volatility)
        
        # Constraints: weights sum to 1 and each weight >= min_weight
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        bounds = tuple((min_weight, 1) for _ in range(len(self.tickers)))
        initial_weights = np.array([1/len(self.tickers)] * len(self.tickers))
        
        optimal = minimize(neg_sharpe, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return {
            'weights': optimal.x.tolist(),
            'return': float(np.dot(optimal.x, self.mean_returns)),
            'volatility': float(np.sqrt(np.dot(optimal.x.T, np.dot(self.cov_matrix, optimal.x)))),
            'sharpe': float(-optimal.fun)
        }
    
    def plot_efficient_frontier(self, results_df, optimal_portfolio):
        """
        Plot the efficient frontier and highlight the optimal portfolio.
        
        Parameters:
        -----------
        results_df : DataFrame
            Results from Monte Carlo simulation
        optimal_portfolio : dict
            Optimal portfolio statistics
        """
        plt.figure(figsize=(12, 8))
        plt.scatter(results_df['Volatility'], results_df['Return'], 
                   c=results_df['Sharpe'], cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], 
                   c='red', s=200, marker='*', label='Optimal Portfolio')
        
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(self.results_dir, 'efficient_frontier.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def save_results(self, results_df, optimal_portfolio, plot_path):
        """Save all results to a markdown file."""
        md_content = f"""# Portfolio Optimization Results - {self.run_id}

## Portfolio Details
- **Assets**: {', '.join(self.tickers)}
- **Date Range**: {self.start_date} to {self.end_date}
- **Risk-Free Rate**: {self.risk_free_rate:.2%}

## Correlation Matrix
```
{self.cor_matrix.to_string()}
```

## Covariance Matrix
```
{self.cov_matrix.to_string()}
```

## Optimal Portfolio
- **Weights**:
"""
        for ticker, weight in zip(self.tickers, optimal_portfolio['weights']):
            md_content += f"  - {ticker}: {weight:.2%}\n"
        
        md_content += f"""
- **Expected Return**: {optimal_portfolio['return']:.2%}
- **Volatility**: {optimal_portfolio['volatility']:.2%}
- **Sharpe Ratio**: {optimal_portfolio['sharpe']:.2f}

## Efficient Frontier Plot
![Efficient Frontier](efficient_frontier.png)

## Procedure
1. Downloaded historical price data for {', '.join(self.tickers)}
2. Calculated daily returns and annualized statistics
3. Performed Monte Carlo simulation with {len(results_df)} random portfolios
4. Optimized portfolio weights to maximize Sharpe ratio
5. Generated efficient frontier plot showing:
   - Scatter points representing different portfolio combinations
   - Color gradient indicating Sharpe ratio
   - Red star marking the optimal portfolio (maximum Sharpe ratio)
"""
        
        # Save markdown file
        with open(os.path.join(self.results_dir, 'results.md'), 'w') as f:
            f.write(md_content)
        
        # Save raw data as JSON
        results_data = {
            'tickers': self.tickers,
            'date_range': {'start': self.start_date, 'end': self.end_date},
            'risk_free_rate': self.risk_free_rate,
            'correlation_matrix': self.cor_matrix.to_dict(),
            'covariance_matrix': self.cov_matrix.to_dict(),
            'optimal_portfolio': optimal_portfolio,
            'monte_carlo_results': {
                'return': results_df['Return'].tolist(),
                'volatility': results_df['Volatility'].tolist(),
                'sharpe': results_df['Sharpe'].tolist()
            }
        }
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results_data, f, indent=4)

def main():
    # Define different stock pairs to analyze
    stock_pairs = [
        ['AAPL', 'XOM'],    # Tech vs Energy
        ['MSFT', 'JNJ'],    # Tech vs Healthcare
        ['GOOGL', 'PG'],    # Tech vs Consumer Staples
        ['AMZN', 'KO'],     # Tech vs Consumer Staples
        ['META', 'WMT'],    # Tech vs Retail
    ]
    
    base_dir = "portfolio_analysis"
    os.makedirs(base_dir, exist_ok=True)
    
    for tickers in stock_pairs:
        # Create folder name from tickers
        pair_name = f"{tickers[0]}_{tickers[1]}"
        optimizer = PortfolioOptimizer(tickers)
        optimizer.results_dir = os.path.join(base_dir, f"{pair_name}_{optimizer.run_id}")
        os.makedirs(optimizer.results_dir, exist_ok=True)
        
        print(f"\nAnalyzing portfolio: {pair_name}")
        
        try:
            # Download and process data
            optimizer.download_data()
            
            # Run Monte Carlo simulation
            results_df = optimizer.monte_carlo_simulation()
            
            # Find optimal portfolio with minimum 20% weight per asset
            optimal_portfolio = optimizer.optimize_portfolio(min_weight=0.2)
            
            # Plot results and get plot path
            plot_path = optimizer.plot_efficient_frontier(results_df, optimal_portfolio)
            
            # Save all results
            optimizer.save_results(results_df, optimal_portfolio, plot_path)
            
            print(f"Results saved in directory: {optimizer.results_dir}")
            
        except Exception as e:
            print(f"Error analyzing {pair_name}: {str(e)}")
            continue
    
    print("\nAnalysis complete. Please check the portfolio_analysis directory for results.")

if __name__ == "__main__":
    main() 