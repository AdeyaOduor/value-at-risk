import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
import logging
from typing import List, Tuple, Dict, Optional
import json
import os

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for pandas_datareader Yahoo issues (using yfinance instead)
try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    logger.error("yfinance not installed. Install with: pip install yfinance")
    YAHOO_AVAILABLE = False
    yf = None

class PortfolioVaRAnalyzer:
    """Portfolio Value at Risk (VaR) Calculator"""
    
    def __init__(self, tickers: List[str], weights: np.ndarray, 
                 initial_investment: float = 1000000):
        """
        Initialize portfolio analyzer
        
        Args:
            tickers: List of stock tickers
            weights: Portfolio weights (must sum to 1)
            initial_investment: Initial investment amount
        """
        self.tickers = tickers
        self.weights = self._validate_weights(weights)
        self.initial_investment = initial_investment
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.port_mean = None
        self.port_stdev = None
        
    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """Validate and normalize portfolio weights"""
        if not np.isclose(weights.sum(), 1.0, rtol=1e-5):
            logger.warning(f"Weights sum to {weights.sum():.4f}, normalizing to 1.0")
            weights = weights / weights.sum()
        return weights
    
    def fetch_data(self, start_date: str = "2018-01-01", 
                   end_date: Optional[str] = None) -> bool:
        """
        Fetch stock price data
        
        Args:
            start_date: Start date for data
            end_date: End date for data (defaults to today)
            
        Returns:
            Boolean indicating success
        """
        if not YAHOO_AVAILABLE:
            logger.error("yfinance is required for data fetching")
            return False
            
        if end_date is None:
            end_date = dt.date.today().strftime("%Y-%m-%d")
        
        try:
            logger.info(f"Fetching data for {self.tickers} from {start_date} to {end_date}")
            
            # Use yfinance directly (more reliable than pandas_datareader)
            data = yf.download(
                self.tickers, 
                start=start_date, 
                end=end_date,
                progress=False
            )['Close']
            
            if data.empty:
                logger.error("No data returned from Yahoo Finance")
                return False
                
            # Handle missing data
            self.data = data.ffill().bfill()  # Forward then backward fill
            missing_pct = self.data.isnull().mean().mean() * 100
            if missing_pct > 5:
                logger.warning(f"High percentage of missing data: {missing_pct:.2f}%")
                
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return False
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate periodic returns from price data"""
        if self.data is None:
            logger.error("No data available. Fetch data first.")
            return pd.DataFrame()
        
        self.returns = self.data.pct_change().dropna()
        return self.returns
    
    def calculate_portfolio_stats(self) -> Tuple[float, float]:
        """Calculate portfolio mean return and standard deviation"""
        if self.returns is None:
            self.calculate_returns()
        
        if self.returns.empty:
            logger.error("Returns data is empty")
            return 0.0, 0.0
        
        # Calculate covariance matrix
        self.cov_matrix = self.returns.cov()
        
        # Calculate mean returns
        avg_rets = self.returns.mean()
        
        # Portfolio statistics
        self.port_mean = avg_rets.dot(self.weights)
        self.port_stdev = np.sqrt(self.weights.T @ self.cov_matrix @ self.weights)
        
        logger.info(f"Portfolio mean: {self.port_mean:.6f}, stdev: {self.port_stdev:.6f}")
        return self.port_mean, self.port_stdev
    
    def calculate_var(self, confidence_level: float = 0.95, 
                      time_horizon_days: int = 1) -> Dict:
        """
        Calculate Value at Risk
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days
            
        Returns:
            Dictionary with VaR results
        """
        if self.port_mean is None or self.port_stdev is None:
            self.calculate_portfolio_stats()
        
        # Convert confidence level to cutoff percentile
        cutoff_percentile = 1 - confidence_level
        
        # Calculate statistics for initial investment
        mean_investment = (1 + self.port_mean) * self.initial_investment
        stdev_investment = self.initial_investment * self.port_stdev
        
        # Calculate cutoff value
        cutoff = norm.ppf(cutoff_percentile, mean_investment, stdev_investment)
        
        # Calculate 1-day VaR
        var_1d = self.initial_investment - cutoff
        
        # Calculate multi-day VaR using square root of time rule
        var_nd = var_1d * np.sqrt(time_horizon_days)
        
        results = {
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon_days,
            'var_1day': var_1d,
            'var_nday': var_nd,
            'mean_investment': mean_investment,
            'stdev_investment': stdev_investment,
            'portfolio_mean_return': self.port_mean,
            'portfolio_stdev': self.port_stdev
        }
        
        return results
    
    def calculate_var_series(self, max_days: int = 15, 
                            confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate VaR for multiple time horizons"""
        # Get 1-day VaR first
        var_results = self.calculate_var(confidence_level, 1)
        var_1d = var_results['var_1day']
        
        # Calculate for multiple days
        days = range(1, max_days + 1)
        var_values = [var_1d * np.sqrt(day) for day in days]
        
        return pd.DataFrame({
            'days': list(days),
            'var': var_values,
            'confidence_level': confidence_level
        })
    
    def plot_var_over_time(self, max_days: int = 15, 
                          confidence_level: float = 0.95,
                          save_path: Optional[str] = None):
        """Plot VaR over time horizon"""
        var_series = self.calculate_var_series(max_days, confidence_level)
        
        plt.figure(figsize=(10, 6))
        plt.plot(var_series['days'], var_series['var'], 'r-', linewidth=2)
        plt.fill_between(var_series['days'], 0, var_series['var'], 
                        alpha=0.3, color='red')
        plt.xlabel("Time Horizon (Days)", fontsize=12)
        plt.ylabel(f"VaR at {confidence_level*100:.0f}% Confidence (USD)", fontsize=12)
        plt.title(f"Portfolio Value at Risk Over {max_days}-Day Period", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (day, var_val) in enumerate(zip(var_series['days'], var_series['var'])):
            if i % 3 == 0 or i == len(var_series) - 1:  # Label every 3rd point plus last
                plt.annotate(f'${var_val:,.0f}', 
                           (day, var_val),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, ticker: str, 
                                 save_path: Optional[str] = None):
        """Plot returns distribution for a specific ticker"""
        if self.returns is None:
            self.calculate_returns()
        
        if ticker not in self.returns.columns:
            logger.error(f"Ticker {ticker} not in portfolio")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of returns
        plt.hist(self.returns[ticker].dropna(), bins=40, 
                density=True, alpha=0.5, 
                label=f'{ticker} Returns', color='steelblue')
        
        # Overlay normal distribution
        ticker_mean = self.returns[ticker].mean()
        ticker_std = self.returns[ticker].std()
        x = np.linspace(ticker_mean - 4*ticker_std, 
                       ticker_mean + 4*ticker_std, 1000)
        plt.plot(x, norm.pdf(x, ticker_mean, ticker_std), 
                'r-', linewidth=2, label='Normal Distribution')
        
        plt.xlabel("Daily Returns", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(f"{ticker} Returns Distribution vs. Normal Distribution", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (f"Mean: {ticker_mean:.4f}\n"
                     f"Std Dev: {ticker_std:.4f}\n"
                     f"Skew: {self.returns[ticker].skew():.4f}\n"
                     f"Kurtosis: {self.returns[ticker].kurtosis():.4f}")
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, confidence_levels: List[float] = [0.95, 0.99],
                       time_horizons: List[int] = [1, 5, 10, 15]) -> Dict:
        """Generate comprehensive VaR report"""
        report = {
            'portfolio': {
                'tickers': self.tickers,
                'weights': self.weights.tolist(),
                'initial_investment': self.initial_investment
            },
            'statistics': {
                'portfolio_mean': float(self.port_mean),
                'portfolio_stdev': float(self.port_stdev)
            },
            'var_results': {}
        }
        
        for conf_level in confidence_levels:
            report['var_results'][f'conf_{int(conf_level*100)}'] = {}
            for days in time_horizons:
                var_result = self.calculate_var(conf_level, days)
                report['var_results'][f'conf_{int(conf_level*100)}'][f'{days}_day'] = {
                    'var': float(var_result['var_nday']),
                    'var_as_percent': float(var_result['var_nday'] / self.initial_investment)
                }
        
        return report

# Example usage
def main():
    """Example usage of the PortfolioVaRAnalyzer"""
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # More liquid stocks
    weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
    initial_investment = 1000000
    
    # Initialize analyzer
    analyzer = PortfolioVaRAnalyzer(tickers, weights, initial_investment)
    
    # Fetch data
    if not analyzer.fetch_data(start_date="2020-01-01"):
        logger.error("Failed to fetch data. Exiting.")
        return
    
    # Calculate portfolio statistics
    analyzer.calculate_portfolio_stats()
    
    # Calculate and print 1-day VaR at 95% confidence
    var_results = analyzer.calculate_var(confidence_level=0.95, time_horizon_days=1)
    print("\n" + "="*50)
    print("PORTFOLIO VALUE AT RISK ANALYSIS")
    print("="*50)
    print(f"Portfolio Value: ${initial_investment:,.2f}")
    print(f"1-Day VaR at 95% Confidence: ${var_results['var_1day']:,.2f}")
    print(f"Potential Loss: {var_results['var_1day']/initial_investment*100:.2f}%")
    
    # Generate VaR series
    print("\n" + "-"*50)
    print("VaR OVER TIME (95% CONFIDENCE)")
    print("-"*50)
    var_series = analyzer.calculate_var_series(max_days=15, confidence_level=0.95)
    for _, row in var_series.iterrows():
        print(f"{row['days']:2d} day VaR: ${row['var']:,.2f}")
    
    # Plot results
    analyzer.plot_var_over_time(max_days=15, confidence_level=0.95,
                              save_path="var_over_time.png")
    
    # Plot returns distribution for first ticker
    analyzer.plot_returns_distribution(tickers[0], 
                                     save_path=f"{tickers[0]}_returns_dist.png")
    
    # Generate comprehensive report
    report = analyzer.generate_report()
    
    # Save report to JSON
    with open('var_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*50)
    print("Analysis complete. Reports saved.")
    print("="*50)

# Quick deployment version
analyzer = PortfolioVaRAnalyzer(['AAPL', 'MSFT'], [0.6, 0.4], 500000)
analyzer.fetch_data(start_date="2022-01-01")
results = analyzer.calculate_var(confidence_level=0.99, time_horizon_days=10)
print(f"10-Day VaR at 99%: ${results['var_nday']:,.2f}")

if __name__ == "__main__":
    main()
