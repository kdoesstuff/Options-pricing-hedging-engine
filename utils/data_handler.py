"""
Data Handler Module

This module handles all data fetching, processing, and volatility calculations.
Integrates with yfinance for real-time market data and provides utility functions
for financial data analysis.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import warnings

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class DataHandler:
    """
    Main class for handling financial data operations.
    """
    
    def __init__(self):
        """Initialize DataHandler with default parameters."""
        self.data_cache = {}
        self.volatility_cache = {}
    
    def fetch_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol (e.g., 'AAPL', 'MSFT')
        period : str
            Time period for data ('1y', '2y', '5y', 'max')
            
        Returns:
        --------
        pd.DataFrame
            Historical stock data with OHLCV columns
        """
        try:
            # Check cache first
            cache_key = f"{ticker}_{period}"
            if cache_key in self.data_cache:
                print(f"Using cached data for {ticker}")
                return self.data_cache[cache_key]
            
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)

            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Cache the data
            self.data_cache[cache_key] = data

            print(f"Successfully fetched {len(data)} days of data for {ticker}")
            return data

        except Exception as e:
            # Live download failed (offline, or Yahoo rate-limiting, which is
            # common on cloud hosts). Fall back to the bundled snapshot CSVs
            # so the app keeps working end-to-end.
            fallback = self._load_bundled_csv(ticker)
            if fallback is not None:
                print(f"Live fetch failed ({e}); using bundled snapshot for {ticker}.")
                self.data_cache[cache_key] = fallback
                return fallback
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")

    @staticmethod
    def _load_bundled_csv(ticker: str) -> Optional[pd.DataFrame]:
        """Load a bundled 1-year snapshot CSV for the ticker, if one exists.

        Snapshots live in <project root>/data/real_data/<TICKER>_1y.csv and
        cover AAPL, MSFT, GOOGL, TSLA and AMZN. Returns None when no snapshot
        is available.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, 'data', 'real_data', f"{ticker.upper()}_1y.csv")
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.set_index('Date').sort_index()
            # Harmonize column names with yfinance output
            if 'Stock_Splits' in df.columns:
                df = df.rename(columns={'Stock_Splits': 'Stock Splits'})
            return df
        except Exception:
            return None
    
    def get_current_price(self, ticker: str) -> float:
        """
        Get the most recent closing price for a stock.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
            
        Returns:
        --------
        float
            Current stock price
        """
        try:
            data = self.fetch_stock_data(ticker, period="5d")
            current_price = data['Close'].iloc[-1]
            return float(current_price)
        except Exception as e:
            raise Exception(f"Error getting current price for {ticker}: {str(e)}")
    
    def calculate_historical_volatility(self, data: pd.DataFrame, 
                                      window: int = 252) -> float:
        """
        Calculate annualized historical volatility using log returns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock price data with 'Close' column
        window : int
            Number of trading days for volatility calculation (default: 252 = 1 year)
            
        Returns:
        --------
        float
            Annualized historical volatility (sigma)
        """
        try:
            if len(data) < 2:
                raise ValueError("Need at least 2 data points for volatility calculation")
            
            # Calculate log returns
            prices = data['Close'].dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # Use the most recent 'window' returns or all available if less
            recent_returns = log_returns.tail(min(window, len(log_returns)))
            
            # Calculate annualized volatility
            volatility = recent_returns.std() * np.sqrt(252)
            
            return float(volatility)
            
        except Exception as e:
            raise Exception(f"Error calculating volatility: {str(e)}")
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information including current price and volatility.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing stock information
        """
        try:
            # Fetch data
            data = self.fetch_stock_data(ticker, period="2y")
            
            # Calculate metrics
            current_price = float(data['Close'].iloc[-1])
            volatility_1y = self.calculate_historical_volatility(data, window=252)
            volatility_6m = self.calculate_historical_volatility(data, window=126)
            volatility_3m = self.calculate_historical_volatility(data, window=63)
            
            # Calculate additional metrics
            returns_1y = self.calculate_returns(data, window=252)
            avg_volume = float(data['Volume'].tail(30).mean())

            # Company metadata from yfinance is a separate network call; it can
            # fail (offline, rate limits) even when price data is available, so
            # never let it break the whole request.
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                info = {}

            stock_info = {
                'ticker': ticker.upper(),
                'current_price': current_price,
                'currency': info.get('currency', 'USD'),
                'volatility_1y': volatility_1y,
                'volatility_6m': volatility_6m,
                'volatility_3m': volatility_3m,
                'returns_1y': returns_1y,
                # Display-ready aliases (percentages)
                'annual_return': returns_1y * 100,
                'volatility_analysis': {
                    '1Y Vol': volatility_1y * 100,
                    '6M Vol': volatility_6m * 100,
                    '3M Vol': volatility_3m * 100,
                },
                'avg_volume_30d': avg_volume,
                'market_cap': info.get('marketCap', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'data_start_date': data.index[0].strftime('%Y-%m-%d'),
                'data_end_date': data.index[-1].strftime('%Y-%m-%d'),
                'total_trading_days': len(data)
            }

            return stock_info
            
        except Exception as e:
            raise Exception(f"Error getting stock info for {ticker}: {str(e)}")
    
    def calculate_returns(self, data: pd.DataFrame, window: int = 252) -> float:
        """
        Calculate annualized returns for the specified window.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock price data with 'Close' column
        window : int
            Number of trading days
            
        Returns:
        --------
        float
            Annualized return
        """
        try:
            prices = data['Close'].dropna()
            if len(prices) < window:
                window = len(prices)
            
            start_price = prices.iloc[-window]
            end_price = prices.iloc[-1]
            
            # Calculate annualized return
            total_return = (end_price / start_price) - 1
            annualized_return = (1 + total_return) ** (252 / window) - 1
            
            return float(annualized_return)
            
        except Exception as e:
            return 0.0
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol exists and has data.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol to validate
            
        Returns:
        --------
        bool
            True if ticker is valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid info
            if not info or 'regularMarketPrice' not in info:
                # Try fetching recent data as fallback
                data = stock.history(period="5d")
                return not data.empty
            
            return True
            
        except:
            return False
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (10-year Treasury yield).
        
        Returns:
        --------
        float
            Current risk-free rate as decimal (e.g., 0.045 for 4.5%)
        """
        try:
            # Fetch 10-year Treasury yield
            treasury = yf.Ticker("^TNX")
            data = treasury.history(period="5d")
            
            if not data.empty:
                # Convert percentage to decimal
                rate = float(data['Close'].iloc[-1]) / 100
                return rate
            else:
                # Default rate if unable to fetch
                print("Warning: Unable to fetch risk-free rate, using default 4.5%")
                return 0.045
                
        except Exception as e:
            print(f"Warning: Error fetching risk-free rate: {str(e)}, using default 4.5%")
            return 0.045
    
    def print_stock_summary(self, ticker: str) -> None:
        """
        Print a formatted summary of stock information.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        """
        try:
            info = self.get_stock_info(ticker)
            
            print(f"\n{'='*60}")
            print(f"STOCK ANALYSIS: {info['ticker']}")
            print(f"{'='*60}")
            
            # Basic info
            print(f"Current Price:      ${info['current_price']:.2f} {info['currency']}")
            print(f"Sector:            {info['sector']}")
            print(f"Industry:          {info['industry']}")
            
            # Volatility analysis
            print(f"\nVOLATILITY ANALYSIS:")
            print(f"3-Month Volatility: {info['volatility_3m']:.2%}")
            print(f"6-Month Volatility: {info['volatility_6m']:.2%}")
            print(f"1-Year Volatility:  {info['volatility_1y']:.2%}")
            
            # Performance
            print(f"\nPERFORMANCE:")
            print(f"1-Year Return:      {info['returns_1y']:.2%}")
            
            # Data info
            print(f"\nDATA INFORMATION:")
            print(f"Data Period:       {info['data_start_date']} to {info['data_end_date']}")
            print(f"Trading Days:      {info['total_trading_days']}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error printing stock summary: {str(e)}")


# Convenience functions for direct use
def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Convenience function to fetch stock data."""
    handler = DataHandler()
    return handler.fetch_stock_data(ticker, period)

def calculate_volatility(data: pd.DataFrame) -> float:
    """Convenience function to calculate volatility."""
    handler = DataHandler()
    return handler.calculate_historical_volatility(data)

def get_current_price(ticker: str) -> float:
    """Convenience function to get current stock price."""
    handler = DataHandler()
    return handler.get_current_price(ticker)

def get_risk_free_rate() -> float:
    """Convenience function to get risk-free rate."""
    handler = DataHandler()
    return handler.get_risk_free_rate()


if __name__ == "__main__":
    # Example usage and testing
    handler = DataHandler()
    
    # Test with AAPL
    try:
        print("Testing DataHandler with AAPL...")
        handler.print_stock_summary("AAPL")
        
        # Test risk-free rate
        rf_rate = handler.get_risk_free_rate()
        print(f"Current Risk-Free Rate: {rf_rate:.3%}")
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")