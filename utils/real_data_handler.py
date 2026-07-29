#!/usr/bin/env python3
"""
Real Financial Data Handler

This module provides comprehensive real-time and historical financial data integration
from multiple sources including Yahoo Finance, Alpha Vantage, FRED, and more.

Usage:
    from utils.real_data_handler import RealDataHandler
    
    handler = RealDataHandler()
    
    # Get real stock data
    data = handler.get_stock_data('AAPL', period='2y')
    
    # Get options data (if available)
    options = handler.get_options_data('AAPL')
    
    # Get economic indicators
    rates = handler.get_risk_free_rate()
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

class RealDataHandler:
    """
    Enhanced data handler for real financial data from multiple sources.
    """
    
    def __init__(self):
        """Initialize the real data handler with API configurations."""
        self.data_cache = {}
        self.api_keys = {}
        
        # Load API keys from environment or config file
        self.api_keys['alpha_vantage'] = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        self.api_keys['quandl'] = os.getenv('QUANDL_API_KEY', '')
        self.api_keys['fred'] = os.getenv('FRED_API_KEY', '')
        
        print("🔗 Real Data Handler initialized")
        print(f"✓ Yahoo Finance: Available (no API key required)")
        print(f"✓ Alpha Vantage: {'Configured' if self.api_keys['alpha_vantage'] else 'Not configured'}")
        print(f"✓ FRED: {'Configured' if self.api_keys['fred'] else 'Not configured'}")
    
    # =====================================================================
    # YAHOO FINANCE DATA (Primary source - free, no API key required)
    # =====================================================================
    
    def get_stock_data_yahoo(self, ticker: str, period: str = "2y", 
                           interval: str = "1d") -> pd.DataFrame:
        """
        Get historical stock data from Yahoo Finance.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period : str
            Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        interval : str
            Data interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        try:
            import yfinance as yf
            
            cache_key = f"yahoo_{ticker}_{period}_{interval}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            print(f"Fetching {ticker} data from Yahoo Finance...")
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Standardize column names
            data.columns = [col.replace(' ', '_') for col in data.columns]
            
            # Cache the data
            self.data_cache[cache_key] = data
            
            print(f"SUCCESS: Successfully fetched {len(data)} data points for {ticker}")
            return data
            
        except ImportError:
            raise ImportError("yfinance package not installed. Install with: pip install yfinance")
        except Exception as e:
            raise Exception(f"Error fetching Yahoo Finance data for {ticker}: {str(e)}")
    
    def get_stock_info_yahoo(self, ticker: str) -> Dict[str, Any]:
        """Get detailed stock information from Yahoo Finance."""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information
            stock_info = {
                'symbol': ticker,
                'company_name': info.get('longName', ticker),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'volume': info.get('averageVolume', 0),
                'description': info.get('longBusinessSummary', 'No description available')[:500]
            }
            
            return stock_info
            
        except Exception as e:
            print(f"Warning: Could not fetch stock info for {ticker}: {e}")
            return {'symbol': ticker, 'current_price': 0}
    
    def get_options_data_yahoo(self, ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get options chain data from Yahoo Finance.
        
        Returns:
        --------
        Dict with 'calls' and 'puts' DataFrames, or None if unavailable
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            if not expirations:
                print(f"No options data available for {ticker}")
                return None
            
            # Get options chain for the first available expiration
            options_chain = stock.option_chain(expirations[0])
            
            return {
                'calls': options_chain.calls,
                'puts': options_chain.puts,
                'expiration': expirations[0],
                'all_expirations': expirations
            }
            
        except Exception as e:
            print(f"Warning: Could not fetch options data for {ticker}: {e}")
            return None
    
    # =====================================================================
    # ALPHA VANTAGE DATA (Premium features)
    # =====================================================================
    
    def get_stock_data_alphavantage(self, ticker: str, outputsize: str = "full") -> pd.DataFrame:
        """Get stock data from Alpha Vantage (requires API key)."""
        if not self.api_keys['alpha_vantage']:
            raise ValueError("Alpha Vantage API key not configured")
        
        try:
            from alpha_vantage.timeseries import TimeSeries
            
            ts = TimeSeries(key=self.api_keys['alpha_vantage'], output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize=outputsize)
            
            # Rename columns to match our standard
            data.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Dividend', 'Split']
            data = data.sort_index()
            
            return data
            
        except ImportError:
            raise ImportError("alpha_vantage package not installed. Install with: pip install alpha-vantage")
        except Exception as e:
            raise Exception(f"Error fetching Alpha Vantage data: {str(e)}")
    
    def get_intraday_data_alphavantage(self, ticker: str, interval: str = "5min") -> pd.DataFrame:
        """Get intraday data from Alpha Vantage."""
        if not self.api_keys['alpha_vantage']:
            raise ValueError("Alpha Vantage API key not configured")
        
        try:
            from alpha_vantage.timeseries import TimeSeries
            
            ts = TimeSeries(key=self.api_keys['alpha_vantage'], output_format='pandas')
            data, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize='full')
            
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data.sort_index()
            
        except Exception as e:
            raise Exception(f"Error fetching intraday data: {str(e)}")
    
    # =====================================================================
    # ECONOMIC DATA (FRED, Risk-free rates, etc.)
    # =====================================================================
    
    def get_risk_free_rate(self, source: str = "fred") -> float:
        """
        Get current risk-free rate from various sources.
        
        Parameters:
        -----------
        source : str
            Data source: 'fred', 'yahoo', 'treasury', or 'manual'
        """
        try:
            if source == "fred" and self.api_keys['fred']:
                return self._get_rate_from_fred()
            elif source == "yahoo":
                return self._get_rate_from_yahoo()
            else:
                # Fallback to manual/estimated rate
                return self._get_manual_rate()
        
        except Exception as e:
            print(f"Warning: Could not fetch risk-free rate from {source}: {e}")
            return 0.05  # Default 5% fallback
    
    def _get_rate_from_fred(self) -> float:
        """Get 10-year Treasury rate from FRED."""
        try:
            import pandas_datareader.data as web
            
            # Get 10-year Treasury rate
            end = datetime.now()
            start = end - timedelta(days=30)
            
            rate_data = web.DataReader('GS10', 'fred', start, end)
            current_rate = rate_data.dropna().iloc[-1, 0] / 100  # Convert to decimal
            
            return float(current_rate)
            
        except Exception as e:
            raise Exception(f"Error fetching FRED data: {e}")
    
    def _get_rate_from_yahoo(self) -> float:
        """Get Treasury rate from Yahoo Finance."""
        try:
            # Use 10-year Treasury ETF as proxy
            treasury_data = self.get_stock_data_yahoo("^TNX", period="5d")
            if not treasury_data.empty:
                return treasury_data['Close'].iloc[-1] / 100
            else:
                return 0.05
        except:
            return 0.05
    
    def _get_manual_rate(self) -> float:
        """Return estimated current risk-free rate."""
        # This could be updated manually or from a config file
        return 0.0525  # Approximate current 10-year Treasury rate
    
    # =====================================================================
    # UNIFIED DATA INTERFACE
    # =====================================================================
    
    def get_stock_data(self, ticker: str, period: str = "2y", 
                      source: str = "yahoo") -> pd.DataFrame:
        """
        Unified interface to get stock data from any source.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        period : str
            Data period
        source : str
            Data source: 'yahoo', 'alphavantage'
        """
        if source == "yahoo":
            return self.get_stock_data_yahoo(ticker, period)
        elif source == "alphavantage":
            return self.get_stock_data_alphavantage(ticker)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def get_multiple_stocks(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks at once."""
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Fetching data for {ticker}...")
                results[ticker] = self.get_stock_data(ticker, period)
            except Exception as e:
                print(f"ERROR: Failed to fetch {ticker}: {e}")
                continue
        
        return results
    
    # =====================================================================
    # DATA PROCESSING AND ANALYSIS
    # =====================================================================
    
    def calculate_realized_volatility(self, data: pd.DataFrame, window: int = 252) -> float:
        """Calculate annualized realized volatility from price data."""
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        # Calculate rolling volatility
        if len(returns) < window:
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = returns.tail(window).std() * np.sqrt(252)
        
        return float(volatility)
    
    def get_stock_statistics(self, ticker: str, period: str = "1y") -> Dict[str, float]:
        """Get comprehensive stock statistics."""
        try:
            data = self.get_stock_data(ticker, period)
            info = self.get_stock_info_yahoo(ticker)
            
            # Calculate statistics
            returns = data['Close'].pct_change().dropna()
            
            stats = {
                'current_price': info['current_price'],
                'volatility': self.calculate_realized_volatility(data),
                'beta': info.get('beta', 1.0),
                'mean_return': returns.mean() * 252,  # Annualized
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(data['Close']),
                'trading_days': len(data),
                'data_start': data.index[0].strftime('%Y-%m-%d'),
                'data_end': data.index[-1].strftime('%Y-%m-%d')
            }
            
            return stats
            
        except Exception as e:
            raise Exception(f"Error calculating statistics for {ticker}: {e}")
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
    
    # =====================================================================
    # DATA EXPORT AND IMPORT
    # =====================================================================
    
    def save_data_to_csv(self, data: pd.DataFrame, ticker: str, 
                        directory: str = "data/real_data") -> str:
        """Save stock data to CSV file."""
        os.makedirs(directory, exist_ok=True)
        
        filename = f"{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(directory, filename)
        
        data.to_csv(filepath)
        print(f"💾 Data saved to {filepath}")
        
        return filepath
    
    def load_data_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load stock data from CSV file."""
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"📁 Loaded data from {filepath}")
            return data
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {e}")
    
    def download_and_save_batch(self, tickers: List[str], period: str = "2y", 
                               directory: str = "data/real_data") -> Dict[str, str]:
        """Download and save data for multiple tickers."""
        print(f"Downloading data for {len(tickers)} tickers...")
        
        saved_files = {}
        for ticker in tickers:
            try:
                data = self.get_stock_data(ticker, period)
                filepath = self.save_data_to_csv(data, ticker, directory)
                saved_files[ticker] = filepath
            except Exception as e:
                print(f"ERROR: Failed to download {ticker}: {e}")
        
        print(f"SUCCESS: Successfully downloaded {len(saved_files)} datasets")
        return saved_files


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def download_popular_stocks(period: str = "2y") -> Dict[str, str]:
    """Download data for popular stocks."""
    popular_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'ZOOM'
    ]
    
    handler = RealDataHandler()
    return handler.download_and_save_batch(popular_tickers, period)

def download_index_etfs(period: str = "2y") -> Dict[str, str]:
    """Download data for major index ETFs."""
    index_tickers = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND']
    
    handler = RealDataHandler()
    return handler.download_and_save_batch(index_tickers, period)

def get_sample_data_for_testing(ticker: str = "AAPL") -> pd.DataFrame:
    """Get sample real data for testing purposes."""
    handler = RealDataHandler()
    return handler.get_stock_data(ticker, period="1y")


# =====================================================================
# MAIN EXECUTION FOR TESTING
# =====================================================================

if __name__ == "__main__":
    print("RUN: Testing Real Data Handler")
    print("=" * 50)
    
    # Initialize handler
    handler = RealDataHandler()
    
    # Test basic functionality
    try:
        # Get AAPL data
        print("\nTesting AAPL data fetch...")
        aapl_data = handler.get_stock_data("AAPL", period="6mo")
        print(f"SUCCESS: AAPL data shape: {aapl_data.shape}")
        print(f"SUCCESS: Date range: {aapl_data.index[0]} to {aapl_data.index[-1]}")
        print(f"SUCCESS: Current price: ${aapl_data['Close'].iloc[-1]:.2f}")
        
        # Get stock info
        print("\n📋 Testing stock info...")
        info = handler.get_stock_info_yahoo("AAPL")
        print(f"SUCCESS: Company: {info.get('company_name', 'N/A')}")
        print(f"SUCCESS: Sector: {info.get('sector', 'N/A')}")
        
        # Calculate statistics
        print("\n📈 Testing statistics calculation...")
        stats = handler.get_stock_statistics("AAPL", period="1y")
        print(f"SUCCESS: Volatility: {stats['volatility']:.2%}")
        print(f"SUCCESS: Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        
        # Test risk-free rate
        print("\n💰 Testing risk-free rate...")
        rate = handler.get_risk_free_rate()
        print(f"SUCCESS: Risk-free rate: {rate:.3%}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")