#!/usr/bin/env python3
"""
Real Data Updater

Simple script to download fresh market data and update the CSV files.
Run this periodically to get latest market prices.
"""

import os
import pandas as pd
import sys
from utils.real_data_handler import RealDataHandler

def update_real_data():
    """Download fresh real data for all tickers and save with standard names."""
    
    print("Updating Real Market Data...")
    print("=" * 40)
    
    # Popular tickers to download
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    # Ensure directory exists
    os.makedirs("data/real_data", exist_ok=True)
    
    # Initialize handler
    handler = RealDataHandler()
    
    # Download fresh data directly with correct names
    updated_files = []
    failed_tickers = []
    
    for ticker in tickers:
        try:
            print(f"Downloading {ticker} data from Yahoo Finance...")
            
            # Get fresh data
            data = handler.get_stock_data(ticker, period="1y")
            
            # Save directly with standard filename (what main app expects)
            standard_filename = f"data/real_data/{ticker}_1y.csv"
            data.to_csv(standard_filename)
            
            # Get latest price for confirmation
            latest_price = data['Close'].iloc[-1]
            data_points = len(data)
            
            print(f"   SUCCESS: Updated {ticker}: ${latest_price:.2f} ({data_points} days)")
            updated_files.append(standard_filename)
            
        except Exception as e:
            print(f"   ERROR: Failed to update {ticker}: {e}")
            failed_tickers.append(ticker)
    
    # Summary
    print(f"\\nUpdate Summary:")
    print(f"   SUCCESS: Successfully updated: {len(updated_files)} tickers")
    if updated_files:
        for file_path in updated_files:
            ticker = os.path.basename(file_path).split('_')[0]
            print(f"      {ticker}_1y.csv")
    
    if failed_tickers:
        print(f"   ERROR: Failed to update: {len(failed_tickers)} tickers")
        for ticker in failed_tickers:
            print(f"      {ticker}")
    
    print(f"\\nINFO: Real data files updated! Main app will now use fresh market prices.")
    print(f"RUN: Run: python project_main.py --nogui --real")

def check_data_status():
    """Check the current status of real data files."""
    
    print("Real Data Status Check")
    print("=" * 30)
    
    real_data_dir = "data/real_data"
    if not os.path.exists(real_data_dir):
        print("ERROR: No real data directory found!")
        return
    
    csv_files = [f for f in os.listdir(real_data_dir) if f.endswith('_1y.csv')]
    
    if not csv_files:
        print("ERROR: No real data files found!")
        return
    
    print(f"Found {len(csv_files)} real data files:")
    
    for filename in sorted(csv_files):
        ticker = filename.split('_')[0]
        filepath = os.path.join(real_data_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            latest_date = df['Date'].iloc[-1] if len(df) > 0 else "Unknown"
            latest_price = df['Close'].iloc[-1] if len(df) > 0 else 0
            
            # Get file modification time
            import datetime
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
            
            print(f"   {ticker}: ${latest_price:.2f} | {latest_date} | Updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            
        except Exception as e:
            print(f"   ERROR: {ticker}: Error reading file - {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        check_data_status()
    else:
        update_real_data()