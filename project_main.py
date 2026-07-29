#!/usr/bin/env python3
"""
Comprehensive Financial Mathematics Simulation - Main Interactive Script

This is the main entry point for the comprehensive option pricing and risk management
system. It provides an interactive terminal interface for exploring all aspects of
financial mathematics including pricing models, Greeks, strategies, and hedging.

Usage:
    python project_main.py

Features:
- Interactive stock selection and data analysis
- Model comparison (BSM, CRR, Monte Carlo, ML)
- Convergence analysis and tree visualization
- Greeks calculation and risk analysis
- Trading strategy backtesting
- Hedging simulation and P&L analysis
- Comprehensive visualization and reporting
"""

import sys
import os
import glob
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
import scipy.stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all modules
try:
    # Data and utilities
    from utils.data_handler import DataHandler, get_risk_free_rate
    from utils.tree_printer import build_and_print_trees, print_tree
    from utils.visualization import (FinancialPlotter, plot_convergence, plot_option_curves, 
                                   plot_strategies, plot_monte_carlo, show_all_plots)
    
    # Models
    from models.pricing_models import (BSMModel, BinomialModel, MonteCarloModel, 
                                     MLModel, ForwardModel, BSM_price, CRR_price, 
                                     MC_price, forward_price)
    from models.greeks import (GreeksCalculator, ImpliedVolatilityCalculator, 
                             calculate_greeks, calculate_iv, print_greeks)
    from models.strategies import (HedgingSimulator, TradingStrategies, 
                                 StrategyComparison, backtest_strategies)
    
    # Options Payoff Strategies
    from strategies.options_payoff_strategies import (OptionsPayoffAnalyzer, 
                                                    get_popular_strategies, 
                                                    create_strategy_comparison_chart)
    
    # External libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class FinancialMathematicsApp:
    """
    Main application class for the Financial Mathematics Simulation.
    """
    
    def __init__(self, force_real=False, force_sample=False):
        """Initialize the application."""
        self.data_handler = DataHandler()
        self.plotter = FinancialPlotter()
        self.current_ticker = None
        self.current_data = None
        self.current_stock_info = None
        self.session_results = {}
        self.force_real = force_real
        self.force_sample = force_sample
        
    def print_header(self, title: str, char: str = "=") -> None:
        """Print a formatted header."""
        width = max(80, len(title) + 4)
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")
    
    def print_section(self, title: str) -> None:
        """Print a section header."""
        print(f"\n\n{'-' * 60}")
        print(f"{title}")
        print(f"{'-' * 60}\n")
    
    def get_user_input(self, prompt: str, input_type: type = str, 
                      default: Any = None, validator: Any = None) -> Any:
        """
        Get validated user input.
        
        Parameters:
        -----------
        prompt : str
            Input prompt message
        input_type : type
            Expected input type (str, int, float, bool)
        default : Any
            Default value if user presses Enter
        validator : callable
            Function to validate input
            
        Returns:
        --------
        Any
            Validated user input
        """
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} [{default}]: ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                if input_type == bool:
                    return user_input.lower() in ['y', 'yes', 'true', '1']
                
                converted = input_type(user_input)
                
                if validator and not validator(converted):
                    print("Invalid input. Please try again.")
                    continue
                
                return converted
                
            except ValueError:
                print(f"Please enter a valid {input_type.__name__}.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    
    def get_available_real_data_tickers(self):
        """Get list of available real data tickers from CSV files."""
        real_data_dir = "data/real_data"
        if not os.path.exists(real_data_dir):
            return []
        
        csv_files = glob.glob(os.path.join(real_data_dir, "*.csv"))
        tickers = [os.path.basename(f).split('_')[0].upper() for f in csv_files]
        return sorted(tickers)
    
    def load_real_data_if_available(self, ticker):
        """Try to load real data first, fallback to API."""
        real_file = f"data/real_data/{ticker}_1y.csv"
        
        # Check force flags
        if self.force_sample:
            print(f"🌐 Forced sample/API mode for {ticker}")
            data = self.data_handler.fetch_stock_data(ticker, period="2y")
            info = self.data_handler.get_stock_info(ticker)
            return data, info
            
        if self.force_real:
            if not os.path.exists(real_file):
                print(f"ERROR: Real data file not found for {ticker}: {real_file}")
                print(f"Available real data: {', '.join(self.get_available_real_data_tickers())}")
                raise FileNotFoundError(f"Real data required but not found for {ticker}")
        
        if os.path.exists(real_file):
            print(f"📁 Using real market data for {ticker}")
            df = pd.read_csv(real_file)
            df['Date'] = pd.to_datetime(df['Date'])
            current_price = df['Close'].iloc[-1]
            
            # Calculate volatility from real data
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            stock_info = {
                'current_price': current_price,
                'volatility': volatility,
                'symbol': ticker,
                'data_source': 'Real CSV File',
                'annual_return': ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            }
            
            print(f"Current {ticker} price: ${current_price:.2f}")
            print(f"Real volatility: {volatility*100:.1f}%")
            return df, stock_info
        else:
            print(f"🌐 Fetching API data for {ticker}")
            data = self.data_handler.fetch_stock_data(ticker, period="2y")
            info = self.data_handler.get_stock_info(ticker)
            return data, info
    
    def select_stock(self) -> bool:
        """
        Interactive stock selection and data fetching.
        
        Returns:
        --------
        bool
            True if stock selected successfully, False otherwise
        """
        self.print_section("STOCK SELECTION AND DATA ANALYSIS")
        
        # Show mode-specific information
        available_real_tickers = self.get_available_real_data_tickers()
        
        if self.force_real:
            print("REAL DATA MODE: Only real market data files allowed")
            if available_real_tickers:
                print(f"Available: {', '.join(available_real_tickers)}")
            else:
                print("ERROR: No real data files found! Run setup to download data first.")
                return False
        elif self.force_sample:
            print("SAMPLE/API MODE: Using sample data and API calls only")
            print("NOTE: Real data files will be ignored")
        else:
            if available_real_tickers:
                print(f"Available Real Market Data: {', '.join(available_real_tickers)}")
                print("TIP: Enter ticker from above list for real data, or any other for API data")
        
        while True:
            ticker = self.get_user_input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)", 
                                       str).upper()
            
            print(f"Validating ticker '{ticker}'...")
            if not self.data_handler.validate_ticker(ticker):
                print(f"ERROR: Invalid ticker: {ticker}")
                if not self.get_user_input("Try another ticker?", bool, True):
                    return False
                continue
            
            try:
                print(f"Loading data for {ticker}...")
                # Try real data first, fallback to API
                self.current_data, self.current_stock_info = self.load_real_data_if_available(ticker)
                self.current_ticker = ticker
                
                # Display stock summary
                if hasattr(self.data_handler, 'print_stock_summary'):
                    self.data_handler.print_stock_summary(ticker)
                else:
                    # Print basic summary for real data
                    print(f"\nSUCCESS: Data loaded for {ticker}")
                    if self.current_stock_info:
                        print(f"Current Price: ${self.current_stock_info['current_price']:.2f}")
                        print(f"Data Source: {self.current_stock_info.get('data_source', 'API')}")
                
                return True
                
            except Exception as e:
                print(f"ERROR: Error fetching data: {str(e)}")
                if not self.get_user_input("Try another ticker?", bool, True):
                    return False
    
    def get_option_parameters(self) -> Dict[str, Any]:
        """
        Get option parameters from user input.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with option parameters
        """
        self.print_section("OPTION PARAMETERS")
        
        if not self.current_stock_info:
            raise ValueError("No stock data available")
        
        S = self.current_stock_info['current_price']
        print(f"Current Stock Price: ${S:.2f}")
        
        # Get parameters
        K = self.get_user_input("Strike Price (K)", float, S * 1.05,
                               lambda x: x > 0)
        
        T = self.get_user_input("Time to Maturity in years (T)", float, 0.25,
                               lambda x: 0 < x <= 5)
        
        # Option to use current risk-free rate or enter manually
        print("\\nRisk-Free Rate Options:")
        print("1. Use current 10-year Treasury rate (recommended)")
        print("2. Enter custom rate")
        
        rate_choice = self.get_user_input("Choose option", int, 1,
                                        lambda x: x in [1, 2])
        
        if rate_choice == 1:
            try:
                r = get_risk_free_rate()
                print(f"Using current Treasury rate: {r:.3%}")
            except:
                print("Unable to fetch current rate, using default")
                r = 0.045
        else:
            r = self.get_user_input("Risk-Free Rate (as decimal, e.g., 0.05 for 5%)", 
                                   float, 0.045, lambda x: 0 <= x <= 1)
        
        # Volatility options
        print("\\nVolatility Options:")
        print("1. Use historical volatility (recommended)")
        print("2. Enter custom volatility")
        
        vol_choice = self.get_user_input("Choose option", int, 1,
                                       lambda x: x in [1, 2])
        
        if vol_choice == 1:
            sigma = self.current_stock_info['volatility_1y']
            print(f"Using 1-year historical volatility: {sigma:.2%}")
        else:
            sigma = self.get_user_input("Volatility (as decimal, e.g., 0.2 for 20%)", 
                                      float, 0.2, lambda x: 0 < x <= 2)
        
        # Market price for IV calculation (optional)
        market_price = None
        if self.get_user_input("Do you have a market option price for implied volatility calculation?", 
                             bool, False):
            market_price = self.get_user_input("Market option price", float, None,
                                             lambda x: x > 0)
        
        return {
            'S': S,
            'K': K, 
            'T': T,
            'r': r,
            'sigma': sigma,
            'market_price': market_price
        }
    
    def model_comparison_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare all pricing models and analyze results.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
            
        Returns:
        --------
        Dict[str, Any]
            Model comparison results
        """
        self.print_section("MODEL COMPARISON ANALYSIS")
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        
        # Get option type
        option_type = 'call'
        if self.get_user_input("Option type (call/put)", str, 'call').lower().startswith('p'):
            option_type = 'put'
        
        print(f"\\nAnalyzing {option_type.upper()} option...")
        print(f"Parameters: S=${S:.2f}, K=${K:.2f}, T={T:.4f}, r={r:.2%}, σ={sigma:.2%}")
        
        results = {}
        
        # 1. Black-Scholes-Merton
        print("\\n🔸 Calculating BSM price...")
        bsm_price = BSM_price(S, K, T, r, sigma, option_type)
        results['BSM'] = bsm_price
        print(f"BSM Price: ${bsm_price:.6f}")
        
        # 2. CRR Binomial (European)
        print("\\n🔸 Calculating CRR price...")
        n_steps = self.get_user_input("Number of steps for CRR", int, 100,
                                     lambda x: 1 <= x <= 10000)
        crr_price = CRR_price(S, K, T, r, sigma, n_steps, option_type, False)
        results['CRR'] = crr_price
        print(f"CRR Price ({n_steps} steps): ${crr_price:.6f}")
        print(f"CRR vs BSM Error: ${abs(crr_price - bsm_price):.6f}")
        
        # 3. American Option (if put)
        if option_type == 'put':
            print("\\n🔸 Calculating American Put price...")
            american_price = CRR_price(S, K, T, r, sigma, n_steps, option_type, True)
            results['American'] = american_price
            early_exercise_premium = american_price - crr_price
            print(f"American Put Price: ${american_price:.6f}")
            print(f"Early Exercise Premium: ${early_exercise_premium:.6f}")
        
        # 4. Monte Carlo
        print("\\n🔸 Calculating Monte Carlo price...")
        n_simulations = self.get_user_input("Number of MC simulations", int, 100000,
                                           lambda x: 1000 <= x <= 10000000)
        mc_result = MonteCarloModel.european_price(S, K, T, r, sigma, n_simulations, option_type, seed=42)
        results['MC'] = mc_result['price']
        print(f"MC Price ({n_simulations:,} sims): ${mc_result['price']:.6f} ± ${mc_result['standard_error']:.6f}")
        print(f"MC vs BSM Error: ${abs(mc_result['price'] - bsm_price):.6f}")
        
        # 5. Machine Learning (if available)
        ml_model = MLModel()  # resolves the bundled data/ml_model.pkl regardless of cwd
        if ml_model.is_trained:
            print("\\n🔸 Calculating ML price...")
            ml_price = ml_model.predict_price(S, K, T, r, sigma, option_type)
            results['ML'] = ml_price
            print(f"ML Price: ${ml_price:.6f}")
            print(f"ML vs BSM Error: ${abs(ml_price - bsm_price):.6f}")
        else:
            print("\\nWARNING: ML model not available, skipping ML prediction")
        
        # 6. Forward Price (for comparison)
        forward = forward_price(S, r, T)
        results['Forward'] = forward
        print(f"\\n📈 Forward Price: ${forward:.2f}")
        
        # Store results for later use
        results['option_type'] = option_type
        results['parameters'] = params
        self.session_results['model_comparison'] = results
        
        return results
    
    def convergence_analysis(self, params: Dict[str, Any]) -> None:
        """
        Perform and visualize convergence analysis.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
        """
        self.print_section("CONVERGENCE ANALYSIS")
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        option_type = self.session_results['model_comparison']['option_type']
        bsm_price = self.session_results['model_comparison']['BSM']
        
        print("🔍 Analyzing model convergence...")
        
        # CRR Convergence
        print("\\n📈 CRR Convergence Analysis...")
        crr_steps = [10, 25, 50, 100, 200, 500, 1000]
        crr_prices = []
        
        for steps in crr_steps:
            price = CRR_price(S, K, T, r, sigma, steps, option_type, False)
            crr_prices.append(price)
            print(f"  N={steps:4d}: ${price:.6f} (error: ${abs(price - bsm_price):.6f})")
        
        crr_data = {'steps': crr_steps, 'prices': crr_prices}
        
        # Monte Carlo Convergence
        print("\\n🎲 Monte Carlo Convergence Analysis...")
        mc_counts = [1000, 5000, 10000, 50000, 100000]
        if self.get_user_input("Include large MC simulation (500K)?", bool, False):
            mc_counts.append(500000)
        
        mc_result = MonteCarloModel.convergence_analysis(S, K, T, r, sigma, option_type, mc_counts, seed=42)
        
        for i, count in enumerate(mc_counts):
            price = mc_result['prices'][i]
            std_err = mc_result['standard_errors'][i]
            print(f"  N={count:6,}: ${price:.6f} ± ${std_err:.6f} (error: ${abs(price - bsm_price):.6f})")
        
        # Plot convergence
        if self.get_user_input("Create convergence plots?", bool, True):
            print("\\nCreating convergence plots...")
            fig = plot_convergence(crr_data, mc_result, bsm_price)
            self.session_results['convergence_plot'] = fig
            print("SUCCESS: Convergence plots created")
    
    def tree_visualization(self, params: Dict[str, Any]) -> None:
        """
        Visualize binomial trees with small number of steps.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
        """
        self.print_section("BINOMIAL TREE VISUALIZATION")
        
        print("🌳 Generating binomial trees for educational purposes...")
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        option_type = self.session_results['model_comparison']['option_type']
        
        # Get small number of steps for visualization
        n_steps = self.get_user_input("Number of steps for tree visualization", int, 5,
                                     lambda x: 1 <= x <= 10)
        
        print(f"\\n🌲 Building {n_steps}-step binomial tree...")
        
        # Build and print trees
        tree_results = build_and_print_trees(S, K, r, T, sigma, n_steps, option_type, False, precision=4)
        
        # American option tree if put
        if option_type == 'put':
            print(f"\\n🌲 Building {n_steps}-step American Put tree...")
            american_results = build_and_print_trees(S, K, r, T, sigma, n_steps, option_type, True, precision=4)
            
            early_exercise_nodes = np.sum(american_results['exercise_decisions'])
            print(f"\\n📋 Early Exercise Analysis:")
            print(f"Nodes with early exercise: {early_exercise_nodes}")
            print(f"Early exercise premium: ${american_results['option_price'] - tree_results['option_price']:.6f}")
        
        self.session_results['tree_results'] = tree_results
    
    def greeks_analysis(self, params: Dict[str, Any]) -> None:
        """
        Calculate and analyze Greeks.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
        """
        self.print_section("GREEKS ANALYSIS")
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        option_type = self.session_results['model_comparison']['option_type']
        option_price = self.session_results['model_comparison']['BSM']
        
        print("Calculating option Greeks...")
        
        # Calculate all Greeks
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        
        # Print formatted Greeks summary
        print_greeks(S, K, T, r, sigma, option_type)
        
        # Greeks interpretation
        print(f"\\n📖 PRACTICAL INTERPRETATION:")
        print(f"Portfolio Impact Analysis (per 1,000 options):")
        print(f"├─ $1 stock move → ${greeks['delta'] * 1000:.0f} portfolio change")
        print(f"├─ 1% vol increase → ${greeks['vega'] * 1000:.0f} portfolio change")
        print(f"├─ 1 day time decay → ${greeks['theta'] * 1000:.0f} portfolio change") 
        print(f"└─ 1% rate increase → ${greeks['rho'] * 1000:.0f} portfolio change")
        
        # Implied Volatility calculation if market price provided
        if params['market_price']:
            print(f"\\n🔍 IMPLIED VOLATILITY ANALYSIS:")
            market_price = params['market_price']
            
            iv_result = ImpliedVolatilityCalculator.calculate_iv(market_price, S, K, T, r, option_type)
            
            if iv_result['success']:
                iv = iv_result['implied_volatility']
                print(f"Market Price:        ${market_price:.4f}")
                print(f"Historical Vol:      {sigma:.2%}")
                print(f"Implied Volatility:  {iv:.2%}")
                print(f"Vol Difference:      {(iv - sigma):.2%} ({'rich' if iv > sigma else 'cheap'})")
                print(f"Method Used:         {iv_result['method_used']}")
                print(f"Price Error:         ${iv_result['price_error']:.6f}")
            else:
                print("ERROR: Unable to calculate implied volatility")
        
        # Option price sensitivity analysis
        if self.get_user_input("\\nPerform sensitivity analysis?", bool, True):
            print("\\n📈 Creating option price sensitivity plots...")
            
            # Stock price range around current price
            S_min, S_max = S * 0.7, S * 1.3
            S_range = np.linspace(S_min, S_max, 100)
            
            fig = plot_option_curves(S_range, K, T, r, sigma, option_type)
            self.session_results['greeks_plot'] = fig
            print("SUCCESS: Sensitivity analysis plots created")
        
        self.session_results['greeks'] = greeks
    
    def hedging_simulation(self, params: Dict[str, Any]) -> None:
        """
        Simulate hedging strategies.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
        """
        self.print_section("HEDGING SIMULATION")
        
        if self.current_data is None or self.current_data.empty or len(self.current_data) < 50:
            print("ERROR: Insufficient historical data for hedging simulation")
            return
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        option_type = self.session_results['model_comparison']['option_type']
        
        print("Simulating delta hedging strategy...")
        
        # Use recent data for simulation
        simulation_days = min(len(self.current_data), int(T * 252 * 1.5))  # 1.5x the option life
        recent_data = self.current_data.tail(simulation_days).copy()
        
        print(f"Using {len(recent_data)} days of recent price data")
        
        # Hedging simulation
        hedge_simulator = HedgingSimulator(initial_cash=0)
        
        # Rebalancing frequency
        print("\\nRebalancing Options:")
        print("1. Daily rebalancing")
        print("2. Weekly rebalancing (every 5 days)")
        print("3. Custom frequency")
        
        rebal_choice = self.get_user_input("Choose rebalancing frequency", int, 1,
                                         lambda x: x in [1, 2, 3])
        
        if rebal_choice == 1:
            rebalance_freq = 1
        elif rebal_choice == 2:
            rebalance_freq = 5
        else:
            rebalance_freq = self.get_user_input("Rebalance every N days", int, 3,
                                               lambda x: 1 <= x <= 20)
        
        print(f"\\nRunning delta hedge simulation (rebalancing every {rebalance_freq} day{'s' if rebalance_freq > 1 else ''})...")
        
        hedge_result = hedge_simulator.simulate_delta_hedge(
            recent_data, K, T, r, sigma, option_type, rebalance_freq
        )
        
        # Print results
        print(f"\\nHEDGING RESULTS:")
        print(f"Final P&L:           ${hedge_result['final_pnl']:.2f}")
        print(f"P&L Volatility:      {hedge_result['volatility_pnl']:.2%} (annualized)")
        print(f"Total Rebalances:    {hedge_result['rebalance_count']}")
        print(f"Transaction Costs:   ${hedge_result['total_transaction_costs']:.2f}")
        print(f"Max Stock Position:  {hedge_result['max_stock_position']:.2f} shares")
        
        # Effectiveness analysis
        unhedged_pnl_vol = hedge_result['volatility_pnl'] / (1 - abs(np.mean(hedge_result['deltas'])))
        hedge_effectiveness = 1 - (hedge_result['volatility_pnl'] / max(unhedged_pnl_vol, hedge_result['volatility_pnl']))
        
        print(f"\\n📈 HEDGE EFFECTIVENESS:")
        print(f"Hedge Effectiveness: {hedge_effectiveness:.1%}")
        print(f"Risk Reduction:      {(1 - hedge_result['volatility_pnl'] / 0.2):.1%}")  # Assuming 20% unhedged vol
        
        # Plot hedging results
        if self.get_user_input("\\nCreate hedging analysis plots?", bool, True):
            print("Creating hedging analysis plots...")
            from utils.visualization import FinancialPlotter
            plotter = FinancialPlotter()
            fig = plotter.plot_hedging_analysis(hedge_result)
            self.session_results['hedging_plot'] = fig
            print("✓ Hedging analysis plots created")
        
        self.session_results['hedging_result'] = hedge_result
    
    def strategy_backtesting(self, params: Dict[str, Any]) -> None:
        """
        Backtest trading strategies.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
        """
        self.print_section("TRADING STRATEGY BACKTESTING")
        
        if self.current_data is None or self.current_data.empty or len(self.current_data) < 100:
            print("ERROR: Insufficient historical data for strategy backtesting")
            return
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        
        # Strategy analysis mode selection
        print("\\nSTRATEGY ANALYSIS MODE:")
        print("1. Compare All Strategies")
        print("2. 🔍 Individual Strategy Analysis")
        
        mode_choice = self.get_user_input("Choose analysis mode", int, 1, lambda x: x in [1, 2])
        
        # Get backtesting parameters
        initial_capital = self.get_user_input("Initial capital for strategies", float, 10000,
                                            lambda x: x > 0)
        
        # Use historical data
        backtest_days = min(len(self.current_data), 252)  # Maximum 1 year
        backtest_data = self.current_data.tail(backtest_days).copy()
        
        print(f"\\nBacktesting strategies using {len(backtest_data)} days of data...")
        print(f"Initial Capital: ${initial_capital:,.0f}")
        
        if mode_choice == 2:
            # Individual strategy analysis
            self.individual_strategy_analysis(backtest_data, params, initial_capital)
        else:
            # Compare all strategies
            print("\\nRunning strategy simulations...")
            strategy_results = backtest_strategies(backtest_data, K, T, r, sigma, initial_capital)
            
            # Print strategy comparison
            StrategyComparison.print_strategy_comparison(strategy_results)
            
            # Additional analysis
            print(f"\\n📈 STRATEGY INSIGHTS:")
            
            # Find best strategy
            sharpe_ratios = [result['sharpe_ratio'] for result in strategy_results]
            best_idx = np.argmax(sharpe_ratios)
            best_strategy = strategy_results[best_idx]
            
            print(f"\\n🏆 Best Strategy: {best_strategy['strategy_name']}")
            print(f"   Return: {best_strategy['total_return']:.2%}")
            print(f"   Sharpe: {best_strategy['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {best_strategy['max_drawdown']:.2%}")
            
            # Risk analysis
            benchmark = next(r for r in strategy_results if 'buy' in r['strategy_name'].lower())
            
            print(f"\\nvs Buy & Hold Benchmark:")
            for result in strategy_results:
                if result['strategy_name'] != benchmark['strategy_name']:
                    excess_return = result['total_return'] - benchmark['total_return']
                    print(f"   {result['strategy_name']}: {excess_return:+.2%} excess return")
            
            # Create strategy plots
            if self.get_user_input("\\nCreate strategy comparison plots?", bool, True):
                print("Creating strategy comparison plots...")
                fig = plot_strategies(strategy_results)
                self.session_results['strategy_plot'] = fig
                print("✓ Strategy comparison plots created")
            
            self.session_results['strategy_results'] = strategy_results

    def individual_strategy_analysis(self, backtest_data: pd.DataFrame, params: Dict[str, Any], initial_capital: float) -> None:
        """
        Perform detailed analysis for a single strategy.
        
        Parameters:
        -----------
        backtest_data : pd.DataFrame
            Historical stock data for backtesting
        params : Dict[str, Any]
            Option parameters
        initial_capital : float
            Initial capital for strategy
        """
        print("\\n🔍 INDIVIDUAL STRATEGY ANALYSIS")
        
        # Strategy selection
        strategies = {
            '1': 'Buy & Hold Benchmark',
            '2': 'Covered Call Strategy',
            '3': 'Long Straddle Strategy',
            '4': 'Delta Neutral Strategy'
        }
        
        print("\\nAvailable Strategies:")
        for key, name in strategies.items():
            print(f"{key}. {name}")
        
        choice = self.get_user_input("Select strategy to analyze", str, "1",
                                   lambda x: x in strategies.keys())
        
        strategy_name = strategies[choice]
        print(f"\\nAnalyzing: {strategy_name}")
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        
        # Initialize trading strategies
        trading_strategies = TradingStrategies(initial_capital)
        
        # Run specific strategy with enhanced parameters
        extra_params = {}
        if choice == '2':  # Covered Call
            result = trading_strategies.covered_call_strategy(backtest_data, K, T, r, sigma)
        elif choice == '3':  # Long Straddle  
            result = trading_strategies.long_straddle_strategy(backtest_data, K, T, r, sigma)
        elif choice == '4':  # Delta Neutral
            vol_forecast = self.get_user_input("Volatility forecast (vs current σ={:.2%})".format(sigma), 
                                             float, sigma, lambda x: 0.01 <= x <= 1.0)
            extra_params['vol_forecast'] = vol_forecast
            result = trading_strategies.delta_neutral_speculation(backtest_data, K, T, r, sigma, vol_forecast)
        else:  # Buy & Hold
            result = trading_strategies.buy_and_hold_benchmark(backtest_data)
        
        # Display comprehensive analysis
        self.print_detailed_strategy_results(result, strategy_name, extra_params)
        
        # Create detailed plots
        if self.get_user_input("\\nCreate detailed strategy plots?", bool, True):
            print("Creating comprehensive strategy analysis plots...")
            fig = self.create_individual_strategy_plots(result, strategy_name)
            self.session_results['individual_strategy_plot'] = fig
            print("✓ Detailed strategy plots created")
        
        self.session_results['individual_strategy_result'] = result

    def print_detailed_strategy_results(self, result: Dict[str, Any], strategy_name: str, extra_params: Dict[str, Any]) -> None:
        """Print comprehensive results for individual strategy analysis"""
        
        print(f"\\n{'='*60}")
        print(f"DETAILED ANALYSIS: {strategy_name.upper()}")
        print(f"{'='*60}")
        
        # Core performance metrics
        print("\\n📈 PERFORMANCE METRICS:")
        print(f"Total Return:        {result['total_return']:+.2%}")
        print(f"Final P&L:           ${result['final_pnl']:+,.2f}")
        print(f"Final Portfolio:     ${result['final_value']:,.2f}")
        print(f"Sharpe Ratio:        {result['sharpe_ratio']:.3f}")
        print(f"Annual Volatility:   {result['volatility_annual']:.2%}")
        print(f"Maximum Drawdown:    {result['max_drawdown']:.2%}")
        print(f"Win Rate:            {result['win_rate']:.1%}")
        
        # Risk metrics
        daily_returns = result.get('daily_returns', np.array([0.0]))
        if len(daily_returns) > 1:
            var_95 = np.percentile(daily_returns, 5) * 100
            var_99 = np.percentile(daily_returns, 1) * 100
            
            print("\\n⚠️ RISK ANALYSIS:")
            print(f"Value at Risk (95%): {var_95:.2f}%")
            print(f"Value at Risk (99%): {var_99:.2f}%")
            print(f"Skewness:            {scipy.stats.skew(daily_returns):.3f}")
            print(f"Kurtosis:            {scipy.stats.kurtosis(daily_returns):.3f}")
        
        # Strategy-specific analysis
        print(f"\\nSTRATEGY-SPECIFIC ANALYSIS:")
        if 'covered call' in strategy_name.lower():
            print("• Income generation through option premium")
            print("• Limited upside, moderate downside protection") 
            print("• Optimal in neutral to slightly bullish markets")
            
        elif 'straddle' in strategy_name.lower():
            realized_vol = result.get('volatility_annual', 0)
            print(f"• Volatility play - realized volatility: {realized_vol:.2%}")
            print("• Profits from large price movements in either direction")
            print("• Main risk: time decay if market stays range-bound")
            
        elif 'delta neutral' in strategy_name.lower():
            vol_forecast = extra_params.get('vol_forecast', 0)
            print(f"• Volatility forecast used: {vol_forecast:.2%}")
            print("• Market direction neutral - pure volatility bet")
            print("• Success depends on volatility prediction accuracy")
            
        elif 'buy' in strategy_name.lower():
            print("• Simple long equity exposure")
            print("• Benchmark for other strategies")
            print("• Benefits from long-term market appreciation")
        
        # Performance assessment
        print(f"\\n🏆 PERFORMANCE ASSESSMENT:")
        if result['sharpe_ratio'] > 1.0:
            print("🟢 EXCELLENT: Outstanding risk-adjusted returns")
        elif result['sharpe_ratio'] > 0.5:
            print("🟡 GOOD: Solid risk-adjusted performance")  
        else:
            print("🔴 POOR: Below-average risk-adjusted returns")
        
        if abs(result['max_drawdown']) < 10:
            print("🟢 LOW RISK: Minimal drawdown exposure")
        elif abs(result['max_drawdown']) < 20:
            print("🟡 MODERATE RISK: Acceptable drawdown levels")
        else:
            print("🔴 HIGH RISK: Significant drawdown potential")

    def create_individual_strategy_plots(self, result: Dict[str, Any], strategy_name: str):
        """Create comprehensive plots for individual strategy"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not available for plotting")
            return None
        
        # Extract data
        portfolio_values = result.get('portfolio_values', np.array([result['final_value']]))
        daily_returns = result.get('daily_returns', np.array([0.0]))
        cumulative_pnl = result.get('cumulative_pnl', np.array([result['final_pnl']]))
        
        # Create comprehensive figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Comprehensive Analysis: {strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Evolution
        axes[0,0].plot(portfolio_values, linewidth=2, color='blue')
        axes[0,0].axhline(y=portfolio_values[0], color='gray', linestyle='--', alpha=0.7, label='Initial Value')
        axes[0,0].set_title('Portfolio Value Evolution')
        axes[0,0].set_xlabel('Trading Days')
        axes[0,0].set_ylabel('Portfolio Value ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Daily Returns Distribution
        if len(daily_returns) > 1:
            axes[0,1].hist(daily_returns*100, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0,1].axvline(x=np.mean(daily_returns)*100, color='red', linestyle='--', label=f'Mean: {np.mean(daily_returns)*100:.2f}%')
            axes[0,1].set_title('Daily Returns Distribution')
            axes[0,1].set_xlabel('Daily Return (%)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Drawdown Analysis  
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max * 100
        axes[1,0].fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.6, color='red')
        axes[1,0].set_title('Drawdown Analysis')
        axes[1,0].set_xlabel('Trading Days')
        axes[1,0].set_ylabel('Drawdown (%)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Cumulative P&L
        axes[1,1].plot(cumulative_pnl, linewidth=2, color='purple')
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].set_title('Cumulative P&L')
        axes[1,1].set_xlabel('Trading Days')
        axes[1,1].set_ylabel('P&L ($)')
        axes[1,1].grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        if len(daily_returns) > 30:
            window = 30
            rolling_sharpe = []
            for i in range(window, len(daily_returns)):
                window_returns = daily_returns[i-window:i]
                if np.std(window_returns) > 0:
                    sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                else:
                    sharpe = 0
                rolling_sharpe.append(sharpe)
            
            axes[2,0].plot(range(window, len(daily_returns)), rolling_sharpe, linewidth=2, color='orange')
            axes[2,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2,0].set_title('Rolling Sharpe Ratio (30-day)')
            axes[2,0].set_xlabel('Trading Days')
            axes[2,0].set_ylabel('Sharpe Ratio')
            axes[2,0].grid(True, alpha=0.3)
        
        # 6. Risk Metrics (VaR)
        if len(daily_returns) > 10:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            var_values = [np.percentile(daily_returns*100, p) for p in percentiles]
            
            axes[2,1].bar([f"{p}%" for p in percentiles], var_values, color='lightcoral', alpha=0.7)
            axes[2,1].set_title('Return Percentiles (VaR Analysis)')
            axes[2,1].set_xlabel('Percentile')
            axes[2,1].set_ylabel('Return (%)')
            axes[2,1].tick_params(axis='x', rotation=45)
            axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def monte_carlo_analysis(self, params: Dict[str, Any]) -> None:
        """
        Detailed Monte Carlo analysis and visualization.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Option parameters
        """
        self.print_section("MONTE CARLO ANALYSIS")
        
        S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']
        option_type = self.session_results['model_comparison']['option_type']
        
        print("🎲 Detailed Monte Carlo simulation analysis...")
        
        # Simulation parameters
        n_paths = self.get_user_input("Number of simulation paths", int, 10000,
                                    lambda x: 1000 <= x <= 1000000)
        n_steps = self.get_user_input("Time steps per path", int, 100,
                                    lambda x: 10 <= x <= 1000)
        
        print(f"\\nGenerating {n_paths:,} paths with {n_steps} time steps...")
        
        # Generate paths
        mc_model = MonteCarloModel()
        paths = mc_model.gbm_paths(S, r, sigma, T, n_steps, n_paths, seed=42)
        
        # Calculate statistics
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        option_value = np.exp(-r * T) * np.mean(payoffs)
        
        print(f"\\nSIMULATION RESULTS:")
        print(f"Option Value:        ${option_value:.6f}")
        print(f"Standard Error:      ${np.std(payoffs) / np.sqrt(n_paths) * np.exp(-r * T):.6f}")
        print(f"\\nFinal Stock Prices:")
        print(f"Mean:                ${np.mean(final_prices):.2f}")
        print(f"Std Dev:             ${np.std(final_prices):.2f}")
        print(f"Min:                 ${np.min(final_prices):.2f}")
        print(f"Max:                 ${np.max(final_prices):.2f}")
        print(f"\\nPayoff Statistics:")
        print(f"ITM Probability:     {np.mean(payoffs > 0):.2%}")
        print(f"Average Payoff:      ${np.mean(payoffs):.2f}")
        print(f"Average ITM Payoff:  ${np.mean(payoffs[payoffs > 0]) if np.any(payoffs > 0) else 0:.2f}")
        
        # Risk metrics
        var_95 = np.percentile(final_prices, 5)
        var_99 = np.percentile(final_prices, 1)
        
        print(f"\\n⚠️  RISK METRICS:")
        print(f"5% VaR (stock):      ${S - var_95:.2f} ({(S - var_95)/S:.1%} loss)")
        print(f"1% VaR (stock):      ${S - var_99:.2f} ({(S - var_99)/S:.1%} loss)")
        
        # Plot Monte Carlo paths and distributions
        if self.get_user_input("\\nCreate Monte Carlo visualization?", bool, True):
            print("Creating Monte Carlo plots...")
            fig = plot_monte_carlo(paths, S, K, option_type)
            self.session_results['montecarlo_plot'] = fig
            print("✓ Monte Carlo plots created")
        
        self.session_results['monte_carlo'] = {
            'paths': paths,
            'final_prices': final_prices,
            'payoffs': payoffs,
            'option_value': option_value
        }
    
    def options_payoff_analysis(self, params: Dict[str, Any]) -> None:
        """
        Comprehensive options payoff analysis for various strategies.
        """
        self.print_header("OPTIONS PAYOFF STRATEGIES ANALYSIS")
        
        spot_price = self.current_stock_info['current_price']
        
        print(f"""
Options Payoff Analysis

This analysis provides comprehensive payoff diagrams and metrics for various options strategies.
You can analyze individual strategies or compare multiple strategies side-by-side.

Current Stock: {self.current_ticker}
Spot Price: ${spot_price:.2f}
""")
        
        # Initialize analyzer
        analyzer = OptionsPayoffAnalyzer(spot_price)
        
        # Analysis mode selection
        analysis_modes = [
            ("Individual Strategy Analysis", self.individual_payoff_analysis),
            ("Strategy Comparison", self.payoff_comparison_analysis),
            ("Popular Strategies Overview", self.popular_strategies_analysis)
        ]
        
        print("📋 Analysis Options:")
        for i, (name, _) in enumerate(analysis_modes, 1):
            print(f"{i}. {name}")
        
        mode_choice = self.get_user_input("\\nSelect analysis mode", int, 1,
                                        lambda x: 1 <= x <= len(analysis_modes))
        
        selected_mode = analysis_modes[mode_choice - 1]
        print(f"\\n🔍 Running {selected_mode[0]}...")
        
        try:
            selected_mode[1](analyzer, params)
        except Exception as e:
            print(f"ERROR: Error in payoff analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def individual_payoff_analysis(self, analyzer: OptionsPayoffAnalyzer, params: Dict[str, Any]) -> None:
        """Analyze individual options strategies with custom parameters."""
        
        strategy_options = {
            1: ("Long Call", analyzer.long_call, ['strike', 'premium']),
            2: ("Long Put", analyzer.long_put, ['strike', 'premium']),
            3: ("Short Call", analyzer.short_call, ['strike', 'premium']),
            4: ("Short Put", analyzer.short_put, ['strike', 'premium']),
            5: ("Bull Call Spread", analyzer.bull_call_spread, ['lower_strike', 'upper_strike', 'lower_premium', 'upper_premium']),
            6: ("Bear Put Spread", analyzer.bear_put_spread, ['lower_strike', 'upper_strike', 'lower_premium', 'upper_premium']),
            7: ("Long Straddle", analyzer.long_straddle, ['strike', 'call_premium', 'put_premium']),
            8: ("Long Strangle", analyzer.long_strangle, ['call_strike', 'put_strike', 'call_premium', 'put_premium']),
            9: ("Butterfly Spread", analyzer.butterfly_spread, ['lower_strike', 'middle_strike', 'upper_strike']),
            10: ("Iron Condor", analyzer.iron_condor, ['put_lower_strike', 'put_upper_strike', 'call_lower_strike', 'call_upper_strike']),
            11: ("Iron Butterfly", analyzer.iron_butterfly, ['strike', 'put_strike', 'call_strike'])
        }
        
        print("\\nAvailable Options Strategies:")
        for key, (name, _, _) in strategy_options.items():
            print(f"{key:2d}. {name}")
        
        strategy_choice = self.get_user_input("\\nSelect strategy to analyze", int, 1,
                                            lambda x: x in strategy_options)
        
        strategy_name, strategy_func, param_names = strategy_options[strategy_choice]
        
        print(f"\\nAnalyzing: {strategy_name}")
        
        # Get custom parameters or use defaults
        use_defaults = self.get_user_input("Use default parameters?", bool, True)
        
        if use_defaults:
            result = strategy_func()
        else:
            # Get custom parameters
            custom_params = {}
            spot_price = analyzer.spot_price
            
            print(f"\\n⚙️ Custom Parameters (Press Enter for defaults):")
            
            for param_name in param_names:
                if 'strike' in param_name.lower():
                    if 'lower' in param_name or 'put' in param_name:
                        default_val = spot_price * 0.95
                    elif 'upper' in param_name or 'call' in param_name:
                        default_val = spot_price * 1.05
                    else:
                        default_val = spot_price
                    
                    prompt = f"{param_name.replace('_', ' ').title()} (default: ${default_val:.2f})"
                    value = self.get_user_input(prompt, float, default_val)
                    custom_params[param_name] = value
                    
                elif 'premium' in param_name.lower():
                    default_val = spot_price * 0.02
                    prompt = f"{param_name.replace('_', ' ').title()} (default: ${default_val:.2f})"
                    value = self.get_user_input(prompt, float, default_val)
                    custom_params[param_name] = value
            
            result = strategy_func(**custom_params)
        
        # Display results
        self.display_payoff_results(result, analyzer)
        
        # Store results
        self.session_results[f'payoff_{strategy_name.lower().replace(" ", "_")}'] = result
    
    def payoff_comparison_analysis(self, analyzer: OptionsPayoffAnalyzer, params: Dict[str, Any]) -> None:
        """Compare multiple strategies side by side."""
        
        print("\\nStrategy Comparison Analysis")
        print("Select multiple strategies to compare their payoff profiles.")
        
        available_strategies = [
            ("Long Call", analyzer.long_call),
            ("Long Put", analyzer.long_put),
            ("Bull Call Spread", analyzer.bull_call_spread),
            ("Bear Put Spread", analyzer.bear_put_spread),
            ("Long Straddle", analyzer.long_straddle),
            ("Long Strangle", analyzer.long_strangle),
            ("Butterfly Spread", analyzer.butterfly_spread),
            ("Iron Condor", analyzer.iron_condor)
        ]
        
        print("\\nAvailable Strategies:")
        for i, (name, _) in enumerate(available_strategies, 1):
            print(f"{i:2d}. {name}")
        
        # Get strategy selections
        selections = []
        while len(selections) < 2:
            prompt = f"Select strategy {len(selections) + 1} (or 0 when done)"
            if len(selections) == 0:
                prompt += " - minimum 2 required"
            
            choice = self.get_user_input(prompt, int, 0,
                                       lambda x: 0 <= x <= len(available_strategies))
            
            if choice == 0 and len(selections) >= 2:
                break
            elif choice == 0:
                print("Minimum 2 strategies required for comparison")
                continue
            elif choice in [s[0] for s in selections]:
                print("Strategy already selected")
                continue
            else:
                selections.append((choice, available_strategies[choice - 1]))
        
        print(f"\\nAnalyzing {len(selections)} strategies...")
        
        # Generate payoff results
        results = []
        for choice, (name, func) in selections:
            try:
                result = func()
                results.append(result)
                print(f"✓ {name} analysis completed")
            except Exception as e:
                print(f"ERROR: Error analyzing {name}: {e}")
        
        if len(results) < 2:
            print("ERROR: Not enough valid results for comparison")
            return
        
        # Create comparison
        print("\\n📈 Creating comparison analysis...")
        
        # Display comparison table
        comparison_df = analyzer.compare_strategies(results)
        print("\\nSTRATEGY COMPARISON TABLE:")
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        if self.get_user_input("\\nCreate comparison chart?", bool, True):
            print("Creating comparison chart...")
            try:
                fig = analyzer.plot_multiple_strategies(results, 
                                                      "Options Strategies Comparison")
                self.session_results['payoff_comparison_plot'] = fig
                print("✓ Comparison chart created")
            except Exception as e:
                print(f"ERROR: Error creating chart: {e}")
        
        # Store results
        self.session_results['payoff_comparison'] = {
            'results': results,
            'comparison_table': comparison_df
        }
    
    def popular_strategies_analysis(self, analyzer: OptionsPayoffAnalyzer, params: Dict[str, Any]) -> None:
        """Analyze popular options strategies with default parameters."""
        
        print("\\n🌟 Popular Options Strategies Analysis")
        print("Quick analysis of commonly used options strategies with market-standard parameters.")
        
        # Get popular strategies
        strategies = get_popular_strategies(analyzer.spot_price)
        
        print(f"\\nAnalyzing {len(strategies)} popular strategies...")
        
        # Display individual results
        for name, result in strategies.items():
            print(f"\\n{'─' * 50}")
            print(f"Analysis: {name}")
            print(f"{'─' * 50}")
            
            print(f"Max Profit: ${'%.2f' % result.max_profit if result.max_profit != float('inf') else 'Unlimited'}")
            print(f"Max Loss: ${'%.2f' % abs(result.max_loss) if result.max_loss != float('-inf') else 'Unlimited'}")
            print(f"Net Premium: ${'%.2f' % result.net_premium}")
            print(f"Breakeven Points: {len(result.breakeven_points)}")
            
            if result.breakeven_points:
                for i, bp in enumerate(result.breakeven_points, 1):
                    print(f"  Breakeven {i}: ${bp:.2f}")
        
        # Create comparison table
        results_list = list(strategies.values())
        comparison_df = analyzer.compare_strategies(results_list)
        
        print(f"\\nPOPULAR STRATEGIES COMPARISON:")
        print(comparison_df.to_string(index=False))
        
        # Plot all strategies
        if self.get_user_input("\\nCreate comprehensive comparison chart?", bool, True):
            print("Creating comprehensive chart...")
            try:
                fig = analyzer.plot_multiple_strategies(results_list, 
                                                      "Popular Options Strategies")
                self.session_results['popular_strategies_plot'] = fig
                print("✓ Comprehensive chart created")
            except Exception as e:
                print(f"ERROR: Error creating chart: {e}")
        
        # Store results
        self.session_results['popular_strategies'] = {
            'strategies': strategies,
            'comparison_table': comparison_df
        }
    
    def display_payoff_results(self, result, analyzer: OptionsPayoffAnalyzer) -> None:
        """Display detailed payoff analysis results."""
        
        print(f"\\n{result.strategy_name.upper()} ANALYSIS RESULTS")
        print("=" * 60)
        
        # Strategy composition
        print("\\n📋 STRATEGY COMPOSITION:")
        for i, leg in enumerate(result.legs, 1):
            position_desc = f"{leg.position.title()} {leg.option_type.title()}"
            if leg.quantity != 1:
                position_desc += f" (x{leg.quantity})"
            
            print(f"  {i}. {position_desc}")
            print(f"     Strike: ${leg.strike:.2f} | Premium: ${leg.premium:.2f}")
        
        # Payoff metrics
        print(f"\\n💰 PAYOFF METRICS:")
        print(f"Net Premium: ${result.net_premium:.2f}")
        
        max_profit_str = f"${result.max_profit:.2f}" if result.max_profit != float('inf') else "Unlimited"
        max_loss_str = f"${abs(result.max_loss):.2f}" if result.max_loss != float('-inf') else "Unlimited"
        
        print(f"Maximum Profit: {max_profit_str}")
        print(f"Maximum Loss: {max_loss_str}")
        
        # Risk/Reward ratio
        if result.max_profit != float('inf') and result.max_loss != float('-inf') and result.max_profit > 0:
            risk_reward = abs(result.max_loss) / result.max_profit
            print(f"Risk/Reward Ratio: {risk_reward:.2f}")
        
        # Breakeven analysis
        print(f"\\n⚖️ BREAKEVEN ANALYSIS:")
        print(f"Number of Breakeven Points: {len(result.breakeven_points)}")
        
        for i, breakeven in enumerate(result.breakeven_points, 1):
            distance_pct = (breakeven - analyzer.spot_price) / analyzer.spot_price * 100
            print(f"  Breakeven {i}: ${breakeven:.2f} ({distance_pct:+.1f}% from spot)")
        
        # Profit/Loss ranges
        if result.profit_range[0] is not None:
            print(f"\\nProfit Range: ${result.profit_range[0]:.2f} - ${result.profit_range[1]:.2f}")
        
        if result.loss_range[0] is not None:
            print(f"Loss Range: ${result.loss_range[0]:.2f} - ${result.loss_range[1]:.2f}")
        
        # Generate detailed report
        if self.get_user_input("\\nGenerate detailed text report?", bool, False):
            report = analyzer.generate_strategy_report(result)
            print(report)
        
        # Create payoff diagram
        if self.get_user_input("\\nCreate payoff diagram?", bool, True):
            print("Creating payoff diagram...")
            try:
                fig = analyzer.plot_payoff(result, show_plot=True)
                self.session_results[f'{result.strategy_name.lower().replace(" ", "_")}_plot'] = fig
                print("✓ Payoff diagram created")
            except Exception as e:
                print(f"ERROR: Error creating diagram: {e}")
    
    def session_summary(self) -> None:
        """
        Print comprehensive session summary.
        """
        self.print_header("SESSION SUMMARY REPORT")
        
        if not self.session_results:
            print("No analysis performed in this session.")
            return
        
        print(f"📈 Stock Analyzed: {self.current_ticker}")
        print(f"Current Price: ${self.current_stock_info['current_price']:.2f}")
        print(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'model_comparison' in self.session_results:
            results = self.session_results['model_comparison']
            params = results['parameters']
            
            print(f"\\n📋 OPTION ANALYZED:")
            print(f"Type: {results['option_type'].upper()}")
            print(f"Strike: ${params['K']:.2f}")
            print(f"Maturity: {params['T']:.4f} years")
            print(f"Risk-free rate: {params['r']:.2%}")
            print(f"Volatility: {params['sigma']:.2%}")
            
            print(f"\\n💰 MODEL PRICES:")
            for model, price in results.items():
                if model not in ['option_type', 'parameters', 'Forward']:
                    print(f"{model:12s}: ${price:.6f}")
            
            if 'Forward' in results:
                print(f"{'Forward':12s}: ${results['Forward']:.2f}")
        
        if 'greeks' in self.session_results:
            greeks = self.session_results['greeks']
            print(f"\\nGREEKS:")
            print(f"Delta:  {greeks['delta']:8.4f}")
            print(f"Gamma:  {greeks['gamma']:8.4f}")
            print(f"Vega:   {greeks['vega']:8.4f}")
            print(f"Theta:  {greeks['theta']:8.4f}")
            print(f"Rho:    {greeks['rho']:8.4f}")
        
        if 'hedging_result' in self.session_results:
            hedge = self.session_results['hedging_result']
            print(f"\\nHEDGING PERFORMANCE:")
            print(f"Final P&L:        ${hedge['final_pnl']:.2f}")
            print(f"Volatility:       {hedge['volatility_pnl']:.2%}")
            print(f"Transaction Costs: ${hedge['total_transaction_costs']:.2f}")
        
        if 'strategy_results' in self.session_results:
            strategies = self.session_results['strategy_results']
            best_strategy = max(strategies, key=lambda x: x['sharpe_ratio'])
            print(f"\\n🏆 BEST STRATEGY:")
            print(f"Name: {best_strategy['strategy_name']}")
            print(f"Return: {best_strategy['total_return']:.2%}")
            print(f"Sharpe: {best_strategy['sharpe_ratio']:.3f}")
        
        # Show plots if available
        plot_count = sum(1 for key in self.session_results.keys() if 'plot' in key)
        if plot_count > 0:
            print(f"\\nGenerated {plot_count} visualization plot{'s' if plot_count > 1 else ''}")
            if self.get_user_input("Show all plots now?", bool, True):
                show_all_plots()
        
        # Save session
        if self.get_user_input("\\nSave session summary to file?", bool, False):
            self.save_session_report()
    
    def save_session_report(self) -> None:
        """Save session report to file."""
        try:
            filename = f"finmath_session_{self.current_ticker}_{int(time.time())}.txt"
            
            with open(filename, 'w') as f:
                f.write("COMPREHENSIVE FINANCIAL MATHEMATICS ANALYSIS REPORT\\n")
                f.write("=" * 60 + "\\n\\n")
                
                f.write(f"Stock: {self.current_ticker}\\n")
                f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Current Price: ${self.current_stock_info['current_price']:.2f}\\n\\n")
                
                # Add detailed results
                for key, value in self.session_results.items():
                    if 'plot' not in key:  # Skip plot objects
                        f.write(f"{key.upper().replace('_', ' ')}:\\n")
                        f.write(str(value))
                        f.write("\\n\\n")
            
            print(f"SUCCESS: Session report saved to: {filename}")
            
        except Exception as e:
            print(f"ERROR: Error saving report: {str(e)}")
    
    def run_interactive_session(self) -> None:
        """
        Main interactive session loop.
        """
        self.print_header("COMPREHENSIVE FINANCIAL MATHEMATICS SIMULATION", "=")
        
        print("""
Welcome to the Comprehensive Financial Mathematics Simulation!

This interactive tool provides complete option pricing analysis including:
* Multiple pricing models (BSM, CRR, Monte Carlo, ML)
* Greeks calculation and risk analysis  
* Convergence analysis and tree visualization
* Trading strategy backtesting
* Hedging simulation and P&L analysis
* Comprehensive visualization and reporting

Let's get started!
        """)
        
        try:
            # 1. Stock Selection
            if not self.select_stock():
                print("Exiting...")
                return
            
            # 2. Get Option Parameters
            params = self.get_option_parameters()
            
            # 3. Model Comparison
            self.model_comparison_analysis(params)
            
            # 4. Optional Advanced Analysis
            advanced_analyses = [
                ("Convergence Analysis", self.convergence_analysis),
                ("Binomial Tree Visualization", self.tree_visualization),
                ("Greeks Analysis", self.greeks_analysis),
                ("Monte Carlo Analysis", self.monte_carlo_analysis),
                ("Options Payoff Analysis", self.options_payoff_analysis),
                ("Hedging Simulation", self.hedging_simulation),
                ("Strategy Backtesting", self.strategy_backtesting),
            ]
            
            print(f"\\n🔬 ADVANCED ANALYSIS OPTIONS:")
            for i, (name, _) in enumerate(advanced_analyses, 1):
                print(f"{i}. {name}")
            print("0. Skip to summary")
            
            while True:
                choice = self.get_user_input("\\nSelect analysis (0 to finish)", int, 0,
                                           lambda x: 0 <= x <= len(advanced_analyses))
                
                if choice == 0:
                    break
                
                name, analysis_func = advanced_analyses[choice - 1]
                print(f"\\n🔍 Running {name}...")
                
                try:
                    analysis_func(params)
                    print(f"SUCCESS: {name} completed successfully")
                except Exception as e:
                    print(f"ERROR: Error in {name}: {str(e)}")
                
                if not self.get_user_input("\\nRun another analysis?", bool, True):
                    break
            
            # 5. Session Summary
            self.session_summary()
            
        except KeyboardInterrupt:
            print("\\n\\nSession interrupted by user.")
        except Exception as e:
            print(f"\\nERROR: Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\\nThank you for using the Financial Mathematics Simulation!")
        print("Session completed. All plots remain open for your review.")


def main(force_real=False, force_sample=False):
    """Main entry point."""
    try:
        app = FinancialMathematicsApp(force_real=force_real, force_sample=force_sample)
        app.run_interactive_session()
    except KeyboardInterrupt:
        print("\\n\\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


def main_with_args():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Financial Mathematics Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_main.py                    # Launch GUI (default)
  python project_main.py --gui              # Launch GUI explicitly  
  python project_main.py --nogui            # Launch terminal interface
  python project_main.py --nogui --real     # Terminal + force real data only
  python project_main.py --nogui --sample   # Terminal + force sample/API data
  python project_main.py --gui --real       # GUI + real data preference
        """
    )
    
    # Create mutually exclusive group for GUI/no-GUI
    mode_group = parser.add_mutually_exclusive_group()
    
    mode_group.add_argument(
        '--gui',
        action='store_true',
        default=True,
        help='Launch Streamlit GUI interface (default)'
    )
    
    mode_group.add_argument(
        '--nogui',
        action='store_true', 
        help='Launch terminal interface'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit GUI (default: 8501)'
    )
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--real',
        action='store_true',
        help='Force use real market data files only (AAPL, MSFT, GOOGL, TSLA, AMZN)'
    )
    data_group.add_argument(
        '--sample',
        action='store_true', 
        help='Force use sample/API data only (no real data files)'
    )
    
    args = parser.parse_args()
    
    if args.nogui:
        # Run terminal interface
        main(force_real=args.real, force_sample=args.sample)
    else:
        # Launch GUI interface (default)
        import subprocess
        import sys
        
        try:
            print("RUN: Launching Financial Mathematics GUI...")
            print(f"   Opening web interface at http://localhost:{args.port}")
            print("   Press Ctrl+C to stop the server")
            
            cmd = [
                sys.executable, '-m', 'streamlit', 'run', 'streamlit_gui.py',
                '--server.port', str(args.port),
                '--server.headless', 'true'
            ]
            
            subprocess.run(cmd, check=True)
            
        except (ImportError, FileNotFoundError):
            print("ERROR: Streamlit not installed. Install with: pip install streamlit plotly")
            print("   Falling back to terminal interface...")
            main()
        except subprocess.CalledProcessError:
            print("ERROR: Error launching GUI. Falling back to terminal interface...")
            main()
        except KeyboardInterrupt:
            print("\n🛑 GUI stopped by user")


if __name__ == "__main__":
    main_with_args()