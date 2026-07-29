"""
Trading Strategies and Hedging Module

This module implements various option trading strategies, hedging simulations,
and portfolio P&L analysis for educational and practical purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

# Import our modules
try:
    from .pricing_models import BSMModel
    from .greeks import GreeksCalculator, PortfolioGreeks
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.pricing_models import BSMModel
    from models.greeks import GreeksCalculator, PortfolioGreeks


@dataclass
class Position:
    """Data class representing a single position in a portfolio."""
    quantity: float  # Number of contracts/shares (positive = long, negative = short)
    asset_type: str  # 'stock', 'call', 'put'
    strike: Optional[float] = None  # Strike price for options
    expiry: Optional[float] = None  # Time to expiry for options
    entry_price: Optional[float] = None  # Price at which position was entered


@dataclass
class StrategyResult:
    """Data class for strategy backtesting results."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    final_pnl: float
    daily_returns: np.ndarray
    cumulative_pnl: np.ndarray
    trade_count: int


class HedgingSimulator:
    """
    Simulator for various hedging strategies.
    """
    
    def __init__(self, initial_cash: float = 0.0):
        """Initialize hedging simulator."""
        self.initial_cash = initial_cash
        self.transaction_cost = 0.01  # $0.01 per share transaction cost
    
    def simulate_delta_hedge(self, stock_data: pd.DataFrame, K: float, T_initial: float, 
                           r: float, sigma: float, option_type: str = 'call',
                           rebalance_frequency: int = 1) -> Dict[str, Any]:
        """
        Simulate delta hedging strategy.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Historical stock price data with 'Close' column
        K : float
            Strike price of the option
        T_initial : float
            Initial time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        rebalance_frequency : int
            Rebalance every N days (1 = daily)
            
        Returns:
        --------
        Dict[str, Any]
            Simulation results including P&L, Greeks, and statistics
        """
        prices = stock_data['Close'].values
        n_days = len(prices)
        
        # Initialize tracking arrays
        portfolio_value = np.zeros(n_days)
        option_positions = np.zeros(n_days)  # Always +1 option (sold)
        stock_positions = np.zeros(n_days)
        cash_positions = np.zeros(n_days)
        daily_pnl = np.zeros(n_days)
        transaction_costs = np.zeros(n_days)
        deltas = np.zeros(n_days)
        
        # Initial setup
        initial_T = T_initial
        days_per_year = 252
        
        # Sell one option at the beginning
        option_positions[0] = -1  # Short one option
        
        for i in range(n_days):
            S = prices[i]
            T = max(0.001, initial_T - i / days_per_year)  # Time decay
            
            # Calculate current option price and delta
            option_price = BSMModel.price(S, K, T, r, sigma, option_type)
            current_delta = GreeksCalculator.delta(S, K, T, r, sigma, option_type)
            deltas[i] = current_delta
            
            if i == 0:
                # Initial setup: sell option, buy delta shares
                cash_positions[i] = self.initial_cash + option_price  # Receive premium
                stock_positions[i] = current_delta  # Buy delta shares
                cash_positions[i] -= stock_positions[i] * S  # Pay for shares
                transaction_costs[i] = abs(stock_positions[i]) * self.transaction_cost
                cash_positions[i] -= transaction_costs[i]
                
            else:
                # Copy previous positions
                cash_positions[i] = cash_positions[i-1]
                stock_positions[i] = stock_positions[i-1]
                option_positions[i] = option_positions[i-1]
                
                # Rebalance if needed
                if i % rebalance_frequency == 0 or T <= 0.001:
                    target_delta = current_delta if T > 0.001 else 0
                    delta_change = target_delta - stock_positions[i]
                    
                    if abs(delta_change) > 0.001:  # Only rebalance if significant change
                        # Adjust stock position
                        stock_positions[i] += delta_change
                        cash_positions[i] -= delta_change * S
                        
                        # Transaction costs
                        transaction_costs[i] = abs(delta_change) * self.transaction_cost
                        cash_positions[i] -= transaction_costs[i]
            
            # Calculate portfolio value
            stock_value = stock_positions[i] * S
            option_value = option_positions[i] * option_price
            portfolio_value[i] = cash_positions[i] + stock_value + option_value
            
            # Calculate daily P&L
            if i > 0:
                daily_pnl[i] = portfolio_value[i] - portfolio_value[i-1]
        
        # Calculate statistics
        total_transaction_costs = np.sum(transaction_costs)
        final_pnl = portfolio_value[-1] - self.initial_cash
        volatility_pnl = np.std(daily_pnl[1:]) * np.sqrt(252)
        
        return {
            'strategy_name': f'Delta Hedge ({option_type.title()})',
            'portfolio_value': portfolio_value,
            'daily_pnl': daily_pnl,
            'cumulative_pnl': portfolio_value - self.initial_cash,
            'stock_positions': stock_positions,
            'option_positions': option_positions,
            'cash_positions': cash_positions,
            'deltas': deltas,
            'final_pnl': final_pnl,
            'total_transaction_costs': total_transaction_costs,
            'volatility_pnl': volatility_pnl,
            'max_stock_position': np.max(np.abs(stock_positions)),
            'rebalance_count': np.sum(transaction_costs > 0)
        }
    
    def simulate_gamma_hedge(self, stock_data: pd.DataFrame, K1: float, K2: float, 
                           T_initial: float, r: float, sigma: float, 
                           option_type: str = 'call') -> Dict[str, Any]:
        """
        Simulate delta-gamma neutral hedging using two options.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Historical stock price data
        K1, K2 : float
            Strike prices of the two options
        T_initial : float
            Initial time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        Dict[str, Any]
            Simulation results
        """
        prices = stock_data['Close'].values
        n_days = len(prices)
        
        # Initialize tracking arrays
        portfolio_value = np.zeros(n_days)
        option1_positions = np.zeros(n_days)  # Primary option (sold)
        option2_positions = np.zeros(n_days)  # Hedge option
        stock_positions = np.zeros(n_days)
        cash_positions = np.zeros(n_days)
        daily_pnl = np.zeros(n_days)
        
        initial_T = T_initial
        days_per_year = 252
        
        # Initial position: sell one option1
        option1_positions[0] = -1
        
        for i in range(n_days):
            S = prices[i]
            T = max(0.001, initial_T - i / days_per_year)
            
            # Calculate option prices and Greeks
            price1 = BSMModel.price(S, K1, T, r, sigma, option_type)
            price2 = BSMModel.price(S, K2, T, r, sigma, option_type)
            
            delta1 = GreeksCalculator.delta(S, K1, T, r, sigma, option_type)
            delta2 = GreeksCalculator.delta(S, K2, T, r, sigma, option_type)
            
            gamma1 = GreeksCalculator.gamma(S, K1, T, r, sigma)
            gamma2 = GreeksCalculator.gamma(S, K2, T, r, sigma)
            
            if i == 0:
                # Initial setup
                cash_positions[i] = self.initial_cash + price1  # Receive premium from selling option1
                
                # Calculate hedge ratios for gamma neutrality
                if abs(gamma2) > 1e-10:
                    option2_quantity = -option1_positions[i] * gamma1 / gamma2
                    option2_positions[i] = option2_quantity
                    cash_positions[i] -= option2_quantity * price2  # Pay for option2
                
                # Delta hedge with stock
                portfolio_delta = (option1_positions[i] * delta1 + 
                                 option2_positions[i] * delta2)
                stock_positions[i] = -portfolio_delta  # Delta neutral
                cash_positions[i] -= stock_positions[i] * S  # Pay for stock
                
            else:
                # Copy previous positions
                cash_positions[i] = cash_positions[i-1]
                stock_positions[i] = stock_positions[i-1]
                option1_positions[i] = option1_positions[i-1]
                option2_positions[i] = option2_positions[i-1]
                
                # Rebalance for delta neutrality (keep gamma hedge static)
                portfolio_delta = (option1_positions[i] * delta1 + 
                                 option2_positions[i] * delta2)
                target_stock_position = -portfolio_delta
                delta_change = target_stock_position - stock_positions[i]
                
                if abs(delta_change) > 0.001:
                    stock_positions[i] = target_stock_position
                    cash_positions[i] -= delta_change * S
            
            # Calculate portfolio value
            stock_value = stock_positions[i] * S
            option1_value = option1_positions[i] * price1
            option2_value = option2_positions[i] * price2
            portfolio_value[i] = cash_positions[i] + stock_value + option1_value + option2_value
            
            # Calculate daily P&L
            if i > 0:
                daily_pnl[i] = portfolio_value[i] - portfolio_value[i-1]
        
        final_pnl = portfolio_value[-1] - self.initial_cash
        
        return {
            'strategy_name': f'Delta-Gamma Hedge ({option_type.title()})',
            'portfolio_value': portfolio_value,
            'daily_pnl': daily_pnl,
            'cumulative_pnl': portfolio_value - self.initial_cash,
            'final_pnl': final_pnl,
            'option1_positions': option1_positions,
            'option2_positions': option2_positions,
            'stock_positions': stock_positions,
            'cash_positions': cash_positions
        }


class TradingStrategies:
    """
    Implementation of various options trading strategies.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize trading strategies simulator."""
        self.initial_capital = initial_capital
        self.transaction_cost_rate = 0.001  # 0.1% transaction cost
    
    def covered_call_strategy(self, stock_data: pd.DataFrame, K: float, T_initial: float,
                            r: float, sigma: float, shares_per_contract: int = 100) -> Dict[str, Any]:
        """
        Simulate covered call strategy.
        
        Strategy: Buy 100 shares, sell 1 call option.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Historical stock price data
        K : float
            Strike price of the call option
        T_initial : float
            Initial time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        shares_per_contract : int
            Number of shares per option contract
            
        Returns:
        --------
        Dict[str, Any]
            Strategy performance results
        """
        prices = stock_data['Close'].values
        n_days = len(prices)
        
        # Calculate number of covered call positions we can afford
        initial_stock_price = prices[0]
        call_premium = BSMModel.call_price(initial_stock_price, K, T_initial, r, sigma)
        
        # Cost per covered call unit = 100 shares - call premium received
        cost_per_unit = shares_per_contract * initial_stock_price - call_premium
        n_units = int(self.initial_capital / cost_per_unit) if cost_per_unit > 0 else 0
        
        if n_units == 0:
            return self._create_empty_result("Covered Call")
        
        # Initialize portfolio
        portfolio_value = np.zeros(n_days)
        daily_returns = np.zeros(n_days)
        
        # Initial setup
        stock_shares = n_units * shares_per_contract
        call_contracts = n_units
        cash = self.initial_capital - (stock_shares * initial_stock_price - call_contracts * call_premium)
        
        initial_T = T_initial
        days_per_year = 252
        option_expired = False
        
        for i in range(n_days):
            S = prices[i]
            T = max(0.0, initial_T - i / days_per_year)
            
            # Calculate portfolio components
            stock_value = stock_shares * S
            
            if not option_expired and T > 0:
                # Option still active
                call_value = BSMModel.call_price(S, K, T, r, sigma)
                short_call_value = -call_contracts * call_value  # We're short calls
            else:
                # Option expired or about to expire
                if not option_expired and T <= 0:
                    # Handle expiration
                    if S > K:
                        # Calls exercised - sell shares at strike price
                        shares_sold = call_contracts * shares_per_contract
                        stock_shares -= shares_sold
                        cash += shares_sold * K
                    option_expired = True
                
                short_call_value = 0  # No option value after expiration
            
            portfolio_value[i] = stock_value + short_call_value + cash
            
            # Calculate daily returns
            if i > 0:
                daily_returns[i] = (portfolio_value[i] - portfolio_value[i-1]) / portfolio_value[i-1]
        
        return self._calculate_strategy_performance(
            "Covered Call", portfolio_value, daily_returns, self.initial_capital
        )
    
    def long_straddle_strategy(self, stock_data: pd.DataFrame, K: float, T_initial: float,
                             r: float, sigma: float) -> Dict[str, Any]:
        """
        Simulate long straddle strategy.
        
        Strategy: Buy call and put at the same strike.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Historical stock price data
        K : float
            Strike price (same for call and put)
        T_initial : float
            Initial time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
            
        Returns:
        --------
        Dict[str, Any]
            Strategy performance results
        """
        prices = stock_data['Close'].values
        n_days = len(prices)
        
        initial_stock_price = prices[0]
        call_price = BSMModel.call_price(initial_stock_price, K, T_initial, r, sigma)
        put_price = BSMModel.put_price(initial_stock_price, K, T_initial, r, sigma)
        
        # Cost per straddle
        straddle_cost = call_price + put_price
        n_straddles = int(self.initial_capital / straddle_cost)
        
        if n_straddles == 0:
            return self._create_empty_result("Long Straddle")
        
        # Initialize portfolio
        portfolio_value = np.zeros(n_days)
        daily_returns = np.zeros(n_days)
        
        cash = self.initial_capital - n_straddles * straddle_cost
        
        initial_T = T_initial
        days_per_year = 252
        
        for i in range(n_days):
            S = prices[i]
            T = max(0.0, initial_T - i / days_per_year)
            
            if T > 0:
                # Options still active
                call_value = BSMModel.call_price(S, K, T, r, sigma)
                put_value = BSMModel.put_price(S, K, T, r, sigma)
                straddle_value = call_value + put_value
            else:
                # Options expired
                call_payoff = max(0, S - K)
                put_payoff = max(0, K - S)
                straddle_value = call_payoff + put_payoff
            
            portfolio_value[i] = cash + n_straddles * straddle_value
            
            # Calculate daily returns
            if i > 0:
                daily_returns[i] = (portfolio_value[i] - portfolio_value[i-1]) / portfolio_value[i-1]
        
        return self._calculate_strategy_performance(
            "Long Straddle", portfolio_value, daily_returns, self.initial_capital
        )
    
    def delta_neutral_speculation(self, stock_data: pd.DataFrame, K: float, T_initial: float,
                                r: float, sigma_model: float, sigma_forecast: float) -> Dict[str, Any]:
        """
        Simulate delta-neutral speculation strategy.
        
        Strategy: Trade volatility by maintaining delta-neutral position
        and betting on volatility forecast vs model volatility.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Historical stock price data
        K : float
            Strike price
        T_initial : float
            Initial time to maturity
        r : float
            Risk-free rate
        sigma_model : float
            Model volatility (used for pricing/hedging)
        sigma_forecast : float
            Forecasted volatility (our bet)
            
        Returns:
        --------
        Dict[str, Any]
            Strategy performance results
        """
        # This is a simplified version - in practice, this would be more complex
        prices = stock_data['Close'].values
        n_days = len(prices)
        
        # Direction of bet: if we think vol is higher than model, buy options
        vol_bet_direction = 1 if sigma_forecast > sigma_model else -1
        
        initial_stock_price = prices[0]
        call_price = BSMModel.call_price(initial_stock_price, K, T_initial, r, sigma_model)
        
        # Use a fraction of capital for this strategy
        capital_fraction = 0.5
        strategy_capital = self.initial_capital * capital_fraction
        n_contracts = int(strategy_capital / call_price)
        
        if n_contracts == 0:
            return self._create_empty_result("Delta-Neutral Speculation")
        
        # Initialize portfolio
        portfolio_value = np.zeros(n_days)
        daily_returns = np.zeros(n_days)
        
        # Start with buying/selling options and hedging with stock
        option_position = n_contracts * vol_bet_direction  # +1 if bullish on vol, -1 if bearish
        
        initial_T = T_initial
        days_per_year = 252
        
        # Initial delta hedge
        initial_delta = GreeksCalculator.delta(initial_stock_price, K, T_initial, r, sigma_model, 'call')
        stock_position = -option_position * initial_delta  # Hedge delta
        
        cash = (self.initial_capital - 
                abs(option_position) * call_price - 
                abs(stock_position) * initial_stock_price)
        
        for i in range(n_days):
            S = prices[i]
            T = max(0.001, initial_T - i / days_per_year)
            
            # Calculate current values using model volatility
            call_value = BSMModel.call_price(S, K, T, r, sigma_model)
            
            # Portfolio value
            option_value = option_position * call_value
            stock_value = stock_position * S
            portfolio_value[i] = cash + option_value + stock_value
            
            # Calculate daily returns
            if i > 0:
                daily_returns[i] = (portfolio_value[i] - portfolio_value[i-1]) / portfolio_value[i-1]
            
            # Rebalance delta hedge periodically (simplified)
            if i % 5 == 0 and T > 0.001:  # Rebalance every 5 days
                current_delta = GreeksCalculator.delta(S, K, T, r, sigma_model, 'call')
                target_stock_position = -option_position * current_delta
                position_change = target_stock_position - stock_position
                
                # Update positions (ignoring transaction costs for simplicity)
                stock_position = target_stock_position
                cash -= position_change * S
        
        return self._calculate_strategy_performance(
            "Delta-Neutral Speculation", portfolio_value, daily_returns, self.initial_capital
        )
    
    def buy_and_hold_benchmark(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple buy and hold benchmark strategy.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Historical stock price data
            
        Returns:
        --------
        Dict[str, Any]
            Benchmark performance results
        """
        prices = stock_data['Close'].values
        n_days = len(prices)
        
        # Buy as many shares as possible with initial capital
        initial_price = prices[0]
        n_shares = self.initial_capital / initial_price
        
        portfolio_value = n_shares * prices
        daily_returns = np.zeros(n_days)
        
        for i in range(1, n_days):
            daily_returns[i] = (portfolio_value[i] - portfolio_value[i-1]) / portfolio_value[i-1]
        
        return self._calculate_strategy_performance(
            "Buy & Hold", portfolio_value, daily_returns, self.initial_capital
        )
    
    def _calculate_strategy_performance(self, strategy_name: str, portfolio_values: np.ndarray,
                                     daily_returns: np.ndarray, initial_capital: float) -> Dict[str, Any]:
        """
        Calculate performance metrics for a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        portfolio_values : np.ndarray
            Daily portfolio values
        daily_returns : np.ndarray
            Daily returns
        initial_capital : float
            Initial capital invested
            
        Returns:
        --------
        Dict[str, Any]
            Performance metrics
        """
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Remove first day (zero return) for calculations
        returns = daily_returns[1:]
        
        # Calculate Sharpe ratio (assuming daily returns)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        cumulative_values = portfolio_values
        peak_value = np.maximum.accumulate(cumulative_values)
        drawdowns = (cumulative_values - peak_value) / peak_value
        max_drawdown = np.min(drawdowns)
        
        # Calculate win rate
        winning_days = np.sum(returns > 0)
        total_trading_days = len(returns)
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0.0
        
        return {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'final_value': final_value,
            'final_pnl': final_value - initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'daily_returns': daily_returns,
            'cumulative_pnl': portfolio_values - initial_capital,
            'portfolio_values': portfolio_values,
            'trade_count': 1,  # Simplified
            'volatility_annual': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        }
    
    def _create_empty_result(self, strategy_name: str) -> Dict[str, Any]:
        """Create empty result for failed strategies."""
        return {
            'strategy_name': strategy_name,
            'total_return': 0.0,
            'final_value': self.initial_capital,
            'final_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'daily_returns': np.array([0.0]),
            'cumulative_pnl': np.array([0.0]),
            'portfolio_values': np.array([self.initial_capital]),
            'trade_count': 0,
            'volatility_annual': 0.0,
            'error': 'Insufficient capital or invalid parameters'
        }


class StrategyComparison:
    """
    Tools for comparing multiple strategies.
    """
    
    @staticmethod
    def compare_strategies(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a comparison table of strategy results.
        
        Parameters:
        -----------
        results : List[Dict[str, Any]]
            List of strategy result dictionaries
            
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Total Return': f"{result['total_return']:.2%}",
                'Final P&L': f"${result['final_pnl']:.2f}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Win Rate': f"{result['win_rate']:.2%}",
                'Annual Volatility': f"{result.get('volatility_annual', 0):.2%}"
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def print_strategy_comparison(results: List[Dict[str, Any]]) -> None:
        """
        Print a formatted comparison of strategies.
        """
        print(f"\n{'='*80}")
        print("STRATEGY PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        comparison_df = StrategyComparison.compare_strategies(results)
        
        # Print table with proper formatting
        for idx, row in comparison_df.iterrows():
            print(f"\n{row['Strategy']}:")
            print(f"  Total Return:     {row['Total Return']:>12}")
            print(f"  Final P&L:        {row['Final P&L']:>12}")
            print(f"  Sharpe Ratio:     {row['Sharpe Ratio']:>12}")
            print(f"  Max Drawdown:     {row['Max Drawdown']:>12}")
            print(f"  Win Rate:         {row['Win Rate']:>12}")
            print(f"  Volatility:       {row['Annual Volatility']:>12}")
        
        # Find best strategy by Sharpe ratio
        sharpe_ratios = [float(row['Sharpe Ratio']) for _, row in comparison_df.iterrows()]
        if sharpe_ratios:
            best_idx = np.argmax(sharpe_ratios)
            best_strategy = comparison_df.iloc[best_idx]['Strategy']
            
            print(f"\n{'='*80}")
            print(f"BEST STRATEGY (by Sharpe Ratio): {best_strategy}")
            print(f"{'='*80}")


# Convenience functions
def simulate_delta_hedge(stock_data: pd.DataFrame, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call') -> Dict[str, Any]:
    """Convenience function for delta hedging simulation."""
    simulator = HedgingSimulator()
    return simulator.simulate_delta_hedge(stock_data, K, T, r, sigma, option_type)

def backtest_strategies(stock_data: pd.DataFrame, K: float, T: float, r: float, 
                       sigma: float, initial_capital: float = 10000.0) -> List[Dict[str, Any]]:
    """
    Backtest multiple strategies and return results.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Historical stock price data
    K : float
        Strike price for options
    T : float
        Time to maturity
    r : float
        Risk-free rate  
    sigma : float
        Volatility
    initial_capital : float
        Initial capital for strategies
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of strategy results
    """
    strategies = TradingStrategies(initial_capital)
    
    results = []
    
    # Buy and hold benchmark
    results.append(strategies.buy_and_hold_benchmark(stock_data))
    
    # Covered call strategy
    results.append(strategies.covered_call_strategy(stock_data, K, T, r, sigma))
    
    # Long straddle strategy  
    results.append(strategies.long_straddle_strategy(stock_data, K, T, r, sigma))
    
    # Delta-neutral speculation (assuming we forecast higher volatility)
    sigma_forecast = sigma * 1.2  # Assume we think volatility will be 20% higher
    results.append(strategies.delta_neutral_speculation(stock_data, K, T, r, sigma, sigma_forecast))
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Trading Strategies and Hedging Simulations")
    print("=" * 60)
    
    # Create sample stock data
    np.random.seed(42)
    n_days = 252  # One year
    S0 = 100
    r = 0.05
    sigma = 0.2
    dt = 1/252
    
    # Generate GBM path
    returns = np.random.normal((r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days-1)
    prices = np.zeros(n_days)
    prices[0] = S0
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * np.exp(returns[i-1])
    
    # Create DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    stock_data = pd.DataFrame({'Close': prices}, index=dates)
    
    print(f"Generated {n_days} days of sample stock data")
    print(f"Initial Price: ${prices[0]:.2f}")
    print(f"Final Price: ${prices[-1]:.2f}")
    print(f"Realized Return: {(prices[-1]/prices[0] - 1):.2%}")
    
    # Test parameters
    K = 105  # Strike price
    T = 0.5  # 6 months
    
    print(f"\nStrategy Parameters:")
    print(f"Strike Price: ${K}")
    print(f"Time to Maturity: {T} years")
    print(f"Risk-free Rate: {r:.1%}")
    print(f"Volatility: {sigma:.1%}")
    
    # Test delta hedging
    print(f"\n{'-'*60}")
    print("DELTA HEDGING SIMULATION")
    print(f"{'-'*60}")
    
    hedge_simulator = HedgingSimulator(initial_cash=0)
    hedge_result = hedge_simulator.simulate_delta_hedge(stock_data, K, T, r, sigma, 'call')
    
    print(f"Delta Hedge Results:")
    print(f"Final P&L: ${hedge_result['final_pnl']:.2f}")
    print(f"P&L Volatility: {hedge_result['volatility_pnl']:.2%}")
    print(f"Max Stock Position: {hedge_result['max_stock_position']:.2f} shares")
    print(f"Rebalance Count: {hedge_result['rebalance_count']}")
    print(f"Transaction Costs: ${hedge_result['total_transaction_costs']:.2f}")
    
    # Test strategies
    print(f"\n{'-'*60}")
    print("STRATEGY BACKTESTING")
    print(f"{'-'*60}")
    
    strategy_results = backtest_strategies(stock_data, K, T, r, sigma, initial_capital=10000)
    
    # Print comparison
    StrategyComparison.print_strategy_comparison(strategy_results)
    
    print(f"\n{'-'*60}")
    print("Testing completed successfully!")
    print(f"{'-'*60}")