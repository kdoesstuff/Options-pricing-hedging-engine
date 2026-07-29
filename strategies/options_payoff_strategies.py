#!/usr/bin/env python3
"""
Options Payoff Strategies

This module provides comprehensive analysis and visualization of various options strategies
including their payoff diagrams, profit/loss calculations, and breakeven points.

Supported Strategies:
- Single Options: Long Call, Long Put, Short Call, Short Put
- Spreads: Bull Call, Bear Put, Bull Put, Bear Call
- Straddles: Long Straddle, Short Straddle, Strangle
- Complex: Butterfly, Iron Condor, Iron Butterfly, Collar
- Ratio: Ratio Spreads, Calendar Spreads

Usage:
    from strategies.options_payoff_strategies import OptionsPayoffAnalyzer
    
    analyzer = OptionsPayoffAnalyzer(spot_price=100)
    payoff_data = analyzer.long_call(strike=105, premium=3)
    analyzer.plot_payoff(payoff_data, "Long Call")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class OptionLeg:
    """Represents a single option leg in a strategy."""
    option_type: str  # 'call' or 'put'
    position: str     # 'long' or 'short'
    strike: float
    premium: float
    quantity: int = 1

@dataclass
class PayoffResult:
    """Contains payoff analysis results for an options strategy."""
    strategy_name: str
    spot_prices: np.ndarray
    payoffs: np.ndarray
    breakeven_points: List[float]
    max_profit: float
    max_loss: float
    profit_range: Tuple[float, float]
    loss_range: Tuple[float, float]
    legs: List[OptionLeg]
    net_premium: float

class OptionsPayoffAnalyzer:
    """
    Comprehensive options payoff analyzer for various strategies.
    """
    
    def __init__(self, spot_price: float = 100, price_range_factor: float = 0.4):
        """
        Initialize the options payoff analyzer.
        
        Parameters:
        -----------
        spot_price : float
            Current stock price
        price_range_factor : float
            Factor to determine price range for analysis (±factor * spot_price)
        """
        self.spot_price = spot_price
        self.price_range_factor = price_range_factor
        
        # Generate price range for analysis
        self.min_price = spot_price * (1 - price_range_factor)
        self.max_price = spot_price * (1 + price_range_factor)
        self.spot_prices = np.linspace(self.min_price, self.max_price, 200)
        
        print(f"Options Payoff Analyzer initialized")
        print(f"   Spot Price: ${spot_price:.2f}")
        print(f"   Analysis Range: ${self.min_price:.2f} - ${self.max_price:.2f}")
    
    def _calculate_option_payoff(self, leg: OptionLeg, spot_prices: np.ndarray) -> np.ndarray:
        """Calculate payoff for a single option leg."""
        if leg.option_type.lower() == 'call':
            intrinsic_value = np.maximum(spot_prices - leg.strike, 0)
        else:  # put
            intrinsic_value = np.maximum(leg.strike - spot_prices, 0)
        
        if leg.position.lower() == 'long':
            payoff = (intrinsic_value - leg.premium) * leg.quantity
        else:  # short
            payoff = (leg.premium - intrinsic_value) * leg.quantity
        
        return payoff
    
    def _find_breakeven_points(self, spot_prices: np.ndarray, payoffs: np.ndarray) -> List[float]:
        """Find breakeven points where payoff crosses zero."""
        breakeven_points = []
        
        for i in range(len(payoffs) - 1):
            if (payoffs[i] <= 0 and payoffs[i + 1] > 0) or (payoffs[i] > 0 and payoffs[i + 1] <= 0):
                # Linear interpolation to find exact breakeven
                x1, x2 = spot_prices[i], spot_prices[i + 1]
                y1, y2 = payoffs[i], payoffs[i + 1]
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(breakeven)
        
        return sorted(breakeven_points)
    
    def _analyze_strategy(self, legs: List[OptionLeg], strategy_name: str) -> PayoffResult:
        """Analyze a complete options strategy."""
        
        # Calculate total payoff
        total_payoff = np.zeros_like(self.spot_prices)
        net_premium = 0
        
        for leg in legs:
            leg_payoff = self._calculate_option_payoff(leg, self.spot_prices)
            total_payoff += leg_payoff
            
            # Calculate net premium (paid is positive, received is negative)
            if leg.position.lower() == 'long':
                net_premium += leg.premium * leg.quantity
            else:
                net_premium -= leg.premium * leg.quantity
        
        # Find breakeven points
        breakeven_points = self._find_breakeven_points(self.spot_prices, total_payoff)
        
        # Calculate profit and loss metrics
        max_profit = np.max(total_payoff) if np.max(total_payoff) != np.inf else float('inf')
        max_loss = np.min(total_payoff) if np.min(total_payoff) != -np.inf else float('-inf')
        
        # Find profit and loss ranges
        profit_mask = total_payoff > 0
        loss_mask = total_payoff < 0
        
        if np.any(profit_mask):
            profit_range = (np.min(self.spot_prices[profit_mask]), np.max(self.spot_prices[profit_mask]))
        else:
            profit_range = (None, None)
        
        if np.any(loss_mask):
            loss_range = (np.min(self.spot_prices[loss_mask]), np.max(self.spot_prices[loss_mask]))
        else:
            loss_range = (None, None)
        
        return PayoffResult(
            strategy_name=strategy_name,
            spot_prices=self.spot_prices,
            payoffs=total_payoff,
            breakeven_points=breakeven_points,
            max_profit=max_profit,
            max_loss=max_loss,
            profit_range=profit_range,
            loss_range=loss_range,
            legs=legs,
            net_premium=net_premium
        )
    
    # =====================================================================
    # SINGLE OPTION STRATEGIES
    # =====================================================================
    
    def long_call(self, strike: float = None, premium: float = None) -> PayoffResult:
        """Long Call strategy."""
        if strike is None:
            strike = self.spot_price * 1.05
        if premium is None:
            premium = max(0.5, self.spot_price * 0.02)
        
        legs = [OptionLeg('call', 'long', strike, premium)]
        return self._analyze_strategy(legs, 'Long Call')
    
    def short_call(self, strike: float = None, premium: float = None) -> PayoffResult:
        """Short Call strategy."""
        if strike is None:
            strike = self.spot_price * 1.05
        if premium is None:
            premium = max(0.5, self.spot_price * 0.02)
        
        legs = [OptionLeg('call', 'short', strike, premium)]
        return self._analyze_strategy(legs, 'Short Call')
    
    def long_put(self, strike: float = None, premium: float = None) -> PayoffResult:
        """Long Put strategy."""
        if strike is None:
            strike = self.spot_price * 0.95
        if premium is None:
            premium = max(0.5, self.spot_price * 0.02)
        
        legs = [OptionLeg('put', 'long', strike, premium)]
        return self._analyze_strategy(legs, 'Long Put')
    
    def short_put(self, strike: float = None, premium: float = None) -> PayoffResult:
        """Short Put strategy."""
        if strike is None:
            strike = self.spot_price * 0.95
        if premium is None:
            premium = max(0.5, self.spot_price * 0.02)
        
        legs = [OptionLeg('put', 'short', strike, premium)]
        return self._analyze_strategy(legs, 'Short Put')
    
    # =====================================================================
    # SPREAD STRATEGIES
    # =====================================================================
    
    def bull_call_spread(self, lower_strike: float = None, upper_strike: float = None,
                        lower_premium: float = None, upper_premium: float = None) -> PayoffResult:
        """Bull Call Spread strategy."""
        if lower_strike is None:
            lower_strike = self.spot_price * 0.98
        if upper_strike is None:
            upper_strike = self.spot_price * 1.08
        if lower_premium is None:
            lower_premium = self.spot_price * 0.03
        if upper_premium is None:
            upper_premium = self.spot_price * 0.01
        
        legs = [
            OptionLeg('call', 'long', lower_strike, lower_premium),
            OptionLeg('call', 'short', upper_strike, upper_premium)
        ]
        return self._analyze_strategy(legs, 'Bull Call Spread')
    
    def bear_put_spread(self, lower_strike: float = None, upper_strike: float = None,
                       lower_premium: float = None, upper_premium: float = None) -> PayoffResult:
        """Bear Put Spread strategy."""
        if lower_strike is None:
            lower_strike = self.spot_price * 0.92
        if upper_strike is None:
            upper_strike = self.spot_price * 1.02
        if lower_premium is None:
            lower_premium = self.spot_price * 0.01
        if upper_premium is None:
            upper_premium = self.spot_price * 0.03
        
        legs = [
            OptionLeg('put', 'long', upper_strike, upper_premium),
            OptionLeg('put', 'short', lower_strike, lower_premium)
        ]
        return self._analyze_strategy(legs, 'Bear Put Spread')
    
    def bull_put_spread(self, lower_strike: float = None, upper_strike: float = None,
                       lower_premium: float = None, upper_premium: float = None) -> PayoffResult:
        """Bull Put Spread strategy."""
        if lower_strike is None:
            lower_strike = self.spot_price * 0.92
        if upper_strike is None:
            upper_strike = self.spot_price * 1.02
        if lower_premium is None:
            lower_premium = self.spot_price * 0.01
        if upper_premium is None:
            upper_premium = self.spot_price * 0.03
        
        legs = [
            OptionLeg('put', 'short', upper_strike, upper_premium),
            OptionLeg('put', 'long', lower_strike, lower_premium)
        ]
        return self._analyze_strategy(legs, 'Bull Put Spread')
    
    def bear_call_spread(self, lower_strike: float = None, upper_strike: float = None,
                        lower_premium: float = None, upper_premium: float = None) -> PayoffResult:
        """Bear Call Spread strategy."""
        if lower_strike is None:
            lower_strike = self.spot_price * 0.98
        if upper_strike is None:
            upper_strike = self.spot_price * 1.08
        if lower_premium is None:
            lower_premium = self.spot_price * 0.03
        if upper_premium is None:
            upper_premium = self.spot_price * 0.01
        
        legs = [
            OptionLeg('call', 'short', lower_strike, lower_premium),
            OptionLeg('call', 'long', upper_strike, upper_premium)
        ]
        return self._analyze_strategy(legs, 'Bear Call Spread')
    
    # =====================================================================
    # STRADDLE AND STRANGLE STRATEGIES
    # =====================================================================
    
    def long_straddle(self, strike: float = None, call_premium: float = None, 
                     put_premium: float = None) -> PayoffResult:
        """Long Straddle strategy."""
        if strike is None:
            strike = self.spot_price
        if call_premium is None:
            call_premium = self.spot_price * 0.025
        if put_premium is None:
            put_premium = self.spot_price * 0.025
        
        legs = [
            OptionLeg('call', 'long', strike, call_premium),
            OptionLeg('put', 'long', strike, put_premium)
        ]
        return self._analyze_strategy(legs, 'Long Straddle')
    
    def short_straddle(self, strike: float = None, call_premium: float = None, 
                      put_premium: float = None) -> PayoffResult:
        """Short Straddle strategy."""
        if strike is None:
            strike = self.spot_price
        if call_premium is None:
            call_premium = self.spot_price * 0.025
        if put_premium is None:
            put_premium = self.spot_price * 0.025
        
        legs = [
            OptionLeg('call', 'short', strike, call_premium),
            OptionLeg('put', 'short', strike, put_premium)
        ]
        return self._analyze_strategy(legs, 'Short Straddle')
    
    def long_strangle(self, call_strike: float = None, put_strike: float = None,
                     call_premium: float = None, put_premium: float = None) -> PayoffResult:
        """Long Strangle strategy."""
        if call_strike is None:
            call_strike = self.spot_price * 1.05
        if put_strike is None:
            put_strike = self.spot_price * 0.95
        if call_premium is None:
            call_premium = self.spot_price * 0.02
        if put_premium is None:
            put_premium = self.spot_price * 0.02
        
        legs = [
            OptionLeg('call', 'long', call_strike, call_premium),
            OptionLeg('put', 'long', put_strike, put_premium)
        ]
        return self._analyze_strategy(legs, 'Long Strangle')
    
    def short_strangle(self, call_strike: float = None, put_strike: float = None,
                      call_premium: float = None, put_premium: float = None) -> PayoffResult:
        """Short Strangle strategy."""
        if call_strike is None:
            call_strike = self.spot_price * 1.05
        if put_strike is None:
            put_strike = self.spot_price * 0.95
        if call_premium is None:
            call_premium = self.spot_price * 0.02
        if put_premium is None:
            put_premium = self.spot_price * 0.02
        
        legs = [
            OptionLeg('call', 'short', call_strike, call_premium),
            OptionLeg('put', 'short', put_strike, put_premium)
        ]
        return self._analyze_strategy(legs, 'Short Strangle')
    
    # =====================================================================
    # COMPLEX STRATEGIES
    # =====================================================================
    
    def butterfly_spread(self, lower_strike: float = None, middle_strike: float = None,
                        upper_strike: float = None, option_type: str = 'call') -> PayoffResult:
        """Butterfly Spread strategy."""
        if lower_strike is None:
            lower_strike = self.spot_price * 0.95
        if middle_strike is None:
            middle_strike = self.spot_price
        if upper_strike is None:
            upper_strike = self.spot_price * 1.05
        
        # Simplified premium calculation
        premium_factor = 0.015
        lower_premium = self.spot_price * premium_factor * 1.5
        middle_premium = self.spot_price * premium_factor
        upper_premium = self.spot_price * premium_factor * 0.5
        
        legs = [
            OptionLeg(option_type, 'long', lower_strike, lower_premium),
            OptionLeg(option_type, 'short', middle_strike, middle_premium, quantity=2),
            OptionLeg(option_type, 'long', upper_strike, upper_premium)
        ]
        return self._analyze_strategy(legs, f'{option_type.title()} Butterfly')
    
    def iron_condor(self, put_lower_strike: float = None, put_upper_strike: float = None,
                   call_lower_strike: float = None, call_upper_strike: float = None) -> PayoffResult:
        """Iron Condor strategy."""
        if put_lower_strike is None:
            put_lower_strike = self.spot_price * 0.90
        if put_upper_strike is None:
            put_upper_strike = self.spot_price * 0.95
        if call_lower_strike is None:
            call_lower_strike = self.spot_price * 1.05
        if call_upper_strike is None:
            call_upper_strike = self.spot_price * 1.10
        
        # Simplified premium calculation
        premium_factor = 0.01
        
        legs = [
            OptionLeg('put', 'long', put_lower_strike, premium_factor * self.spot_price),
            OptionLeg('put', 'short', put_upper_strike, premium_factor * self.spot_price * 2),
            OptionLeg('call', 'short', call_lower_strike, premium_factor * self.spot_price * 2),
            OptionLeg('call', 'long', call_upper_strike, premium_factor * self.spot_price)
        ]
        return self._analyze_strategy(legs, 'Iron Condor')
    
    def iron_butterfly(self, strike: float = None, put_strike: float = None, 
                      call_strike: float = None) -> PayoffResult:
        """Iron Butterfly strategy."""
        if strike is None:
            strike = self.spot_price
        if put_strike is None:
            put_strike = self.spot_price * 0.95
        if call_strike is None:
            call_strike = self.spot_price * 1.05
        
        premium_factor = 0.015
        
        legs = [
            OptionLeg('put', 'long', put_strike, premium_factor * self.spot_price),
            OptionLeg('put', 'short', strike, premium_factor * self.spot_price * 1.5),
            OptionLeg('call', 'short', strike, premium_factor * self.spot_price * 1.5),
            OptionLeg('call', 'long', call_strike, premium_factor * self.spot_price)
        ]
        return self._analyze_strategy(legs, 'Iron Butterfly')
    
    def protective_collar(self, put_strike: float = None, call_strike: float = None,
                         put_premium: float = None, call_premium: float = None) -> PayoffResult:
        """Protective Collar strategy (includes stock position)."""
        if put_strike is None:
            put_strike = self.spot_price * 0.90
        if call_strike is None:
            call_strike = self.spot_price * 1.10
        if put_premium is None:
            put_premium = self.spot_price * 0.02
        if call_premium is None:
            call_premium = self.spot_price * 0.02
        
        # This strategy includes a stock position
        # We'll simulate it with a synthetic stock position using put-call parity
        legs = [
            OptionLeg('put', 'long', put_strike, put_premium),
            OptionLeg('call', 'short', call_strike, call_premium)
        ]
        
        # Add synthetic stock position (long stock = long call + short put at same strike)
        synthetic_premium = 0  # Net zero for synthetic stock
        legs.extend([
            OptionLeg('call', 'long', self.spot_price, synthetic_premium),
            OptionLeg('put', 'short', self.spot_price, synthetic_premium)
        ])
        
        return self._analyze_strategy(legs, 'Protective Collar')
    
    # =====================================================================
    # ANALYSIS AND VISUALIZATION
    # =====================================================================
    
    def compare_strategies(self, strategies: List[PayoffResult]) -> pd.DataFrame:
        """Compare multiple strategies in a table format."""
        
        comparison_data = []
        
        for strategy in strategies:
            data = {
                'Strategy': strategy.strategy_name,
                'Max Profit': f"${strategy.max_profit:.2f}" if strategy.max_profit != float('inf') else "Unlimited",
                'Max Loss': f"${abs(strategy.max_loss):.2f}" if strategy.max_loss != float('-inf') else "Unlimited",
                'Breakeven Points': len(strategy.breakeven_points),
                'Net Premium': f"${strategy.net_premium:.2f}",
                'Risk/Reward': f"{abs(strategy.max_loss)/strategy.max_profit:.2f}" if strategy.max_profit > 0 and strategy.max_loss != float('-inf') else "N/A"
            }
            comparison_data.append(data)
        
        return pd.DataFrame(comparison_data)
    
    def plot_payoff(self, payoff_result: PayoffResult, title: str = None, 
                   save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """Plot payoff diagram for a strategy."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot payoff line
        ax.plot(payoff_result.spot_prices, payoff_result.payoffs, 
               linewidth=3, label=payoff_result.strategy_name, color='blue')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add current spot price line
        ax.axvline(x=self.spot_price, color='red', linestyle='--', alpha=0.7, 
                  label=f'Current Price: ${self.spot_price:.2f}')
        
        # Mark breakeven points
        for i, breakeven in enumerate(payoff_result.breakeven_points):
            ax.axvline(x=breakeven, color='green', linestyle=':', alpha=0.7)
            ax.annotate(f'Breakeven: ${breakeven:.2f}', 
                       xy=(breakeven, 0), xytext=(10, 20),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Highlight profit and loss areas
        profit_mask = payoff_result.payoffs > 0
        loss_mask = payoff_result.payoffs < 0
        
        if np.any(profit_mask):
            ax.fill_between(payoff_result.spot_prices, payoff_result.payoffs, 0, 
                          where=profit_mask, alpha=0.3, color='green', label='Profit Area')
        
        if np.any(loss_mask):
            ax.fill_between(payoff_result.spot_prices, payoff_result.payoffs, 0, 
                          where=loss_mask, alpha=0.3, color='red', label='Loss Area')
        
        # Formatting
        ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax.set_title(title or f'{payoff_result.strategy_name} Payoff Diagram', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add strategy details text box
        details = f"""Strategy Details:
Max Profit: ${payoff_result.max_profit:.2f} {'(Unlimited)' if payoff_result.max_profit == float('inf') else ''}
Max Loss: ${abs(payoff_result.max_loss):.2f} {'(Unlimited)' if payoff_result.max_loss == float('-inf') else ''}
Net Premium: ${payoff_result.net_premium:.2f}
Breakevens: {len(payoff_result.breakeven_points)}"""
        
        ax.text(0.02, 0.98, details, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_multiple_strategies(self, strategies: List[PayoffResult], 
                               title: str = "Options Strategies Comparison",
                               save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """Plot multiple strategies on the same chart."""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        for i, strategy in enumerate(strategies):
            ax.plot(strategy.spot_prices, strategy.payoffs, 
                   linewidth=2.5, label=strategy.strategy_name, color=colors[i])
            
            # Mark breakeven points for each strategy
            for breakeven in strategy.breakeven_points:
                ax.plot(breakeven, 0, 'o', color=colors[i], markersize=6, alpha=0.7)
        
        # Add zero line and current price line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=self.spot_price, color='red', linestyle='--', alpha=0.7, 
                  label=f'Current Price: ${self.spot_price:.2f}')
        
        # Formatting
        ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def generate_strategy_report(self, payoff_result: PayoffResult) -> str:
        """Generate a comprehensive text report for a strategy."""
        
        report = f"""
{'='*60}
OPTIONS STRATEGY ANALYSIS REPORT
{'='*60}

Strategy: {payoff_result.strategy_name}
Current Stock Price: ${self.spot_price:.2f}

STRATEGY COMPOSITION:
{'-'*30}
"""
        
        for i, leg in enumerate(payoff_result.legs, 1):
            position_desc = f"{leg.position.title()} {leg.option_type.title()}"
            if leg.quantity != 1:
                position_desc += f" (x{leg.quantity})"
            
            report += f"{i}. {position_desc}\n"
            report += f"   Strike: ${leg.strike:.2f}\n"
            report += f"   Premium: ${leg.premium:.2f}\n\n"
        
        report += f"""PAYOFF ANALYSIS:
{'-'*30}
Net Premium Paid/Received: ${payoff_result.net_premium:.2f}
Maximum Profit: ${payoff_result.max_profit:.2f} {'(Unlimited)' if payoff_result.max_profit == float('inf') else ''}
Maximum Loss: ${abs(payoff_result.max_loss):.2f} {'(Unlimited)' if payoff_result.max_loss == float('-inf') else ''}

BREAKEVEN ANALYSIS:
{'-'*30}
Number of Breakeven Points: {len(payoff_result.breakeven_points)}
"""
        
        for i, breakeven in enumerate(payoff_result.breakeven_points, 1):
            report += f"Breakeven {i}: ${breakeven:.2f}\n"
        
        if payoff_result.profit_range[0] is not None:
            report += f"\nProfit Range: ${payoff_result.profit_range[0]:.2f} - ${payoff_result.profit_range[1]:.2f}\n"
        
        if payoff_result.loss_range[0] is not None:
            report += f"Loss Range: ${payoff_result.loss_range[0]:.2f} - ${payoff_result.loss_range[1]:.2f}\n"
        
        report += f"\n{'='*60}\n"
        
        return report

# =====================================================================
# CONVENIENCE FUNCTIONS AND PRESETS
# =====================================================================

def get_popular_strategies(spot_price: float = 100) -> Dict[str, PayoffResult]:
    """Get payoff results for popular options strategies."""
    
    analyzer = OptionsPayoffAnalyzer(spot_price)
    
    strategies = {
        'Long Call': analyzer.long_call(),
        'Long Put': analyzer.long_put(),
        'Bull Call Spread': analyzer.bull_call_spread(),
        'Bear Put Spread': analyzer.bear_put_spread(),
        'Long Straddle': analyzer.long_straddle(),
        'Long Strangle': analyzer.long_strangle(),
        'Butterfly Spread': analyzer.butterfly_spread(),
        'Iron Condor': analyzer.iron_condor()
    }
    
    return strategies

def create_strategy_comparison_chart(spot_price: float = 100, 
                                   strategies: List[str] = None) -> plt.Figure:
    """Create a comparison chart for selected strategies."""
    
    if strategies is None:
        strategies = ['Long Call', 'Long Put', 'Bull Call Spread', 'Long Straddle']
    
    analyzer = OptionsPayoffAnalyzer(spot_price)
    payoff_results = []
    
    strategy_methods = {
        'Long Call': analyzer.long_call,
        'Short Call': analyzer.short_call,
        'Long Put': analyzer.long_put,
        'Short Put': analyzer.short_put,
        'Bull Call Spread': analyzer.bull_call_spread,
        'Bear Put Spread': analyzer.bear_put_spread,
        'Bull Put Spread': analyzer.bull_put_spread,
        'Bear Call Spread': analyzer.bear_call_spread,
        'Long Straddle': analyzer.long_straddle,
        'Short Straddle': analyzer.short_straddle,
        'Long Strangle': analyzer.long_strangle,
        'Short Strangle': analyzer.short_strangle,
        'Butterfly Spread': analyzer.butterfly_spread,
        'Iron Condor': analyzer.iron_condor,
        'Iron Butterfly': analyzer.iron_butterfly
    }
    
    for strategy_name in strategies:
        if strategy_name in strategy_methods:
            result = strategy_methods[strategy_name]()
            payoff_results.append(result)
    
    return analyzer.plot_multiple_strategies(payoff_results)

# =====================================================================
# MAIN EXECUTION FOR TESTING
# =====================================================================

if __name__ == "__main__":
    print("RUN: Testing Options Payoff Strategies")
    print("=" * 50)
    
    # Initialize analyzer
    spot_price = 100
    analyzer = OptionsPayoffAnalyzer(spot_price)
    
    # Test individual strategies
    print("\nTesting Individual Strategies:")
    
    # Long Call
    long_call_result = analyzer.long_call(strike=105, premium=3)
    print(f"\n{long_call_result.strategy_name}:")
    print(f"  Max Profit: ${long_call_result.max_profit:.2f}")
    print(f"  Max Loss: ${abs(long_call_result.max_loss):.2f}")
    print(f"  Breakevens: {len(long_call_result.breakeven_points)}")
    
    # Bull Call Spread
    bull_spread_result = analyzer.bull_call_spread(lower_strike=95, upper_strike=105)
    print(f"\n{bull_spread_result.strategy_name}:")
    print(f"  Max Profit: ${bull_spread_result.max_profit:.2f}")
    print(f"  Max Loss: ${abs(bull_spread_result.max_loss):.2f}")
    print(f"  Breakevens: {len(bull_spread_result.breakeven_points)}")
    
    # Long Straddle
    straddle_result = analyzer.long_straddle(strike=100)
    print(f"\n{straddle_result.strategy_name}:")
    print(f"  Max Profit: Unlimited" if straddle_result.max_profit == float('inf') else f"  Max Profit: ${straddle_result.max_profit:.2f}")
    print(f"  Max Loss: ${abs(straddle_result.max_loss):.2f}")
    print(f"  Breakevens: {len(straddle_result.breakeven_points)}")
    
    # Butterfly Spread
    butterfly_result = analyzer.butterfly_spread()
    print(f"\n{butterfly_result.strategy_name}:")
    print(f"  Max Profit: ${butterfly_result.max_profit:.2f}")
    print(f"  Max Loss: ${abs(butterfly_result.max_loss):.2f}")
    print(f"  Breakevens: {len(butterfly_result.breakeven_points)}")
    
    print(f"\n🎉 All strategy tests completed successfully!")
    print(f"Ready for integration with GUI and terminal interfaces!")