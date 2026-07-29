"""
Visualization Module

This module provides comprehensive plotting and visualization functions
for all aspects of the financial mathematics project including:
- Convergence analysis plots
- P&L and strategy comparison charts
- Greeks visualization
- Option pricing curves
- Monte Carlo path simulation plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

# Set plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)

# Import our modules for type checking
try:
    from ..models.pricing_models import BSMModel, MonteCarloModel
    from ..models.greeks import GreeksCalculator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class FinancialPlotter:
    """
    Main class for creating financial mathematics visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'default'):
        """
        Initialize plotter with default settings.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        style : str
            Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_convergence_analysis(self, crr_data: Dict[str, Any], mc_data: Dict[str, Any], 
                                bsm_price: float, title: str = "Model Convergence Analysis") -> plt.Figure:
        """
        Plot convergence analysis for CRR and Monte Carlo models.
        
        Parameters:
        -----------
        crr_data : Dict[str, Any]
            Dictionary with 'steps' and 'prices' for CRR convergence
        mc_data : Dict[str, Any]  
            Dictionary with 'simulation_counts' and 'prices' for MC convergence
        bsm_price : float
            BSM benchmark price
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # CRR Convergence Plot
        ax1.plot(crr_data['steps'], crr_data['prices'], 'o-', color=self.colors[0], 
                linewidth=2, markersize=6, label='CRR Price')
        ax1.axhline(y=bsm_price, color=self.colors[1], linestyle='--', 
                   linewidth=2, label=f'BSM Price: ${bsm_price:.4f}')
        
        ax1.set_xlabel('Number of Steps', fontsize=12)
        ax1.set_ylabel('Option Price ($)', fontsize=12)
        ax1.set_title('CRR Convergence to BSM', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add convergence error annotation
        final_crr_price = crr_data['prices'][-1]
        error = abs(final_crr_price - bsm_price)
        ax1.text(0.05, 0.95, f'Final Error: ${error:.6f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Monte Carlo Convergence Plot
        ax2.semilogx(mc_data['simulation_counts'], mc_data['prices'], 's-', 
                    color=self.colors[2], linewidth=2, markersize=6, label='MC Price')
        ax2.axhline(y=bsm_price, color=self.colors[1], linestyle='--', 
                   linewidth=2, label=f'BSM Price: ${bsm_price:.4f}')
        
        # Add confidence bands if available
        if 'std_errors' in mc_data:
            mc_prices = np.array(mc_data['prices'])
            std_errors = np.array(mc_data['std_errors'])
            ax2.fill_between(mc_data['simulation_counts'], 
                           mc_prices - 1.96 * std_errors, 
                           mc_prices + 1.96 * std_errors, 
                           alpha=0.3, color=self.colors[2], label='95% Confidence')
        
        ax2.set_xlabel('Number of Simulations', fontsize=12)
        ax2.set_ylabel('Option Price ($)', fontsize=12)
        ax2.set_title('Monte Carlo Convergence', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add convergence info
        final_mc_price = mc_data['prices'][-1]
        mc_error = abs(final_mc_price - bsm_price)
        ax2.text(0.05, 0.95, f'Final Error: ${mc_error:.6f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_option_price_surface(self, S_range: np.ndarray, K: float, T: float, 
                                r: float, sigma: float, option_type: str = 'call',
                                title: Optional[str] = None) -> plt.Figure:
        """
        Plot option price as a function of stock price.
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of stock prices
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        title : str, optional
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        from models.pricing_models import BSMModel
        from models.greeks import GreeksCalculator
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        if title is None:
            title = f'{option_type.title()} Option Price Analysis'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Calculate option prices and Greeks
        option_prices = []
        deltas = []
        gammas = []
        
        for S in S_range:
            price = BSMModel.price(S, K, T, r, sigma, option_type)
            delta = GreeksCalculator.delta(S, K, T, r, sigma, option_type)
            gamma = GreeksCalculator.gamma(S, K, T, r, sigma)
            
            option_prices.append(price)
            deltas.append(delta)
            gammas.append(gamma)
        
        # Plot option prices
        ax1.plot(S_range, option_prices, color=self.colors[0], linewidth=3, label=f'{option_type.title()} Price')
        
        # Add intrinsic value line
        if option_type.lower() == 'call':
            intrinsic = np.maximum(S_range - K, 0)
        else:
            intrinsic = np.maximum(K - S_range, 0)
        
        ax1.plot(S_range, intrinsic, '--', color=self.colors[1], linewidth=2, label='Intrinsic Value')
        ax1.axvline(x=K, color='red', linestyle=':', alpha=0.7, label=f'Strike: ${K}')
        
        ax1.set_xlabel('Stock Price ($)', fontsize=12)
        ax1.set_ylabel('Option Price ($)', fontsize=12)
        ax1.set_title(f'{option_type.title()} Option Price vs Stock Price', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot Greeks
        ax2_twin = ax2.twinx()
        
        # Delta on left axis
        line1 = ax2.plot(S_range, deltas, color=self.colors[2], linewidth=2, label='Delta (Δ)')
        ax2.set_ylabel('Delta', fontsize=12, color=self.colors[2])
        ax2.tick_params(axis='y', labelcolor=self.colors[2])
        
        # Gamma on right axis
        line2 = ax2_twin.plot(S_range, gammas, color=self.colors[3], linewidth=2, label='Gamma (Γ)')
        ax2_twin.set_ylabel('Gamma', fontsize=12, color=self.colors[3])
        ax2_twin.tick_params(axis='y', labelcolor=self.colors[3])
        
        ax2.axvline(x=K, color='red', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Stock Price ($)', fontsize=12)
        ax2.set_title('Greeks vs Stock Price', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_strategy_comparison(self, strategy_results: List[Dict[str, Any]], 
                               title: str = "Strategy Performance Comparison") -> plt.Figure:
        """
        Plot comprehensive strategy comparison.
        
        Parameters:
        -----------
        strategy_results : List[Dict[str, Any]]
            List of strategy result dictionaries
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        strategy_names = [result['strategy_name'] for result in strategy_results]
        total_returns = [result['total_return'] for result in strategy_results]
        sharpe_ratios = [result['sharpe_ratio'] for result in strategy_results]
        max_drawdowns = [result['max_drawdown'] for result in strategy_results]
        
        # 1. Cumulative P&L Plot
        ax1 = fig.add_subplot(gs[0, :])
        
        for i, result in enumerate(strategy_results):
            if 'cumulative_pnl' in result:
                days = range(len(result['cumulative_pnl']))
                ax1.plot(days, result['cumulative_pnl'], 
                        linewidth=2, label=result['strategy_name'], 
                        color=self.colors[i % len(self.colors)])
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Trading Days', fontsize=12)
        ax1.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax1.set_title('Cumulative P&L Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Total Returns Bar Chart
        ax2 = fig.add_subplot(gs[1, 0])
        
        bars = ax2.bar(range(len(strategy_names)), [r * 100 for r in total_returns], 
                      color=self.colors[:len(strategy_names)])
        ax2.set_xlabel('Strategy', fontsize=12)
        ax2.set_ylabel('Total Return (%)', fontsize=12)
        ax2.set_title('Total Returns Comparison', fontsize=14)
        ax2.set_xticks(range(len(strategy_names)))
        ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, return_val in zip(bars, total_returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{return_val:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 3. Sharpe Ratio Bar Chart
        ax3 = fig.add_subplot(gs[1, 1])
        
        bars = ax3.bar(range(len(strategy_names)), sharpe_ratios, 
                      color=self.colors[:len(strategy_names)])
        ax3.set_xlabel('Strategy', fontsize=12)
        ax3.set_ylabel('Sharpe Ratio', fontsize=12)
        ax3.set_title('Risk-Adjusted Returns', fontsize=14)
        ax3.set_xticks(range(len(strategy_names)))
        ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, sharpe in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01 if height >= 0 else height - abs(height)*0.05,
                    f'{sharpe:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 4. Risk Metrics Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create table data
        table_data = []
        for result in strategy_results:
            table_data.append([
                result['strategy_name'],
                f"{result['total_return']:.2%}",
                f"${result['final_pnl']:.0f}",
                f"{result['sharpe_ratio']:.3f}",
                f"{result['max_drawdown']:.2%}",
                f"{result['win_rate']:.1%}",
                f"{result.get('volatility_annual', 0):.1%}"
            ])
        
        table_headers = ['Strategy', 'Return', 'Final P&L', 'Sharpe', 'Max DD', 'Win Rate', 'Volatility']
        
        table = ax4.table(cellText=table_data, colLabels=table_headers,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        return fig
    
    def plot_hedging_analysis(self, hedge_result: Dict[str, Any], 
                            title: str = "Delta Hedging Analysis") -> plt.Figure:
        """
        Plot hedging simulation results.
        
        Parameters:
        -----------
        hedge_result : Dict[str, Any]
            Hedging simulation results
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        days = range(len(hedge_result['portfolio_value']))
        ax1.plot(days, hedge_result['portfolio_value'], color=self.colors[0], linewidth=2)
        ax1.set_xlabel('Trading Days', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Portfolio Value', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. Daily P&L
        ax2 = axes[0, 1]
        ax2.plot(days, hedge_result['daily_pnl'], color=self.colors[1], linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Trading Days', fontsize=12)
        ax2.set_ylabel('Daily P&L ($)', fontsize=12)
        ax2.set_title('Daily P&L', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. Stock Position Over Time
        ax3 = axes[1, 0]
        ax3.plot(days, hedge_result['stock_positions'], color=self.colors[2], linewidth=2)
        ax3.set_xlabel('Trading Days', fontsize=12)
        ax3.set_ylabel('Stock Position (Shares)', fontsize=12)
        ax3.set_title('Dynamic Stock Position', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. Delta Over Time
        ax4 = axes[1, 1]
        if 'deltas' in hedge_result:
            ax4.plot(days, hedge_result['deltas'], color=self.colors[3], linewidth=2)
            ax4.set_xlabel('Trading Days', fontsize=12)
            ax4.set_ylabel('Delta', fontsize=12)
            ax4.set_title('Option Delta Evolution', fontsize=14)
            ax4.grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = f"""
Final P&L: ${hedge_result['final_pnl']:.2f}
P&L Volatility: {hedge_result['volatility_pnl']:.2%}
Transaction Costs: ${hedge_result['total_transaction_costs']:.2f}
Rebalances: {hedge_result['rebalance_count']}
        """.strip()
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_monte_carlo_paths(self, paths: np.ndarray, S0: float, K: float, 
                             option_type: str = 'call', n_paths_show: int = 100,
                             title: str = "Monte Carlo Price Paths") -> plt.Figure:
        """
        Plot Monte Carlo simulation paths.
        
        Parameters:
        -----------
        paths : np.ndarray
            Stock price paths from Monte Carlo simulation
        S0 : float
            Initial stock price
        K : float
            Strike price
        option_type : str
            'call' or 'put'
        n_paths_show : int
            Number of paths to display
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Sample Price Paths
        n_steps = paths.shape[1] - 1
        time_steps = np.linspace(0, 1, n_steps + 1)  # Assume 1 year
        
        # Show a subset of paths
        show_indices = np.random.choice(paths.shape[0], min(n_paths_show, paths.shape[0]), replace=False)
        
        for i in show_indices:
            alpha = 0.1 if n_paths_show > 50 else 0.3
            ax1.plot(time_steps, paths[i], color='blue', alpha=alpha, linewidth=0.5)
        
        # Highlight mean path
        mean_path = np.mean(paths, axis=0)
        ax1.plot(time_steps, mean_path, color='red', linewidth=3, label='Mean Path')
        
        # Add strike line
        ax1.axhline(y=K, color='green', linestyle='--', linewidth=2, label=f'Strike: ${K}')
        ax1.axhline(y=S0, color='orange', linestyle=':', linewidth=2, label=f'Initial: ${S0}')
        
        ax1.set_xlabel('Time (Years)', fontsize=12)
        ax1.set_ylabel('Stock Price ($)', fontsize=12)
        ax1.set_title(f'Sample Paths (showing {len(show_indices)} of {paths.shape[0]})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Final Price Distribution
        final_prices = paths[:, -1]
        
        ax2.hist(final_prices, bins=50, alpha=0.7, color=self.colors[0], density=True, label='Final Prices')
        
        # Add vertical lines for strike and initial price
        ax2.axvline(x=K, color='green', linestyle='--', linewidth=2, label=f'Strike: ${K}')
        ax2.axvline(x=S0, color='orange', linestyle=':', linewidth=2, label=f'Initial: ${S0}')
        ax2.axvline(x=np.mean(final_prices), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean Final: ${np.mean(final_prices):.2f}')
        
        ax2.set_xlabel('Final Stock Price ($)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Final Price Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add statistics
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        itm_percentage = np.mean(payoffs > 0) * 100
        avg_payoff = np.mean(payoffs)
        
        stats_text = f"""
Paths: {paths.shape[0]:,}
ITM: {itm_percentage:.1f}%
Avg Payoff: ${avg_payoff:.2f}
Final Mean: ${np.mean(final_prices):.2f}
Final Std: ${np.std(final_prices):.2f}
        """.strip()
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_volatility_surface(self, S_range: np.ndarray, T_range: np.ndarray, 
                              K: float, r: float, sigma: float,
                              title: str = "Option Price Volatility Surface") -> plt.Figure:
        """
        Plot 3D volatility surface for option prices.
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of stock prices
        T_range : np.ndarray
            Range of times to maturity
        K : float
            Strike price
        r : float
            Risk-free rate
        sigma : float
            Volatility
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        from models.pricing_models import BSMModel
        
        fig = plt.figure(figsize=(14, 10))
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        S_mesh, T_mesh = np.meshgrid(S_range, T_range)
        
        # Calculate option prices
        call_prices = np.zeros_like(S_mesh)
        put_prices = np.zeros_like(S_mesh)
        
        for i in range(len(T_range)):
            for j in range(len(S_range)):
                S = S_range[j]
                T = T_range[i]
                if T > 0:
                    call_prices[i, j] = BSMModel.call_price(S, K, T, r, sigma)
                    put_prices[i, j] = BSMModel.put_price(S, K, T, r, sigma)
                else:
                    call_prices[i, j] = max(0, S - K)
                    put_prices[i, j] = max(0, K - S)
        
        # Plot surfaces
        surf1 = ax.plot_surface(S_mesh, T_mesh, call_prices, alpha=0.7, 
                               cmap='viridis', label='Call Prices')
        surf2 = ax.plot_surface(S_mesh, T_mesh, put_prices, alpha=0.7, 
                               cmap='plasma', label='Put Prices')
        
        ax.set_xlabel('Stock Price ($)', fontsize=12)
        ax.set_ylabel('Time to Maturity (Years)', fontsize=12)
        ax.set_zlabel('Option Price ($)', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add colorbar
        fig.colorbar(surf1, shrink=0.5, aspect=5, label='Call Price ($)')
        
        return fig


# Convenience functions for easy plotting
def plot_convergence(crr_data: Dict[str, Any], mc_data: Dict[str, Any], bsm_price: float) -> plt.Figure:
    """Convenience function for convergence plotting."""
    plotter = FinancialPlotter()
    return plotter.plot_convergence_analysis(crr_data, mc_data, bsm_price)

def plot_option_curves(S_range: np.ndarray, K: float, T: float, r: float, 
                      sigma: float, option_type: str = 'call') -> plt.Figure:
    """Convenience function for option price curves."""
    plotter = FinancialPlotter()
    return plotter.plot_option_price_surface(S_range, K, T, r, sigma, option_type)

def plot_strategies(strategy_results: List[Dict[str, Any]]) -> plt.Figure:
    """Convenience function for strategy comparison."""
    plotter = FinancialPlotter()
    return plotter.plot_strategy_comparison(strategy_results)

def plot_monte_carlo(paths: np.ndarray, S0: float, K: float, option_type: str = 'call') -> plt.Figure:
    """Convenience function for Monte Carlo plots."""
    plotter = FinancialPlotter()
    return plotter.plot_monte_carlo_paths(paths, S0, K, option_type)

def show_all_plots() -> None:
    """Show all matplotlib plots."""
    plt.show()

def save_all_plots(directory: str = "plots") -> None:
    """Save all open plots to directory."""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f"{directory}/plot_{i}.png", dpi=300, bbox_inches='tight')
    
    print(f"Saved {len(plt.get_fignums())} plots to {directory}/")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Visualization Module")
    print("=" * 50)
    
    # Create sample data for testing
    np.random.seed(42)
    
    # Test parameters
    S0 = 100
    K = 105
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    print(f"Test Parameters: S=${S0}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
    
    # 1. Test convergence plot
    print("\n1. Testing convergence analysis plot...")
    
    # Mock CRR convergence data
    crr_data = {
        'steps': [10, 25, 50, 100, 200, 500, 1000],
        'prices': [4.8521, 4.8789, 4.8856, 4.8877, 4.8885, 4.8889, 4.8891]
    }
    
    # Mock MC convergence data
    mc_data = {
        'simulation_counts': [1000, 5000, 10000, 50000, 100000, 500000],
        'prices': [4.8956, 4.8842, 4.8891, 4.8887, 4.8889, 4.8890],
        'std_errors': [0.0156, 0.0098, 0.0069, 0.0031, 0.0022, 0.0010]
    }
    
    bsm_price = 4.8890
    
    fig1 = plot_convergence(crr_data, mc_data, bsm_price)
    print("✓ Convergence analysis plot created")
    
    # 2. Test option price curves
    print("\n2. Testing option price curves...")
    
    S_range = np.linspace(70, 130, 100)
    fig2 = plot_option_curves(S_range, K, T, r, sigma, 'call')
    print("✓ Option price curves created")
    
    # 3. Test strategy comparison
    print("\n3. Testing strategy comparison...")
    
    # Mock strategy results
    days = 252
    mock_returns1 = np.random.normal(0.0008, 0.02, days)  # Buy & Hold
    mock_returns2 = np.random.normal(0.0003, 0.015, days)  # Covered Call
    mock_returns3 = np.random.normal(0.0001, 0.025, days)  # Straddle
    
    strategy_results = [
        {
            'strategy_name': 'Buy & Hold',
            'total_return': 0.12,
            'final_pnl': 1200,
            'sharpe_ratio': 0.85,
            'max_drawdown': -0.08,
            'win_rate': 0.52,
            'volatility_annual': 0.18,
            'cumulative_pnl': np.cumsum(mock_returns1) * 10000,
            'daily_returns': mock_returns1
        },
        {
            'strategy_name': 'Covered Call',
            'total_return': 0.08,
            'final_pnl': 800,
            'sharpe_ratio': 1.02,
            'max_drawdown': -0.05,
            'win_rate': 0.58,
            'volatility_annual': 0.14,
            'cumulative_pnl': np.cumsum(mock_returns2) * 10000,
            'daily_returns': mock_returns2
        },
        {
            'strategy_name': 'Long Straddle',
            'total_return': 0.03,
            'final_pnl': 300,
            'sharpe_ratio': 0.22,
            'max_drawdown': -0.12,
            'win_rate': 0.48,
            'volatility_annual': 0.22,
            'cumulative_pnl': np.cumsum(mock_returns3) * 10000,
            'daily_returns': mock_returns3
        }
    ]
    
    fig3 = plot_strategies(strategy_results)
    print("✓ Strategy comparison plot created")
    
    # 4. Test Monte Carlo paths
    print("\n4. Testing Monte Carlo paths...")
    
    # Generate sample MC paths
    n_paths = 1000
    n_steps = 100
    dt = T / n_steps
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_paths)
        paths[:, i] = paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    fig4 = plot_monte_carlo(paths, S0, K, 'call')
    print("✓ Monte Carlo paths plot created")
    
    print(f"\n{'='*50}")
    print(f"Created {len(plt.get_fignums())} visualization plots")
    print("Call show_all_plots() to display them or save_all_plots() to save them")
    print(f"{'='*50}")
    
    # Uncomment to show plots immediately
    # show_all_plots()