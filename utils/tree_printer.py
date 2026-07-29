"""
Tree Printer Utility

This module provides functions to visualize binomial option pricing trees
in a clean, readable format for educational purposes.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any


class TreePrinter:
    """
    Class for printing binomial trees in various formats.
    """
    
    def __init__(self, precision: int = 4):
        """
        Initialize TreePrinter with display settings.
        
        Parameters:
        -----------
        precision : int
            Number of decimal places to display
        """
        self.precision = precision
    
    def print_stock_tree(self, tree: np.ndarray, title: str = "Stock Price Tree") -> None:
        """
        Print a stock price tree with formatted layout.
        
        Parameters:
        -----------
        tree : np.ndarray
            2D array representing the binomial tree (steps x nodes)
        title : str
            Title for the tree display
        """
        print(f"\n{title}")
        print("=" * len(title))
        
        n_steps = len(tree)
        
        for i in range(n_steps):
            # Create indentation for tree structure
            indent = "  " * (n_steps - i - 1)
            
            # Print step number
            step_label = f"Step {i}:"
            print(f"\n{step_label:<8}", end="")
            
            # Print values for this step
            values = []
            for j in range(i + 1):
                if not np.isnan(tree[i][j]):
                    values.append(f"{tree[i][j]:.{self.precision}f}")
            
            # Print with proper spacing
            if values:
                print(f"{indent}{' ' * 8}".join(values))
    
    def print_option_tree(self, tree: np.ndarray, exercise_decisions: Optional[np.ndarray] = None,
                         title: str = "Option Value Tree") -> None:
        """
        Print an option value tree with optional early exercise indicators.
        
        Parameters:
        -----------
        tree : np.ndarray
            2D array representing option values
        exercise_decisions : np.ndarray, optional
            Boolean array indicating early exercise nodes
        title : str
            Title for the tree display
        """
        print(f"\n{title}")
        print("=" * len(title))
        
        n_steps = len(tree)
        
        for i in range(n_steps):
            # Print step number
            step_label = f"Step {i}:"
            print(f"\n{step_label:<8}", end="")
            
            # Print values for this step
            for j in range(i + 1):
                if not np.isnan(tree[i][j]):
                    value_str = f"{tree[i][j]:.{self.precision}f}"
                    
                    # Add exercise indicator if provided
                    if exercise_decisions is not None and exercise_decisions[i][j]:
                        value_str += "*"  # Asterisk indicates early exercise
                    
                    # Add spacing
                    if j > 0:
                        print("        ", end="")
                    print(f"{value_str:<12}", end="")
            
        # Print legend if exercise decisions are shown
        if exercise_decisions is not None:
            print(f"\n\n* indicates optimal early exercise")
    
    def print_tree_comparison(self, tree1: np.ndarray, tree2: np.ndarray, 
                            title1: str = "Tree 1", title2: str = "Tree 2") -> None:
        """
        Print two trees side by side for comparison.
        
        Parameters:
        -----------
        tree1, tree2 : np.ndarray
            Trees to compare
        title1, title2 : str
            Titles for each tree
        """
        print(f"\nTREE COMPARISON: {title1} vs {title2}")
        print("=" * 60)
        
        n_steps = max(len(tree1), len(tree2))
        
        for i in range(n_steps):
            print(f"\nStep {i}:")
            
            # Print tree1 values
            print(f"  {title1}:", end="")
            if i < len(tree1):
                for j in range(i + 1):
                    if not np.isnan(tree1[i][j]):
                        print(f"  {tree1[i][j]:.{self.precision}f}", end="")
            print()
            
            # Print tree2 values
            print(f"  {title2}:", end="")
            if i < len(tree2):
                for j in range(i + 1):
                    if not np.isnan(tree2[i][j]):
                        print(f"  {tree2[i][j]:.{self.precision}f}", end="")
            print()
    
    def print_tree_with_probabilities(self, price_tree: np.ndarray, 
                                    prob_up: float, prob_down: float,
                                    title: str = "Binomial Tree with Probabilities") -> None:
        """
        Print tree with probability information.
        
        Parameters:
        -----------
        price_tree : np.ndarray
            Stock price tree
        prob_up : float
            Risk-neutral probability of up move
        prob_down : float
            Risk-neutral probability of down move
        title : str
            Title for display
        """
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Risk-Neutral Probabilities: Up = {prob_up:.4f}, Down = {prob_down:.4f}")
        
        self.print_stock_tree(price_tree, "")
    
    def create_tree_summary(self, stock_tree: np.ndarray, option_tree: np.ndarray,
                           S0: float, K: float, r: float, T: float, sigma: float,
                           option_type: str, american: bool = False) -> str:
        """
        Create a comprehensive summary of the tree calculation.
        
        Parameters:
        -----------
        stock_tree : np.ndarray
            Stock price tree
        option_tree : np.ndarray
            Option value tree
        S0, K, r, T, sigma : float
            Option parameters
        option_type : str
            'call' or 'put'
        american : bool
            Whether option is American style
            
        Returns:
        --------
        str
            Formatted summary string
        """
        n_steps = len(stock_tree) - 1
        option_price = option_tree[0][0]
        
        # Calculate tree parameters
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        prob_up = (np.exp(r * dt) - d) / (u - d)
        
        summary = f"""
BINOMIAL TREE CALCULATION SUMMARY
{'='*50}
Parameters:
  Initial Stock Price (S0):    ${S0:.2f}
  Strike Price (K):            ${K:.2f}
  Risk-Free Rate (r):          {r:.2%}
  Time to Maturity (T):        {T:.4f} years
  Volatility (σ):              {sigma:.2%}
  Option Type:                 {option_type.title()} {'(American)' if american else '(European)'}

Tree Parameters:
  Number of Steps (N):         {n_steps}
  Time Step (Δt):              {dt:.6f}
  Up Factor (u):               {u:.6f}
  Down Factor (d):             {d:.6f}
  Risk-Neutral Prob (p):       {prob_up:.6f}
  Risk-Neutral Prob (1-p):     {1-prob_up:.6f}

Results:
  Option Price:                ${option_price:.{self.precision}f}
  
Final Stock Prices:
  Maximum (u^N * S0):          ${stock_tree[-1][0]:.2f}
  Minimum (d^N * S0):          ${stock_tree[-1][-1]:.2f}
  
Final Option Values:
  Maximum:                     ${option_tree[-1][0]:.{self.precision}f}
  Minimum:                     ${option_tree[-1][-1]:.{self.precision}f}
"""
        
        return summary


class TreeBuilder:
    """
    Helper class to build different types of binomial trees.
    """
    
    @staticmethod
    def build_stock_price_tree(S0: float, u: float, d: float, n_steps: int) -> np.ndarray:
        """
        Build a stock price tree.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        u : float
            Up factor
        d : float
            Down factor
        n_steps : int
            Number of time steps
            
        Returns:
        --------
        np.ndarray
            Stock price tree
        """
        # Initialize tree with NaN values
        tree = np.full((n_steps + 1, n_steps + 1), np.nan)
        
        # Fill the tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                tree[i][j] = S0 * (u ** (i - j)) * (d ** j)
        
        return tree
    
    @staticmethod
    def build_option_tree_european(stock_tree: np.ndarray, K: float, r: float, 
                                 dt: float, prob_up: float, option_type: str) -> np.ndarray:
        """
        Build European option value tree using backward induction.
        
        Parameters:
        -----------
        stock_tree : np.ndarray
            Stock price tree
        K : float
            Strike price
        r : float
            Risk-free rate
        dt : float
            Time step
        prob_up : float
            Risk-neutral probability of up move
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        np.ndarray
            Option value tree
        """
        n_steps = len(stock_tree) - 1
        option_tree = np.full_like(stock_tree, np.nan)
        prob_down = 1 - prob_up
        discount_factor = np.exp(-r * dt)
        
        # Terminal values (payoff at expiration)
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_tree[n_steps][j] = max(0, stock_tree[n_steps][j] - K)
            else:  # put
                option_tree[n_steps][j] = max(0, K - stock_tree[n_steps][j])
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                expected_value = (prob_up * option_tree[i + 1][j] + 
                                prob_down * option_tree[i + 1][j + 1])
                option_tree[i][j] = discount_factor * expected_value
        
        return option_tree
    
    @staticmethod
    def build_option_tree_american(stock_tree: np.ndarray, K: float, r: float, 
                                 dt: float, prob_up: float, option_type: str) -> tuple:
        """
        Build American option value tree with early exercise detection.
        
        Parameters:
        -----------
        stock_tree : np.ndarray
            Stock price tree
        K : float
            Strike price
        r : float
            Risk-free rate
        dt : float
            Time step
        prob_up : float
            Risk-neutral probability of up move
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        tuple
            (option_tree, exercise_decisions)
        """
        n_steps = len(stock_tree) - 1
        option_tree = np.full_like(stock_tree, np.nan)
        exercise_decisions = np.full_like(stock_tree, False, dtype=bool)
        prob_down = 1 - prob_up
        discount_factor = np.exp(-r * dt)
        
        # Terminal values (payoff at expiration)
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_tree[n_steps][j] = max(0, stock_tree[n_steps][j] - K)
            else:  # put
                option_tree[n_steps][j] = max(0, K - stock_tree[n_steps][j])
        
        # Backward induction with early exercise check
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                expected_value = (prob_up * option_tree[i + 1][j] + 
                                prob_down * option_tree[i + 1][j + 1])
                continuation_value = discount_factor * expected_value
                
                # Intrinsic value (immediate exercise)
                if option_type.lower() == 'call':
                    intrinsic_value = max(0, stock_tree[i][j] - K)
                else:  # put
                    intrinsic_value = max(0, K - stock_tree[i][j])
                
                # Choose maximum and record exercise decision
                if intrinsic_value > continuation_value:
                    option_tree[i][j] = intrinsic_value
                    exercise_decisions[i][j] = True
                else:
                    option_tree[i][j] = continuation_value
                    exercise_decisions[i][j] = False
        
        return option_tree, exercise_decisions


# Convenience functions for easy use
def print_tree(tree: np.ndarray, title: str = "Binomial Tree", precision: int = 4) -> None:
    """Convenience function to print a tree."""
    printer = TreePrinter(precision=precision)
    printer.print_stock_tree(tree, title)

def print_option_tree_with_exercise(tree: np.ndarray, exercise_decisions: np.ndarray = None,
                                   title: str = "Option Tree", precision: int = 4) -> None:
    """Convenience function to print option tree with exercise indicators."""
    printer = TreePrinter(precision=precision)
    printer.print_option_tree(tree, exercise_decisions, title)

def build_and_print_trees(S0: float, K: float, r: float, T: float, sigma: float,
                         n_steps: int, option_type: str = 'call', american: bool = False,
                         precision: int = 4) -> Dict[str, Any]:
    """
    Build and print complete tree analysis.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all trees and results
    """
    # Calculate tree parameters
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    prob_up = (np.exp(r * dt) - d) / (u - d)
    
    # Build trees
    stock_tree = TreeBuilder.build_stock_price_tree(S0, u, d, n_steps)
    
    if american:
        option_tree, exercise_decisions = TreeBuilder.build_option_tree_american(
            stock_tree, K, r, dt, prob_up, option_type)
    else:
        option_tree = TreeBuilder.build_option_tree_european(
            stock_tree, K, r, dt, prob_up, option_type)
        exercise_decisions = None
    
    # Print results
    printer = TreePrinter(precision=precision)
    
    # Print summary
    summary = printer.create_tree_summary(stock_tree, option_tree, S0, K, r, T, 
                                        sigma, option_type, american)
    print(summary)
    
    # Print trees
    printer.print_tree_with_probabilities(stock_tree, prob_up, 1-prob_up)
    
    if american and exercise_decisions is not None:
        printer.print_option_tree(option_tree, exercise_decisions, 
                                f"American {option_type.title()} Option Tree")
    else:
        printer.print_option_tree(option_tree, title=f"European {option_type.title()} Option Tree")
    
    return {
        'stock_tree': stock_tree,
        'option_tree': option_tree,
        'exercise_decisions': exercise_decisions,
        'option_price': option_tree[0][0],
        'tree_parameters': {
            'u': u, 'd': d, 'prob_up': prob_up, 'dt': dt
        }
    }


if __name__ == "__main__":
    # Example usage
    print("Testing Tree Printer with sample data...")
    
    # Example: European Call Option
    S0 = 100    # Initial stock price
    K = 105     # Strike price
    r = 0.05    # Risk-free rate
    T = 0.25    # Time to maturity (3 months)
    sigma = 0.2 # Volatility
    N = 5       # Number of steps
    
    print("\n" + "="*60)
    print("EXAMPLE: European Call Option Tree Analysis")
    print("="*60)
    
    results = build_and_print_trees(S0, K, r, T, sigma, N, 'call', False, precision=3)
    
    print(f"\nFinal Option Price: ${results['option_price']:.3f}")
    
    print("\n" + "="*60)
    print("EXAMPLE: American Put Option Tree Analysis")  
    print("="*60)
    
    results_american = build_and_print_trees(S0, K, r, T, sigma, N, 'put', True, precision=3)
    
    print(f"\nFinal American Put Price: ${results_american['option_price']:.3f}")