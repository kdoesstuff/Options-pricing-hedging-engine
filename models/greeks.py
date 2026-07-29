"""
Greeks and Implied Volatility Module

This module implements the calculation of option Greeks (Delta, Gamma, Vega, Theta, Rho)
and provides numerical methods for calculating implied volatility from market prices.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
import warnings

# Import our pricing models
try:
    from .pricing_models import BSMModel
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.pricing_models import BSMModel


class GreeksCalculator:
    """
    Calculator for option Greeks using Black-Scholes-Merton model.
    """
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate Delta: ∂V/∂S (price sensitivity to underlying price changes).
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility of the underlying asset
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Delta value
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = BSMModel.d1(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma: ∂²V/∂S² (rate of change of Delta).
        
        Note: Gamma is the same for calls and puts.
        
        Returns:
        --------
        float
            Gamma value
        """
        if T <= 0:
            return 0.0
        
        d1 = BSMModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega: ∂V/∂σ (price sensitivity to volatility changes).
        
        Note: Vega is the same for calls and puts.
        Returns vega as price change per 1% volatility change.
        
        Returns:
        --------
        float
            Vega value (per 1% volatility change)
        """
        if T <= 0:
            return 0.0
        
        d1 = BSMModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate Theta: ∂V/∂T (time decay).
        
        Returns theta as daily time decay (negative for long positions).
        
        Returns:
        --------
        float
            Theta value (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = BSMModel.d1(S, K, T, r, sigma)
        d2 = BSMModel.d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            theta_annual = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            )
        else:  # put
            theta_annual = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            )
        
        return theta_annual / 365  # Convert to daily theta
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate Rho: ∂V/∂r (sensitivity to interest rate changes).
        
        Returns rho as price change per 1% interest rate change.
        
        Returns:
        --------
        float
            Rho value (per 1% rate change)
        """
        if T <= 0:
            return 0.0
        
        d2 = BSMModel.d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    @staticmethod
    def calculate_all_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all Greeks
        """
        return {
            'delta': GreeksCalculator.delta(S, K, T, r, sigma, option_type),
            'gamma': GreeksCalculator.gamma(S, K, T, r, sigma),
            'vega': GreeksCalculator.vega(S, K, T, r, sigma),
            'theta': GreeksCalculator.theta(S, K, T, r, sigma, option_type),
            'rho': GreeksCalculator.rho(S, K, T, r, sigma, option_type)
        }
    
    @staticmethod
    def print_greeks_summary(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str, option_price: Optional[float] = None) -> None:
        """
        Print a formatted summary of all Greeks.
        """
        greeks = GreeksCalculator.calculate_all_greeks(S, K, T, r, sigma, option_type)
        
        if option_price is None:
            option_price = BSMModel.price(S, K, T, r, sigma, option_type)
        
        print(f"\n{'='*60}")
        print(f"GREEKS ANALYSIS: {option_type.upper()} OPTION")
        print(f"{'='*60}")
        print(f"Option Price:       ${option_price:.4f}")
        print(f"\nRisk Sensitivities:")
        print(f"Delta (Δ):          {greeks['delta']:>8.4f}  (per $1 stock move)")
        print(f"Gamma (Γ):          {greeks['gamma']:>8.4f}  (delta change per $1 stock move)")
        print(f"Vega (ν):           {greeks['vega']:>8.4f}  (per 1% volatility change)")
        print(f"Theta (Θ):          {greeks['theta']:>8.4f}  (per day time decay)")
        print(f"Rho (ρ):            {greeks['rho']:>8.4f}  (per 1% rate change)")
        print(f"{'='*60}")
        
        # Add interpretations
        print(f"\nInterpretation:")
        delta_interp = f"Price changes by ${abs(greeks['delta']):.3f} for each $1 stock move"
        if option_type.lower() == 'put' and greeks['delta'] < 0:
            delta_interp += " (inverse to stock)"
        print(f"• {delta_interp}")
        
        if abs(greeks['theta']) > 0.001:
            print(f"• Option loses ${abs(greeks['theta']):.3f} per day due to time decay")
        
        if abs(greeks['vega']) > 0.001:
            vega_direction = "increases" if greeks['vega'] > 0 else "decreases"
            print(f"• Price {vega_direction} by ${abs(greeks['vega']):.3f} per 1% volatility increase")


class ImpliedVolatilityCalculator:
    """
    Calculator for implied volatility using numerical methods.
    """
    
    @staticmethod
    def newton_raphson_iv(market_price: float, S: float, K: float, T: float, r: float, 
                         option_type: str, max_iterations: int = 100, 
                         tolerance: float = 1e-6, initial_guess: float = 0.2) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Parameters:
        -----------
        market_price : float
            Observed market price of the option
        S, K, T, r : float
            Option parameters
        option_type : str
            'call' or 'put'
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        initial_guess : float
            Initial guess for volatility
            
        Returns:
        --------
        Optional[float]
            Implied volatility if found, None if not converged
        """
        if T <= 0:
            return None
        
        sigma = initial_guess
        
        for i in range(max_iterations):
            try:
                # Calculate price and vega at current sigma
                price = BSMModel.price(S, K, T, r, sigma, option_type)
                vega = GreeksCalculator.vega(S, K, T, r, sigma) * 100  # Convert back to percentage
                
                # Price difference
                price_diff = price - market_price
                
                # Check convergence
                if abs(price_diff) < tolerance:
                    return sigma
                
                # Newton-Raphson update: sigma_new = sigma - f(sigma)/f'(sigma)
                if abs(vega) < 1e-10:  # Avoid division by zero
                    break
                
                sigma_new = sigma - price_diff / vega
                
                # Ensure volatility stays positive and reasonable
                sigma_new = max(0.001, min(5.0, sigma_new))
                
                # Check for convergence in sigma
                if abs(sigma_new - sigma) < tolerance:
                    return sigma_new
                
                sigma = sigma_new
                
            except (ValueError, ZeroDivisionError, OverflowError):
                break
        
        return None
    
    @staticmethod
    def brent_method_iv(market_price: float, S: float, K: float, T: float, r: float, 
                       option_type: str) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method (more robust).
        
        Parameters:
        -----------
        market_price : float
            Observed market price of the option
        S, K, T, r : float
            Option parameters
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        Optional[float]
            Implied volatility if found, None if not found
        """
        if T <= 0:
            return None
        
        def objective(sigma):
            """Objective function: model_price - market_price"""
            try:
                return BSMModel.price(S, K, T, r, sigma, option_type) - market_price
            except (ValueError, OverflowError):
                return float('inf')
        
        try:
            # Find bounds where objective function changes sign
            sigma_low, sigma_high = 0.001, 5.0
            
            # Check if market price is reasonable
            intrinsic_value = max(0, S - K) if option_type.lower() == 'call' else max(0, K - S)
            if market_price < intrinsic_value:
                return None
            
            # Check bounds
            if objective(sigma_low) * objective(sigma_high) > 0:
                # Try to find better bounds
                for sigma_test in [0.1, 0.5, 1.0, 2.0]:
                    if objective(sigma_low) * objective(sigma_test) < 0:
                        sigma_high = sigma_test
                        break
                    elif objective(sigma_test) * objective(sigma_high) < 0:
                        sigma_low = sigma_test
                        break
                else:
                    return None
            
            # Use Brent's method
            iv = brentq(objective, sigma_low, sigma_high, xtol=1e-6, maxiter=100)
            return iv
            
        except (ValueError, RuntimeError):
            return None
    
    @staticmethod
    def calculate_iv(market_price: float, S: float, K: float, T: float, r: float, 
                    option_type: str, method: str = 'brent') -> Dict[str, Any]:
        """
        Calculate implied volatility with multiple methods and return comprehensive results.
        
        Parameters:
        -----------
        market_price : float
            Market price of the option
        method : str
            'brent', 'newton', or 'both'
            
        Returns:
        --------
        Dict[str, Any]
            Results including IV, method used, and diagnostics
        """
        results = {
            'market_price': market_price,
            'implied_volatility': None,
            'method_used': None,
            'model_price': None,
            'price_error': None,
            'success': False
        }
        
        # Try Brent's method first (more robust)
        if method in ['brent', 'both']:
            iv_brent = ImpliedVolatilityCalculator.brent_method_iv(market_price, S, K, T, r, option_type)
            if iv_brent is not None:
                results['implied_volatility'] = iv_brent
                results['method_used'] = 'brent'
                results['success'] = True
        
        # Try Newton-Raphson if Brent failed or if requested
        if not results['success'] and method in ['newton', 'both']:
            iv_newton = ImpliedVolatilityCalculator.newton_raphson_iv(market_price, S, K, T, r, option_type)
            if iv_newton is not None:
                results['implied_volatility'] = iv_newton
                results['method_used'] = 'newton_raphson'
                results['success'] = True
        
        # Calculate model price and error if we found IV
        if results['success'] and results['implied_volatility'] is not None:
            model_price = BSMModel.price(S, K, T, r, results['implied_volatility'], option_type)
            results['model_price'] = model_price
            results['price_error'] = abs(model_price - market_price)
        
        return results


class PortfolioGreeks:
    """
    Calculator for portfolio-level Greeks and hedging ratios.
    """
    
    @staticmethod
    def calculate_portfolio_greeks(positions: list) -> Dict[str, float]:
        """
        Calculate Greeks for a portfolio of options.
        
        Parameters:
        -----------
        positions : list
            List of dictionaries, each containing:
            {
                'quantity': float,
                'S': float, 'K': float, 'T': float, 'r': float, 'sigma': float,
                'option_type': str
            }
            
        Returns:
        --------
        Dict[str, float]
            Portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            quantity = position['quantity']
            greeks = GreeksCalculator.calculate_all_greeks(
                position['S'], position['K'], position['T'], 
                position['r'], position['sigma'], position['option_type']
            )
            
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += quantity * greeks[greek]
        
        return portfolio_greeks
    
    @staticmethod
    def delta_hedge_ratio(option_delta: float, hedge_delta: float = 1.0) -> float:
        """
        Calculate hedge ratio to achieve delta neutrality.
        
        Parameters:
        -----------
        option_delta : float
            Delta of the option position
        hedge_delta : float
            Delta of the hedging instrument (1.0 for stock)
            
        Returns:
        --------
        float
            Number of hedge units needed (negative means short position)
        """
        return -option_delta / hedge_delta
    
    @staticmethod
    def gamma_hedge_ratios(target_gamma: float, option1_gamma: float, 
                          option2_gamma: float) -> Tuple[float, float]:
        """
        Calculate ratios for gamma hedging using two options.
        
        Parameters:
        -----------
        target_gamma : float
            Desired portfolio gamma
        option1_gamma : float
            Gamma of first option
        option2_gamma : float
            Gamma of second option
            
        Returns:
        --------
        Tuple[float, float]
            Quantities of option1 and option2 needed
        """
        if abs(option1_gamma - option2_gamma) < 1e-10:
            return (0.0, 0.0)  # Cannot hedge if gammas are equal
        
        # Simple two-option gamma hedging
        ratio1 = (target_gamma - option2_gamma) / (option1_gamma - option2_gamma)
        ratio2 = 1 - ratio1
        
        return (ratio1, ratio2)


# Convenience functions
def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
    """Convenience function to calculate all Greeks."""
    return GreeksCalculator.calculate_all_greeks(S, K, T, r, sigma, option_type)

def calculate_iv(market_price: float, S: float, K: float, T: float, r: float, option_type: str) -> Optional[float]:
    """Convenience function to calculate implied volatility."""
    result = ImpliedVolatilityCalculator.calculate_iv(market_price, S, K, T, r, option_type)
    return result['implied_volatility'] if result['success'] else None

def print_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> None:
    """Convenience function to print Greeks summary."""
    GreeksCalculator.print_greeks_summary(S, K, T, r, sigma, option_type)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Greeks and Implied Volatility Calculations")
    print("=" * 60)
    
    # Test parameters
    S = 100    # Current stock price
    K = 105    # Strike price
    T = 0.25   # Time to maturity (3 months)
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    print(f"Test Parameters:")
    print(f"S=${S}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
    
    # Calculate option price
    call_price = BSMModel.price(S, K, T, r, sigma, 'call')
    put_price = BSMModel.price(S, K, T, r, sigma, 'put')
    
    print(f"\nOption Prices:")
    print(f"Call Price: ${call_price:.4f}")
    print(f"Put Price:  ${put_price:.4f}")
    
    # Test Greeks
    print(f"\n{'='*60}")
    print("GREEKS TESTING")
    print(f"{'='*60}")
    
    # Call Greeks
    GreeksCalculator.print_greeks_summary(S, K, T, r, sigma, 'call', call_price)
    
    # Put Greeks  
    GreeksCalculator.print_greeks_summary(S, K, T, r, sigma, 'put', put_price)
    
    # Test Implied Volatility
    print(f"\n{'='*60}")
    print("IMPLIED VOLATILITY TESTING")
    print(f"{'='*60}")
    
    # Use the calculated call price as "market price"
    market_price = call_price + 0.01  # Add small noise
    
    iv_result = ImpliedVolatilityCalculator.calculate_iv(market_price, S, K, T, r, 'call')
    
    print(f"Market Price:         ${market_price:.4f}")
    print(f"True Volatility:      {sigma:.2%}")
    
    if iv_result['success']:
        print(f"Implied Volatility:   {iv_result['implied_volatility']:.4f} ({iv_result['implied_volatility']:.2%})")
        print(f"Method Used:          {iv_result['method_used']}")
        print(f"Model Price:          ${iv_result['model_price']:.4f}")
        print(f"Price Error:          ${iv_result['price_error']:.6f}")
        print(f"IV Error:             {abs(iv_result['implied_volatility'] - sigma):.6f}")
    else:
        print("Failed to calculate implied volatility")
    
    # Test portfolio Greeks
    print(f"\n{'='*60}")
    print("PORTFOLIO GREEKS TESTING")
    print(f"{'='*60}")
    
    # Example portfolio: Long call, short put
    portfolio = [
        {'quantity': 1, 'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'option_type': 'call'},
        {'quantity': -1, 'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'option_type': 'put'}
    ]
    
    port_greeks = PortfolioGreeks.calculate_portfolio_greeks(portfolio)
    
    print("Portfolio: +1 Call, -1 Put (Synthetic Long Stock)")
    for greek, value in port_greeks.items():
        print(f"{greek.capitalize()}:    {value:>8.4f}")
    
    # Delta hedge ratio
    call_delta = GreeksCalculator.delta(S, K, T, r, sigma, 'call')
    hedge_ratio = PortfolioGreeks.delta_hedge_ratio(call_delta)
    
    print(f"\nDelta Hedging:")
    print(f"Call Delta:           {call_delta:.4f}")
    print(f"Hedge Ratio:          {hedge_ratio:.4f} (short {abs(hedge_ratio):.0f} shares per call)")
    
    print(f"\n{'='*60}")
    print("Testing completed successfully!")
    print(f"{'='*60}")