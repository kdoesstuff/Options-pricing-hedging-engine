"""
Pricing Models Module

This module implements all core option pricing models:
- Black-Scholes-Merton (BSM) for European options
- Binomial Tree models (Pure Binomial and Cox-Ross-Rubinstein)
- Monte Carlo simulation using Geometric Brownian Motion
- Machine Learning based pricing
- Forward/Futures pricing
"""

import numpy as np
import pickle
import os
from typing import Optional, Union, Tuple, Dict, Any
from scipy.stats import norm
from scipy.optimize import brentq


class BSMModel:
    """
    Black-Scholes-Merton model for European option pricing.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for BSM formula."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for BSM formula."""
        return BSMModel.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price using BSM formula.
        
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
            
        Returns:
        --------
        float
            Call option price
        """
        if T <= 0:
            return max(0, S - K)
        
        d1 = BSMModel.d1(S, K, T, r, sigma)
        d2 = BSMModel.d2(S, K, T, r, sigma)
        
        call_value = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return call_value
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price using BSM formula.
        
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
            
        Returns:
        --------
        float
            Put option price
        """
        if T <= 0:
            return max(0, K - S)
        
        d1 = BSMModel.d1(S, K, T, r, sigma)
        d2 = BSMModel.d2(S, K, T, r, sigma)
        
        put_value = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        return put_value
    
    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option price for given type.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option price
        """
        if option_type.lower() == 'call':
            return BSMModel.call_price(S, K, T, r, sigma)
        elif option_type.lower() == 'put':
            return BSMModel.put_price(S, K, T, r, sigma)
        else:
            raise ValueError("option_type must be 'call' or 'put'")


class BinomialModel:
    """
    Binomial Tree models for option pricing.
    """
    
    @staticmethod
    def pure_binomial_price(S: float, K: float, T: float, r: float, 
                           u: float, d: float, n_steps: int, option_type: str) -> float:
        """
        Pure binomial model with arbitrary up/down factors.
        
        Parameters:
        -----------
        u : float
            Up factor (multiplicative)
        d : float
            Down factor (multiplicative)
        n_steps : int
            Number of time steps
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option price
        """
        dt = T / n_steps
        discount_factor = np.exp(-r * dt)
        
        # Risk-neutral probabilities
        prob_up = (np.exp(r * dt) - d) / (u - d)
        prob_down = 1 - prob_up
        
        # Validate probabilities
        if prob_up < 0 or prob_up > 1:
            raise ValueError("Invalid parameters: risk-neutral probabilities must be between 0 and 1")
        
        # Initialize stock price tree
        stock_prices = np.zeros(n_steps + 1)
        
        # Calculate stock prices at maturity
        for j in range(n_steps + 1):
            stock_prices[j] = S * (u ** (n_steps - j)) * (d ** j)
        
        # Calculate option values at maturity
        option_values = np.zeros(n_steps + 1)
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_values[j] = max(0, stock_prices[j] - K)
            else:
                option_values[j] = max(0, K - stock_prices[j])
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j] = discount_factor * (
                    prob_up * option_values[j] + prob_down * option_values[j + 1]
                )
        
        return option_values[0]
    
    @staticmethod
    def crr_european_price(S: float, K: float, T: float, r: float, sigma: float, 
                          n_steps: int, option_type: str) -> float:
        """
        Cox-Ross-Rubinstein model for European options.
        
        Parameters:
        -----------
        sigma : float
            Volatility of the underlying asset
        n_steps : int
            Number of time steps
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option price
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        return BinomialModel.pure_binomial_price(S, K, T, r, u, d, n_steps, option_type)
    
    @staticmethod
    def crr_american_price(S: float, K: float, T: float, r: float, sigma: float, 
                          n_steps: int, option_type: str) -> float:
        """
        Cox-Ross-Rubinstein model for American options with early exercise.
        
        Parameters:
        -----------
        sigma : float
            Volatility of the underlying asset
        n_steps : int
            Number of time steps
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            American option price
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        discount_factor = np.exp(-r * dt)
        
        # Risk-neutral probabilities
        prob_up = (np.exp(r * dt) - d) / (u - d)
        prob_down = 1 - prob_up
        
        # Initialize arrays for stock prices and option values
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Build stock price tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_tree[i][j] = S * (u ** (i - j)) * (d ** j)
        
        # Calculate option values at maturity
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_tree[n_steps][j] = max(0, stock_tree[n_steps][j] - K)
            else:
                option_tree[n_steps][j] = max(0, K - stock_tree[n_steps][j])
        
        # Backward induction with early exercise check
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation_value = discount_factor * (
                    prob_up * option_tree[i + 1][j] + prob_down * option_tree[i + 1][j + 1]
                )
                
                # Intrinsic value (early exercise value)
                if option_type.lower() == 'call':
                    intrinsic_value = max(0, stock_tree[i][j] - K)
                else:
                    intrinsic_value = max(0, K - stock_tree[i][j])
                
                # American option value is maximum of continuation and intrinsic
                option_tree[i][j] = max(continuation_value, intrinsic_value)
        
        return option_tree[0][0]


class MonteCarloModel:
    """
    Monte Carlo simulation for option pricing using Geometric Brownian Motion.
    """
    
    @staticmethod
    def gbm_paths(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int, 
                  seed: Optional[int] = None) -> np.ndarray:
        """
        Generate stock price paths using Geometric Brownian Motion.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        r : float
            Risk-free rate
        sigma : float
            Volatility
        T : float
            Time to maturity
        n_steps : int
            Number of time steps per path
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Array of stock price paths (n_paths x n_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        
        # Generate random normal variables
        Z = np.random.standard_normal((n_paths, n_steps))
        
        # Initialize price paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate paths using GBM formula
        for i in range(1, n_steps + 1):
            paths[:, i] = paths[:, i-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i-1]
            )
        
        return paths
    
    @staticmethod
    def european_price(S: float, K: float, T: float, r: float, sigma: float, 
                      n_simulations: int, option_type: str, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Price European option using Monte Carlo simulation.
        
        Parameters:
        -----------
        n_simulations : int
            Number of Monte Carlo simulations
        option_type : str
            'call' or 'put'
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with price, standard error, and confidence interval
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate final stock prices
        Z = np.random.standard_normal(n_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # 95% confidence interval
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        return {
            'price': price,
            'standard_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_simulations': n_simulations
        }
    
    @staticmethod
    def convergence_analysis(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str, simulation_counts: list, 
                           seed: Optional[int] = None) -> Dict[str, list]:
        """
        Analyze Monte Carlo convergence for different numbers of simulations.
        
        Parameters:
        -----------
        simulation_counts : list
            List of simulation counts to test
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Dict[str, list]
            Dictionary with prices and standard errors for each simulation count
        """
        prices = []
        std_errors = []
        
        for n_sims in simulation_counts:
            result = MonteCarloModel.european_price(S, K, T, r, sigma, n_sims, option_type, seed)
            prices.append(result['price'])
            std_errors.append(result['standard_error'])
        
        return {
            'simulation_counts': simulation_counts,
            'prices': prices,
            'standard_errors': std_errors,
            'bsm_benchmark': BSMModel.price(S, K, T, r, sigma, option_type)
        }


class MLModel:
    """
    Machine Learning based option pricing model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to pre-trained model file
        """
        self.model = None
        self.metadata = {}
        self.is_trained = False

        if model_path is None:
            # Default: bundled model, resolved relative to the project root so
            # it works regardless of the current working directory.
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'ml_model.pkl')
            if os.path.exists(default_path):
                model_path = default_path

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from file.

        Supports both raw estimator pickles and dict payloads of the form
        {'model': estimator, 'feature_names': [...], ...} as produced by
        setup_sample_data.py.
        """
        try:
            with open(model_path, 'rb') as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and 'model' in payload:
                self.model = payload['model']
                self.metadata = {k: v for k, v in payload.items() if k != 'model'}
            else:
                self.model = payload
                self.metadata = {}
            self.is_trained = True
            print(f"Loaded ML model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False
    
    def save_model(self, model_path: str) -> None:
        """Save trained model to file."""
        if self.model is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Saved ML model to {model_path}")
    
    def prepare_features(self, S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = 'call') -> np.ndarray:
        """
        Prepare features for ML model prediction.

        Feature order must match the training pipeline in setup_sample_data.py
        (14 features).

        Parameters:
        -----------
        S, K, T, r, sigma : float
            Option parameters
        option_type : str
            'call' or 'put'

        Returns:
        --------
        np.ndarray
            Feature array for model input
        """
        is_call = 1 if option_type.lower() == 'call' else 0
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        features = [
            S,                      # Stock price
            K,                      # Strike price
            T,                      # Time to maturity
            r,                      # Risk-free rate
            sigma,                  # Volatility
            is_call,                # Option type (0=put, 1=call)
            S / K,                  # Moneyness
            np.log(S / K),          # Log moneyness
            sigma * np.sqrt(T),     # Volatility * sqrt(time)
            r * T,                  # Interest component
            T * 365,                # Days to expiration
            intrinsic,              # Intrinsic value
            1 if S > K else 0,      # ITM indicator (matches training convention)
            abs(S - K) / K,         # Relative moneyness
        ]

        return np.array(features).reshape(1, -1)
    
    def predict_price(self, S: float, K: float, T: float, r: float, sigma: float,
                      option_type: str = 'call') -> float:
        """
        Predict option price using ML model.

        Parameters:
        -----------
        S, K, T, r, sigma : float
            Option parameters
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float
            Predicted option price
        """
        if not self.is_trained or self.model is None:
            # Fallback to BSM if no model available
            print("Warning: ML model not available, using BSM as fallback")
            return BSMModel.price(S, K, T, r, sigma, option_type)

        try:
            features = self.prepare_features(S, K, T, r, sigma, option_type)
            prediction = self.model.predict(features)[0]
            return max(0, float(prediction))  # Ensure non-negative price
        except Exception as e:
            print(f"Error in ML prediction: {e}, using BSM fallback")
            return BSMModel.price(S, K, T, r, sigma, option_type)
    
    def create_training_data(self, n_samples: int = 10000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data using BSM model.
        
        Parameters:
        -----------
        n_samples : int
            Number of training samples to generate
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Features (X) and targets (y) for training
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random parameters
        S_range = np.random.uniform(50, 200, n_samples)
        K_range = np.random.uniform(50, 200, n_samples)
        T_range = np.random.uniform(0.02, 2.0, n_samples)  # 1 week to 2 years
        r_range = np.random.uniform(0.01, 0.1, n_samples)  # 1% to 10%
        sigma_range = np.random.uniform(0.1, 0.8, n_samples)  # 10% to 80%
        
        X = []
        y = []
        
        for i in range(n_samples):
            S, K, T, r, sigma = S_range[i], K_range[i], T_range[i], r_range[i], sigma_range[i]
            
            # Generate both call and put prices
            call_price = BSMModel.call_price(S, K, T, r, sigma)
            put_price = BSMModel.put_price(S, K, T, r, sigma)
            
            # Add call option data
            call_features = self.prepare_features(S, K, T, r, sigma).flatten().tolist()
            call_features.append(1)  # Call indicator
            X.append(call_features)
            y.append(call_price)
            
            # Add put option data
            put_features = self.prepare_features(S, K, T, r, sigma).flatten().tolist()
            put_features.append(0)  # Put indicator
            X.append(put_features)
            y.append(put_price)
        
        return np.array(X), np.array(y)


class ForwardModel:
    """
    Forward and Futures pricing model.
    """
    
    @staticmethod
    def forward_price(S: float, r: float, T: float, dividend_yield: float = 0.0) -> float:
        """
        Calculate forward price using no-arbitrage relationship.
        
        Parameters:
        -----------
        S : float
            Current spot price
        r : float
            Risk-free interest rate
        T : float
            Time to maturity
        dividend_yield : float
            Continuous dividend yield (default: 0)
            
        Returns:
        --------
        float
            Forward price
        """
        return S * np.exp((r - dividend_yield) * T)
    
    @staticmethod
    def futures_price(S: float, r: float, T: float, dividend_yield: float = 0.0) -> float:
        """
        Calculate futures price (same as forward for simplicity).
        
        Parameters are the same as forward_price.
        """
        return ForwardModel.forward_price(S, r, T, dividend_yield)


# Convenience functions for easy access to all models
def BSM_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Convenience function for BSM pricing."""
    return BSMModel.price(S, K, T, r, sigma, option_type)

def CRR_price(S: float, K: float, T: float, r: float, sigma: float, 
              n_steps: int, option_type: str, american: bool = False) -> float:
    """Convenience function for CRR pricing."""
    if american:
        return BinomialModel.crr_american_price(S, K, T, r, sigma, n_steps, option_type)
    else:
        return BinomialModel.crr_european_price(S, K, T, r, sigma, n_steps, option_type)

def MC_price(S: float, K: float, T: float, r: float, sigma: float, 
             n_simulations: int, option_type: str, seed: Optional[int] = None) -> float:
    """Convenience function for Monte Carlo pricing."""
    result = MonteCarloModel.european_price(S, K, T, r, sigma, n_simulations, option_type, seed)
    return result['price']

def forward_price(S: float, r: float, T: float) -> float:
    """Convenience function for forward pricing."""
    return ForwardModel.forward_price(S, r, T)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Option Pricing Models")
    print("=" * 50)
    
    # Test parameters
    S = 100    # Current stock price
    K = 105    # Strike price
    T = 0.25   # Time to maturity (3 months)
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    print(f"Parameters: S=${S}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
    print(f"{'='*50}")
    
    # Test BSM
    bsm_call = BSM_price(S, K, T, r, sigma, 'call')
    bsm_put = BSM_price(S, K, T, r, sigma, 'put')
    
    print(f"BSM Call Price:     ${bsm_call:.4f}")
    print(f"BSM Put Price:      ${bsm_put:.4f}")
    
    # Test CRR
    crr_call = CRR_price(S, K, T, r, sigma, 100, 'call', False)
    crr_put = CRR_price(S, K, T, r, sigma, 100, 'put', False)
    
    print(f"CRR Call Price:     ${crr_call:.4f}")
    print(f"CRR Put Price:      ${crr_put:.4f}")
    
    # Test Monte Carlo
    mc_call = MC_price(S, K, T, r, sigma, 100000, 'call', seed=42)
    mc_put = MC_price(S, K, T, r, sigma, 100000, 'put', seed=42)
    
    print(f"MC Call Price:      ${mc_call:.4f}")
    print(f"MC Put Price:       ${mc_put:.4f}")
    
    # Test Forward
    forward = forward_price(S, r, T)
    print(f"Forward Price:      ${forward:.4f}")
    
    # Test American option
    american_put = CRR_price(S, K, T, r, sigma, 100, 'put', True)
    print(f"American Put:       ${american_put:.4f}")
    
    print(f"\nPrice Differences (vs BSM):")
    print(f"CRR Call Diff:      ${crr_call - bsm_call:.6f}")
    print(f"MC Call Diff:       ${mc_call - bsm_call:.6f}")
    print(f"American Premium:   ${american_put - bsm_put:.6f}")