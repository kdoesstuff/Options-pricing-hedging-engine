"""
Sample Data Generation Script

This script creates sample data and a basic ML model for demonstration purposes.
It generates:
1. A simple ML model trained on BSM-generated data
2. Sample historical options data for backtesting
3. Configuration files for the project
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add project path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from models.pricing_models import BSMModel, MLModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("This script is for initial setup and may not work without dependencies installed.")


def create_sample_ml_model(n_samples: int = 50000, save_path: str = "data/ml_model.pkl") -> None:
    """
    Create and train a sample ML model for option pricing.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples to generate
    save_path : str
        Path to save the trained model
    """
    print(f"Creating sample ML model with {n_samples:,} training samples...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random option parameters
    print("Generating training data...")
    
    # Parameter ranges (realistic values)
    S_range = np.random.uniform(20, 300, n_samples)      # Stock price: $20-$300
    K_range = np.random.uniform(20, 300, n_samples)      # Strike price: $20-$300
    T_range = np.random.uniform(0.02, 2.0, n_samples)   # Time: 1 week to 2 years
    r_range = np.random.uniform(0.001, 0.15, n_samples) # Rate: 0.1% to 15%
    sigma_range = np.random.uniform(0.05, 1.0, n_samples) # Vol: 5% to 100%
    
    # Option types (0 = put, 1 = call)
    option_types = np.random.choice([0, 1], n_samples)
    
    # Prepare features and targets
    features = []
    targets = []
    
    print("Calculating BSM prices for training data...")
    
    for i in range(n_samples):
        S, K, T, r, sigma = S_range[i], K_range[i], T_range[i], r_range[i], sigma_range[i]
        option_type = 'call' if option_types[i] == 1 else 'put'
        
        # Calculate BSM price as target
        try:
            bsm_price = BSMModel.price(S, K, T, r, sigma, option_type)
            
            # Create feature vector
            # Basic features: S, K, T, r, sigma, option_type
            # Derived features: moneyness, log_moneyness, vol*sqrt(T), etc.
            feature_vector = [
                S,                      # Stock price
                K,                      # Strike price
                T,                      # Time to maturity
                r,                      # Risk-free rate
                sigma,                  # Volatility
                option_types[i],        # Option type (0=put, 1=call)
                S / K,                  # Moneyness
                np.log(S / K),          # Log moneyness
                sigma * np.sqrt(T),     # Volatility * sqrt(time)
                r * T,                  # Interest component
                T * 365,                # Days to expiration
                max(0, S - K) if option_types[i] == 1 else max(0, K - S),  # Intrinsic value
                1 if S > K else 0,      # ITM indicator
                abs(S - K) / K          # Relative moneyness
            ]
            
            features.append(feature_vector)
            targets.append(bsm_price)
            
        except (ValueError, OverflowError, ZeroDivisionError):
            # Skip invalid parameter combinations
            continue
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} samples...")
    
    # Convert to arrays
    X = np.array(features)
    y = np.array(targets)
    
    print(f"Generated {len(X):,} valid training samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target stats: min=${np.min(y):.2f}, max=${np.max(y):.2f}, mean=${np.mean(y):.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model performance...")
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\\nModel Performance:")
    print(f"  Training RMSE:   ${train_rmse:.4f}")
    print(f"  Testing RMSE:    ${test_rmse:.4f}")
    print(f"  Training R²:     {train_r2:.4f}")
    print(f"  Testing R²:      {test_r2:.4f}")
    
    # Feature importance
    feature_names = [
        'Stock_Price', 'Strike_Price', 'Time_to_Maturity', 'Risk_Free_Rate', 
        'Volatility', 'Option_Type', 'Moneyness', 'Log_Moneyness', 
        'Vol_Sqrt_Time', 'Interest_Component', 'Days_to_Expiry', 
        'Intrinsic_Value', 'ITM_Indicator', 'Relative_Moneyness'
    ]
    
    importances = model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\\nTop 5 Feature Importances:")
    for name, importance in feature_importance[:5]:
        print(f"  {name}: {importance:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create model wrapper with metadata
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_samples': len(X_train),
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'created_date': datetime.now().isoformat(),
        'model_type': 'RandomForestRegressor',
        'target': 'BSM_Option_Price'
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"SUCCESS: Model saved to: {save_path}")
    
    return model_data


def create_sample_options_data(save_path: str = "data/sample_options_data.csv") -> None:
    """
    Create sample historical options data for backtesting.
    
    Parameters:
    -----------
    save_path : str
        Path to save the sample data
    """
    print("Creating sample historical options data...")
    
    np.random.seed(42)
    
    # Generate sample data for a few months
    n_days = 126  # ~6 months of trading days
    
    # Base parameters
    base_stock_price = 100
    strike_prices = [90, 95, 100, 105, 110]
    base_vol = 0.25
    base_rate = 0.05
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate stock price path (GBM)
    dt = 1/252
    stock_prices = [base_stock_price]
    
    for i in range(1, n_days):
        shock = np.random.normal(0, 1)
        price_change = stock_prices[-1] * np.exp((base_rate - 0.5 * base_vol**2) * dt + base_vol * np.sqrt(dt) * shock)
        stock_prices.append(price_change)
    
    # Create options data
    options_data = []
    
    for i, date in enumerate(dates):
        S = stock_prices[i]
        
        # Varying time to maturity (options with different expiration dates)
        for T in [0.08, 0.25, 0.5]:  # 1 month, 3 months, 6 months
            for K in strike_prices:
                for option_type in ['call', 'put']:
                    
                    # Add some noise to volatility
                    vol = base_vol + np.random.normal(0, 0.05)
                    vol = max(0.1, min(1.0, vol))  # Keep reasonable bounds
                    
                    # Calculate theoretical price
                    theoretical_price = BSMModel.price(S, K, T, base_rate, vol, option_type)
                    
                    # Add bid-ask spread and market noise
                    spread = theoretical_price * 0.02  # 2% spread
                    market_price = theoretical_price + np.random.normal(0, theoretical_price * 0.01)
                    bid_price = market_price - spread/2
                    ask_price = market_price + spread/2
                    
                    # Volume (higher for ATM options)
                    moneyness = S / K
                    volume_factor = np.exp(-5 * (moneyness - 1)**2)  # Peak at ATM
                    volume = max(1, int(np.random.poisson(50 * volume_factor)))
                    
                    options_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'stock_price': round(S, 2),
                        'strike_price': K,
                        'time_to_maturity': round(T, 4),
                        'option_type': option_type,
                        'theoretical_price': round(theoretical_price, 4),
                        'market_price': round(market_price, 4),
                        'bid_price': round(bid_price, 4),
                        'ask_price': round(ask_price, 4),
                        'implied_volatility': round(vol, 4),
                        'volume': volume,
                        'open_interest': volume * np.random.randint(5, 20),
                        'risk_free_rate': base_rate
                    })
    
    # Create DataFrame and save
    df = pd.DataFrame(options_data)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"SUCCESS: Sample options data saved to: {save_path}")
    print(f"   Records: {len(df):,}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Strike prices: {sorted(df['strike_price'].unique())}")
    print(f"   Option types: {sorted(df['option_type'].unique())}")
    
    return df


def create_project_config(save_path: str = "data/config.json") -> None:
    """
    Create project configuration file.
    
    Parameters:
    -----------
    save_path : str
        Path to save the configuration
    """
    import json
    
    config = {
        "project": {
            "name": "Comprehensive Financial Mathematics Simulation",
            "version": "1.0.0",
            "description": "Complete option pricing and risk management system",
            "created": datetime.now().isoformat()
        },
        "models": {
            "bsm": {
                "description": "Black-Scholes-Merton analytical model",
                "use_cases": ["European options", "benchmark pricing"]
            },
            "crr": {
                "description": "Cox-Ross-Rubinstein binomial model", 
                "use_cases": ["American options", "convergence analysis"],
                "default_steps": 100
            },
            "monte_carlo": {
                "description": "Monte Carlo simulation using GBM",
                "use_cases": ["Path-dependent options", "risk analysis"],
                "default_simulations": 100000
            },
            "ml": {
                "description": "Machine learning based pricing",
                "model_file": "ml_model.pkl",
                "use_cases": ["Market price prediction", "alternative pricing"]
            }
        },
        "strategies": {
            "covered_call": {
                "description": "Buy stock, sell call option",
                "risk_profile": "Limited upside, downside protection"
            },
            "long_straddle": {
                "description": "Buy call and put at same strike",
                "risk_profile": "Profits from high volatility"
            },
            "delta_neutral": {
                "description": "Volatility trading with delta hedge",
                "risk_profile": "Market neutral, volatility exposure"
            }
        },
        "default_parameters": {
            "initial_capital": 10000,
            "transaction_cost_rate": 0.001,
            "rebalance_frequency": 1,
            "risk_free_rate_source": "10Y_Treasury",
            "volatility_window": 252
        },
        "visualization": {
            "default_figsize": [12, 8],
            "color_scheme": "husl",
            "save_plots": False,
            "plot_directory": "plots"
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"SUCCESS: Configuration saved to: {save_path}")


def setup_sample_data_and_models():
    """
    Main function to set up all sample data and models.
    """
    print("RUN: Setting up sample data and models for Financial Mathematics project...")
    print("=" * 70)
    
    try:
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # 1. Create ML Model
        print("\\n1️⃣  Creating sample ML model...")
        try:
            model_data = create_sample_ml_model(n_samples=25000)  # Smaller for faster setup
            print("SUCCESS: ML model creation completed")
        except Exception as e:
            print(f"⚠️  ML model creation failed: {e}")
            print("   The project will still work without the ML model")
        
        # 2. Create sample options data
        print("\\n2️⃣  Creating sample options data...")
        try:
            options_df = create_sample_options_data()
            print("SUCCESS: Sample options data creation completed")
        except Exception as e:
            print(f"⚠️  Options data creation failed: {e}")
        
        # 3. Create configuration
        print("\\n3️⃣  Creating project configuration...")
        try:
            create_project_config()
            print("SUCCESS: Configuration creation completed")
        except Exception as e:
            print(f"⚠️  Configuration creation failed: {e}")
        
        print("\\n" + "=" * 70)
        print("🎉 Sample data and models setup completed!")
        print("\\nCreated files:")
        print("├── data/ml_model.pkl           (Machine learning model)")
        print("├── data/sample_options_data.csv (Historical options data)")
        print("└── data/config.json            (Project configuration)")
        print("\\nThe project is now ready to run!")
        print("Execute: python project_main.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR: Setup failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    setup_sample_data_and_models()