import sys
print('Script train_and_predict_true_bart.py is running')

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = {
        'xarray': 'xarray',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pymc': 'pymc',
        'pymc_bart': 'pymc-bart',
        'arviz': 'arviz',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {package} is NOT installed")
    
    if missing_packages:
        print("\nMissing packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'data/processed/processed_race_data.nc',
        'data/processed/processed_results_data.nc',
        'data/rawDataForPrediction/CDX0515_filtered.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"✗ {file_path} is missing")
        else:
            print(f"✓ {file_path} exists")
    
    if missing_files:
        print("\nMissing required data files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

# Check dependencies and data files before proceeding
if not check_dependencies():
    print("Please install missing dependencies and try again.")
    sys.exit(1)

if not check_data_files():
    print("Please ensure all required data files are present and try again.")
    sys.exit(1)

print("\nAll dependencies and data files are available. Proceeding with script execution...\n")

import xarray as xr
import pandas as pd
import numpy as np
import pymc as pm
import pymc_bart as pmb
import arviz as az
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
from typing import List, Tuple, Dict
import logging

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate data quality and required columns."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for infinite values
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    if inf_mask.any().any():
        logger.warning("Found infinite values in numeric columns")
        df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check for extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        extreme_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        if extreme_mask.any():
            logger.warning(f"Found extreme values in column {col}")
            df.loc[extreme_mask, col] = df[col].median()
    
    return True

def load_training_data() -> pd.DataFrame:
    """Load and prepare training data from NetCDF files."""
    logger.info("Loading training data...")
    
    try:
        # Load NetCDF files
        race_data = xr.open_dataset('data/processed/processed_race_data.nc')
        results_data = xr.open_dataset('data/processed/processed_results_data.nc')
        
        # Convert to DataFrames
        race_df = race_data.to_dataframe().reset_index()
        results_df = results_data.to_dataframe().reset_index()
        
        # Merge race and results data
        race_df['race_id'] = race_df['race'].astype(str).str.extract(r'(\d{2}-\d{2}-\d{2}-R\d{2})')
        results_df['race_id'] = results_df['race'].astype(str).str.extract(r'(\d{2}-\d{2}-\d{2}-R\d{2})')
        
        training_df = pd.merge(
            race_df,
            results_df,
            on=['race_id', 'starter'],
            how='inner',
            suffixes=('_race', '_result')
        )
        
        logger.info(f"Training data shape: {training_df.shape}")
        return training_df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def get_common_features(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> List[str]:
    """Get features that are common between training and prediction data."""
    # Define the features we want to use with their expected types
    feature_specs = {
        # Basic race information
        'distance': float, 'post_position': int, 'field_size': int, 'purse': float,
        
        # Horse information
        'speed_rating': float, 'odds': float, 'weight': float,
        
        # Recent performance (last 3 races)
        'recentFinishPosition1': float, 'recentFinishPosition2': float, 'recentFinishPosition3': float,
        'recentBrisSpeed1': float, 'recentBrisSpeed2': float, 'recentBrisSpeed3': float,
        'recentPostPosition1': float, 'recentPostPosition2': float, 'recentPostPosition3': float,
        
        # Jockey and trainer statistics
        'jockey_starts': int, 'jockey_wins': int, 'jockey_places': int, 'jockey_shows': int,
        'trainer_starts': int, 'trainer_wins': int, 'trainer_places': int, 'trainer_shows': int,
        'jockeyPrevYrStarts': int, 'jockeyPrevYrWins': int,
        'trainerPrevYrStarts': int, 'trainerPrevYrWins': int,
        
        # Pedigree information
        'BrisDirtPedigree': float, 'BrisMudPedigree': float, 'BrisTurfPedigree': float, 'BrisDistPedigree': float
    }
    
    # Get common numeric features
    train_numeric = train_df.select_dtypes(include=[np.number]).columns
    pred_numeric = pred_df.select_dtypes(include=[np.number]).columns
    common_features = list(set(train_numeric) & set(pred_numeric))
    
    # Filter to only include our desired features that are available
    common_features = [f for f in feature_specs.keys() if f in common_features]
    
    # Validate feature types (build a new list instead of removing while iterating)
    valid_features = []
    for feature in common_features:
        if pd.api.types.is_numeric_dtype(train_df[feature]) and pd.api.types.is_numeric_dtype(pred_df[feature]):
            valid_features.append(feature)
        else:
            logger.warning(f"Feature {feature} is not numeric in both datasets")
    
    return valid_features

def prepare_features(df: pd.DataFrame, is_training: bool = True, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for training or prediction."""
    if feature_cols is None:
        raise ValueError("feature_cols must be provided")
    
    # Handle missing values
    df_numeric = df[feature_cols].copy()
    
    # Use different strategies for different types of features
    for col in feature_cols:
        if 'position' in col.lower():
            # For position-related features, use median
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())
        elif 'speed' in col.lower() or 'rating' in col.lower():
            # For speed-related features, use mean
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].mean())
        else:
            # For other features, use 0
            df_numeric[col] = df_numeric[col].fillna(0)
    
    if is_training:
        y = df['finish_position']
        return df_numeric, y
    return df_numeric

def train_bart_model_pymc(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pm.Model, az.InferenceData, StandardScaler]:
    """
    Train a Bayesian Additive Regression Trees (BART) model using PyMC.
    
    The BART model combines multiple regression trees in a Bayesian framework:
    1. Each tree is a weak learner that captures local patterns in the data
    2. The ensemble of trees provides robust predictions
    3. Bayesian inference allows for uncertainty quantification
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Target variable (finish positions)
    
    Returns:
    --------
    Tuple containing:
    - PyMC model
    - Inference data
    - Feature scaler
    """
    logger.info("Training BART model with PyMC...")
    
    # Scale features for better numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    with pm.Model() as model:
        # BART prior for the mean function
        μ = pmb.BART(
            'μ',
            X_scaled,
            y_train,
            m=100,          # Number of trees in the ensemble
            alpha=0.95,     # Tree depth prior (higher = deeper trees)
            beta=2.0,       # Tree depth prior
            k=2.0,          # Leaf value prior (higher = more conservative)
            split_prior=0.5 # Prior probability of splitting
        )
        
        # Observation noise
        σ = pm.HalfNormal('σ', 5.0)
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=μ, sigma=σ, observed=y_train)
        
        # MCMC sampling
        idata = pm.sample(
            draws=1000,     # Number of posterior samples
            tune=500,       # Number of tuning steps
            target_accept=0.9,  # Target acceptance rate
            cores=1,        # Number of CPU cores to use
            random_seed=42, # For reproducibility
            progressbar=True
        )
    
    return model, idata, scaler

def make_predictions_pymc(
    model: pm.Model,
    idata: az.InferenceData,
    scaler: StandardScaler,
    X_pred: pd.DataFrame,
    pred_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Make predictions using the trained BART model.
    
    The prediction process:
    1. Scale the prediction features
    2. Generate posterior predictions
    3. Calculate mean and uncertainty
    4. Convert to win probabilities
    
    Parameters:
    -----------
    model : pm.Model
        Trained PyMC model
    idata : az.InferenceData
        Inference data from model training
    scaler : StandardScaler
        Feature scaler
    X_pred : pd.DataFrame
        Features for prediction
    pred_df : pd.DataFrame
        Original prediction dataframe
    
    Returns:
    --------
    Tuple containing:
    - Updated prediction dataframe
    - Posterior predictive samples
    """
    logger.info("Making predictions with BART model...")
    
    # Scale prediction features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Generate posterior predictions
    with model:
        post_pred = pmb.predict(idata, X_pred_scaled)
    
    # Calculate prediction statistics
    pred_mean = post_pred.mean(axis=0)  # Mean prediction
    pred_std = post_pred.std(axis=0)    # Prediction uncertainty
    
    # Convert to win probabilities using softmax
    exp_preds = np.exp(-pred_mean)  # Negative because lower finish positions are better
    win_probabilities = exp_preds / np.sum(exp_preds)
    
    # Add predictions to dataframe
    pred_df['predicted_finish_position_mean'] = pred_mean
    pred_df['predicted_finish_position_std'] = pred_std
    pred_df['win_probability'] = win_probabilities * 100
    
    return pred_df, post_pred

def main():
    try:
        logger.info("Starting BART model training and prediction process...")
        
        # Load and prepare training data
        logger.info("Step 1: Loading training data...")
        training_df = load_training_data()
        logger.info(f"Training data loaded successfully. Shape: {training_df.shape}")
        
        # Load prediction data
        logger.info("Step 2: Loading prediction data...")
        pred_df = pd.read_csv('data/rawDataForPrediction/CDX0515_filtered.csv')
        logger.info(f"Prediction data loaded successfully. Shape: {pred_df.shape}")
        
        # Get common features
        logger.info("Step 3: Identifying common features...")
        common_features = get_common_features(training_df, pred_df)
        logger.info(f"Found {len(common_features)} common features")
        logger.info(f"Common features: {common_features}")
        
        # Validate data
        logger.info("Step 4: Validating data...")
        if not validate_data(training_df, common_features + ['finish_position']):
            raise ValueError("Training data validation failed")
        if not validate_data(pred_df, common_features):
            raise ValueError("Prediction data validation failed")
        logger.info("Data validation completed successfully")
        
        # Prepare features for training
        logger.info("Step 5: Preparing features for training...")
        X_train, y_train = prepare_features(training_df, is_training=True, feature_cols=common_features)
        logger.info(f"Training features prepared. X shape: {X_train.shape}, y shape: {y_train.shape}")
        
        # Train the BART model
        logger.info("Step 6: Training BART model (this may take a while)...")
        model, idata, scaler = train_bart_model_pymc(X_train, y_train)
        logger.info("BART model training completed")
        
        # Prepare features for prediction
        logger.info("Step 7: Preparing features for prediction...")
        X_pred = prepare_features(pred_df, is_training=False, feature_cols=common_features)
        logger.info(f"Prediction features prepared. Shape: {X_pred.shape}")
        
        # Make predictions
        logger.info("Step 8: Making predictions...")
        pred_df, post_pred = make_predictions_pymc(model, idata, scaler, X_pred, pred_df)
        logger.info("Predictions completed")
        
        # Save predictions
        logger.info("Step 9: Saving results...")
        output_file = 'data/rawDataForPrediction/new_CDX0515_bart_predictions.csv'
        pred_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")
        
        # Save posterior predictive samples
        posterior_file = 'data/rawDataForPrediction/new_CDX0515_bart_posterior_predictions.nc'
        xr.DataArray(post_pred, dims=["draw", "horse"]).to_netcdf(posterior_file)
        logger.info(f"Posterior predictive samples saved to: {posterior_file}")
        
        # Save model artifacts
        model_file = 'data/processed/new_bart_model.joblib'
        joblib.dump((scaler, common_features), model_file)
        logger.info("Model artifacts saved")
        
        # Save inference data
        idata_file = 'data/processed/new_bart_inference_data.nc'
        idata.to_netcdf(idata_file)
        logger.info("Inference data saved")
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 