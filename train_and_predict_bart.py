import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def load_training_data():
    """Load and prepare training data from NetCDF files."""
    print("Loading training data...")
    
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
    
    print(f"Training data shape: {training_df.shape}")
    return training_df

def get_common_features(train_df, pred_df):
    """Get features that are common between training and prediction data."""
    # Define the features we want to use
    desired_features = [
        # Basic race information
        'distance', 'post_position', 'field_size', 'purse',
        
        # Horse information
        'speed_rating', 'odds', 'weight',
        
        # Recent performance (last 3 races)
        'recentFinishPosition1', 'recentFinishPosition2', 'recentFinishPosition3',
        'recentBrisSpeed1', 'recentBrisSpeed2', 'recentBrisSpeed3',
        'recentPostPosition1', 'recentPostPosition2', 'recentPostPosition3',
        
        # Jockey and trainer statistics
        'jockey_starts', 'jockey_wins', 'jockey_places', 'jockey_shows',
        'trainer_starts', 'trainer_wins', 'trainer_places', 'trainer_shows',
        'jockeyPrevYrStarts', 'jockeyPrevYrWins',
        'trainerPrevYrStarts', 'trainerPrevYrWins',
        
        # Pedigree information
        'BrisDirtPedigree', 'BrisMudPedigree', 'BrisTurfPedigree', 'BrisDistPedigree'
    ]
    
    # Get numeric columns from both datasets
    train_numeric = train_df.select_dtypes(include=[np.number]).columns
    pred_numeric = pred_df.select_dtypes(include=[np.number]).columns
    
    # Get common numeric features
    common_features = list(set(train_numeric) & set(pred_numeric))
    
    # Filter to only include our desired features that are available
    common_features = [f for f in desired_features if f in common_features]
    
    # Remove target and identifier columns
    common_features = [f for f in common_features if f not in ['finish_position', 'starter']]
    
    return common_features

def prepare_features(df, is_training=True, feature_cols=None):
    """Prepare features for training or prediction."""
    if is_training:
        # For training data, separate features and target
        df_numeric = df[feature_cols + ['finish_position']].fillna(df[feature_cols + ['finish_position']].mean())
        X = df_numeric[feature_cols]
        y = df_numeric['finish_position']
        return X, y
    else:
        # For prediction data, use the same features as training
        if feature_cols is None:
            raise ValueError("feature_cols must be provided for prediction data")
        df_numeric = df[feature_cols].fillna(df[feature_cols].mean())
        return df_numeric

def train_bart_model(X_train, y_train):
    """Train BART-like model using RandomForest with BART-like parameters."""
    print("Training BART-like model...")
    
    # Initialize and train the model with BART-like parameters
    model = RandomForestRegressor(
        n_estimators=200,  # Increased number of trees for better uncertainty estimates
        max_depth=None,    # Allow trees to grow fully (like BART)
        min_samples_split=2,  # Minimum samples required to split (like BART's alpha)
        min_samples_leaf=1,   # Minimum samples required in leaf nodes
        max_features='sqrt',  # Number of features to consider for best split
        bootstrap=True,       # Use bootstrap samples (like BART)
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model training completed")
    return model

def make_predictions(model, X_pred, n_samples=1000):
    """Make predictions with uncertainty estimates."""
    print("Making predictions...")
    
    # Get predictions from all trees
    predictions = np.array([tree.predict(X_pred) for tree in model.estimators_])
    
    # Calculate statistics
    mean_preds = np.mean(predictions, axis=0)
    median_preds = np.median(predictions, axis=0)
    q05_preds = np.percentile(predictions, 5, axis=0)
    q95_preds = np.percentile(predictions, 95, axis=0)
    
    return mean_preds, median_preds, q05_preds, q95_preds, predictions

def main():
    # Load and prepare training data
    training_df = load_training_data()
    
    # Load prediction data
    print("\nLoading prediction data...")
    pred_df = pd.read_csv('data/rawDataForPrediction/CDX0515_filtered.csv')
    
    # Get common features
    common_features = get_common_features(training_df, pred_df)
    print(f"\nNumber of common features: {len(common_features)}")
    print("Common features:", common_features)
    
    # Prepare features for training
    X_train, y_train = prepare_features(training_df, is_training=True, feature_cols=common_features)
    
    # Train the model
    model = train_bart_model(X_train, y_train)
    
    # Prepare features for prediction
    X_pred = prepare_features(pred_df, is_training=False, feature_cols=common_features)
    
    # Make predictions with uncertainty estimates
    mean_preds, median_preds, q05_preds, q95_preds, all_predictions = make_predictions(model, X_pred)
    
    # Add predictions to original dataframe
    pred_df['predicted_finish_position_mean'] = mean_preds
    pred_df['predicted_finish_position_median'] = median_preds
    pred_df['predicted_finish_position_q05'] = q05_preds
    pred_df['predicted_finish_position_q95'] = q95_preds
    
    # Save predictions
    output_file = 'data/rawDataForPrediction/CDX0515_predictions.csv'
    pred_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # Save full posterior distribution
    np.save('data/rawDataForPrediction/CDX0515_posterior_predictions.npy', all_predictions)
    
    # Create and save xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "predictions": (["tree", "race"], all_predictions)
        },
        coords={
            "tree": np.arange(1, len(model.estimators_) + 1),
            "race": pred_df.index + 1
        }
    )
    
    ds.attrs["description"] = "Posterior predictive draws for race predictions"
    ds.attrs["model"] = "BART-like RandomForest"
    ds.attrs["date_created"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    ds.to_netcdf("data/rawDataForPrediction/CDX0515_posterior_predictions.nc")
    print("Full posterior distribution saved to NetCDF file")

if __name__ == "__main__":
    main() 