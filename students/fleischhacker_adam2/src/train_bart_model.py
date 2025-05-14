import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pymc_bart as pmb
import os
import xarray as xr

def load_data():
    # Load training and prediction data
    train_df = pd.read_csv('students/fleischhacker_adam2/data/features/training_features.csv')
    pred_df = pd.read_csv('students/fleischhacker_adam2/data/features/prediction_features.csv')
    
    # Determine number of starters from the data
    n_starters = 14 
    # TO: Fix this .. hardcoded for now
    print(f"Number of starters found in data: {n_starters}")
    
    # Get feature columns (excluding race_id and target columns)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['race_id'] + [f'st{i}_pts' for i in range(1, n_starters + 1)]]
    
    # Get target columns (only from training data)
    target_cols = [f'st{i}_pts' for i in range(1, n_starters + 1)]
    
    # Handle any remaining NaN values in features
    # For numeric columns, use appropriate imputation values
    for col in feature_cols:
        if 'odds' in col:
            # For odds, use 999.0 (worst possible odds)
            train_df[col] = train_df[col].fillna(999.0)
            pred_df[col] = pred_df[col].fillna(999.0)
        elif 'str' in col:  # stretch position
            # For stretch position, use -1.0 (indicates no position)
            train_df[col] = train_df[col].fillna(-1.0)
            pred_df[col] = pred_df[col].fillna(-1.0)
        elif 'ent' in col:  # number of entrants
            # For number of entrants, use 0.0
            train_df[col] = train_df[col].fillna(0.0)
            pred_df[col] = pred_df[col].fillna(0.0)
        elif 'pts' in col:  # points
            # For points, use 0.0
            train_df[col] = train_df[col].fillna(0.0)
            pred_df[col] = pred_df[col].fillna(0.0)
        else:
            # For any other numeric columns, use 0.0
            train_df[col] = train_df[col].fillna(0.0)
            pred_df[col] = pred_df[col].fillna(0.0)
    
    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values
    
    # Prepare prediction data (only features, no targets)
    X_pred = pred_df[feature_cols].values
    
    # Extract race_ids from prediction data if available
    if 'race_id' in pred_df.columns:
        race_ids = pred_df['race_id'].values
    else:
        # If race_ids aren't available, just use indices 1-10
        race_ids = np.arange(1, len(pred_df) + 1)
    
    return X_train, y_train, X_pred, feature_cols, target_cols, n_starters, race_ids

def create_multi_output_bart_model(X_train, y_train, n_starters):
    # Set up coordinates for our dimensions
    n_obs = X_train.shape[0]
    starter_names = [f"st{i}" for i in range(1, n_starters + 1)]
    
    # Define coordinates for the model
    coords = {
        "n_obs": np.arange(n_obs),
        "starters": starter_names
    }
    
    with pm.Model(coords=coords) as model:
        # Define X as a variable we can update later
        X = pm.Data("X", X_train)  # Already using the right approach
        
        # Create a BART model with dims parameter for multi-output
        μ = pmb.BART(
            "μ", 
            X,
            y_train,  
            m=200,     # Reduced number of trees to prevent overfitting
            dims=["starters", "n_obs"],
            separate_trees=True  # Use separate trees for each starter
        )
        
        # More flexible error term with appropriate dimensions
        # Allow different error terms for each starter and observation
        σ = pm.HalfNormal("σ", 2, dims=["starters", "n_obs"])
        
        # Make sure the observation matches dimensions properly
        # No need for dimshuffle since dimensions already match
        obs = pm.Normal("obs", mu=μ, sigma=σ, observed=y_train.T)
    
    return model, X  # Return both the model and the X variable

def main():
    # Load and preprocess data
    print("Loading data...")
    X_train, y_train, X_pred, feature_cols, target_cols, n_starters, race_ids = load_data()
    
    # Create and train model
    print("Creating multi-output BART model...")
    model, X = create_multi_output_bart_model(X_train, y_train, n_starters)
    
    # Sample from posterior
    print("Sampling from posterior...")
    with model:
        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            cores=4,
            return_inferencedata=True
        )
    
    # Generate predictions for training data
    print("Sampling posterior predictive for training data...")
    with model:
        posterior_predictive_train = pm.sample_posterior_predictive(
            trace=idata,
            random_seed=123
        )
    
    # Generate predictions for new data
    print("Generating predictions for new data...")
    
    # Create a padded version of X_pred to match the original dimensions
    n_train = X_train.shape[0]  # 179
    n_pred = X_pred.shape[0]    # 10
    n_features = X_train.shape[1]
    
    # Create a padded matrix filled with zeros
    X_padded = np.zeros((n_train, n_features))
    
    # Fill in the first n_pred rows with your actual prediction data
    X_padded[:n_pred] = X_pred
    
    with model:
        # Replace X with padded version
        X.set_value(X_padded)
        
        # Use return_inferencedata=False to get raw samples
        raw_predictions = pm.sample_posterior_predictive(
            trace=idata,
            var_names=["μ"],  # Only predict the mean parameter
            random_seed=123,
            return_inferencedata=False  # Get raw samples instead of InferenceData
        )
    
    # Now we know the shape is (chains, draws, starters, obs)
    mu_preds = raw_predictions["μ"]  # Expected shape: (4, 1000, 14, 179)
    print(f"Shape of μ predictions: {mu_preds.shape}")
    
    # Combine chains and draws to get all samples
    # Reshape to (4000, 14, 179) by merging chains and draws dimensions
    n_chains = mu_preds.shape[0]
    n_draws = mu_preds.shape[1]
    total_samples = n_chains * n_draws
    combined_samples = mu_preds.reshape(total_samples, mu_preds.shape[2], mu_preds.shape[3])
    print(f"Combined samples shape: {combined_samples.shape}")
    
    # Extract only the predictions for our 10 observations
    # Shape will be (4000, 14, 10)
    predictions_subset = combined_samples[:, :, :n_pred]
    print(f"Predictions subset shape: {predictions_subset.shape}")
    
    # Save the full posterior predictive distribution
    print("Saving full posterior predictive distribution...")
    
    # Create output directory
    output_dir = 'students/fleischhacker_adam2/model_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Save as numpy array - contains all 4000 samples for all starters and observations
    np.save(os.path.join(output_dir, 'posterior_predictive_full.npy'), predictions_subset)
    
    # Option 2: Save CSV files for a subset of draws to avoid creating too many files
    # Let's save 100 random draws instead of all 4000
    draws_dir = os.path.join(output_dir, 'posterior_draws')
    os.makedirs(draws_dir, exist_ok=True)
    
    # Select 100 random sample indices
    random_indices = np.random.choice(total_samples, size=100, replace=False)
    
    for i, idx in enumerate(random_indices):
        # Extract a single draw from the posterior - shape (14, 10)
        single_draw = predictions_subset[idx]
        
        # Transpose to get (10, 14) shape for DataFrame - observations as rows, starters as columns
        draw_df = pd.DataFrame(single_draw.T, columns=target_cols)
        draw_df.to_csv(os.path.join(draws_dir, f'draw_{i+1}.csv'), index=False)
    
    # Option 3: Calculate summary statistics and save them
    # Calculate across the samples dimension (axis=0)
    mean_preds = np.mean(predictions_subset, axis=0)       # Shape: (14, 10)
    median_preds = np.median(predictions_subset, axis=0)   # Shape: (14, 10)
    q05_preds = np.quantile(predictions_subset, 0.05, axis=0)
    q95_preds = np.quantile(predictions_subset, 0.95, axis=0)
    
    # Transpose to get (10, 14) shape for DataFrames
    mean_df = pd.DataFrame(mean_preds.T, columns=target_cols)
    median_df = pd.DataFrame(median_preds.T, columns=target_cols)
    q05_df = pd.DataFrame(q05_preds.T, columns=target_cols)
    q95_df = pd.DataFrame(q95_preds.T, columns=target_cols)
    
    # Add diagnostic analysis of odds vs predictions
    print("\nAnalyzing relationship between odds and predictions...")
    
    # Load the prediction features to get odds
    pred_features = pd.read_csv('students/fleischhacker_adam2/data/features/prediction_features.csv')
    odds_cols = [col for col in pred_features.columns if 'odds' in col]
    
    # For each race, compare odds with predictions
    for race_idx in range(len(mean_df)):
        print(f"\nRace {race_idx + 1}:")
        race_odds = pred_features.iloc[race_idx][odds_cols].values
        race_preds = mean_df.iloc[race_idx].values
        
        # Get the lowest odds horse and highest predicted points horse
        lowest_odds_idx = np.argmin(race_odds)
        highest_pred_idx = np.argmax(race_preds)
        
        print(f"Lowest odds horse (starter {lowest_odds_idx + 1}): {race_odds[lowest_odds_idx]:.2f}")
        print(f"Highest predicted points horse (starter {highest_pred_idx + 1}): {race_preds[highest_pred_idx]:.2f}")
        print(f"Predicted points for lowest odds horse: {race_preds[lowest_odds_idx]:.2f}")
        
        # Print all horses' odds and predictions
        print("\nAll horses in this race:")
        for i in range(len(race_odds)):
            print(f"Starter {i+1}: Odds = {race_odds[i]:.2f}, Predicted Points = {race_preds[i]:.2f}")
    
    # Save summary statistics
    mean_df.to_csv(os.path.join(output_dir, 'predictions_mean.csv'), index=False)
    median_df.to_csv(os.path.join(output_dir, 'predictions_median.csv'), index=False)
    q05_df.to_csv(os.path.join(output_dir, 'predictions_q05.csv'), index=False)
    q95_df.to_csv(os.path.join(output_dir, 'predictions_q95.csv'), index=False)
    
    # Option 4: Create and save an xarray Dataset with the structure you requested
    print("Creating xarray Dataset...")
    
    # Transpose predictions_subset to get (draws, starters, races) shape
    # Original shape: (4000, 14, 10) -> Desired shape: (4000, 14, 10)
    # So actually no transposition needed, just reshape for clarity
    xr_predictions = predictions_subset
    
    # Create coordinates for the xarray Dataset
    draw_ids = np.arange(1, total_samples + 1)  # 1 to 4000
    starter_ids = np.arange(1, n_starters + 1)  # 1 to 14
    race_ids_subset = race_ids[:n_pred]  # First 10 race ids from prediction data
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "predictions": (["draw", "starter", "race"], xr_predictions)
        },
        coords={
            "draw": draw_ids,
            "starter": starter_ids,
            "race": race_ids_subset
        }
    )
    
    # Add some metadata
    ds.attrs["description"] = "Posterior predictive draws for race predictions"
    ds.attrs["model"] = "BART"
    ds.attrs["date_created"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # Save the xarray Dataset
    print("Saving xarray Dataset...")
    ds.to_netcdf(os.path.join(output_dir, "posterior_predictions.nc"))
    
    print("Training and prediction complete!")

if __name__ == "__main__":
    main()