import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pymc_bart as pmb

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
    
    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values
    
    # Prepare prediction data (only features, no targets)
    X_pred = pred_df[feature_cols].values
    
    return X_train, y_train, X_pred, feature_cols, target_cols, n_starters

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
        # Create a BART model with dims parameter for multi-output
        # Using separate_trees=False for better compatibility
        μ = pmb.BART(
            "μ", 
            X_train, 
            y_train,  
            m=50,     
            dims=["starters", "n_obs"],
            separate_trees=False  # Share tree structure across outputs
        )
        
        # Error term with appropriate dimensions
        σ = pm.HalfNormal("σ", 1, dims=["starters"])
        
        # Make sure the observation matches dimensions properly
        # The key is to ensure obs has the same dimensions as μ
        obs = pm.Normal("obs", mu=μ, sigma=σ.dimshuffle(0, 'x'), observed=y_train.T)
    
    return model

def main():
    # Load and preprocess data
    print("Loading data...")
    X_train, y_train, X_pred, feature_cols, target_cols, n_starters = load_data()
    
    # Create and train model
    print("Creating multi-output BART model...")
    model = create_multi_output_bart_model(X_train, y_train, n_starters)
    
    # Sample from posterior
    print("Sampling from posterior...")
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            return_inferencedata=True
        )
    
    # Generate predictions
    print("Generating predictions...")
    with model:
        # Use pm.sample_posterior_predictive for out-of-sample prediction
        posterior_pred = pm.sample_posterior_predictive(
            idata,
            var_names=["μ"],
            predictions=True,
            X=X_pred
        )
    
    # Get mean predictions across posterior samples
    y_pred = posterior_pred.posterior_predictive["μ"].mean(axis=(0, 1))
    
    # Convert predictions to DataFrame - transpose to get original orientation
    pred_df = pd.DataFrame(y_pred.T, columns=target_cols)
    
    # Save predictions and model
    print("Saving predictions and model...")
    pred_df.to_csv('predictions.csv', index=False)
    pm.save_trace(idata, 'bart_model')
    
    print("Training and prediction complete!")

if __name__ == "__main__":
    main()