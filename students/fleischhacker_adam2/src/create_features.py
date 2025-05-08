"""
This script creates feature dataframes for both training and prediction data.
The dataframes will have identical columns (except for target variables) to ensure compatibility with PYMC-BART.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_data(input_path, is_training_data=False):
    """
    Load and process the dataset to create initial features.
    
    Parameters:
    -----------
    input_path : str or Path
        Path to input netCDF file
    is_training_data : bool
        If True, convert distance from furlongs to yards
        If False, assume distance is already in yards
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the processed features
    """
    # Load the dataset
    ds = xr.load_dataset(input_path)
    
    # Initialize lists to store data
    race_ids = []
    distances = []
    purses = []
    
    # Process each race
    for race_idx in range(len(ds.race)):
        # Get race identifier
        race_id = ds.race.values[race_idx]
        
        # Get distance
        if is_training_data:
            # Training data uses furlongs
            distance_furlongs = ds.distance_f.values[race_idx]
            distance_yards = distance_furlongs * 220  # Convert furlongs to yards
        else:
            # Prediction data is already in yards
            # Get the first valid distance value for the race
            race_distances = ds.distance.values[race_idx]
            # Convert to float, handling empty strings
            try:
                # Try to convert the first non-empty value
                for dist in race_distances:
                    if dist != '':
                        distance_yards = float(dist)
                        break
                else:
                    # If all values are empty
                    distance_yards = np.nan
            except (ValueError, TypeError):
                distance_yards = np.nan
        
        # Get purse value
        try:
            if is_training_data:
                purse = float(ds.purse.values[race_idx])
            else:
                # For prediction data, get the first non-null purse value for the race
                race_purses = ds.purse.values[race_idx]
                purse = np.nan
                for p in race_purses:
                    if not pd.isna(p) and p != '':
                        purse = float(p)
                        break
        except (ValueError, TypeError, AttributeError):
            purse = np.nan  # Use NaN for missing purse values
        
        # Store the data
        race_ids.append(race_id)
        distances.append(distance_yards)
        purses.append(purse)
    
    # Create DataFrame
    df = pd.DataFrame({
        'race_id': race_ids,
        'distance_yards': distances,
        'purse': purses
    })
    
    return df

def create_feature_dataframes():
    """
    Create feature dataframes for both training and prediction data.
    """
    # Define paths
    training_input = Path("students/fleischhacker_adam2/data/processed/processed_race_data.nc")
    prediction_input = Path("students/fleischhacker_adam2/data/processed/processed_prediction_data.nc")
    
    # Create output directory if it doesn't exist
    output_dir = Path("students/fleischhacker_adam2/data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    training_df = load_and_process_data(training_input, is_training_data=True)
    training_df.to_csv(output_dir / "training_features.csv", index=False)
    print(f"Training features saved to {output_dir / 'training_features.csv'}")
    
    # Process prediction data
    print("\nProcessing prediction data...")
    prediction_df = load_and_process_data(prediction_input, is_training_data=False)
    prediction_df.to_csv(output_dir / "prediction_features.csv", index=False)
    print(f"Prediction features saved to {output_dir / 'prediction_features.csv'}")
    
    # Print summary of the dataframes
    print("\nDataFrame Summary:")
    print("\nTraining DataFrame:")
    print(training_df.info())
    print("\nPrediction DataFrame:")
    print(prediction_df.info())

if __name__ == "__main__":
    create_feature_dataframes() 