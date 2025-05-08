"""
This script creates feature dataframes for both training and prediction data.
The dataframes will have identical columns (except for target variables) to ensure compatibility with PYMC-BART.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_points(finish_pos):
    """
    Calculate points based on finish position:
    - 1st place: 6 points
    - 2nd place: 2 points
    - 3rd place: 1 point
    - Other positions: 0 points
    
    Parameters:
    -----------
    finish_pos : float or int
        Finish position of the horse
        
    Returns:
    --------
    float
        Points earned
    """
    if pd.isna(finish_pos):
        return np.nan
    
    finish_pos = float(finish_pos)
    if finish_pos == 1:
        return 6.0
    elif finish_pos == 2:
        return 2.0
    elif finish_pos == 3:
        return 1.0
    else:
        return 0.0

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
    
    # Initialize dictionaries to store data for each starter
    points = {f'st{i+1}_r1_pts': [] for i in range(23)}  # 23 starters max, r1 = recent1, pts = points
    stretch_pos = {f'st{i+1}_r1_str': [] for i in range(23)}  # str = stretch position
    num_entrants = {f'st{i+1}_r1_ent': [] for i in range(23)}  # ent = number of entrants
    
    # Initialize dictionary for finish positions (target variables)
    finish_positions = {f'st{i+1}_fin': [] for i in range(23)}  # fin = finish position
    
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
        
        # Get most recent race data for all starters
        if is_training_data:
            # For training data, get the most recent data (index 0 in past_race dimension)
            race_finish_positions = ds.recent_finish_pos.isel(race=race_idx, past_race=0).values
            race_stretch_positions = ds.recent_last_call_pos.isel(race=race_idx, past_race=0).values
            race_num_entrants = ds.recent_num_starters.isel(race=race_idx, past_race=0).values
        else:
            # For prediction data, get the first post position
            race_finish_positions = ds.recentPostPosition1.isel(race=race_idx).values
            race_stretch_positions = ds.recentStretchPosition1.isel(race=race_idx).values
            race_num_entrants = ds.recentNumEntrants1.isel(race=race_idx).values
            
        # Store the data
        race_ids.append(race_id)
        distances.append(distance_yards)
        purses.append(purse)
        
        # Store data for each starter
        for starter_idx in range(23):  # Process all 23 possible starter positions
            if starter_idx < len(race_finish_positions):
                # Calculate points from finish position
                finish_pos = race_finish_positions[starter_idx]
                points[f'st{starter_idx+1}_r1_pts'].append(calculate_points(finish_pos))
                
                # Store stretch position
                stretch_pos[f'st{starter_idx+1}_r1_str'].append(race_stretch_positions[starter_idx])
                
                # Store number of entrants
                num_entrants[f'st{starter_idx+1}_r1_ent'].append(race_num_entrants[starter_idx])
                
                # Store finish position (target variable)
                if not is_training_data:
                    finish_positions[f'st{starter_idx+1}_fin'].append(finish_pos)
            else:
                # Add NaN for missing starters
                points[f'st{starter_idx+1}_r1_pts'].append(np.nan)
                stretch_pos[f'st{starter_idx+1}_r1_str'].append(np.nan)
                num_entrants[f'st{starter_idx+1}_r1_ent'].append(np.nan)
                if not is_training_data:
                    finish_positions[f'st{starter_idx+1}_fin'].append(np.nan)
    
    # Create DataFrame with all columns
    df_dict = {
        'race_id': race_ids,
        'distance_yards': distances,
        'purse': purses,
        **points,  # Add all starter points columns
        **stretch_pos,  # Add all starter stretch position columns
        **num_entrants  # Add all starter number of entrants columns
    }
    
    # Add finish position columns for prediction data
    if not is_training_data:
        df_dict.update(finish_positions)
    
    df = pd.DataFrame(df_dict)
    
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