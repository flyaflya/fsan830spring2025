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
    # Handle any invalid input by returning 0
    if pd.isna(finish_pos) or finish_pos == '' or finish_pos is None:
        return 0.0
    
    try:
        finish_pos = float(finish_pos)
        if finish_pos == 1:
            return 6.0
        elif finish_pos == 2:
            return 2.0
        elif finish_pos == 3:
            return 1.0
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

def load_and_process_data(input_path, is_training_data=False, max_starters=14):
    """
    Load and process the dataset to create initial features.
    
    Parameters:
    -----------
    input_path : str or Path
        Path to input netCDF file
    is_training_data : bool
        If True, convert distance from furlongs to yards
        If False, assume distance is already in yards
    max_starters : int
        Maximum number of starters to include. Default is 14.
        
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
    # points = {f'st{i+1}_r1_pts': [] for i in range(max_starters)}  # r1 = recent1, pts = points
    # surge = {f'st{i+1}_surge': [] for i in range(max_starters)}  # surge = improvement in final stretch
    # num_entrants = {f'st{i+1}_r1_ent': [] for i in range(max_starters)}  # ent = number of entrants
    odds = {f'st{i+1}_odds': [] for i in range(max_starters)}  # odds for each starter
    
    # Initialize dictionary for target points (6-2-1 system)
    target_points = {f'st{i+1}_pts': [] for i in range(max_starters)}
    
    def convert_odds(odds_str):
        """Convert odds to decimal format consistently."""
        try:
            if pd.isna(odds_str) or str(odds_str).strip() == '0':
                return 999.0  # Use high number for missing odds
            odds_str = str(odds_str).strip()
            # Handle decimal odds (e.g., "4.00", "30.00")
            if odds_str.replace('.', '').isdigit():
                return float(odds_str)
            # Handle fractional odds (e.g., "6/1")
            if '/' in odds_str:
                num, denom = odds_str.split('/')
                return float(num) / float(denom) + 1
            return float(odds_str)
        except (ValueError, ZeroDivisionError):
            return 999.0  # Use high number for invalid odds
    
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
                    # If all values are empty, use 0
                    distance_yards = 0.0
            except (ValueError, TypeError):
                distance_yards = 0.0
        
        # Get purse value
        try:
            if is_training_data:
                purse = float(ds.purse.values[race_idx])
            else:
                # For prediction data, get the first non-null purse value for the race
                race_purses = ds.purse.values[race_idx]
                purse = 0.0  # Default to 0 for missing purse values
                for p in race_purses:
                    if not pd.isna(p) and p != '':
                        purse = float(p)
                        break
        except (ValueError, TypeError, AttributeError):
            purse = 0.0  # Use 0 for missing purse values
        
        # Get most recent race data for all starters
        if is_training_data:
            # For training data, use the most recent prior race's finish and stretch positions
            prior_finish_positions = ds.recent_finish_pos.isel(race=race_idx, past_race=0).values
            prior_stretch_positions = ds.recent_last_call_pos.isel(race=race_idx, past_race=0).values
            race_odds = ds.odds.isel(race=race_idx).values if 'odds' in ds else np.full(max_starters, 999.0)
        else:
            # For prediction data, use the most recent prior race's finish and stretch positions
            prior_finish_positions = ds.recentFinishPosition1.isel(race=race_idx).values
            prior_stretch_positions = ds.recentStretchPosition1.isel(race=race_idx).values
            race_odds = ds.odds.isel(race=race_idx).values if 'odds' in ds else np.full(max_starters, 999.0)
        
        # Store the data
        race_ids.append(race_id)
        distances.append(distance_yards)
        purses.append(purse)
        
        # Store data for each starter
        for starter_idx in range(max_starters):  # Process only up to max_starters
            if starter_idx < len(prior_finish_positions):
                finish_pos = prior_finish_positions[starter_idx]
                stretch_pos_value = float(prior_stretch_positions[starter_idx]) if not pd.isna(prior_stretch_positions[starter_idx]) else -1.0
                # Calculate surge (improvement in final stretch)
                # if stretch_pos_value == -1.0 or pd.isna(finish_pos):
                #     surge_value = 0.0
                # else:
                #     # Only calculate surge for horses that finished in top 2 in the prior race
                #     if finish_pos <= 2:
                #         surge_value = max(0, stretch_pos_value - finish_pos)
                #     else:
                #         surge_value = 0.0
                # surge[f'st{starter_idx+1}_surge'].append(surge_value)
                
                # Store odds (use 999.0 for missing values - worst possible odds)
                odds[f'st{starter_idx+1}_odds'].append(
                    convert_odds(race_odds[starter_idx])
                )
                
                # Store target points (6-2-1 system) only for training data
                if is_training_data:
                    target_points[f'st{starter_idx+1}_pts'].append(calculate_points(finish_pos))
            else:
                # Add default values for missing starters
                # surge[f'st{starter_idx+1}_surge'].append(0.0)  # 0 surge for missing starters
                odds[f'st{starter_idx+1}_odds'].append(999.0)  # 999 for missing odds
                if is_training_data:
                    target_points[f'st{starter_idx+1}_pts'].append(0.0)  # 0 points for missing starters
    
    # Create DataFrame with all columns
    df_dict = {
        'race_id': race_ids,
        'distance_yards': distances,
        'purse': purses,
        # **points,  # Add all starter points columns
        # **surge,  # Add all starter surge columns
        # **num_entrants,  # Add all starter number of entrants columns
        **odds  # Add all starter odds columns
    }
    
    # Add target points columns only for training data
    if is_training_data:
        df_dict.update(target_points)
    
    df = pd.DataFrame(df_dict)
    
    # Ensure all points columns have 0.0 instead of NaN
    points_cols = [col for col in df.columns if col.endswith('_pts')]
    df[points_cols] = df[points_cols].fillna(0.0)
    
    return df

def create_feature_dataframes():
    """
    Create feature dataframes for both training and prediction data.
    """
    # Define paths
    training_input = Path("students/fleischhacker_adam2/data/processed/processed_race_data_with_results.nc")
    prediction_input = Path("students/fleischhacker_adam2/data/processed/processed_prediction_data.nc")
    
    # Create output directory if it doesn't exist
    output_dir = Path("students/fleischhacker_adam2/data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training data with fixed 14 starters
    print("\nProcessing training data...")
    training_df = load_and_process_data(training_input, is_training_data=True, max_starters=14)
    training_df.to_csv(output_dir / "training_features.csv", index=False)
    print(f"Training features saved to {output_dir / 'training_features.csv'}")
    
    # Process prediction data with actual number of starters per race
    print("\nProcessing prediction data...")
    prediction_df = load_and_process_data(prediction_input, is_training_data=False, max_starters=14)
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