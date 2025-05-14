"""
This script processes racing data from the following locations:
- Input: ../data/rawDataForPrediction/CDX0426.csv (raw racing data)
- Output: ../data/processed/CDX0426_processed.csv (processed data with all columns)
- Output: ../data/processed/CDX0426_filtered.csv (filtered data with only mapped columns)
- Mapping: ../data/column_mapping.csv (column header mappings)
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import csv

def load_column_mapping():
    """Load column mapping from CSV file."""
    try:
        mapping_df = pd.read_csv('students/fleischhacker_adam2/data/column_mapping.csv')
        # Convert column_number to integer and create dictionary
        mapping_dict = dict(zip(mapping_df['column_number'].astype(int), mapping_df['header_name']))
        return mapping_dict
    except Exception as e:
        print(f"Error loading column mapping: {e}")
        return {}

def convert_odds(odds_str):
    """Convert odds to decimal format consistently."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '0':
            return np.nan
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
        return np.nan

def create_prediction_dataset(data_dir):
    """Create an xarray dataset from the prediction data CSV file."""
    # Process the CSV file
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    csv_file = data_dir_path / 'CDX0426.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    print(f"Processing {csv_file.name}...")
    
    # Load column mapping
    column_mapping = load_column_mapping()
    
    # Read CSV file
    df = pd.read_csv(csv_file, header=None, quoting=csv.QUOTE_MINIMAL)
    
    # Create default column names (col_1, col_2, etc.)
    default_columns = [f'col_{i+1}' for i in range(len(df.columns))]
    df.columns = default_columns
    
    # Apply headers from mapping where available
    for col_num, header_name in column_mapping.items():
        if col_num <= len(df.columns):
            df.rename(columns={f'col_{col_num}': header_name}, inplace=True)
    print("\n[DEBUG] Columns after renaming:", df.columns.tolist())
    if 'odds' in df.columns:
        print("[DEBUG] First 10 raw odds values:", df['odds'].head(10).tolist())
        df['odds'] = df['odds'].apply(convert_odds)
        print("[DEBUG] First 10 processed odds values:", df['odds'].head(10).tolist())
    else:
        print("[DEBUG] 'odds' column not found after renaming.")
    # Save processed CSV with all columns
    processed_file = Path("data/processed") / 'CDX0426_processed.csv'
    print("[DEBUG] Saving processed DataFrame to:", processed_file)
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_file, index=False)
    
    # Create a filtered dataframe with only the columns that have headers from the mapping
    mapped_columns = []
    for col_num, header_name in column_mapping.items():
        if col_num <= len(df.columns):
            mapped_columns.append(header_name)
    
    filtered_df = df[mapped_columns]
    
    # Sort by race number and post position
    filtered_df = filtered_df.sort_values(['race_number', 'post_position'])
    
    # Group by race to get number of starters per race
    race_groups = filtered_df.groupby('race_number')
    max_starters = race_groups.size().max()
    
    # Create xarray dataset with proper dimensions
    ds = xr.Dataset()
    
    # Add dimensions
    ds.coords['race'] = np.arange(1, len(race_groups) + 1)  # Race numbers
    ds.coords['starter'] = np.arange(max_starters)  # Maximum number of starters
    
    # Initialize arrays for each variable
    for col in filtered_df.columns:
        if col not in ['race_number', 'post_position']:  # Skip these as they're used for indexing
            # Determine if the column contains numeric or string data
            sample_value = filtered_df[col].iloc[0]
            if isinstance(sample_value, (int, float)) or (isinstance(sample_value, str) and sample_value.replace('.', '').isdigit()):
                # Numeric data
                arr = np.full((len(ds.race), len(ds.starter)), np.nan, dtype=float)
            else:
                # String data
                arr = np.full((len(ds.race), len(ds.starter)), '', dtype=object)
            
            # Fill in values for each race
            for race_idx, (race_num, race_df) in enumerate(race_groups):
                for starter_idx, (_, row) in enumerate(race_df.iterrows()):
                    value = row[col]
                    if pd.isna(value):
                        continue
                    if isinstance(arr, np.ndarray) and arr.dtype == float:
                        try:
                            arr[race_idx, starter_idx] = float(value)
                        except (ValueError, TypeError):
                            arr[race_idx, starter_idx] = np.nan
                    else:
                        arr[race_idx, starter_idx] = str(value).strip()
            
            ds[col] = (['race', 'starter'], arr)
    
    # Add race number and post position as coordinates
    race_nums = np.array([race_num for race_num, _ in race_groups])
    ds.coords['race_number'] = ('race', race_nums)
    
    post_positions = np.full((len(ds.race), len(ds.starter)), np.nan)
    for race_idx, (_, race_df) in enumerate(race_groups):
        for starter_idx, (_, row) in enumerate(race_df.iterrows()):
            post_positions[race_idx, starter_idx] = row['post_position']
    ds.coords['post_position'] = (['race', 'starter'], post_positions)
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print("=" * 80)
    print(f"Total races: {len(ds.race)}")
    print(f"Maximum starters per race: {len(ds.starter)}")
    print(f"Total entries: {filtered_df.shape[0]}")
    
    # Print odds statistics if odds data exists
    if 'odds' in ds:
        print("\nOdds Data Statistics:")
        print("-" * 50)
        total_entries = ds.odds.size
        valid_odds = np.sum(~np.isnan(ds.odds))
        print(f"Total possible odds entries: {total_entries}")
        print(f"Valid odds entries found: {valid_odds}")
        print(f"Percentage of valid odds: {(valid_odds/total_entries)*100:.2f}%")
        
        # Calculate statistics for valid odds
        valid_odds_values = ds.odds.values[~np.isnan(ds.odds.values)]
        if len(valid_odds_values) > 0:
            print(f"\nOdds Statistics (for valid entries):")
            print(f"Minimum odds: {np.min(valid_odds_values):.2f}")
            print(f"Maximum odds: {np.max(valid_odds_values):.2f}")
            print(f"Mean odds: {np.mean(valid_odds_values):.2f}")
            print(f"Median odds: {np.median(valid_odds_values):.2f}")
    
    print("=" * 80)
    
    return ds

def save_dataset(ds, output_dir):
    """Save the xarray dataset to a netCDF file in the specified directory."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    output_file = output_path / "processed_prediction_data.nc"
    ds.to_netcdf(output_file)
    print(f"\nDataset saved to {output_file}")

if __name__ == "__main__":
    # Process prediction data
    data_dir = "data/rawDataForPrediction"
    ds = create_prediction_dataset(data_dir)
    
    # Save the dataset
    output_dir = "students/fleischhacker_adam2/data/processed"
    save_dataset(ds, output_dir)
