import xarray as xr
import pandas as pd
from pathlib import Path

def get_top_3_horses(dataset_path):
    """Load the dataset and retrieve the top 3 horses for each race."""
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    
    # Load dataset
    ds = xr.load_dataset(dataset_path)

    # Convert to pandas DataFrame
    df = pd.DataFrame({
        'race_number': ds['race_number'].values,
        'horse': ds['horse'].values,
        'jockey': ds['jockey'].values,
        'trainer': ds['trainer'].values,
        'finish_pos': ds['finish_pos'].values,
        'odds': ds['odds'].values
    })

    # Ensure finish_pos is numeric
    df = df.dropna(subset=['finish_pos'])  # Remove any NaN finish positions
    df['finish_pos'] = df['finish_pos'].astype(int)

    # Sort by race number and finishing position, then get top 3 per race
    top_3_df = df.sort_values(by=['race_number', 'finish_pos']).groupby('race_number').head(3)

    return top_3_df

if __name__ == "__main__":
    dataset_path = Path.cwd() / 'race_results.nc'
    try:
        top_3_results = get_top_3_horses(dataset_path)
        print("Top 3 horses in each race:")
        print(top_3_results)
    except Exception as e:
        print(f"Error: {e}")
