import xarray as xr
import pandas as pd
from pathlib import Path

def find_horse_data(horse_name, data_dir="students/fleischhacker_adam2/data/processed"):
    """Find and display all data for a specific horse."""
    # Load the dataset
    data_path = Path(data_dir) / "processed_race_data.nc"
    if not data_path.exists():
        print(f"Error: Could not find data file at {data_path}")
        return
    
    ds = xr.open_dataset(data_path)
    
    # Find all races where the horse appears
    horse_mask = ds.horse == horse_name
    if not horse_mask.any():
        print(f"No data found for horse: {horse_name}")
        return
    
    # Get the race indices where the horse appears
    race_indices = horse_mask.any(dim='starter').values.nonzero()[0]
    
    # Create a DataFrame to store the results
    results = []
    
    for race_idx in race_indices:
        # Get the race ID
        race_id = ds.race.values[race_idx]
        
        # Find the starter index for this horse in this race
        starter_idx = horse_mask[race_idx].values.nonzero()[0][0]
        
        # Get all the data for this race and starter
        race_data = {
            'Race ID': race_id,
            'Surface': ds.surface[race_idx].values,
            'Distance (f)': ds.distance_f[race_idx].values,
            'Purse ($)': f"${ds.purse[race_idx].values:,.0f}",
            'Horse': ds.horse[race_idx, starter_idx].values,
            'Jockey': ds.jockey[race_idx, starter_idx].values,
            'Trainer': ds.trainer[race_idx, starter_idx].values,
            'Finish Position': ds.finish_pos[race_idx, starter_idx].values,
            'Lengths Back': f"{ds.lengths_back[race_idx, starter_idx].values:.1f}",
            'Speed Figure': ds.speed_fig[race_idx, starter_idx].values,
            'ML Odds': f"{ds.odds_mline[race_idx, starter_idx].values:.2f}"
        }
        results.append(race_data)
    
    # Convert to DataFrame and display
    df = pd.DataFrame(results)
    
    # Sort by Race ID to show races in chronological order
    df = df.sort_values('Race ID')
    
    # Set display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(f"\nFound {len(results)} races for {horse_name}:")
    print("\nRace History:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    find_horse_data("Lady Arsinoe") 