import xarray as xr
import numpy as np
import pandas as pd
import os

def compute_rankings(ds):
    """
    Compute rankings for each draw and race based on predictions.
    Returns the top 3 starters for each draw and race.
    """
    # Get the predictions array
    predictions = ds.predictions
    
    # Initialize arrays for top 3 rankings
    n_draws = len(ds.draw)
    n_races = len(ds.race)
    
    # Create arrays to store rankings
    first_place = np.zeros((n_draws, n_races), dtype=int)
    second_place = np.zeros((n_draws, n_races), dtype=int)
    third_place = np.zeros((n_draws, n_races), dtype=int)
    
    # For each draw and race, find top 3 starters
    for d in range(n_draws):
        for r in range(n_races):
            # Get predictions for this draw and race
            race_preds = predictions[d, :, r].values
            
            # Get indices of top 3 predictions
            top_3_indices = np.argsort(race_preds)[-3:][::-1]
            
            # Store rankings (add 1 since starter IDs are 1-based)
            first_place[d, r] = top_3_indices[0] + 1
            second_place[d, r] = top_3_indices[1] + 1
            third_place[d, r] = top_3_indices[2] + 1
    
    return first_place, second_place, third_place

def main():
    # Create output directory if it doesn't exist
    output_dir = "students/fleischhacker_adam2/model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the xarray dataset
    print("Loading xarray dataset...")
    ds = xr.open_dataset(os.path.join(output_dir, "posterior_predictions.nc"))
    
    # Compute rankings
    print("Computing rankings...")
    first_place, second_place, third_place = compute_rankings(ds)
    
    # Add ranking variables to the dataset
    ds["first_place"] = xr.DataArray(
        first_place,
        dims=["draw", "race"],
        coords={"draw": ds.draw, "race": ds.race}
    )
    
    ds["second_place"] = xr.DataArray(
        second_place,
        dims=["draw", "race"],
        coords={"draw": ds.draw, "race": ds.race}
    )
    
    ds["third_place"] = xr.DataArray(
        third_place,
        dims=["draw", "race"],
        coords={"draw": ds.draw, "race": ds.race}
    )
    
    # Add descriptions for the new variables
    ds.first_place.attrs["description"] = "First place starter ID for each draw and race"
    ds.second_place.attrs["description"] = "Second place starter ID for each draw and race"
    ds.third_place.attrs["description"] = "Third place starter ID for each draw and race"
    
    # Save the updated dataset
    print("Saving updated dataset...")
    ds.to_netcdf(os.path.join(output_dir, "posterior_predictions_with_rankings.nc"))
    
    # Also save summary statistics of rankings
    print("Computing and saving ranking summary statistics...")
    
    # Calculate frequency of each starter appearing in top 3 positions
    n_draws = len(ds.draw)
    n_starters = len(ds.starter)
    n_races = len(ds.race)
    
    # Initialize counters
    first_place_counts = np.zeros((n_races, n_starters))
    place_counts = np.zeros((n_races, n_starters))  # For top 2
    show_counts = np.zeros((n_races, n_starters))   # For top 3
    
    # Count occurrences
    for r in range(n_races):
        # Get predictions for this race across all draws
        race_preds = ds.predictions[:, :, r].values
        
        # Find actual starters (those with non-zero predictions in at least one draw)
        actual_starters = np.any(race_preds != 0, axis=0)
        n_actual_starters = np.sum(actual_starters)
        
        for d in range(n_draws):
            # Only count rankings for actual starters
            if first_place[d, r] <= n_actual_starters:
                first_place_counts[r, first_place[d, r] - 1] += 1
                place_counts[r, first_place[d, r] - 1] += 1
                show_counts[r, first_place[d, r] - 1] += 1
                
            if second_place[d, r] <= n_actual_starters:
                place_counts[r, second_place[d, r] - 1] += 1
                show_counts[r, second_place[d, r] - 1] += 1
                
            if third_place[d, r] <= n_actual_starters:
                show_counts[r, third_place[d, r] - 1] += 1
    
    # Convert to probabilities
    first_place_probs = first_place_counts / n_draws
    place_probs = place_counts / n_draws
    show_probs = show_counts / n_draws
    
    # Create DataFrames for each position
    races = ds.race.values
    starters = ds.starter.values
    
    # Create DataFrames
    first_df = pd.DataFrame(first_place_probs, index=races, columns=starters)
    place_df = pd.DataFrame(place_probs, index=races, columns=starters)
    show_df = pd.DataFrame(show_probs, index=races, columns=starters)
    
    # Save to CSV
    first_df.to_csv(os.path.join(output_dir, "first_place_probabilities.csv"))
    place_df.to_csv(os.path.join(output_dir, "place_probabilities.csv"))
    show_df.to_csv(os.path.join(output_dir, "show_probabilities.csv"))
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 