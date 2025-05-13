import xarray as xr
import numpy as np
import pandas as pd

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
    # Load the xarray dataset
    print("Loading xarray dataset...")
    ds = xr.open_dataset("posterior_predictions.nc")
    
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
    ds.to_netcdf("posterior_predictions_with_rankings.nc")
    
    # Also save summary statistics of rankings
    print("Computing and saving ranking summary statistics...")
    
    # Calculate frequency of each starter appearing in top 3 positions
    n_draws = len(ds.draw)
    n_starters = len(ds.starter)
    n_races = len(ds.race)
    
    # Initialize counters
    first_place_counts = np.zeros((n_races, n_starters))
    second_place_counts = np.zeros((n_races, n_starters))
    third_place_counts = np.zeros((n_races, n_starters))
    
    # Count occurrences
    for r in range(n_races):
        for d in range(n_draws):
            first_place_counts[r, first_place[d, r] - 1] += 1
            second_place_counts[r, second_place[d, r] - 1] += 1
            third_place_counts[r, third_place[d, r] - 1] += 1
    
    # Convert to probabilities
    first_place_probs = first_place_counts / n_draws
    second_place_probs = second_place_counts / n_draws
    third_place_probs = third_place_counts / n_draws
    
    # Create DataFrames for each position
    races = ds.race.values
    starters = ds.starter.values
    
    # Create DataFrames
    first_df = pd.DataFrame(first_place_probs, index=races, columns=starters)
    second_df = pd.DataFrame(second_place_probs, index=races, columns=starters)
    third_df = pd.DataFrame(third_place_probs, index=races, columns=starters)
    
    # Save to CSV
    first_df.to_csv("first_place_probabilities.csv")
    second_df.to_csv("second_place_probabilities.csv")
    third_df.to_csv("third_place_probabilities.csv")
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 