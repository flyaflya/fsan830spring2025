import xarray as xr
import pandas as pd
import numpy as np

def get_top_three_horses(dataset):
    """
    Query the race dataset to get the top 3 horses in each race along with their details.
    
    Args:
        dataset (xarray.Dataset): The race dataset containing all race information
        
    Returns:
        pandas.DataFrame: A DataFrame containing the top 3 horses from each race
                         with their details
    """
    # Initialize lists to store results
    results = []
    
    # Iterate through each race
    for race_num in dataset.race.values:
        # Get finish positions for current race
        finish_pos = dataset.finish_pos.sel(race=race_num).values
        
        # Create a mask for valid positions (exclude NaN values)
        valid_mask = ~np.isnan(finish_pos)
        
        if np.any(valid_mask):
            # Get valid positions and corresponding horse numbers
            valid_positions = finish_pos[valid_mask]
            horse_numbers = dataset.horse.values[valid_mask]
            
            # Sort horses by finish position
            sorted_indices = np.argsort(valid_positions)
            top_3_indices = sorted_indices[:3]  # Get indices of top 3 finishers
            
            # Get horse numbers for top 3 finishers
            top_3_horses = horse_numbers[top_3_indices]
            
            # Extract details for each top 3 horse
            for horse_num in top_3_horses:
                horse_data = {
                    'race_number': race_num,
                    'horse_name': dataset.horse_names.sel(
                        race=race_num, horse=horse_num).item(),
                    'jockey_name': dataset.jockeys.sel(
                        race=race_num, horse=horse_num).item(),
                    'trainer_name': dataset.trainers.sel(
                        race=race_num, horse=horse_num).item(),
                    'finish_position': dataset.finish_pos.sel(
                        race=race_num, horse=horse_num).item(),
                    'odds': dataset.odds.sel(
                        race=race_num, horse=horse_num).item()
                }
                results.append(horse_data)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by race number and finish position
    df = df.sort_values(['race_number', 'finish_position'])
    
    return df

def main():
    # Load the dataset
    dataset = xr.open_dataset('race_data.nc')
    
    # Get top 3 horses for each race
    top_horses = get_top_three_horses(dataset)
    
    # Display results
    print("\nTop 3 Horses in Each Race:")
    print("==========================")
    
    # Group by race number and display results for each race
    for race_num, race_group in top_horses.groupby('race_number'):
        print(f"\nRace {race_num}:")
        print("---------")
        for _, horse in race_group.iterrows():
            print(f"Position: {horse['finish_position']}")
            print(f"Horse: {horse['horse_name']}")
            print(f"Jockey: {horse['jockey_name']}")
            print(f"Trainer: {horse['trainer_name']}")
            print(f"Odds: {horse['odds']}")
            print()

if __name__ == '__main__':
    main()