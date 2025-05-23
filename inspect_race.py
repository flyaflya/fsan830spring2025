import xarray as xr
import sys

# Path to the processed NetCDF file
nc_path = 'students/fleischhacker_adam2/data/processed/processed_race_data_with_results.nc'

def main():
    ds = xr.open_dataset(nc_path)
    race_ids = list(ds['race'].values)
    
    # Get race_id from command line or prompt
    if len(sys.argv) > 1:
        race_id = sys.argv[1]
    else:
        print('Available race IDs (first 10):', race_ids[:10])
        race_id = input('Enter race ID (e.g., CD-05-03-23-R01): ').strip()
    
    if race_id not in race_ids:
        print(f'Race ID {race_id} not found!')
        return
    race_idx = race_ids.index(race_id)
    horses = ds['horse'][race_idx].values
    finish_positions = ds['finish_position'][race_idx].values
    print(f'Race: {race_id}')
    print('Horses and finish positions:')
    for h, pos in zip(horses, finish_positions):
        print(f'  Horse: {h}, Finish Position: {pos}')
    # Check for multiple or missing winners
    num_winners = (finish_positions == 1).sum()
    print(f'Number of winners (finish_position == 1): {num_winners}')
    if num_winners != 1:
        print('WARNING: There should be exactly one winner!')

if __name__ == '__main__':
    main() 