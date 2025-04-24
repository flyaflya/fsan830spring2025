"""
Example script for converting XML race results to xarray dataset.
This is a starting point - students should modify and expand this code.
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import xarray as xr
import os

def parse_xml_to_xarray(xml_path):
    """
    Parse XML race results file and convert to xarray dataset.
    
    Parameters:
    -----------
    xml_path : str
        Path to the XML file
        
    Returns:
    --------
    xr.Dataset
        xarray Dataset with race results
    """
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
     # Debug print to see what we're working with
    print(f"Root tag: {root.tag}")
    race_date = root.get('RACE_DATE')

    # Extract basic information
    # chart = root.find('.//CHART')
    #race_date = chart.get('RACE_DATE')
    
    # Find track information
    track_elem = root.find('.//TRACK')
    track_id = track_elem.find('CODE').text
    track_name = track_elem.find('NAME').text
    
    # Initialize lists to store data
    race_numbers = []
    horse_names = []
    jockey_names = []
    trainer_names = []
    finishing_positions = []
    odds = []
    purses = []
    distances = []
    track_conditions = []
    
    # Extract race information
    for race in root.findall('.//RACE'):
        race_number = race.get('NUMBER')
        
        # Extract race details
        purse = race.find('PURSE').text if race.find('PURSE') is not None else None
        distance = race.find('DISTANCE').text if race.find('DISTANCE') is not None else None
        track_condition = race.find('TRK_COND').text if race.find('TRK_COND') is not None else None
        
        # Extract entry information
        for entry in race.findall('.//ENTRY'):
            # Horse name
            horse_name = entry.find('NAME').text if entry.find('NAME') is not None else None
            
            # Jockey name
            jockey = entry.find('.//JOCKEY')
            if jockey is not None:
                first_name = jockey.find('FIRST_NAME').text if jockey.find('FIRST_NAME') is not None else ""
                middle_name = jockey.find('MIDDLE_NAME').text if jockey.find('MIDDLE_NAME') is not None else ""
                last_name = jockey.find('LAST_NAME').text if jockey.find('LAST_NAME') is not None else ""
                jockey_name = f"{first_name} {middle_name} {last_name}".strip().replace("  ", " ")
            else:
                jockey_name = None
            
            # Trainer name
            trainer = entry.find('.//TRAINER')
            if trainer is not None:
                first_name = trainer.find('FIRST_NAME').text if trainer.find('FIRST_NAME') is not None else ""
                middle_name = trainer.find('MIDDLE_NAME').text if trainer.find('MIDDLE_NAME') is not None else ""
                last_name = trainer.find('LAST_NAME').text if trainer.find('LAST_NAME') is not None else ""
                trainer_name = f"{first_name} {middle_name} {last_name}".strip().replace("  ", " ")
            else:
                trainer_name = None
            
            # Finishing position and odds
            fin_pos = entry.find('OFFICIAL_FIN').text if entry.find('OFFICIAL_FIN') is not None else None
            dollar_odds = entry.find('DOLLAR_ODDS').text if entry.find('DOLLAR_ODDS') is not None else None
            
            # Append data to lists
            race_numbers.append(race_number)
            horse_names.append(horse_name)
            jockey_names.append(jockey_name)
            trainer_names.append(trainer_name)
            finishing_positions.append(fin_pos)
            odds.append(dollar_odds)
            purses.append(purse)
            distances.append(distance)
            track_conditions.append(track_condition)
    
    # Create a pandas DataFrame
    data = {
        'race_number': race_numbers,
        'horse': horse_names,
        'jockey': jockey_names,
        'trainer': trainer_names,
        'finishing_position': finishing_positions,
        'odds': odds,
        'purse': purses,
        'distance': distances,
        'track_condition': track_conditions
    }
    df = pd.DataFrame(data)
    
    # Convert to numeric where appropriate
    df['race_number'] = pd.to_numeric(df['race_number'])
    df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    
    # Create xarray dataset
    ds = xr.Dataset()
    
    # Add coordinates
    ds.coords['TRACK'] = [track_id]
    ds.coords['RACE_DATE'] = [race_date]
    ds.coords['RACE_NUMBER'] = sorted(df['race_number'].unique())
    
    # Add variables
    # Note: This is a simplified approach - students should expand this
    for race_num in ds.coords['RACE_NUMBER'].values:
        race_df = df[df['race_number'] == race_num]
        
        # Add horse information as a variable
        ds[f'race_{race_num}_horses'] = xr.DataArray(
            race_df['horse'].values,
            dims=['entry'],
            coords={'entry': np.arange(len(race_df))}
        )
        
        # Add finishing positions
        ds[f'race_{race_num}_positions'] = xr.DataArray(
            race_df['finishing_position'].values,
            dims=['entry'],
            coords={'entry': np.arange(len(race_df))}
        )
        
        # Add odds
        ds[f'race_{race_num}_odds'] = xr.DataArray(
            race_df['odds'].values,
            dims=['entry'],
            coords={'entry': np.arange(len(race_df))}
        )
        
        # Add jockey name
        ds[f'race_{race_num}_jockey'] = xr.DataArray(
            race_df['jockey'].values,
            dims=['entry'],
            coords={'entry': np.arange(len(race_df))}
        )

        # Add trainer name
        ds[f'race_{race_num}_trainer'] = xr.DataArray(
            race_df['trainer'].values,
            dims=['entry'],
            coords={'entry': np.arange(len(race_df))}
        )

    # Add track metadata
    ds.attrs['track_name'] = track_name
    
    return ds

def query_top_horses(ds, n=3):
    """
    Query the top n horses in each race.
    
    Parameters:
    -----------
    ds : xr.Dataset
        xarray Dataset with race results
    n : int
        Number of top horses to return
        
    Returns:
    --------
    dict
        Dictionary with race numbers as keys and top horses as values
    """
    results = {}
    
    for race_num in ds.coords['RACE_NUMBER'].values:
        # Get positions and horse names
        horses = ds[f'race_{race_num}_horses'].values
        jockey_names = ds[f'race_{race_num}_jockey'].values
        trainer_names = ds[f'race_{race_num}_trainer'].values
        positions = ds[f'race_{race_num}_positions'].values
        odds = ds[f'race_{race_num}_odds'].values
        
        # Create a DataFrame for this race
        race_df = pd.DataFrame({
            'horse': horses,
            'jockey_name': jockey_names,
            'trainer_name': trainer_names,
            'position': positions,
            'odds': odds
        })
        
        # Sort by position and get top n
        top_horses = race_df.sort_values('position').head(n)
        
        # Store in results
        results[int(race_num)] = top_horses
    
    return results

def main():
    # Path to XML file (relative to repository root)
    xml_path = os.path.join('..', '..', 'data', 'sampleRaceResults', 'del20230708tch.xml')

    # Parse XML to xarray
    ds = parse_xml_to_xarray(xml_path)
    
    # Print dataset information
    print("Dataset information:")
    print(ds)
    
    # Query top 3 horses in each race
    top_horses = query_top_horses(ds, n=3)
    
    # Print results
    print("\nTop 3 horses in each race:")
    for race_num, horses in top_horses.items():
        print(f"\nRace {race_num}:")
        print(horses)
    
    # Save dataset to NetCDF file
    output_dir = '.'
    ds.to_netcdf(os.path.join(output_dir, 'race_results.nc'))
    print(f"\nDataset saved to {os.path.join(output_dir, 'race_results.nc')}")

if __name__ == "__main__":
    main() 