import xml.etree.ElementTree as ET
import xarray as xr
import pandas as pd
import numpy as np
import os

def parse_xml_to_xarray(xml_path):
    """
    Parse XML race results file and convert to an xarray Dataset.
    
    Parameters:
    -----------
    xml_path : str
        Path to the XML file.
        
    Returns:
    --------
    xr.Dataset
        xarray Dataset with race results.
    """
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Debug: Print XML structure
    print("Root tag:", root.tag)
    for child in root:
        print("Child tag:", child.tag, "attributes:", child.attrib)
        
    # Extract race date directly from the root (which is <CHART>)
    race_date = root.get('RACE_DATE')
    print("Found chart with race_date:", race_date)
    
    # Find track information
    track_elem = root.find('TRACK')
    if track_elem is None:
        print("Track element not found")
        track_id = "UNKNOWN"
        track_name = "UNKNOWN"
    else:
        track_id = track_elem.find('CODE').text if track_elem.find('CODE') is not None else "UNKNOWN"
        track_name = track_elem.find('NAME').text if track_elem.find('NAME') is not None else "UNKNOWN"
        print(f"Found track: {track_id} - {track_name}")
    
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
    
    # Extract race and entry information
    for race in root.findall('.//RACE'):
        race_number = race.get('NUMBER')
        
        # Extract race details
        purse = race.find('PURSE').text if race.find('PURSE') is not None else None
        distance = race.find('DISTANCE').text if race.find('DISTANCE') is not None else None
        track_condition = race.find('TRK_COND').text if race.find('TRK_COND') is not None else None
        
        # Process each ENTRY element in this race
        for entry in race.findall('.//ENTRY'):
            # Horse name (using tag "NAME")
            horse_name = entry.find('NAME').text if entry.find('NAME') is not None else None
            
            # Jockey name
            jockey = entry.find('.//JOCKEY')
            if jockey is not None:
                first_name = jockey.find('FIRST_NAME').text if jockey.find('FIRST_NAME') is not None else ""
                middle_name = jockey.find('MIDDLE_NAME').text if jockey.find('MIDDLE_NAME') is not None else ""
                last_name = jockey.find('LAST_NAME').text if jockey.find('LAST_NAME') is not None else ""
                jockey_full = f"{first_name} {middle_name} {last_name}".strip().replace("  ", " ")
            else:
                jockey_full = None
            
            # Trainer name
            trainer = entry.find('.//TRAINER')
            if trainer is not None:
                first_name = trainer.find('FIRST_NAME').text if trainer.find('FIRST_NAME') is not None else ""
                middle_name = trainer.find('MIDDLE_NAME').text if trainer.find('MIDDLE_NAME') is not None else ""
                last_name = trainer.find('LAST_NAME').text if trainer.find('LAST_NAME') is not None else ""
                trainer_full = f"{first_name} {middle_name} {last_name}".strip().replace("  ", " ")
            else:
                trainer_full = None
            
            # Finishing position and odds
            fin_pos = entry.find('OFFICIAL_FIN').text if entry.find('OFFICIAL_FIN') is not None else None
            dollar_odds = entry.find('DOLLAR_ODDS').text if entry.find('DOLLAR_ODDS') is not None else None
            
            # Append data to lists
            race_numbers.append(race_number)
            horse_names.append(horse_name)
            jockey_names.append(jockey_full)
            trainer_names.append(trainer_full)
            finishing_positions.append(fin_pos)
            odds.append(dollar_odds)
            purses.append(purse)
            distances.append(distance)
            track_conditions.append(track_condition)
    
    # Create a pandas DataFrame from the extracted data
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
    
    # Convert numeric columns where appropriate
    df['race_number'] = pd.to_numeric(df['race_number'])
    df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    
    # Create an xarray Dataset
    ds = xr.Dataset()
    
    # Add coordinates
    ds.coords['TRACK'] = [track_id]
    ds.coords['RACE_DATE'] = [race_date]
    ds.coords['RACE_NUMBER'] = sorted(df['race_number'].unique())
    
    # Add variables for each race
    for race_num in ds.coords['RACE_NUMBER'].values:
        race_df = df[df['race_number'] == race_num]
        entries = np.arange(len(race_df))
        
        ds[f'race_{race_num}_horses'] = xr.DataArray(
            race_df['horse'].values,
            dims=['entry'],
            coords={'entry': entries}
        )
        ds[f'race_{race_num}_jockey'] = xr.DataArray(
            race_df['jockey'].values,
            dims=['entry'],
            coords={'entry': entries}
        )
        ds[f'race_{race_num}_trainer'] = xr.DataArray(
            race_df['trainer'].values,
            dims=['entry'],
            coords={'entry': entries}
        )
        ds[f'race_{race_num}_positions'] = xr.DataArray(
            race_df['finishing_position'].values,
            dims=['entry'],
            coords={'entry': entries}
        )
        ds[f'race_{race_num}_odds'] = xr.DataArray(
            race_df['odds'].values,
            dims=['entry'],
            coords={'entry': entries}
        )
    
    # Add track metadata as an attribute
    ds.attrs['track_name'] = track_name
    
    return ds

def query_top_horses(ds, n=3):
    """
    Query the top n horses in each race along with:
      - Horse name
      - Jockey name
      - Trainer name
      - Finishing position
      - Odds
    
    Parameters:
    -----------
    ds : xr.Dataset
        xarray Dataset with race results.
    n : int
        Number of top horses to return.
        
    Returns:
    --------
    dict
        Dictionary with race numbers as keys and a DataFrame of top horses as values.
    """
    results = {}
    
    for race_num in ds.coords['RACE_NUMBER'].values:
        positions = ds[f'race_{race_num}_positions'].values
        horses = ds[f'race_{race_num}_horses'].values
        jockeys = ds[f'race_{race_num}_jockey'].values
        trainers = ds[f'race_{race_num}_trainer'].values
        odds = ds[f'race_{race_num}_odds'].values
        
        # Create a DataFrame for this race with all required information
        race_df = pd.DataFrame({
            'horse': horses,
            'jockey': jockeys,
            'trainer': trainers,
            'finishing_position': positions,
            'odds': odds
        })
        
        # Sort by finishing position (assuming lower is better) and select top n
        top_horses = race_df.sort_values('finishing_position').head(n)
        results[int(race_num)] = top_horses
    
    return results

def main():
    # Path to the XML file (adjust path as needed)
    xml_path = os.path.join('data', 'sampleRaceResults', 'del20230708tch.xml')
    
    # Parse the XML into an xarray Dataset
    ds = parse_xml_to_xarray(xml_path)
    
    # Print the dataset information
    print("Dataset information:")
    print(ds)
    
    # Query the top 3 horses in each race along with the desired details
    top_horses = query_top_horses(ds, n=3)
    
    print("\nTop 3 horses in each race:")
    for race_num, df in top_horses.items():
        print(f"\nRace {race_num}:")
        print(df)
    
    # Save the dataset to a NetCDF file
    output_dir = 'students/Buskin_Evgeny/'
    netcdf_path = os.path.join(output_dir, 'race_results.nc')
    ds.to_netcdf(netcdf_path)
    print(f"\nDataset saved to {netcdf_path}")

if __name__ == "__main__":
    main()
