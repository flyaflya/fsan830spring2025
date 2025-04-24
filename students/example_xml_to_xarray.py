import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import xarray as xr
import os

def get_xml_text(element, tag, default="Unknown"):
    """
    Helper function to safely extract text from an XML element.

    Parameters:
    -----------
    element : xml.etree.ElementTree.Element or None
        The XML element to extract text from.
    tag : str
        The tag name to search for within the element.
    default : str
        The default value to return if the element or tag is missing.

    Returns:
    --------
    str
        The extracted text or the default value.
    """
    found_element = element.find(tag) if element is not None else None
    return found_element.text if found_element is not None else default

def parse_xml_to_xarray(xml_path):
    """
    Parse an XML race results file and convert it to an xarray dataset.

    Parameters:
    -----------
    xml_path : str
        Path to the XML file.

    Returns:
    --------
    xr.Dataset
        xarray Dataset with race results.
    """
    # Check if the file exists
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Error: The file '{xml_path}' does not exist. Check the path.")

    try:
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file '{xml_path}': {e}")

    # Extract race details
    chart = root.find('.//CHART')
    race_date = chart.get('RACE_DATE') if chart is not None else "Unknown"

    # Extract track information
    track_elem = root.find('.//TRACK')
    track_id = get_xml_text(track_elem, 'CODE', "Unknown")
    track_name = get_xml_text(track_elem, 'n', "Unknown")

    # Initialize lists to store race data
    race_numbers, horse_names, jockey_names, trainer_names = [], [], [], []
    finishing_positions, odds, purses, distances, track_conditions = [], [], [], [], []

    # Extract race information
    for race in root.findall('.//RACE'):
        race_number = race.get('NUMBER', "0")
        purse = get_xml_text(race, 'PURSE', "N/A")
        distance = get_xml_text(race, 'DISTANCE', "N/A")
        track_condition = get_xml_text(race, 'TRK_COND', "Unknown")

        # Extract entry information
        for entry in race.findall('.//ENTRY'):
            # Handling missing horse names with debugging
            horse_name = get_xml_text(entry, 'n', "Unknown Horse")
            if horse_name == "Unknown Horse":
                print(f"⚠️ Missing horse name in race {race_number}, entry: {ET.tostring(entry, encoding='unicode')}")

            # Extract Jockey information
            jockey = entry.find('.//JOCKEY')
            jockey_first = get_xml_text(jockey, 'FIRST_NAME', "")
            jockey_middle = get_xml_text(jockey, 'MIDDLE_NAME', "")
            jockey_last = get_xml_text(jockey, 'LAST_NAME', "")
            jockey_name = f"{jockey_first} {jockey_middle} {jockey_last}".strip().replace("  ", " ")

            # Extract Trainer information
            trainer = entry.find('.//TRAINER')
            trainer_first = get_xml_text(trainer, 'FIRST_NAME', "")
            trainer_middle = get_xml_text(trainer, 'MIDDLE_NAME', "")
            trainer_last = get_xml_text(trainer, 'LAST_NAME', "")
            trainer_name = f"{trainer_first} {trainer_middle} {trainer_last}".strip().replace("  ", " ")

            # Finishing position and odds
            fin_pos = get_xml_text(entry, 'OFFICIAL_FIN', "N/A")
            dollar_odds = get_xml_text(entry, 'DOLLAR_ODDS', "N/A")

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
    df = pd.DataFrame({
        'race_number': race_numbers,
        'horse': horse_names,
        'jockey': jockey_names,
        'trainer': trainer_names,
        'finishing_position': finishing_positions,
        'odds': odds,
        'purse': purses,
        'distance': distances,
        'track_condition': track_conditions
    })

    # Convert numeric columns
    df['race_number'] = pd.to_numeric(df['race_number'], errors='coerce')
    df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce')
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')

    # Create xarray dataset
    ds = xr.Dataset()

    # Add coordinates
    ds.coords['TRACK'] = [track_id]
    ds.coords['RACE_DATE'] = [race_date]
    ds.coords['RACE_NUMBER'] = sorted(df['race_number'].dropna().unique())

    # Add variables for each race
    for race_num in ds.coords['RACE_NUMBER'].values:
        race_df = df[df['race_number'] == race_num]

        ds[f'race_{race_num}_horses'] = xr.DataArray(
            race_df['horse'].values, dims=['entry'], coords={'entry': np.arange(len(race_df))}
        )
        ds[f'race_{race_num}_positions'] = xr.DataArray(
            race_df['finishing_position'].values, dims=['entry'], coords={'entry': np.arange(len(race_df))}
        )
        ds[f'race_{race_num}_odds'] = xr.DataArray(
            race_df['odds'].values, dims=['entry'], coords={'entry': np.arange(len(race_df))}
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
        xarray Dataset with race results.
    n : int
        Number of top horses to return.

    Returns:
    --------
    dict
        Dictionary with race numbers as keys and top horses as values.
    """
    results = {}

    for race_num in ds.coords['RACE_NUMBER'].values:
        positions = ds[f'race_{race_num}_positions'].values
        horses = ds[f'race_{race_num}_horses'].values
        odds = ds[f'race_{race_num}_odds'].values

        race_df = pd.DataFrame({'horse': horses, 'position': positions, 'odds': odds})
        top_horses = race_df.sort_values('position').head(n)

        results[int(race_num)] = top_horses

    return results

def main():
    xml_path = 'data/sampleRaceResults/del20230708tch.xml'

    try:
        ds = parse_xml_to_xarray(xml_path)
        
        print("Dataset information:")
        print(ds)

        top_horses = query_top_horses(ds, n=3)

        print("\nTop 3 horses in each race:")
        for race_num, horses in top_horses.items():
            print(f"\nRace {race_num}:")
            print(horses)

        output_dir = '.'
        ds.to_netcdf(os.path.join(output_dir, 'race_results.nc'))
        print(f"\nDataset saved to {os.path.join(output_dir, 'race_results.nc')}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
