import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
import numpy as np

def parse_name(person_elem):
    """Helper function to combine name parts"""
    first = person_elem.find('FIRST_NAME').text or ''
    middle = person_elem.find('MIDDLE_NAME').text or ''
    last = person_elem.find('LAST_NAME').text or ''
    return ' '.join(filter(None, [first, middle, last])).strip()

def convert_xml_to_xarray(xml_path):
    print("Converting XML to xarray dataset...")
    try:
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract track information
        track = root.find('TRACK')
        track_id = track.find('CODE').text
        track_name = track.find('n').text
        race_date = root.get('RACE_DATE')
        
        # Initialize lists to store data
        data = []
        
        # Process each race
        for race in root.findall('RACE'):
            race_num = int(race.get('NUMBER'))
            purse = float(race.find('PURSE').text)
            distance = float(race.find('DISTANCE').text)
            trk_cond = race.find('TRK_COND').text
            
            # Process each entry in the race
            for entry in race.findall('ENTRY'):
                horse_name = entry.find('n').text
                jockey = parse_name(entry.find('JOCKEY'))
                trainer = parse_name(entry.find('TRAINER'))
                
                # Get finishing position and odds
                fin_pos = int(entry.find('OFFICIAL_FIN').text)
                odds = float(entry.find('DOLLAR_ODDS').text)
                
                data.append({
                    'trackID': track_id,
                    'trackName': track_name,
                    'RACE_DATE': race_date,
                    'RACE_NUMBER': race_num,
                    'horse': horse_name,
                    'jockey': jockey,
                    'trainer': trainer,
                    'OFFICIAL_FIN': fin_pos,
                    'DOLLAR_ODDS': odds,
                    'PURSE': purse,
                    'DISTANCE': distance,
                    'TRK_COND': trk_cond
                })
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Convert to xarray Dataset
        ds = df.set_index(['trackID', 'trackName', 'RACE_DATE', 'RACE_NUMBER', 'horse', 'jockey', 'trainer']).to_xarray()
        
        return ds
        
    except Exception as e:
        print(f"Error processing XML file: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        xml_path = '/Users/jerry/Desktop/fsan830spring2025/data/sampleRaceResults/del20230708tch.xml'
        dataset = convert_xml_to_xarray(xml_path)
        print("\nDataset created successfully!")
        print("\nDataset dimensions:")
        print(dataset.dims)
        print("\nDataset variables:")
        print(list(dataset.data_vars))
        
    except Exception as e:
        print(f"Error: {str(e)}") 