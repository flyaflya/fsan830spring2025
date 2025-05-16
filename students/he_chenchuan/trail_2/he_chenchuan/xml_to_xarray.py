import xml.etree.ElementTree as ET
import xarray as xr
import pandas as pd
import numpy as np

def xml_to_xarray(xml_file_path):
    # Parse XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Get track and race date information
    track = root.find('.//TRACK')
    track_id = track.find('CODE').text
    track_name = track.find('NAME').text
    race_date = root.get('RACE_DATE')
    
    # Lists to store race data
    race_data = []
    
    # Extract race information
    for race in root.findall('.//RACE'):
        race_info = {}
        
        # Get race attributes
        race_info['race_num'] = int(race.get('NUMBER', 0))
        race_info['distance'] = float(race.find('DISTANCE').text if race.find('DISTANCE') is not None else 0)
        race_info['purse'] = float(race.find('PURSE').text if race.find('PURSE') is not None else 0)
        race_info['track_condition'] = race.find('TRK_COND').text if race.find('TRK_COND') is not None else ''
        race_info['surface'] = race.find('SURFACE').text if race.find('SURFACE') is not None else ''
        
        # Get entry information for each race
        for entry in race.findall('.//ENTRY'):
            entry_data = race_info.copy()
            
            # Add entry-specific information
            entry_data['horse_name'] = entry.find('NAME').text if entry.find('NAME') is not None else ''
            entry_data['program_num'] = int(entry.find('PROGRAM_NUM').text if entry.find('PROGRAM_NUM') is not None else 0)
            
            # Get jockey information
            jockey = entry.find('JOCKEY')
            if jockey is not None:
                jockey_name_parts = [jockey.find('FIRST_NAME').text or '',
                                   jockey.find('MIDDLE_NAME').text or '',
                                   jockey.find('LAST_NAME').text or '']
                entry_data['jockey'] = ' '.join(filter(None, jockey_name_parts))
            else:
                entry_data['jockey'] = ''
            
            # Get trainer information
            trainer = entry.find('TRAINER')
            if trainer is not None:
                trainer_name_parts = [trainer.find('FIRST_NAME').text or '',
                                    trainer.find('MIDDLE_NAME').text or '',
                                    trainer.find('LAST_NAME').text or '']
                entry_data['trainer'] = ' '.join(filter(None, trainer_name_parts))
            else:
                entry_data['trainer'] = ''
            
            # Get finish position and odds
            entry_data['finish_pos'] = int(entry.find('OFFICIAL_FIN').text if entry.find('OFFICIAL_FIN') is not None else np.nan)
            entry_data['odds'] = float(entry.find('DOLLAR_ODDS').text if entry.find('DOLLAR_ODDS') is not None else np.nan)
            
            race_data.append(entry_data)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(race_data)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'finish_pos': (['race', 'horse'], df.pivot(index='race_num', columns='program_num', values='finish_pos').values),
            'odds': (['race', 'horse'], df.pivot(index='race_num', columns='program_num', values='odds').values),
            'distance': (['race'], df.groupby('race_num')['distance'].first().values),
            'purse': (['race'], df.groupby('race_num')['purse'].first().values),
            'track_condition': (['race'], df.groupby('race_num')['track_condition'].first().values),
            'surface': (['race'], df.groupby('race_num')['surface'].first().values),
        },
        coords={
            'race': df['race_num'].unique(),
            'horse': sorted(df['program_num'].unique()),
            'horse_names': (['race', 'horse'], df.pivot(index='race_num', columns='program_num', values='horse_name').values),
            'jockeys': (['race', 'horse'], df.pivot(index='race_num', columns='program_num', values='jockey').values),
            'trainers': (['race', 'horse'], df.pivot(index='race_num', columns='program_num', values='trainer').values),
            'track_id': track_id,
            'track_name': track_name,
            'race_date': race_date
        }
    )
    
    return ds

if __name__ == '__main__':
    # Path to the XML file
    xml_file = 'data/sampleRaceResults/del20230708tch.xml'
    
    # Convert XML to xarray Dataset
    dataset = xml_to_xarray(xml_file)
    
    # Print dataset information
    print("Dataset Information:")
    print(dataset)
    
    # Save the dataset to a NetCDF file
    output_file = 'race_data.nc'
    print(f"Saving dataset to {output_file}...")
    dataset.to_netcdf(output_file)
    print("Dataset saved successfully!")