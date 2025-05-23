"""
This script processes XML race results data from the following locations:
- Input: data/rawDataForTraining/resultsData/*.xml (raw XML race results files)
- Output: data/processed/processed_results_data.nc (processed results data in netCDF format)
The script parses XML files containing race results and creates a structured dataset
for analysis and model validation.
"""

import xml.etree.ElementTree as ET
import xarray as xr
import numpy as np
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Update paths to match your directory structure
BASE_DIR = Path(__file__).parent
TRAINING_DATA_DIR = BASE_DIR / "data" / "rawDataForTraining"
RESULTS_DATA_DIR = TRAINING_DATA_DIR / "resultsData"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_race_results_xml(xml_file):
    """Parse a race results XML file and extract race and horse information."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get race date and track from filename
        filename = xml_file.name
        try:
            date_str = filename[2:10]  # Extract YYYYMMDD
            current_year = date_str[:4]
            current_month = date_str[4:6]
            current_day = date_str[6:8]
            current_year_short = current_year[-2:]
            current_track = filename[10:12]  # Extract track code
        except:
            print(f"Warning: Could not parse date/track from filename {filename}")
            return None
        
        # Initialize data structures
        race_data = {
            'race_ids': [],
            'race_types': [],
            'surfaces': [],
            'distances': [],
            'purses': [],
            'class_ratings': [],
            'track_conditions': [],
            'weather': [],
            'win_times': [],
            'fraction_1': [],
            'fraction_2': [],
            'fraction_3': [],
            'pace_call1': [],
            'pace_call2': [],
            'pace_final': [],
            'par_time': [],
            'horses': [],
            'jockeys': [],
            'trainers': [],
            'program_numbers': [],
            'post_positions': [],
            'start_positions': [],
            'finish_positions': [],
            'lengths_back': [],
            'odds': [],
            'speed_ratings': [],
            'comments': [],
            'win_payoffs': [],
            'place_payoffs': [],
            'show_payoffs': []
        }
        
        # Process each race
        for race in root.findall('.//RACE'):
            try:
                # Get race number from attribute
                race_number = race.get('NUMBER')
                if race_number is None:
                    print(f"Warning: No race number found in {xml_file.name}")
                    continue
                
                # Create race ID
                race_id = f"{current_track}-{current_month}-{current_day}-{current_year_short}-R{int(race_number):02d}"
                
                # Get race details
                race_type = race.find('TYPE').text if race.find('TYPE') is not None else ''
                surface = race.find('SURFACE').text if race.find('SURFACE') is not None else ''
                distance = float(race.find('DISTANCE').text) if race.find('DISTANCE') is not None else np.nan
                purse = float(race.find('PURSE').text) if race.find('PURSE') is not None else 0.0
                class_rating = int(race.find('CLASS_RATING').text) if race.find('CLASS_RATING') is not None else 0
                track_condition = race.find('TRK_COND').text if race.find('TRK_COND') is not None else ''
                weather = race.find('WEATHER').text if race.find('WEATHER') is not None else ''
                
                # Get race times
                win_time = float(race.find('WIN_TIME').text) if race.find('WIN_TIME') is not None else np.nan
                fraction_1 = float(race.find('FRACTION_1').text) if race.find('FRACTION_1') is not None else np.nan
                fraction_2 = float(race.find('FRACTION_2').text) if race.find('FRACTION_2') is not None else np.nan
                fraction_3 = float(race.find('FRACTION_3').text) if race.find('FRACTION_3') is not None else np.nan
                pace_call1 = int(race.find('PACE_CALL1').text) if race.find('PACE_CALL1') is not None else 0
                pace_call2 = int(race.find('PACE_CALL2').text) if race.find('PACE_CALL2') is not None else 0
                pace_final = int(race.find('PACE_FINAL').text) if race.find('PACE_FINAL') is not None else 0
                par_time = float(race.find('PAR_TIME').text) if race.find('PAR_TIME') is not None else np.nan
                
                # Process each horse in the race
                for entry in race.findall('.//ENTRY'):
                    try:
                        # Get horse info
                        horse = entry.find('n').text if entry.find('n') is not None else ''
                        jockey = entry.find('.//JOCKEY/LAST_NAME').text if entry.find('.//JOCKEY/LAST_NAME') is not None else ''
                        trainer = entry.find('.//TRAINER/LAST_NAME').text if entry.find('.//TRAINER/LAST_NAME') is not None else ''
                        program_number = entry.find('PROGRAM_NUM').text if entry.find('PROGRAM_NUM') is not None else ''
                        post_position = int(entry.find('POST_POS').text) if entry.find('POST_POS') is not None else 0
                        start_position = int(entry.find('START_POSITION').text) if entry.find('START_POSITION') is not None else 0
                        finish_position = int(entry.find('OFFICIAL_FIN').text) if entry.find('OFFICIAL_FIN') is not None else 0
                        odds = float(entry.find('DOLLAR_ODDS').text) if entry.find('DOLLAR_ODDS') is not None else 0.0
                        speed_rating = int(entry.find('SPEED_RATING').text) if entry.find('SPEED_RATING') is not None else 0
                        comment = entry.find('COMMENT').text if entry.find('COMMENT') is not None else ''
                        
                        # Get payoffs
                        win_payoff = float(entry.find('WIN_PAYOFF').text) if entry.find('WIN_PAYOFF') is not None else 0.0
                        place_payoff = float(entry.find('PLACE_PAYOFF').text) if entry.find('PLACE_PAYOFF') is not None else 0.0
                        show_payoff = float(entry.find('SHOW_PAYOFF').text) if entry.find('SHOW_PAYOFF') is not None else 0.0
                        
                        # Get lengths back at finish
                        lengths_back = 0.0
                        for call in entry.findall('.//POINT_OF_CALL'):
                            if call.get('WHICH') == 'FINAL':
                                lengths_back = float(call.find('LENGTHS').text) if call.find('LENGTHS') is not None else 0.0
                                break
                        
                        # Store all data
                        race_data['race_ids'].append(race_id)
                        race_data['race_types'].append(race_type)
                        race_data['surfaces'].append(surface)
                        race_data['distances'].append(distance)
                        race_data['purses'].append(purse)
                        race_data['class_ratings'].append(class_rating)
                        race_data['track_conditions'].append(track_condition)
                        race_data['weather'].append(weather)
                        race_data['win_times'].append(win_time)
                        race_data['fraction_1'].append(fraction_1)
                        race_data['fraction_2'].append(fraction_2)
                        race_data['fraction_3'].append(fraction_3)
                        race_data['pace_call1'].append(pace_call1)
                        race_data['pace_call2'].append(pace_call2)
                        race_data['pace_final'].append(pace_final)
                        race_data['par_time'].append(par_time)
                        race_data['horses'].append(horse)
                        race_data['jockeys'].append(jockey)
                        race_data['trainers'].append(trainer)
                        race_data['program_numbers'].append(program_number)
                        race_data['post_positions'].append(post_position)
                        race_data['start_positions'].append(start_position)
                        race_data['finish_positions'].append(finish_position)
                        race_data['lengths_back'].append(lengths_back)
                        race_data['odds'].append(odds)
                        race_data['speed_ratings'].append(speed_rating)
                        race_data['comments'].append(comment)
                        race_data['win_payoffs'].append(win_payoff)
                        race_data['place_payoffs'].append(place_payoff)
                        race_data['show_payoffs'].append(show_payoff)
                        
                    except Exception as e:
                        print(f"Error processing horse in race {race_id}: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error processing race in {xml_file.name}: {str(e)}")
                continue
        
        return race_data
        
    except Exception as e:
        print(f"Error processing {xml_file.name}: {str(e)}")
        return None

def create_results_dataset(data_dir):
    """Create an xarray dataset from all XML files in the directory."""
    # Initialize data structures
    all_race_data = defaultdict(list)
    
    # Process all XML files in the directory
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    xml_files = list(data_dir_path.glob('*.xml'))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in directory: {data_dir}")
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Track statistics for each file
    file_stats = {}
    
    for xml_file in xml_files:
        print(f"\nProcessing {xml_file.name}...")
        data = parse_race_results_xml(xml_file)
        if data is not None:
            # Track statistics
            file_stats[xml_file.name] = {
                'races': len(set(data['race_ids'])),
                'horses': len(data['horses'])
            }
            
            # Extend data
            for key, value in data.items():
                all_race_data[key].extend(value)
    
    if not all_race_data['race_ids']:
        raise ValueError("No race data was extracted from the XML files")
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print("=" * 80)
    for filename, stats in file_stats.items():
        print(f"\n{filename}:")
        print(f"  Races: {stats['races']}")
        print(f"  Horses: {stats['horses']}")
    
    print("\nOverall Statistics:")
    print(f"Total races: {len(set(all_race_data['race_ids']))}")
    print(f"Total horses: {len(all_race_data['horses'])}")
    print("=" * 80)
    
    # Create xarray dataset
    unique_races = sorted(set(all_race_data['race_ids']))
    max_starters = max(len([h for h, r in zip(all_race_data['horses'], all_race_data['race_ids']) if r == race]) 
                      for race in unique_races)
    
    # Create coordinate arrays
    race_coords = np.array(unique_races)
    starter_coords = np.arange(max_starters)
    
    # Create data arrays
    def create_data_array(values, dtype):
        if dtype in ['U35', 'U250', 'U10', 'U5', 'U1']:
            values = ['' if x is None else str(x) for x in values]
            arr = np.full((len(unique_races), max_starters), '', dtype=dtype)
        else:
            arr = np.full((len(unique_races), max_starters), np.nan, dtype=dtype)
        
        for race_idx, race in enumerate(unique_races):
            race_mask = np.array(all_race_data['race_ids']) == race
            race_values = np.array(values)[race_mask]
            arr[race_idx, :len(race_values)] = race_values
        return arr
    
    # Create the dataset
    ds = xr.Dataset(
        {
            # Race information
            'race_type': (['race'], [all_race_data['race_types'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'surface': (['race'], [all_race_data['surfaces'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'distance': (['race'], [all_race_data['distances'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'purse': (['race'], [all_race_data['purses'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'class_rating': (['race'], [all_race_data['class_ratings'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'track_condition': (['race'], [all_race_data['track_conditions'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'weather': (['race'], [all_race_data['weather'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'win_time': (['race'], [all_race_data['win_times'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'fraction_1': (['race'], [all_race_data['fraction_1'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'fraction_2': (['race'], [all_race_data['fraction_2'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'fraction_3': (['race'], [all_race_data['fraction_3'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'pace_call1': (['race'], [all_race_data['pace_call1'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'pace_call2': (['race'], [all_race_data['pace_call2'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'pace_final': (['race'], [all_race_data['pace_final'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            'par_time': (['race'], [all_race_data['par_time'][np.where(np.array(all_race_data['race_ids']) == race)[0][0]] for race in unique_races]),
            
            # Horse information
            'horse': (['race', 'starter'], create_data_array(all_race_data['horses'], 'U35')),
            'jockey': (['race', 'starter'], create_data_array(all_race_data['jockeys'], 'U250')),
            'trainer': (['race', 'starter'], create_data_array(all_race_data['trainers'], 'U250')),
            'program_number': (['race', 'starter'], create_data_array(all_race_data['program_numbers'], 'U10')),
            'post_position': (['race', 'starter'], create_data_array(all_race_data['post_positions'], np.int16)),
            'start_position': (['race', 'starter'], create_data_array(all_race_data['start_positions'], np.int16)),
            'finish_position': (['race', 'starter'], create_data_array(all_race_data['finish_positions'], np.int16)),
            'lengths_back': (['race', 'starter'], create_data_array(all_race_data['lengths_back'], np.float32)),
            'odds': (['race', 'starter'], create_data_array(all_race_data['odds'], np.float32)),
            'speed_rating': (['race', 'starter'], create_data_array(all_race_data['speed_ratings'], np.int16)),
            'comment': (['race', 'starter'], create_data_array(all_race_data['comments'], 'U250')),
            'win_payoff': (['race', 'starter'], create_data_array(all_race_data['win_payoffs'], np.float32)),
            'place_payoff': (['race', 'starter'], create_data_array(all_race_data['place_payoffs'], np.float32)),
            'show_payoff': (['race', 'starter'], create_data_array(all_race_data['show_payoffs'], np.float32))
        },
        coords={
            'race': race_coords,
            'starter': starter_coords
        }
    )
    
    return ds

def save_dataset(ds, output_dir):
    """Save the xarray dataset to a netCDF file in the specified directory."""
    # Save the dataset
    output_file = output_dir / "processed_results_data.nc"
    print(f"\nSaving dataset to {output_file}...")
    ds.to_netcdf(output_file)
    print(f"Dataset saved successfully")

if __name__ == "__main__":
    # Create and save the dataset
    print("Starting results data processing...")
    ds = create_results_dataset(RESULTS_DATA_DIR)
    save_dataset(ds, OUTPUT_DIR)
    print("\nDataset summary:")
    print(ds) 