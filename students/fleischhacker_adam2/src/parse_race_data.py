"""
This script processes XML racing data from the following locations:
- Input: ../data/rawDataForTraining/pastPerformanceData/*.xml (raw XML racing data files)
- Output: ../data/processed/processed_race_data.nc (processed data in netCDF format)
The script parses XML files containing past performance data and creates a structured dataset
for analysis.
"""

import xml.etree.ElementTree as ET
import xarray as xr
import numpy as np
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
from datetime import datetime

def convert_fractional_odds(odds_str):
    """Convert fractional odds (e.g., '6/1') to decimal odds."""
    try:
        if not odds_str or odds_str == '0':
            return 0.0
        if '/' in odds_str:
            numerator, denominator = odds_str.split('/')
            return float(numerator) / float(denominator) + 1
        return float(odds_str)
    except (ValueError, ZeroDivisionError):
        return 0.0

def parse_distance(distance_str):
    """Convert distance string (e.g., '6f', '1 1/16m') to furlongs as float."""
    try:
        if not distance_str:
            return np.nan
            
        # Convert to lowercase for case-insensitive matching
        distance_str = distance_str.lower()
        
        # Split the string into number and unit
        if 'm' in distance_str:
            # Handle miles (e.g., '1 1/16m', '1m')
            parts = distance_str.split('m')[0].strip().split()
            if len(parts) == 1:
                # Whole miles (e.g., '1m')
                distance = float(parts[0])
            else:
                # Fractional miles (e.g., '1 1/16m')
                whole = float(parts[0])
                fraction = parts[1]
                if '/' in fraction:
                    num, denom = map(float, fraction.split('/'))
                    distance = whole + (num / denom)
                else:
                    distance = whole
            # Convert miles to furlongs (1 mile = 8 furlongs)
            distance *= 8
        elif 'f' in distance_str:
            # Handle furlongs (e.g., '6f', '5 1/2f')
            parts = distance_str.split('f')[0].strip().split()
            if len(parts) == 1:
                # Whole furlongs (e.g., '6f')
                distance = float(parts[0])
            else:
                # Fractional furlongs (e.g., '5 1/2f')
                whole = float(parts[0])
                fraction = parts[1]
                if '/' in fraction:
                    num, denom = map(float, fraction.split('/'))
                    distance = whole + (num / denom)
                else:
                    distance = whole
        else:
            return np.nan
            
        return distance
    except (ValueError, IndexError):
        return np.nan

def parse_past_performance_xml(xml_file):
    """Parse a past performance XML file and extract current race and past performance information."""
    try:
        # Try to parse the XML file with error handling
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            print(f"XML parsing error in {xml_file.name}: {str(e)}")
            return None
            
        root = tree.getroot()
        
        # Print the root tag to verify we're reading the XML correctly
        print(f"Root tag: {root.tag}")
        
        # Get current race date and track from filename
        filename = xml_file.name
        try:
            date_str = filename[4:12]  # Extract YYYYMMDD
            current_year = date_str[:4]
            current_month = date_str[4:6]
            current_day = date_str[6:8]
            current_year_short = current_year[-2:]
            current_track = filename[12:14]  # Extract track code
        except:
            print(f"Warning: Could not parse date/track from filename {filename}")
            return None
        
        # Initialize data structures for current races
        current_races = {
            'race_ids': [],
            'surfaces': [],
            'distances': [],
            'purses': [],
            'class_ratings': [],
            'horses': [],
            'jockeys': [],
            'trainers': [],
            'program_numbers': []
        }
        
        # Initialize data structures for past performances
        past_performances = {
            'horse': [],
            'recent_race_ids': [],
            'recent_finish_positions': [],
            'recent_lengths_back_finish': [],
            'recent_lengths_back_last_call': [],
            'recent_speed_figs': [],
            'recent_surfaces': [],
            'recent_distances': [],
            'recent_dates': [],
            'recent_purses': [],
            'recent_start_positions': [],
            'recent_numStarters': [],
            'recent_jockeys': [],
            'recent_trainers': [],
            'recent_last_call_positions': []
        }
        
        # Process each race in the current card
        race_elements = root.findall('.//Race')
        print(f"Found {len(race_elements)} race elements in {xml_file.name}")
        
        for race_idx, race in enumerate(race_elements):
            try:
                # Get race number
                race_number_elem = race.find('RaceNumber')
                race_number = race_number_elem.text if race_number_elem is not None else ''
                
                # Create current race ID
                current_race_id = f"{current_track}-{current_month}-{current_day}-{current_year_short}-R{int(race_number):02d}"
                
                # Get surface
                course = race.find('Course')
                surface = 'Unknown'
                if course is not None:
                    surface_elem = course.find('Surface/Value')
                    if surface_elem is not None:
                        surface = surface_elem.text
                
                # Get distance
                distance_elem = race.find('Distance/PublishedValue')
                distance = parse_distance(distance_elem.text) if distance_elem is not None else np.nan
                
                # Get purse and class rating
                purse_elem = race.find('PurseUSA')
                purse = float(purse_elem.text) if purse_elem is not None else 0.0
                
                # Process each starter in the current race
                starter_elements = race.findall('Starters')
                for starter_idx, starter in enumerate(starter_elements):
                    try:
                        # Get horse info
                        horse_elem = starter.find('Horse')
                        horse = horse_elem.find('HorseName').text if horse_elem is not None and horse_elem.find('HorseName') is not None else ''
                        
                        # Get jockey info
                        jockey_elem = starter.find('Jockey')
                        jockey = jockey_elem.find('LastName').text if jockey_elem is not None and jockey_elem.find('LastName') is not None else ''
                        
                        # Get trainer info
                        trainer_elem = starter.find('Trainer')
                        trainer = trainer_elem.find('LastName').text if trainer_elem is not None and trainer_elem.find('LastName') is not None else ''
                        
                        # Get program number
                        program_number = starter.find('ProgramNumber').text if starter.find('ProgramNumber') is not None else ''
                        
                        # Store current race data
                        current_races['race_ids'].append(current_race_id)
                        current_races['surfaces'].append(surface)
                        current_races['distances'].append(distance)
                        current_races['purses'].append(purse)  # Store purse per race
                        current_races['horses'].append(horse)
                        current_races['jockeys'].append(jockey)
                        current_races['trainers'].append(trainer)
                        current_races['program_numbers'].append(program_number)
                        
                        # Process past performances for this horse
                        past_races = []
                        past_start_positions = []
                        past_finish_positions = []
                        past_lengths_back_finish = []
                        past_lengths_back_last_call = []
                        past_last_call_positions = []
                        past_surfaces = []
                        past_distances = []
                        past_dates = []
                        past_purses = []
                        past_numStarters = []
                        past_jockeys = []
                        past_trainers = []
                        
                        # Get past performances (up to 5 most recent)
                        past_perf_elements = starter.findall('PastPerformance')
                        
                        for perf in past_perf_elements[:5]:  # Only take the 5 most recent
                            try:
                                # Get the Start element which contains the race details
                                start_elem = perf.find('Start')
                                if start_elem is None:
                                    continue
                                
                                # Get lengths back at finish and last printed call point
                                lengths_finish = 0.0
                                lengths_last_call = 0.0
                                start_position = 0
                                finish_position = 0
                                last_call_position = 0
                                
                                point_of_calls = start_elem.findall('PointOfCall')
                                last_printed_call = None
                                
                                for call in point_of_calls:
                                    point_of_call = call.find('PointOfCall').text
                                    if point_of_call == 'S':
                                        # Get starting position
                                        position = call.find('Position')
                                        if position is not None:
                                            start_position = int(position.text)
                                    elif point_of_call == 'F':
                                        # Get finish position and lengths behind at finish
                                        position = call.find('Position')
                                        if position is not None:
                                            finish_position = int(position.text)
                                            # If horse is in first position, lengths behind should be 0
                                            if finish_position == 1:
                                                lengths_finish = 0.0
                                            else:
                                                lengths_behind = call.find('LengthsBehind')
                                                if lengths_behind is not None:
                                                    lengths_finish = float(lengths_behind.text)
                                        else:
                                            lengths_behind = call.find('LengthsBehind')
                                            if lengths_behind is not None:
                                                lengths_finish = float(lengths_behind.text)
                                    elif call.find('PointOfCallPrint').text == 'Y':
                                        # Track the last printed call point
                                        last_printed_call = call
                                        position = call.find('Position')
                                        if position is not None:
                                            last_call_position = int(position.text)
                                
                                # Get lengths behind at last printed call point
                                if last_printed_call is not None:
                                    lengths_behind = last_printed_call.find('LengthsBehind')
                                    if lengths_behind is not None:
                                        lengths_last_call = float(lengths_behind.text)
                                
                                # Get number of starters
                                num_starters = int(perf.find('NumberOfStarters').text) if perf.find('NumberOfStarters') is not None else 0
                                
                                # Get surface and distance
                                perf_surface = perf.find('Course/Surface/Value').text if perf.find('Course/Surface/Value') is not None else 'Unknown'
                                perf_distance_elem = perf.find('Distance/PublishedValue')
                                if perf_distance_elem is not None:
                                    perf_distance = parse_distance(perf_distance_elem.text)
                                else:
                                    perf_distance = np.nan
                                
                                # Get purse
                                perf_purse = float(perf.find('PurseUSA').text) if perf.find('PurseUSA') is not None else 0.0
                                
                                # Get race date
                                perf_date = perf.find('RaceDate').text
                                perf_date_parts = perf_date.split('-')
                                perf_year = perf_date_parts[2][-2:]
                                perf_month = perf_date_parts[0]
                                perf_day = perf_date_parts[1]
                                
                                # Get track and race number
                                perf_track = perf.find('Track/TrackID').text if perf.find('Track/TrackID') is not None else ''
                                perf_race_num = perf.find('RaceNumber').text if perf.find('RaceNumber') is not None else ''
                                
                                # Get jockey and trainer info
                                perf_jockey = start_elem.find('Jockey/LastName').text if start_elem.find('Jockey/LastName') is not None else ''
                                perf_trainer = start_elem.find('Trainer/LastName').text if start_elem.find('Trainer/LastName') is not None else ''
                                
                                # Create past race ID
                                past_race_id = f"{perf_track}-{perf_month}-{perf_day}-{perf_year}-R{int(perf_race_num):02d}" if perf_track and perf_race_num else None
                                
                                past_races.append(past_race_id)
                                past_finish_positions.append(finish_position)  # Store actual finish position
                                past_lengths_back_finish.append(lengths_finish)  # Store lengths behind at finish
                                past_lengths_back_last_call.append(lengths_last_call)  # Store lengths behind at last printed call
                                past_last_call_positions.append(last_call_position)  # Store position at last printed call
                                past_surfaces.append(perf_surface)
                                past_distances.append(perf_distance)
                                past_dates.append(perf_date)
                                past_purses.append(perf_purse)
                                past_start_positions.append(start_position)
                                past_numStarters.append(num_starters)
                                past_jockeys.append(perf_jockey)
                                past_trainers.append(perf_trainer)
                                
                            except Exception as e:
                                print(f"Error processing past performance for horse {horse}: {str(e)}")
                                continue
                        
                        # Pad with None if less than 5 races
                        while len(past_races) < 5:
                            past_races.append(None)
                            past_finish_positions.append(None)
                            past_lengths_back_finish.append(None)
                            past_lengths_back_last_call.append(None)
                            past_last_call_positions.append(None)
                            past_surfaces.append(None)
                            past_distances.append(None)
                            past_dates.append(None)
                            past_purses.append(None)
                            past_start_positions.append(None)
                            past_numStarters.append(None)
                            past_jockeys.append(None)
                            past_trainers.append(None)
                        
                        # Store past performance data
                        past_performances['horse'].append(horse)
                        past_performances['recent_race_ids'].append(past_races)
                        past_performances['recent_finish_positions'].append(past_finish_positions)
                        past_performances['recent_lengths_back_finish'].append(past_lengths_back_finish)
                        past_performances['recent_lengths_back_last_call'].append(past_lengths_back_last_call)
                        past_performances['recent_last_call_positions'].append(past_last_call_positions)
                        past_performances['recent_surfaces'].append(past_surfaces)
                        past_performances['recent_distances'].append(past_distances)
                        past_performances['recent_dates'].append(past_dates)
                        past_performances['recent_purses'].append(past_purses)
                        past_performances['recent_start_positions'].append(past_start_positions)
                        past_performances['recent_numStarters'].append(past_numStarters)
                        past_performances['recent_jockeys'].append(past_jockeys)
                        past_performances['recent_trainers'].append(past_trainers)
                        
                    except Exception as e:
                        print(f"Error processing starter {starter_idx + 1}: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error processing race {race_idx + 1}: {str(e)}")
                continue
        
        print(f"Extracted data for {len(current_races['race_ids'])} current races from {xml_file.name}")
        print(f"Extracted past performances for {len(past_performances['horse'])} horses")
        
        return {
            'current_races': current_races,
            'past_performances': past_performances
        }
    except Exception as e:
        print(f"Error processing {xml_file.name}: {str(e)}")
        return None

def create_xarray_dataset(data_dir):
    """Create an xarray dataset from all XML files in the directory."""
    # Initialize data structures
    current_races = {
        'race_ids': [],
        'surfaces': [],
        'distances': [],
        'purses': [],
        'class_ratings': [],
        'horses': [],
        'jockeys': [],
        'trainers': [],
        'program_numbers': []
    }
    
    past_performances = {
        'horse': [],
        'recent_race_ids': [],
        'recent_finish_positions': [],
        'recent_lengths_back_finish': [],
        'recent_lengths_back_last_call': [],
        'recent_speed_figs': [],
        'recent_surfaces': [],
        'recent_distances': [],
        'recent_dates': [],
        'recent_purses': [],
        'recent_start_positions': [],
        'recent_numStarters': [],
        'recent_jockeys': [],
        'recent_trainers': [],
        'recent_last_call_positions': []
    }
    
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
        data = parse_past_performance_xml(xml_file)
        if data is not None:
            # Track statistics
            file_stats[xml_file.name] = {
                'current_races': len(data['current_races']['race_ids']),
                'horses': len(data['past_performances']['horse'])
            }
            
            # Extend current races data
            for key in current_races:
                current_races[key].extend(data['current_races'][key])
            
            # Extend past performances data
            for key in past_performances:
                past_performances[key].extend(data['past_performances'][key])
    
    if not current_races['race_ids']:
        raise ValueError("No current race data was extracted from the XML files")
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print("=" * 80)
    for filename, stats in file_stats.items():
        print(f"\n{filename}:")
        print(f"  Current races: {stats['current_races']}")
        print(f"  Horses with past performances: {stats['horses']}")
    
    print("\nOverall Statistics:")
    print(f"Total current races: {len(set(current_races['race_ids']))}")
    print(f"Total horses: {len(set(current_races['horses']))}")
    print("=" * 80)
    
    # Create xarray dataset for current races
    unique_races = sorted(set(current_races['race_ids']))
    max_starters = max(len([h for h, r in zip(current_races['horses'], current_races['race_ids']) if r == race]) 
                      for race in unique_races)
    
    # Create coordinate arrays
    race_coords = np.array(unique_races)
    starter_coords = np.arange(max_starters)
    
    # Create data arrays for current races
    def create_current_data_array(values, dtype):
        if dtype in ['U35', 'U250', 'U10', 'U5', 'U1']:
            # For string types, ensure we have empty strings instead of None
            values = ['' if x is None else str(x) for x in values]
            arr = np.full((len(unique_races), max_starters), '', dtype=dtype)
        else:
            arr = np.full((len(unique_races), max_starters), np.nan, dtype=dtype)
        
        for race_idx, race in enumerate(unique_races):
            race_mask = np.array(current_races['race_ids']) == race
            starter_indices = np.arange(len(race_mask))[race_mask]
            race_values = np.array(values)[race_mask]
            arr[race_idx, :len(race_values)] = race_values
        return arr
    
    # Create data arrays for past performances
    def create_past_data_array(values, dtype):
        # Convert None values to appropriate fill values before creating numpy array
        if dtype == np.int16:
            fill_value = -1
            converted_values = [[fill_value if x is None else x for x in row] for row in values]
        elif dtype == np.float32:
            fill_value = np.nan
            converted_values = [[fill_value if x is None else x for x in row] for row in values]
        elif dtype in ['U35', 'U250', 'U10', 'U5', 'U1']:
            fill_value = ''
            converted_values = [[fill_value if x is None else str(x) if x is not None else fill_value for x in row] for row in values]
        else:
            converted_values = values
        
        # Create numpy array from converted values
        if dtype in ['U35', 'U250', 'U10', 'U5', 'U1']:
            arr = np.full((len(unique_races), max_starters, 5), '', dtype=dtype)
        else:
            arr = np.full((len(unique_races), max_starters, 5), np.nan, dtype=dtype)
            
        for race_idx, race in enumerate(unique_races):
            race_mask = np.array(current_races['race_ids']) == race
            starter_indices = np.arange(len(race_mask))[race_mask]
            race_values = np.array(converted_values)[race_mask]
            arr[race_idx, :len(race_values)] = race_values
        return arr
    
    # Create the dataset
    ds = xr.Dataset(
        {
            # Current race data
            'horse': (['race', 'starter'], create_current_data_array(current_races['horses'], 'U35')),
            'jockey': (['race', 'starter'], create_current_data_array(current_races['jockeys'], 'U250')),
            'trainer': (['race', 'starter'], create_current_data_array(current_races['trainers'], 'U250')),
            'program_number': (['race', 'starter'], create_current_data_array(current_races['program_numbers'], 'U10')),
            'surface': (['race'], np.array([current_races['surfaces'][np.where(np.array(current_races['race_ids']) == race)[0][0]] for race in unique_races], dtype='U1')),
            'distance_f': (['race'], [current_races['distances'][np.where(np.array(current_races['race_ids']) == race)[0][0]] for race in unique_races]),
            'purse': (['race'], [current_races['purses'][np.where(np.array(current_races['race_ids']) == race)[0][0]] for race in unique_races]),
            
            # Past performance data
            'recent_race_id': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_race_ids'], 'U35')),
            'recent_finish_pos': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_finish_positions'], np.float32)),
            'recent_lengths_back_finish': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_lengths_back_finish'], np.float32)),
            'recent_lengths_back_last_call': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_lengths_back_last_call'], np.float32)),
            'recent_last_call_pos': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_last_call_positions'], np.int16)),
            'recent_surface': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_surfaces'], 'U5')),
            'recent_distance': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_distances'], np.float32)),
            'recent_date': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_dates'], 'U10')),
            'recent_purse': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_purses'], np.float32)),
            'recent_start_pos': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_start_positions'], np.int16)),
            'recent_num_starters': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_numStarters'], np.int16)),
            'recent_jockey': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_jockeys'], 'U250')),
            'recent_trainer': (['race', 'starter', 'past_race'], create_past_data_array(past_performances['recent_trainers'], 'U250'))
        },
        coords={
            'race': race_coords,
            'starter': starter_coords,
            'past_race': np.arange(5)  # 0 = most recent, 4 = 5th most recent
        }
    )
    
    return ds

def save_dataset(ds, output_dir):
    """Save the xarray dataset to a netCDF file in the specified directory."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    output_file = output_path / "processed_race_data.nc"
    ds.to_netcdf(output_file)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    data_dir = "data/rawDataForTraining/pastPerformanceData"
    output_dir = "students/fleischhacker_adam2/data/processed"
    
    # Create and save the dataset
    ds = create_xarray_dataset(data_dir)
    save_dataset(ds, output_dir)
    print(ds) 