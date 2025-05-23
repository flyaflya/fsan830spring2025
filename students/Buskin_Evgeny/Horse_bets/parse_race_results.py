import xml.etree.ElementTree as ET
import xarray as xr
import numpy as np
from pathlib import Path
from parse_race_data import create_xarray_dataset

def parse_race_results_xml(xml_file):
    """Parse a race results XML file and extract race IDs and finish positions."""
    try:
        # Try to parse the XML file with error handling
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            print(f"XML parsing error in {xml_file.name}: {str(e)}")
            return None
            
        root = tree.getroot()
        
        # Get track and race date information
        track_elem = root.find('.//TRACK')
        track_info = {
            'track_id': track_elem.find('CODE').text if track_elem is not None and track_elem.find('CODE') is not None else 'Unknown',
            'track_name': track_elem.find('NAME').text if track_elem is not None and track_elem.find('NAME') is not None else 'Unknown'
        }
        
        race_date = root.get('RACE_DATE', 'Unknown')
        # Convert date from YYYY-MM-DD to MM-DD-YY format
        if race_date != 'Unknown':
            year, month, day = race_date.split('-')
            short_year = year[-2:]
            race_date = f"{month}-{day}-{short_year}"
        
        # Initialize data structures
        race_data = {
            'race_ids': [],
            'horses': [],
            'finish_positions': []
        }
        
        # Process each race
        for race in root.findall('.//RACE'):
            try:
                race_number = int(race.get('NUMBER', 0))
                
                # Process each entry in the race
                for entry in race.findall('ENTRY'):
                    try:
                        # Create race ID in the format CD-05-02-23-R01
                        race_id = f"{track_info['track_id']}-{race_date}-R{race_number:02d}"
                        
                        # Get horse information
                        horse_name = entry.find('NAME').text if entry.find('NAME') is not None else ''
                        finish_position = int(entry.find('OFFICIAL_FIN').text) if entry.find('OFFICIAL_FIN') is not None else np.nan
                        
                        # Store the data
                        race_data['race_ids'].append(race_id)
                        race_data['horses'].append(horse_name)
                        race_data['finish_positions'].append(finish_position)
                        
                    except Exception as e:
                        print(f"Error processing entry in race {race_number}: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error processing race {race_number}: {str(e)}")
                continue
        
        return race_data
        
    except Exception as e:
        print(f"Error processing {xml_file.name}: {str(e)}")
        return None

def create_race_results_dataset(data_dir):
    """Create an xarray dataset from all race results XML files in the directory."""
    # Initialize data structures
    race_data = {
        'race_ids': [],
        'horses': [],
        'finish_positions': []
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
        data = parse_race_results_xml(xml_file)
        if data is not None:
            # Track statistics
            file_stats[xml_file.name] = {
                'races': len(set(data['race_ids'])),
                'entries': len(data['race_ids'])
            }
            
            # Extend data arrays
            for key in race_data:
                race_data[key].extend(data[key])
    
    if not race_data['race_ids']:
        raise ValueError("No race data was extracted from the XML files")
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print("=" * 80)
    for filename, stats in file_stats.items():
        print(f"\n{filename}:")
        print(f"  Races: {stats['races']}")
        print(f"  Entries: {stats['entries']}")
    
    print("\nOverall Statistics:")
    print(f"Total races: {len(set(race_data['race_ids']))}")
    print(f"Total entries: {len(race_data['race_ids'])}")
    print("=" * 80)
    
    # Create xarray dataset
    unique_races = sorted(set(race_data['race_ids']))
    max_entries = max(len([h for h, r in zip(race_data['horses'], race_data['race_ids']) if r == race]) 
                     for race in unique_races)
    
    # Create coordinate arrays
    race_coords = np.array(unique_races)
    entry_coords = np.arange(max_entries)
    
    # Create data arrays
    def create_data_array(values, dtype):
        if dtype in ['U35', 'U250', 'U10', 'U5', 'U1']:
            # For string types, ensure we have empty strings instead of None
            values = ['' if x is None else str(x) for x in values]
            arr = np.full((len(unique_races), max_entries), '', dtype=dtype)
        else:
            arr = np.full((len(unique_races), max_entries), np.nan, dtype=dtype)
        
        for race_idx, race in enumerate(unique_races):
            race_mask = np.array(race_data['race_ids']) == race
            entry_indices = np.arange(len(race_mask))[race_mask]
            race_values = np.array(values)[race_mask]
            arr[race_idx, :len(race_values)] = race_values
        return arr
    
    # Create the dataset
    ds = xr.Dataset(
        {
            'horse': (['race', 'entry'], create_data_array(race_data['horses'], 'U35')),
            'finish_position': (['race', 'entry'], create_data_array(race_data['finish_positions'], np.int16))
        },
        coords={
            'race': race_coords,
            'entry': entry_coords
        }
    )
    
    return ds

def standardize_race_id(race_id):
    """
    Convert race ID to a standard format: CD-YYYY-MM-DD-RXX
    """
    try:
        # Convert numpy string to Python string
        race_id = str(race_id)
        
        # Handle format like CD-05-02-23-R01
        if len(race_id.split('-')) == 4:
            track, month, day, rest = race_id.split('-')
            year = '20' + rest[:2]  # Assuming 20xx for year
            race_num = rest[3:]     # Extract R01 from 23-R01
            # Pad month and day with leading zeros if needed
            month = month.zfill(2)
            day = day.zfill(2)
            return f"{track}-{year}-{month}-{day}-{race_num}"
        
        # Handle format like CD-2023-05-02-R01
        parts = race_id.split('-')
        if len(parts) == 5:
            track, year, month, day, race_num = parts
            # Pad month and day with leading zeros if needed
            month = month.zfill(2)
            day = day.zfill(2)
            return f"{track}-{year}-{month}-{day}-{race_num}"
        
        return race_id
    except Exception as e:
        print(f"Error standardizing race ID {race_id}: {str(e)}")
        return race_id

def normalize_horse_name(name):
    """Normalize horse name by removing 'dh-' prefix and standardizing case."""
    if not name:
        return name
    name = str(name).strip()
    if name.lower().startswith('dh-'):
        name = name[3:]
    return name

def is_dead_heat(name):
    """Check if a horse name has the dead heat prefix."""
    if not name:
        return False
    return str(name).lower().startswith('dh-')

def merge_race_data_with_results(race_ds, results_ds):
    """
    Merge the race data dataset with the race results dataset.
    
    Parameters:
    -----------
    race_ds : xarray.Dataset
        The dataset from parse_race_data.py containing race and past performance information
    results_ds : xarray.Dataset
        The dataset from create_race_results_dataset containing race results
        
    Returns:
    --------
    xarray.Dataset
        The merged dataset with finish positions added to the race data
    """
    print("\nMerging datasets...")
    print(f"Race dataset has {len(race_ds.race)} races")
    print(f"Results dataset has {len(results_ds.race)} races")
    
    # Standardize race IDs in both datasets
    race_ids_standardized = [standardize_race_id(race_id) for race_id in race_ds.race.values]
    results_race_ids_standardized = [standardize_race_id(race_id) for race_id in results_ds.race.values]
    
    # Create a mapping of standardized race IDs to their indices in the results dataset
    results_race_map = {race_id: idx for idx, race_id in enumerate(results_race_ids_standardized)}
    
    # Initialize arrays
    finish_positions = np.full_like(race_ds.horse, np.nan, dtype=np.int16)
    scratched = np.zeros_like(race_ds.horse, dtype=bool)  # True if horse was scratched
    
    # Track statistics
    matched_races = 0
    matched_horses = 0
    total_horses = 0
    scratched_horses = 0
    dead_heat_count = 0
    unmatched_horses = []
    
    # For each race in the race dataset
    for race_idx, race_id in enumerate(race_ids_standardized):
        if race_id in results_race_map:
            matched_races += 1
            # Get the corresponding race from results
            results_race_idx = results_race_map[race_id]
            
            # Get the horses and finish positions for this race
            results_horses = results_ds.horse[results_race_idx].values
            results_finish_positions = results_ds.finish_position[results_race_idx].values
            
            # Create a mapping of normalized horse names to their finish positions
            horse_finish_map = {}
            dead_heat_groups = {}  # Track groups of horses in dead heats
            
            # First pass: identify dead heat groups
            for horse, pos in zip(results_horses, results_finish_positions):
                if horse != '' and not np.isnan(pos):
                    normalized_horse = normalize_horse_name(horse)
                    if is_dead_heat(horse):
                        if pos not in dead_heat_groups:
                            dead_heat_groups[pos] = []
                        dead_heat_groups[pos].append(normalized_horse)
                        dead_heat_count += 1
                    horse_finish_map[normalized_horse] = pos
            
            # Second pass: ensure all horses in dead heat groups have the same position
            for pos, horses in dead_heat_groups.items():
                for horse in horses:
                    horse_finish_map[horse] = pos
            
            # For each horse in the race dataset
            for entry_idx, horse in enumerate(race_ds.horse[race_idx].values):
                # Only count non-empty entries
                if horse != '':
                    total_horses += 1
                    normalized_horse = normalize_horse_name(horse)
                    if normalized_horse in horse_finish_map:
                        matched_horses += 1
                        finish_positions[race_idx, entry_idx] = horse_finish_map[normalized_horse]
                    else:
                        # If horse is in race dataset but not in results, it was scratched
                        scratched[race_idx, entry_idx] = True
                        scratched_horses += 1
                        unmatched_horses.append((race_id, horse))
                        # Print detailed comparison for the first few scratched horses
                        if len(unmatched_horses) <= 3:
                            print(f"\nDetailed comparison for race {race_id}:")
                            print("Race dataset horses:")
                            for h in race_ds.horse[race_idx].values:
                                if h != '':
                                    print(f"  - {h}")
                            print("\nResults dataset horses:")
                            for h in results_horses:
                                if h != '':
                                    print(f"  - {h}")
    
    # Print matching statistics
    print("\nMatching Statistics:")
    print(f"Matched races: {matched_races} out of {len(race_ds.race)}")
    print(f"Total actual entries: {total_horses}")  # Changed from total_horses to total actual entries
    print(f"Matched horses: {matched_horses} out of {total_horses}")
    print(f"Scratched horses: {scratched_horses}")
    print(f"Dead heats found: {dead_heat_count}")
    
    # Print sample of scratched horses for debugging
    if unmatched_horses:
        print("\nSample of scratched horses:")
        for race_id, horse in unmatched_horses[:10]:
            print(f"Race {race_id}: {horse}")
    
    # Add finish positions and scratched flag to the race dataset
    race_ds['finish_position'] = (['race', 'starter'], finish_positions)
    race_ds['scratched'] = (['race', 'starter'], scratched)
    
    return race_ds

def save_dataset(ds, output_dir):
    """Save the xarray dataset to a netCDF file in the specified directory."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    output_file = output_path / "processed_race_data_with_results.nc"
    ds.to_netcdf(output_file)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    # Process race data
    race_data_dir = "data/rawDataForTraining/pastPerformanceData"
    race_ds = create_xarray_dataset(race_data_dir)
    
    # Process race results
    results_data_dir = "data/rawDataForTraining/resultsData"
    results_ds = create_race_results_dataset(results_data_dir)
    
    # Merge the datasets
    merged_ds = merge_race_data_with_results(race_ds, results_ds)
    
    # Save the merged dataset
    output_dir = "students/Buskin_Evgeny/Horse_bets/data/processed"
    save_dataset(merged_ds, output_dir) 