import xml.etree.ElementTree as ET
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Define XML file path
xml_path = Path(r'C:\Users\cgzgl\fsanVersion2\data\sampleRaceResults\del20230708tch.xml')

print(f"Checking if file exists: {xml_path}")
print(f"File exists: {xml_path.exists()}")  # Debugging check

def parse_name(person_elem):
    """Extract full name from FIRST_NAME, MIDDLE_NAME, LAST_NAME elements."""
    if person_elem is None:
        return "Unknown"

    first = person_elem.find('FIRST_NAME')
    middle = person_elem.find('MIDDLE_NAME')
    last = person_elem.find('LAST_NAME')

    return ' '.join(filter(None, [
        first.text if first is not None else '',
        middle.text if middle is not None else '',
        last.text if last is not None else ''
    ]))

def parse_race_results():
    """Parse XML race results and convert them into an xarray dataset."""
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found at: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    track_elem = root.find('.//TRACK')
    track_info = {
        'trackID': track_elem.find('CODE').text if track_elem is not None and track_elem.find('CODE') is not None else 'Unknown',
        'trackName': track_elem.find('NAME').text if track_elem is not None and track_elem.find('NAME') is not None else 'Unknown'
    }

    chart_elem = root.find('.//CHART')
    race_date = chart_elem.get('RACE_DATE') if chart_elem is not None else 'Unknown'

    race_data = []
    for race in root.findall('.//RACE'):
        race_num = int(race.get('NUMBER', -1))

        purse = float(race.findtext('PURSE', np.nan))
        distance = float(race.findtext('DISTANCE', np.nan))
        track_cond = race.findtext('TRK_COND', '')

        for entry in race.findall('ENTRY'):
            horse_name = entry.findtext('NAME', 'Unknown')
            jockey = parse_name(entry.find('.//JOCKEY'))
            trainer = parse_name(entry.find('.//TRAINER'))
            finish_pos = entry.findtext('OFFICIAL_FIN')
            odds = entry.findtext('DOLLAR_ODDS')

            # Ensure data integrity before appending
            if finish_pos is None or odds is None:
                print(f"⚠️ Missing data for race {race_num}, skipping entry.")
                continue  # Skip incomplete entries

            race_data.append({
                'race_num': race_num,
                'horse': horse_name,
                'jockey': jockey,
                'trainer': trainer,
                'finish_pos': int(finish_pos) if finish_pos.isdigit() else np.nan,
                'odds': float(odds) if odds.replace('.', '', 1).isdigit() else np.nan,
                'purse': purse,
                'distance': distance,
                'track_cond': track_cond
            })

    df = pd.DataFrame(race_data)

    # Debugging: Print length of each column
    print("🔍 DataFrame lengths before xarray conversion:")
    for col in df.columns:
        print(f"{col}: {len(df[col])}")

    # Convert to xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'finish_pos': ('entry', df['finish_pos'].values),
            'odds': ('entry', df['odds'].values),
        },
        coords={
            'track': ('track', [track_info['trackID']]),
            'trackName': ('track', [track_info['trackName']]),
            'race_date': ('race_date', [race_date]),
            'race_number': ('race_number', df['race_num'].values),
            'entry': ('entry', np.arange(len(df))),
            'horse': ('entry', df['horse'].values),
            'jockey': ('entry', df['jockey'].values),
            'trainer': ('entry', df['trainer'].values),
        }
    )

    output_path = Path.cwd() / 'race_results.nc'
    ds.to_netcdf(output_path)
    print(f"✅ Dataset created successfully at {output_path}")

    return ds
    
if __name__ == "__main__":
    parse_race_results()


