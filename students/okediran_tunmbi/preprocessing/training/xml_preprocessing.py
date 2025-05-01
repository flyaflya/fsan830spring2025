import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import os
from datetime import datetime

def calculate_age_2025(foaling_date):
    try:
        birth = datetime.strptime(foaling_date.split('+')[0], "%Y-%m-%d")
        reference_date = datetime(2025, 5, 1)  # May 1, 2025
        return (reference_date - birth).days / 365.25  # Using 365.25 to account for leap years
    except:
        return np.nan

def flatten_xml(element, parent_path=''):
    data = {}
    for child in element:
        path = f"{parent_path}/{child.tag}" if parent_path else child.tag
        if len(child):  # has children, recurse
            data.update(flatten_xml(child, path))
        else:
            data[path] = child.text
    return data

def parse_odds(odds_str):
    try:
        num, denom = map(int, odds_str.split('/'))
        return num / denom
    except:
        return np.nan

def process_past_performance(xml_file):
    """
    Process past performance XML file and create features DataFrame.
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
        
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        data = []
        for race in root.findall('.//Race'):
            race_number = race.findtext('RaceNumber')
            for starter in race.findall('.//Starters'):
                flat = flatten_xml(starter)
                flat['RaceNumber'] = race_number
                data.append(flat)

        if not data:
            raise ValueError("No starter data found in XML")

        df = pd.DataFrame(data)

        # Feature Engineering
        df['PostPosition'] = pd.to_numeric(df['PostPosition'], errors='coerce')
        df['OddsNumeric'] = df['Odds'].apply(parse_odds)
        df['JockeyName'] = df['Jockey/FirstName'].fillna('') + ' ' + df['Jockey/LastName'].fillna('')
        df['JockeyName'] = df['JockeyName'].str.strip()
        df['TrainerName'] = df['Trainer/FirstName'].fillna('') + ' ' + df['Trainer/LastName'].fillna('')
        df['TrainerName'] = df['TrainerName'].str.strip()
        
        df['Age2025'] = df['Horse/FoalingDate'].apply(calculate_age_2025)

        # Rename columns for consistency
        df = df.rename(columns={
            'Horse/HorseName': 'HorseName',
            'Horse/FoalingDate': 'FoalingDate'
        })

        final_cols = [
            'HorseName', 'RaceNumber', 'Age2025', 'PostPosition', 'OddsNumeric',
            'JockeyName', 'TrainerName', 'FoalingDate'
        ]
        
        return df[final_cols]
        
    except Exception as e:
        raise Exception(f"Error processing past performance data: {str(e)}")

def process_race_results(xml_file):
    """
    Process race results XML file and create results DataFrame.
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        data = []
        for race in root.findall('.//RACE'):
            race_number = race.get('NUMBER')
            for entry in race.findall('./ENTRY'):
                horse_name = entry.findtext('NAME')
                official_finish = entry.findtext('OFFICIAL_FIN')
                
                data.append({
                    'HorseName': horse_name,
                    'RaceNumber': race_number,
                    'OfficialFinish': pd.to_numeric(official_finish, errors='coerce')
                })

        df = pd.DataFrame(data)
        df['OfficialFinish'] = df['OfficialFinish'].astype(int)
        return df 
    
    except Exception as e:
        raise Exception(f"Error processing race results: {str(e)}")

if __name__ == "__main__":
    # File paths
    pp_xml_path = "data/rawDataForTraining/pastPerformanceData/SIMD20230502CD_USA.xml"
    results_xml_path = "data/rawDataForTraining/resultsData/cd20230502tch.xml"
    output_path = "merged_race_data.csv"

    try:
        # Process both datasets
        print("Processing past performance data...")
        df_pp = process_past_performance(pp_xml_path)
        print(f"Found {len(df_pp)} past performance entries")

        print("\nProcessing race results...")
        df_results = process_race_results(results_xml_path)
        print(f"Found {len(df_results)} race results")

        # Merge datasets
        print("\nMerging datasets...")
        df_merged = pd.merge(
            df_pp,
            df_results,
            on=['HorseName', 'RaceNumber'],
            how='inner'
        )
        print(f"Merged dataset has {len(df_merged)} entries")

        # Display sample of merged data
        print("\nSample of merged data:")
        print(df_merged.head())

        # Save merged data
        df_merged.to_csv(output_path, index=False)
        print(f"\nMerged data saved to {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")