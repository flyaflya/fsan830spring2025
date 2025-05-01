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

def print_xml_structure(element, indent=""):
    """Print the structure of the XML tree."""
    print(f"{indent}{element.tag}")
    for child in element:
        print_xml_structure(child, indent + "  ")

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

def extract_most_recent_finish(row, finish_cols):
    for col in sorted(finish_cols):  # assumes order reflects recency
        val = row.get(col)
        if pd.notnull(val):
            try:
                return int(val)
            except:
                return None
    return None

def process_xml_data(xml_file):
    """
    Process XML file and create engineered features DataFrame.
    
    Args:
        xml_file (str): Path to the XML file
        
    Returns:
        pd.DataFrame: Processed DataFrame with engineered features
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
        
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print XML structure to help debug
        print("\nXML Structure:")
        print_xml_structure(root)

        # Flatten all <Starters> entries
        data = []
        for race in root.findall('.//Race'):
            race_number = race.findtext('RaceNumber')  # Get race number
            for starter in race.findall('.//Starters'):
                flat = flatten_xml(starter)
                flat['RaceNumber'] = race_number  # Add race number to the data
                data.append(flat)

        if not data:
            raise ValueError("No starter data found in XML")

        # Create DataFrame
        df = pd.DataFrame(data)

        # Feature Engineering
        df['PostPosition'] = pd.to_numeric(df['PostPosition'], errors='coerce')
        df['OddsNumeric'] = df['Odds'].apply(parse_odds)
        df['JockeyName'] = df['Jockey/FirstName'].fillna('') + ' ' + df['Jockey/LastName'].fillna('')
        df['JockeyName'] = df['JockeyName'].str.strip()
        df['TrainerName'] = df['Trainer/FirstName'].fillna('') + ' ' + df['Trainer/LastName'].fillna('')
        df['TrainerName'] = df['TrainerName'].str.strip()
        
        # Calculate age as of May 1, 2025
        df['Age2025'] = df['Horse/FoalingDate'].apply(calculate_age_2025)

        # Past Performance: Most Recent Finish Position
        finish_cols = [col for col in df.columns if 'PastPerformance/Start/FinishPosition' in col]
        df['MostRecentFinishPosition'] = df.apply(lambda row: extract_most_recent_finish(row, finish_cols), axis=1)

        # Select final columns
        final_cols = [
            'Horse/HorseName', 'RaceNumber', 'Age2025', 'PostPosition', 'OddsNumeric',
            'JockeyName', 'TrainerName', 'MostRecentFinishPosition'
        ]
        
        # Verify all required columns exist
        missing_cols = [col for col in final_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            
        return df[final_cols]
        
    except ET.ParseError as e:
        raise Exception(f"Error parsing XML file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Path to your XML file
    xml_path = "data/rawDataForTraining/pastPerformanceData/SIMD20230502CD_USA.xml"
    output_path = "engineered_race_features_1.csv"

    try:
        # Process XML and create features
        df_final = process_xml_data(xml_path)
        
        # Display basic information
        print("\nProcessed Data Summary:")
        print(f"Number of horses: {len(df_final)}")
        print(f"Number of features: {len(df_final.columns)}")
        print("\nFirst 5 rows:")
        print(df_final.head())
        
        # Save to CSV
        df_final.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
