import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import os
from datetime import datetime
from pathlib import Path

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
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Flatten all <Starters> entries
        data = []
        for race in root.findall('.//Race'):
            for starter in race.findall('.//Starters'):
                flat = flatten_xml(starter)
                data.append(flat)

        if not data:
            print(f"Warning: No starter data found in {xml_file}")
            return None

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

        # Add source file information
        df['SourceFile'] = os.path.basename(xml_file)

        # Select final columns
        final_cols = [
            'Horse/HorseName', 'Age2025', 'PostPosition', 'OddsNumeric',
            'JockeyName', 'TrainerName', 'MostRecentFinishPosition', 'SourceFile'
        ]
        
        # Verify all required columns exist
        missing_cols = [col for col in final_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {xml_file}: {missing_cols}")
            
        return df[final_cols]
        
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {xml_file}: {str(e)}")
        return None

if __name__ == "__main__":
    # Directory containing XML files
    xml_dir = "data/rawDataForTraining/pastPerformanceData"
    output_path = "engineered_race_features_all.csv"

    try:
        # Get list of all XML files
        xml_files = list(Path(xml_dir).glob("*.xml"))
        
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {xml_dir}")
            
        print(f"Found {len(xml_files)} XML files to process")
        
        # Process each file and combine results
        all_dfs = []
        for xml_file in xml_files:
            print(f"\nProcessing {xml_file.name}...")
            df = process_xml_data(str(xml_file))
            if df is not None:
                all_dfs.append(df)
                print(f"Successfully processed {len(df)} horses from {xml_file.name}")
        
        if not all_dfs:
            raise ValueError("No data was successfully processed from any file")
            
        # Combine all DataFrames
        df_final = pd.concat(all_dfs, ignore_index=True)
        
        # Display basic information
        print("\nFinal Data Summary:")
        print(f"Total number of horses: {len(df_final)}")
        print(f"Number of features: {len(df_final.columns)}")
        print("\nFirst 5 rows:")
        print(df_final.head())
        
        # Save to CSV
        df_final.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
        
        # Print summary by source file
        print("\nRecords per source file:")
        print(df_final['SourceFile'].value_counts())
        
    except Exception as e:
        print(f"Error: {str(e)}")