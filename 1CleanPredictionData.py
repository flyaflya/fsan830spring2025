"""
This script processes racing data from the following locations:
- Input: data/rawDataForPrediction/CDX0515.csv (raw racing data)
- Output: data/rawDataForPrediction/CDX0515_processed.csv (processed data with all columns)
- Output: data/rawDataForPrediction/CDX0515_filtered.csv (filtered data with only mapped columns)
- Mapping: data/rawDataForPrediction/column_mapping.csv (column header mappings)
"""

import pandas as pd
import json
import csv
from pathlib import Path

def load_column_mapping():
    """Load column mapping from CSV file."""
    try:
        mapping_path = Path("data/rawDataForPrediction/column_mapping.csv")
        mapping_df = pd.read_csv(mapping_path)
        # Convert column_number to integer and create dictionary
        mapping_dict = dict(zip(mapping_df['column_number'].astype(int), mapping_df['header_name']))
        return mapping_dict
    except Exception as e:
        print(f"Error loading column mapping: {e}")
        return {}


def process_racing_data():
    """Process the racing data CSV file."""
    # Define input and output paths
    data_dir = Path("data/rawDataForPrediction")
    input_file = data_dir / "CDX0515.csv"
    output_file = data_dir / "CDX0515_processed.csv"
    filtered_output_file = data_dir / "CDX0515_filtered.csv"

    # Load column mapping
    column_mapping = load_column_mapping()
    
    # Read CSV file
    df = pd.read_csv(input_file, header=None, quoting=csv.QUOTE_MINIMAL)
    
    # Create default column names (col_1, col_2, etc.)
    default_columns = [f'col_{i+1}' for i in range(len(df.columns))]
    df.columns = default_columns
    
    # Apply headers from mapping where available
    for col_num, header_name in column_mapping.items():
        if col_num <= len(df.columns):
            df.rename(columns={f'col_{col_num}': header_name}, inplace=True)
    
    # Save processed CSV with all columns
    df.to_csv(output_file, index=False)
    
    # Create a filtered dataframe with only the columns that have headers from the mapping
    mapped_columns = []
    for col_num, header_name in column_mapping.items():
        if col_num <= len(df.columns):
            mapped_columns.append(header_name)
    
    filtered_df = df[mapped_columns]
    
    # Save filtered CSV with only mapped columns
    filtered_df.to_csv(filtered_output_file, index=False)
    
    print(f"Processed file saved to: {output_file}")
    print(f"Filtered file (mapped columns only) saved to: {filtered_output_file}")
    print(f"Applied {len(column_mapping)} column headers from mapping file")
    print(f"Filtered file contains {len(mapped_columns)} columns with defined headers")

if __name__ == "__main__":
    process_racing_data()
