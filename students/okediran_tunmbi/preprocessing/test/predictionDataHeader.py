import pandas as pd
import os

# Define paths
base_path = r"C:\inClassTemp\fsan830spring2025\students\okediran_tunmbi\preprocessing\test"
mapping_path = os.path.join(base_path, "column_mapping.csv")
cdx0515_path = os.path.join(base_path, "CDX0515.csv")
cdx0426_path = os.path.join(base_path, "CDX0426.csv")

# Load files
mapping_df = pd.read_csv(mapping_path)
cdx0515_df = pd.read_csv(cdx0515_path, header=None)
cdx0426_df = pd.read_csv(cdx0426_path, header=None)

# Create mapping from 1-based to 0-based column indices
mapping_df["column_index"] = mapping_df["column_number"] - 1
column_mapping = dict(zip(mapping_df["column_index"], mapping_df["header_name"]))

# Filter only the columns present in the mapping
mapped_indices = sorted(column_mapping.keys())
cdx0515_df = cdx0515_df.iloc[:, mapped_indices]
cdx0426_df = cdx0426_df.iloc[:, mapped_indices]

# Rename columns using the mapping
cdx0515_df.columns = [column_mapping[i] for i in mapped_indices]
cdx0426_df.columns = [column_mapping[i] for i in mapped_indices]

# Save processed files
cdx0515_df.to_csv(os.path.join(base_path, "CDX0515_processed.csv"), index=False)
cdx0426_df.to_csv(os.path.join(base_path, "CDX0426_processed.csv"), index=False)


cdx0515_df['odds'].unique()
