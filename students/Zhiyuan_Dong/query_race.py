import pandas as pd
import xarray as xr

# Load the NetCDF file
file_path = "students/Zhiyuan_Dong/race_results.nc"
ds = xr.open_dataset(file_path)
# Convert the dataset to a DataFrame for easier querying
df = ds.to_dataframe().reset_index()

# Convert relevant columns to numeric for proper sorting
df["officialFinish"] = pd.to_numeric(df["officialFinish"], errors="coerce")

# Filter out rows where finishing position is greater than 3 or missing
top_3_horses = df[df["officialFinish"] <= 3].copy()

# Select required columns
top_3_horses = top_3_horses[["raceNumber", "horse", "jockey", "trainer", "officialFinish", "dollarOdds"]]

# Sort by raceNumber and finishing position
top_3_horses = top_3_horses.sort_values(by=["raceNumber", "officialFinish"])

# Display the results
print(top_3_horses)