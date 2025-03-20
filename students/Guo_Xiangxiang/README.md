# Race Results Data Processing - Xiangxiang Guo

## Project Description
This project processes XML race results data into an xarray dataset for FSAN830 Business Process Innovation Class.

## Files in this Directory
- `race_data_processor.py`: Python script that converts XML race data to xarray dataset
- `race_results.nc`: Output file containing the processed race data in NetCDF format

## Data Structure
The processed dataset includes the following dimensions:
- TRACK: Track information (ID and name)
- RACE_DATE: Date of the races
- RACE_NUMBER: Race number
- ENTRY: Horse entry information (horse, jockey, trainer)

Variables include:
- Finishing position
- Odds
- Race purse
- Race distance
- Track condition

## Usage
To run the data processing:
```python
python race_data_processor.py
```

## Input Data
Uses XML data from: `data/sampleRaceResults/del20230708tch.xml` 