---
output:
  word_document: default
  html_document: default
  pdf_document: default
---
# FSAN830 Business Process Innovation - Spring 2025

This repository documents the 'Innovation Track' of UD's Spring 2025 FSAN830 Business Process Innovation class.

## Environment Setup for XML to xarray Challenge

To set up your Python environment for the XML to xarray challenge:

1. **Clone this repository** (or your fork of it)
2. **Navigate to the students directory** and follow the instructions in the README.md file there
3. **Create your personal directory** under students/lastname_firstname/
4. **Run the test script** to verify your environment: `python test_env.py`

For detailed instructions, see [students/README.md](students/README.md).

## Repository Structure

```
fsan830spring2025/
├── data/                  # Shared data files
│   └── sampleRaceResults/ # Sample race results XML files
├── docs/                  # Documentation and class website
├── students/              # Student work directories
│   ├── lastname_firstname/ # Your personal directory
│   ├── README.md          # Environment setup instructions
│   ├── test_env.py        # Environment test script
│   └── example_xml_to_xarray.py # Example script
└── requirements.txt       # Python package requirements
```

## Current Mission: XML to xarray Challenge

See [docs/README.md](docs/README.md) for details on the current mission.

See The Site At: [https://flyaflya.github.io/fsan830spring2025/](https://flyaflya.github.io/fsan830spring2025/)

# Horse Racing Data Processor

This project processes horse racing data from raw CSV files by adding meaningful headers and creating a data dictionary for reference.

## Features

- Processes raw racing data CSV files
- Adds descriptive headers to the data
- Creates a comprehensive data dictionary
- Saves processed data in a clean format

## Requirements

- Python 3.7+
- pandas >= 2.0.0

## Installation

1. Clone this repository
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your raw racing data CSV file in the `data/rawDataForPrediction/` directory
2. Run the processing script:
```bash
python process_racing_data.py
```
3. Find the processed output in:
   - Processed CSV: `data/processedData/CDX0426_processed.csv`
   - Data Dictionary: `data/processedData/data_dictionary.json`

## Data Dictionary

The data dictionary provides detailed descriptions of each field in the processed CSV file. Key fields include:

- track_code: Two-letter code representing the track
- race_date: Date of the race in YYYYMMDD format
- race_number: Race number at the track for the day
- horse_name: Name of the horse
- trainer_name: Name of the horse's trainer
- jockey_name: Name of the horse's jockey
- odds: Morning line odds

For a complete list of fields and their descriptions, refer to the generated data_dictionary.json file.

## Directory Structure

```
.
├── README.md
├── requirements.txt
├── process_racing_data.py
├── data/
│   ├── rawDataForPrediction/
│   │   └── CDX0426.csv
│   └── processedData/
│       ├── CDX0426_processed.csv
│       └── data_dictionary.json
```

