# Student Environment Setup Guide

This guide will help you set up your Python environment for the XML to xarray challenge.

## Setting Up Your Environment

### Option 1: Using venv (Recommended)

1. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r ../requirements.txt
   ```

### Option 2: Using conda

1. **Create a conda environment**:
   ```bash
   conda create -n fsan830 python=3.10
   conda activate fsan830
   ```

2. **Install required packages**:
   ```bash
   conda install -c conda-forge numpy pandas xarray netcdf4 matplotlib lxml pytest jupyter
   ```

## Directory Structure

Create your personal directory under the `students/` folder using your last name and first name:

```
students/lastname_firstname/
```

Place your Python scripts in this directory. Use relative paths in your code to access the sample data:

```python
# Example path to access the sample data
xml_path = '../../data/sampleRaceResults/del20230708tch.xml'
```

## Testing Your Environment

To verify your environment is set up correctly, create a simple test script:

```python
# test_env.py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

print("Environment test successful!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"xarray version: {xr.__version__}")

# Test XML parsing
try:
    tree = ET.parse('../../data/sampleRaceResults/del20230708tch.xml')
    root = tree.getroot()
    print("XML parsing successful!")
except Exception as e:
    print(f"XML parsing error: {e}")
```

Run this script to confirm your environment is working:

```bash
python test_env.py
```

## Common Issues and Solutions

### Package Installation Errors

If you encounter errors installing packages:

1. Ensure you're using Python 3.8 or newer
2. Try updating pip: `pip install --upgrade pip`
3. For Windows users with netCDF4 installation issues, consider using conda instead

### Path Issues

If you encounter path-related errors:
1. Verify you're running your scripts from your personal directory
2. Check that you're using relative paths correctly
3. Ensure the XML file exists in the expected location

## Getting Help

If you encounter issues setting up your environment, please:
1. Check if others have reported similar issues in the class discussion
2. Document the exact error message and steps to reproduce
3. Reach out to the instructor with this information 