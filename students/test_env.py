"""
Environment test script for the XML to xarray challenge.
Run this script to verify your Python environment is set up correctly.
"""

import sys
import os
import platform

def print_separator():
    print("-" * 60)

def main():
    print_separator()
    print("ENVIRONMENT TEST SCRIPT")
    print_separator()
    
    # Python version
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print_separator()
    
    # Check required packages
    packages = [
        "numpy", 
        "pandas", 
        "xarray", 
        "matplotlib", 
        "xml.etree.ElementTree"
    ]
    
    print("Checking required packages:")
    all_packages_installed = True
    
    for package in packages:
        try:
            if package == "xml.etree.ElementTree":
                import xml.etree.ElementTree as ET
                print(f"✓ {package} is available (built-in)")
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                print(f"✓ {package} is installed (version: {version})")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            all_packages_installed = False
    
    print_separator()
    
    # Check if sample data is accessible
    print("Checking sample data:")
    xml_path = os.path.join('..', '..', 'data', 'sampleRaceResults', 'del20230708tch.xml')
    
    if os.path.exists(xml_path):
        print(f"✓ Sample XML file found at: {xml_path}")
        
        # Try to parse the XML file
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            print("✓ XML file parsed successfully")
            
            # Get some basic info from the XML
            chart = root.find('.//CHART')
            if chart is not None:
                race_date = chart.get('RACE_DATE')
                print(f"  Race date: {race_date}")
            
            track = root.find('.//TRACK')
            if track is not None:
                track_name = track.find('n').text
                print(f"  Track name: {track_name}")
            
            races = root.findall('.//RACE')
            print(f"  Number of races: {len(races)}")
        except Exception as e:
            print(f"✗ Error parsing XML file: {e}")
    else:
        print(f"✗ Sample XML file NOT found at: {xml_path}")
        print("  Make sure you're running this script from your personal directory")
        print("  under the students/ folder and the data file exists.")
    
    print_separator()
    
    # Summary
    if all_packages_installed and os.path.exists(xml_path):
        print("✓ Environment test PASSED! You're ready to start the challenge.")
    else:
        print("✗ Environment test FAILED. Please fix the issues above.")
    
    print_separator()

if __name__ == "__main__":
    main() 