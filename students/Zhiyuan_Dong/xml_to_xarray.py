import xml.etree.ElementTree as ET
import xarray as xr
import pandas as pd

# Load the XML file
xml_file = 'data/sampleRaceResults/del20230708tch.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

# Extract track information
track_info = root.find("TRACK")
track_id = track_info.find("CODE").text
track_name = track_info.find("NAME").text
race_date = root.attrib["RACE_DATE"]

# Initialize data lists
data = []

# Iterate through races
for race in root.findall("RACE"):
    race_number = race.attrib["NUMBER"]
    purse = race.find("PURSE").text
    distance = race.find("DISTANCE").text
    track_condition = race.find("TRK_COND").text
    
    # Iterate through entries (horses)
    for entry in race.findall("ENTRY"):
        horse_name = entry.find("NAME").text
        official_finish = entry.find("OFFICIAL_FIN").text if entry.find("OFFICIAL_FIN") is not None else None
        dollar_odds = entry.find("DOLLAR_ODDS").text if entry.find("DOLLAR_ODDS") is not None else None
        
        # Extract jockey and trainer details
        jockey = entry.find("JOCKEY")
        trainer = entry.find("TRAINER")
        jockey_name = " ".join(filter(None, [
            jockey.find("FIRST_NAME").text if jockey is not None else None,
            jockey.find("MIDDLE_NAME").text if jockey is not None else None,
            jockey.find("LAST_NAME").text if jockey is not None else None
        ]))
        trainer_name = " ".join(filter(None, [
            trainer.find("FIRST_NAME").text if trainer is not None else None,
            trainer.find("MIDDLE_NAME").text if trainer is not None else None,
            trainer.find("LAST_NAME").text if trainer is not None else None
        ]))
        
        data.append([
            track_id, track_name, race_date, race_number, horse_name, 
            jockey_name, trainer_name, official_finish, dollar_odds, 
            purse, distance, track_condition
        ])

# Convert data to DataFrame
df = pd.DataFrame(data, columns=[
    "trackID", "trackName", "raceDate", "raceNumber", "horse", 
    "jockey", "trainer", "officialFinish", "dollarOdds", 
    "purse", "distance", "trackCondition"
])

# Convert DataFrame to xarray dataset
dataset = df.set_index(["trackID", "trackName", "raceDate", "raceNumber", "horse"]).to_xarray()

# Save to netCDF file
dataset.to_netcdf("students/Zhiyuan_Dong/race_results.nc")

# Display dataset
print(dataset)

