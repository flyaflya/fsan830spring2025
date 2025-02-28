import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
# Path is relative to the repository root
xml_path = 'data/sampleRaceResults/del20230708tch.xml'
tree = ET.parse(xml_path)
root = tree.getroot()

print(root)

# Extract data from the XML structure
race_data = []

# Extract race date and track info
race_date = root.get("RACE_DATE")
track = root.find("TRACK")
track_id = track.find("CODE").text if track.find("CODE") is not None else "Unknown"
track_name = track.find("NAME").text if track.find("NAME") is not None else "Unknown"

for race in root.findall("RACE"):
    race_number = int(race.get("NUMBER", -1))
    purse = float(race.find("PURSE").text) if race.find("PURSE") is not None else 0.0
    distance = int(race.find("DISTANCE").text) if race.find("DISTANCE") is not None else 0
    track_condition = race.find("TRK_COND").text if race.find("TRK_COND") is not None else "Unknown"

    for entry in race.findall("ENTRY"):
        horse = entry.find("NAME").text if entry.find("NAME") is not None else "Unknown"

        # Jockey details
        jockey_elem = entry.find("JOCKEY")
        if jockey_elem is not None:
            jockey = " ".join(filter(None, [
                jockey_elem.find("FIRST_NAME").text if jockey_elem.find("FIRST_NAME") is not None else "",
                jockey_elem.find("MIDDLE_NAME").text if jockey_elem.find("MIDDLE_NAME") is not None else "",
                jockey_elem.find("LAST_NAME").text if jockey_elem.find("LAST_NAME") is not None else ""
            ])).strip()
        else:
            jockey = "Unknown"

        # Trainer details
        trainer_elem = entry.find("TRAINER")
        if trainer_elem is not None:
            trainer = " ".join(filter(None, [
                trainer_elem.find("FIRST_NAME").text if trainer_elem.find("FIRST_NAME") is not None else "",
                trainer_elem.find("MIDDLE_NAME").text if trainer_elem.find("MIDDLE_NAME") is not None else "",
                trainer_elem.find("LAST_NAME").text if trainer_elem.find("LAST_NAME") is not None else ""
            ])).strip()
        else:
            trainer = "Unknown"

        finishing_position = int(entry.find("OFFICIAL_FIN").text) if entry.find("OFFICIAL_FIN") is not None else -1
        odds = float(entry.find("DOLLAR_ODDS").text) if entry.find("DOLLAR_ODDS") is not None else 0.0

        race_data.append([
            track_id, track_name, race_date, race_number, horse, jockey, trainer,
            finishing_position, odds, purse, distance, track_condition
        ])

# Convert extracted data to DataFrame
df = pd.DataFrame(race_data, columns=[
    "trackID", "trackName", "raceDate", "raceNumber", "horse", "jockey", "trainer",
    "finishingPosition", "odds", "purse", "distance", "trackCondition"
])

# Convert DataFrame to xarray Dataset
ds = df.set_index(["trackID", "trackName", "raceDate", "raceNumber", "horse"]).to_xarray()


if __name__ == "__main__":
    # Save the dataset to NetCDF
    ds.to_netcdf("students/shi_chen/race_results.nc")


    # Display the dataset summary
    print(ds)
