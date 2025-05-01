from students.okediran_tunmbi.preprocessing.training.xml_to_xarray import ds
import xarray as xr
# print(ds)


# Function to get top 3 horses per race
def get_top_3(ds):
    df = ds.to_dataframe().reset_index()
    top_3 = df.groupby(["trackID", "trackName", "raceDate", "raceNumber"]).apply(
        lambda x: x.nsmallest(3, "finishingPosition")
    ).reset_index(drop=True)
    return top_3

# Get top 3 horses
top_3_horses = get_top_3(ds)

# Display the result
print(top_3_horses)

# Viewing all races separately
for race, group in top_3_horses.groupby("raceNumber"):
    print(f"Race {race} - Top 3 horses:")
    print(group[["horse", "jockey", "trainer", "finishingPosition", "odds"]])
    print("-" * 40)