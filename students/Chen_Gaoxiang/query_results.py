from xml_to_xarray import ds
import xarray as xr
# print(ds)



def get_top_3(ds):
    df = ds.to_dataframe().reset_index()
    top_3 = df.groupby(["trackID", "trackName", "raceDate", "raceNumber"]).apply(
        lambda x: x.nsmallest(3, "finishingPosition")
    ).reset_index(drop=True)
    return top_3

top_3_horses = get_top_3(ds)


print(top_3_horses)

