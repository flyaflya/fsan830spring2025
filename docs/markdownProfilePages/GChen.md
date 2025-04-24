# Gaoxiang Chen

![Profile Picture](../images/GChen11111.jpg)

## About Me
[I am Gaoxiang (Gao) Chen, who loves sports and food.]

## Research Interests And/Or Favorite Three Topics Covered In Other Classes
- Retail Return
- Recommendation System
- Business Analytics

## XML to xarray Challenge
- Explanation
The dataset is imported from the xml_to_xarray module, which likely parses XML data into an xarray Dataset. The dataset is then converted to a Pandas DataFrame using .to_dataframe(), which allows for easy manipulation and querying.
- top 3
 from xml_to_xarray import ds
import xarray as xr

def get_top_3(ds):
    df = ds.to_dataframe().reset_index()
    top_3 = df.groupby(["trackID", "trackName", "raceDate", "raceNumber"]).apply(
        lambda x: x.nsmallest(3, "finishingPosition")
    ).reset_index(drop=True)
    return top_3

top_3_horses = get_top_3(ds)
print(top_3_horses)

- Sample race 10
27     DEL  DELAWARE PARK  2023-07-08          10           Dialherup  ...               1.0   0.9   38000.0     550.0              FT
28     DEL  DELAWARE PARK  2023-07-08          10          Mo Traffic  ...               2.0   1.8   38000.0     550.0              FT
29     DEL  DELAWARE PARK  2023-07-08          10            Kilmaley  ...               3.0   5.5   38000.0     550.0              FT


