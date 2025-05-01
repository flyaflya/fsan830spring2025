# Shi Chen
![Profile Picture](../images/ShiChen.png)

## About Me
Hello World!

## Research Interests And/Or Favorite Three Topics Covered In Other Classes
- MATHEMATICS
- FINTACH
- NATURAL LANGUAGE PROCESSING



## XML to xarray Challenge
1 To convert XML data to an Xarray structure, I first used the xml.etree.ElementTree module to parse the XML file and extract relevant information. I then structured the data into a Pandas DataFrame, ensuring the hierarchical race information was properly mapped. Finally, I converted the DataFrame into an Xarray dataset using xarray.Dataset, which allowed for efficient querying and analysis.

2 query method in query_results.py
def get_top_3(ds):
    df = ds.to_dataframe().reset_index()
    top_3 = df.groupby(["trackID", "trackName", "raceDate", "raceNumber"]).apply(
        lambda x: x.nsmallest(3, "finishingPosition")
    ).reset_index(drop=True)
    return top_3

3 Top 3 Horses in race 9:
Idiomatic  – Jockey: Florent Geroux, Trainer: Brad H. Cox, Odds: 1.0 (Winner)
Classy Edition – Jockey: Kendrick Carmouche, Trainer: Todd A. Pletcher, Odds: 9.4 (Runner-up)
Morning Matcha – Jockey: Paco Lopez, Trainer: Robert E. Reid, Odds: 5.9 (Third Place)