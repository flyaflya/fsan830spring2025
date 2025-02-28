# Zhiyuan Dong
![Profile Picture](../images/ZhiyuanDong.jpeg)
## About Me
I am a PhD student at the University of Delaware, and a very good table tennis player.
## Research Interests
- Graph Neural Networks
- Machine Learning
- Causal Inference
## XML to xarray Challenge
- I interacted with AI tool to complete this task. 
- First, I input the data structure and the features inside in to help the coding agent to understand the data structure it is wroking with.
- Second, I told him the variables I am expected in the output dataset and my goal is to get the results of each race.
- Third, I wrote a query and I asked AI to do as well. And to see if we got the same result.
## My Query Code
import pandas as pd
import xarray as xr
file_path = "students/Zhiyuan_Dong/race_results.nc"
ds = xr.open_dataset(file_path)
df = ds.to_dataframe().reset_index()
df["officialFinish"] = pd.to_numeric(df["officialFinish"], errors="coerce")
top_3_horses = df[df["officialFinish"] <= 3].copy()
top_3_horses = top_3_horses[["raceNumber", "horse", "jockey", "trainer", "officialFinish", "dollarOdds"]]
top_3_horses = top_3_horses.sort_values(by=["raceNumber", "officialFinish"])
print(top_3_horses)
## Sample Output
- 9th Race: Idiomatic, Classy Edition, Morning Matcha