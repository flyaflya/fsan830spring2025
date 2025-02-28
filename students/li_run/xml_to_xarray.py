import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr

# 解析 XML 文件
xml_path = 'data/sampleRaceResults/del20230708tch.xml'
tree = ET.parse(xml_path)
root = tree.getroot()

# 存储数据的列表
race_data = []

# 遍历所有比赛
for race in root.findall(".//RACE"):
    race_number = race.get("NUMBER")
    
    # 先检查 CHART 是否存在
    chart = race.find(".//CHART")
    race_date = chart.get("RACE_DATE") if chart is not None else None

    track = race.find(".//TRACK")
    track_id = track.get("CODE") if track is not None else None
    track_name = track.find("n").text if track is not None and track.find("n") is not None else None


    # 遍历所有马匹参赛信息
    for entry in race.findall(".//ENTRY"):
        horse_name = entry.find("n").text if entry.find("n") is not None else None
        
        # 提取骑师信息
        jockey = entry.find(".//JOCKEY")
        jockey_name = " ".join(
            filter(None, [
                jockey.findtext("FIRST_NAME"),
                jockey.findtext("MIDDLE_NAME"),
                jockey.findtext("LAST_NAME")
            ])
        ) if jockey is not None else None
        
        # 提取训练师信息
        trainer = entry.find(".//TRAINER")
        trainer_name = " ".join(
            filter(None, [
                trainer.findtext("FIRST_NAME"),
                trainer.findtext("MIDDLE_NAME"),
                trainer.findtext("LAST_NAME")
            ])
        ) if trainer is not None else None
        
        # 提取其他变量
        finishing_position = entry.findtext(".//OFFICIAL_FIN")
        odds = entry.findtext(".//DOLLAR_ODDS")
        purse = entry.findtext(".//PURSE")
        distance = entry.findtext(".//DISTANCE")
        track_condition = entry.findtext(".//TRK_COND")

        # 添加到 race_data
        race_data.append({
            "race_number": race_number,
            "race_date": race_date,
            "track_id": track_id,
            "track_name": track_name,
            "horse_name": horse_name,
            "jockey_name": jockey_name,
            "trainer_name": trainer_name,
            "finishing_position": int(finishing_position) if finishing_position else None,
            "odds": float(odds) if odds else None,
            "purse": float(purse) if purse else None,
            "distance": float(distance) if distance else None,
            "track_condition": track_condition
        })

# 转换为 Pandas DataFrame
df = pd.DataFrame(race_data)

# 转换为 xarray 数据集
ds = xr.Dataset.from_dataframe(df)

# 保存为 NetCDF 格式
ds.to_netcdf("race_results.nc")

print("数据集已成功转换并保存为 NetCDF 格式！")

# 查询：获取每场比赛排名前三的马匹
top3_horses = df.sort_values(by="finishing_position").groupby("race_number").head(3)
print("前 3 名马匹信息：")
print(top3_horses[["race_number", "horse_name", "jockey_name", "trainer_name", "finishing_position", "odds"]])

