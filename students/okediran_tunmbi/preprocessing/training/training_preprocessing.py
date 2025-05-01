import xml.etree.ElementTree as ET
import pandas as pd

# --- 1. Parse the past performance dataset ---
pp_tree = ET.parse('data/rawDataForTraining/pastPerformanceData/SIMD20230502CD_USA.xml')
pp_root = pp_tree.getroot()

past_performances = []

for race in pp_root.findall('Race'):
    race_number = race.findtext('RaceNumber')
    for starter in race.findall('./Starters/Horse'):
        horse_name = starter.findtext('HorseName')
        jockey = starter.findtext('./Jockey/LastName')
        trainer = starter.findtext('./Trainer/LastName')
        odds = starter.findtext('Odds')
        post_position = starter.findtext('../PostPosition')  # relative to <Horse>
        # You can add more fields as needed

        past_performances.append({
            'HorseName': horse_name,
            'RaceNumber': race_number,
            'PostPosition': post_position,
            'Jockey': jockey,
            'Trainer': trainer,
            'Odds': odds
        })

df_pp = pd.DataFrame(past_performances)

# --- 2. Parse the result dataset ---
result_tree = ET.parse('data/rawDataForTraining/resultsData/cd20230502tch.xml')
result_root = result_tree.getroot()

results = []

for race in result_root.findall('RACE'):
    race_number = race.get('NUMBER')
    for entry in race.findall('ENTRY'):
        horse_name = entry.findtext('NAME')
        finish = entry.findtext('OFFICIAL_FIN')

        results.append({
            'HorseName': horse_name,
            'RaceNumber': race_number,
            'OfficialFinish': finish
        })

df_result = pd.DataFrame(results)

print(df_result)
print(df_pp)

# --- 3. Merge datasets on horse name and race number ---
df_merged = pd.merge(df_pp, df_result, on=['HorseName', 'RaceNumber'], how='inner')

print(df_merged)