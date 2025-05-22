import pandas as pd

# Read the CSV file
df = pd.read_csv('data/rawDataForPrediction/CDX0426_processed.csv')

# Select and rename relevant columns
columns_to_keep = {
    'track_code': 'track_code',
    'race_date': 'race_date',
    'race_number': 'race_number',
    'post_position': 'post_position',
    'distance': 'distance',
    'surface_code': 'surface_code',
    'race_type': 'race_type',
    'claiming_price': 'claiming_price',
    'speed_rating': 'speed_rating',
    'trainer_name': 'trainer_name',
    'trainer_starts': 'trainer_starts',
    'trainer_wins': 'trainer_wins',
    'trainer_places': 'trainer_places',
    'trainer_shows': 'trainer_shows',
    'jockey_name': 'jockey_name',
    'jockey_starts': 'jockey_starts',
    'jockey_wins': 'jockey_wins',
    'jockey_places': 'jockey_places',
    'jockey_shows': 'jockey_shows',
    'program_number': 'program_number',
    'odds': 'odds',
    'horse_name': 'horse_name',
    'age': 'age',
    'sex': 'sex',
    'weight': 'weight',
    'sire': 'sire',
    'dam': 'dam',
    'recentRaceDate1': 'recent_race_date_1',
    'recentRaceTrackCode1': 'recent_track_code_1',
    'recentRaceNumber1': 'recent_race_number_1',
    'recentSurfaceCode1': 'recent_surface_code_1',
    'recentNumEntrants1': 'recent_num_entrants_1',
    'recentPostPosition1': 'recent_post_position_1',
    'recentStartCallPosition1': 'recent_start_call_position_1',
    'recent1stCallPosition1': 'recent_1st_call_position_1',
    'recentStretchPosition1': 'recent_stretch_position_1',
    'recentFinishPosition1': 'recent_finish_position_1',
    'recentBrisSpeed1': 'recent_bris_speed_1',
    'recentRaceType1': 'recent_race_type_1'
}

# Create new dataframe with selected columns
df_transformed = df[columns_to_keep.keys()].rename(columns=columns_to_keep)

# Save transformed data
df_transformed.to_csv('data/transformed_data.csv', index=False)

print("Data transformation complete. Output saved to 'data/transformed_data.csv'") 