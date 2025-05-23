import pandas as pd
import numpy as np

# Load predictions
print("Loading predictions...")
pred_df = pd.read_csv('data/rawDataForPrediction/CDX0515_predictions.csv')

# Filter for races 4, 5, 6, and 7
target_races = [4, 5, 6, 7]
race_predictions = pred_df[pred_df['race_number'].isin(target_races)]

# For each race, show the top 3 predicted finishers
print("\nTop 3 predicted finishers for each race:")
for race in target_races:
    race_data = race_predictions[race_predictions['race_number'] == race].sort_values('predicted_finish_position_mean')
    print(f"\nRace {race}:")
    print("=" * 50)
    for idx, row in race_data.head(3).iterrows():
        print(f"Horse: {row['horse_name']}")
        print(f"  Predicted Finish: {row['predicted_finish_position_mean']:.2f}")
        print(f"  Confidence Interval: [{row['predicted_finish_position_q05']:.2f}, {row['predicted_finish_position_q95']:.2f}]")
        print(f"  Post Position: {row['post_position']}")
        print(f"  Speed Rating: {row['speed_rating']}")
        print(f"  Odds: {row['odds']}")
        print("-" * 30)

# Calculate win probabilities based on predicted finish positions
print("\nWin probabilities for each race:")
for race in target_races:
    race_data = race_predictions[race_predictions['race_number'] == race]
    # Convert predicted finish positions to win probabilities (lower finish position = higher win probability)
    total_inverse = sum(1/race_data['predicted_finish_position_mean'])
    race_data['win_probability'] = (1/race_data['predicted_finish_position_mean']) / total_inverse
    
    print(f"\nRace {race} Win Probabilities:")
    print("=" * 50)
    for idx, row in race_data.sort_values('win_probability', ascending=False).head(3).iterrows():
        print(f"Horse: {row['horse_name']}")
        print(f"  Win Probability: {row['win_probability']*100:.1f}%")
        print(f"  Predicted Finish: {row['predicted_finish_position_mean']:.2f}")
        print(f"  Speed Rating: {row['speed_rating']}")
        print(f"  Odds: {row['odds']}")
        print("-" * 30) 