import pandas as pd
import numpy as np

def analyze_lowest_odds_performance():
    # Read the training features CSV file
    df = pd.read_csv('data/features/training_features.csv')
    
    # Initialize counters
    total_races = len(df)
    wins = 0
    places = 0
    shows = 0
    exact_places = 0  # exactly 2 points
    exact_shows = 0   # exactly 1 point
    
    # Process each race
    for _, race in df.iterrows():
        # Get all odds columns and find the lowest odds
        odds_columns = [col for col in race.index if 'odds' in col]
        odds_values = race[odds_columns].values
        lowest_odds_idx = np.argmin(odds_values)
        
        # Get corresponding points for the lowest odds horse
        points_columns = [col for col in race.index if 'pts' in col]
        points_values = race[points_columns].values
        points = points_values[lowest_odds_idx]
        
        # Count wins (6 points), places (2+ points), and shows (1+ points)
        if points == 6:
            wins += 1
        if points >= 2:
            places += 1
            if points == 2:
                exact_places += 1
        if points >= 1:
            shows += 1
            if points == 1:
                exact_shows += 1
    
    # Calculate percentages
    win_pct = (wins / total_races) * 100
    place_pct = (places / total_races) * 100
    show_pct = (shows / total_races) * 100
    exact_place_pct = (exact_places / total_races) * 100
    exact_show_pct = (exact_shows / total_races) * 100
    
    # Print results
    print(f"Analysis of lowest odds horse performance:")
    print(f"Total races analyzed: {total_races}")
    print(f"\nWin statistics:")
    print(f"Win percentage (6 points): {win_pct:.1f}%")
    print(f"\nPlace statistics:")
    print(f"Place percentage (2+ points): {place_pct:.1f}%")
    print(f"  - Exactly 2 points: {exact_place_pct:.1f}%")
    print(f"  - More than 2 points: {place_pct - exact_place_pct:.1f}%")
    print(f"\nShow statistics:")
    print(f"Show percentage (1+ points): {show_pct:.1f}%")
    print(f"  - Exactly 1 point: {exact_show_pct:.1f}%")
    print(f"  - More than 1 point: {show_pct - exact_show_pct:.1f}%")
    print(f"\nTotal in-the-money percentage: {(wins + places + shows) / total_races * 100:.1f}%")

if __name__ == "__main__":
    analyze_lowest_odds_performance() 