import pandas as pd
import numpy as np
import os
from datetime import datetime

def convert_odds(odds):
    """
    Convert odds to decimal format consistently with data extraction.
    Handles both fractional and decimal odds.
    """
    try:
        if pd.isna(odds) or odds == 0:
            return np.nan
        # If odds is a string with fractional format
        if isinstance(odds, str) and '/' in odds:
            numerator, denominator = odds.split('/')
            return float(numerator) / float(denominator) + 1
        # If odds is American format (>100)
        if float(odds) > 100:
            return float(odds) / 100
        # Otherwise assume decimal odds
        return float(odds)
    except (ValueError, ZeroDivisionError):
        return np.nan

def calculate_kelly_fraction(odds, prob):
    """
    Calculate Kelly fraction given odds and probability
    odds: decimal odds (e.g., 2.0 means bet 1 to win 1)
    prob: probability of winning
    """
    # Convert odds to decimal form
    decimal_odds = convert_odds(odds)
    if pd.isna(decimal_odds):
        return 0.0
    
    # Calculate b (net odds received on win)
    b = decimal_odds - 1
    
    # Calculate q (probability of losing)
    q = 1 - prob
    
    # Calculate Kelly fraction
    kelly = (b * prob - q) / b if b > 0 else 0
    
    # Only return positive Kelly fractions
    return max(0, kelly)

def analyze_kelly_bets():
    # Create output directory if it doesn't exist
    output_dir = "students/fleischhacker_adam2/model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a report file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"kelly_criterion_report_{timestamp}.txt")
    
    # Load the data
    probs_df = pd.read_csv('students/fleischhacker_adam2/model_outputs/first_place_probabilities.csv')
    odds_df = pd.read_csv('students/fleischhacker_adam2/data/features/prediction_features.csv')
    
    # Get odds columns
    odds_cols = [col for col in odds_df.columns if 'odds' in col]
    
    with open(report_file, 'w') as f:
        f.write("Kelly Criterion Analysis Report (Condensed)\n")
        f.write("========================================\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary_bets = []  # To store best Kelly bet info for summary
        # Process each race
        for race_idx, race_id in enumerate(probs_df.index):
            f.write(f"Race {race_idx + 1}:\n")
            f.write("-" * 40 + "\n")
            
            # Get probabilities and odds for this race
            race_probs = probs_df.iloc[race_idx].values[1:]  # Skip the race_id column
            race_odds = odds_df.iloc[race_idx][odds_cols].values
            
            # Pair up starter number, prob, odds
            horses = [(starter_num, prob, odds) for starter_num, (prob, odds) in enumerate(zip(race_probs, race_odds), 1) if odds < 999]
            # Sort by probability
            horses_by_prob = sorted(horses, key=lambda x: x[1], reverse=True)
            
            f.write("Top 3 by Win Probability:\n")
            for i, (starter, prob, odds) in enumerate(horses_by_prob[:3], 1):
                f.write(f"  {i}. Starter #{starter} - odds {odds:.1f}, win prob: {prob:.1%}\n")
            
            # Kelly bet
            kelly_fractions = []
            for starter_num, prob, odds in horses:
                kelly = calculate_kelly_fraction(odds, prob)
                kelly_fractions.append((kelly, prob, odds, starter_num))
            kelly_fractions.sort(reverse=True)
            best_kelly, best_prob, best_odds, best_starter = kelly_fractions[0]
            if best_kelly > 0:
                f.write("Best Kelly Bet:\n")
                f.write(f"  Starter #{best_starter} - odds {best_odds:.1f}, win prob: {best_prob:.1%}, Kelly: {best_kelly:.1%} of bankroll\n")
                summary_bets.append((race_idx + 1, best_starter, best_odds, best_prob, best_kelly))
            else:
                f.write("No positive expectation bets available for this race\n")
                summary_bets.append((race_idx + 1, None, None, None, None))
            f.write("\n")
        # Add summary section
        f.write("Summary of Best Kelly Bets:\n")
        f.write("===========================\n")
        for race_num, starter, odds, prob, kelly in summary_bets:
            if starter is not None:
                f.write(f"Race {race_num}: Starter #{starter} - odds {odds:.1f}, win prob: {prob:.1%}, Kelly: {kelly:.1%} of bankroll\n")
            else:
                f.write(f"Race {race_num}: No positive expectation bet available\n")
    print(f"Report has been written to: {report_file}")

if __name__ == "__main__":
    analyze_kelly_bets() 