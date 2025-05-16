import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bart_model import BARTModel
from data.prediction_processor import PredictionDataProcessor
from config.model_config import ModelConfig

def save_predictions(predictions, race_ids, output_dir):
    """Save predictions to CSV file.
    
    Args:
        predictions: Array of predictions
        race_ids: List of race identifiers
        output_dir: Directory to save predictions
    """
    # Create DataFrame
    df = pd.DataFrame({
        'race_id': race_ids,
        'predicted_position': predictions
    })
    
    # Group by race and sort by predicted position
    df = df.sort_values(['race_id', 'predicted_position'])
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'predictions_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def plot_predictions(predictions, race_ids, save_path=None):
    """Plot predictions for each race.
    
    Args:
        predictions: Array of predictions
        race_ids: List of race identifiers
        save_path: Path to save the plot
    """
    # Create DataFrame
    df = pd.DataFrame({
        'race_id': race_ids,
        'predicted_position': predictions
    })
    
    # Group by race
    races = df.groupby('race_id')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    for race_id, race_data in races:
        plt.scatter(
            [race_id] * len(race_data),
            race_data['predicted_position'],
            alpha=0.5
        )
    
    plt.title('Predicted Positions by Race')
    plt.xlabel('Race ID')
    plt.ylabel('Predicted Position')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    """Main prediction script."""
    print("Starting prediction process...")
    
    # Initialize configuration
    config = ModelConfig()
    
    # Load model
    model_path = os.path.join(config.get_full_path('model_dir'), config.paths['model_file'])
    print(f"Loading model from {model_path}")
    try:
        model = BARTModel(config.get_model_params())
        model.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load new data
    data_path = config.get_full_path('processed_data_file')
    print(f"Loading data from {data_path}")
    try:
        data = xr.open_dataset(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize data processor
    processor = PredictionDataProcessor(
        config.get_data_params(),
        model.model_data
    )
    
    # Process data
    print("Processing data...")
    try:
        X, race_ids = processor.process(data)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # Generate predictions
    print("Generating predictions...")
    try:
        predictions = model.predict(X)
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return
    
    # Save predictions
    output_dir = config.get_full_path('prediction_dir')
    save_predictions(predictions, race_ids, output_dir)
    
    # Plot predictions
    plot_path = os.path.join(output_dir, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plot_predictions(predictions, race_ids, plot_path)
    
    print("\nPrediction completed successfully!")

if __name__ == "__main__":
    main() 