import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from models.bart_model import BARTModel
from data.training_processor import TrainingDataProcessor
from config.model_config import ModelConfig

def main():
    """Main training script."""
    print("Starting training process...")
    
    # Initialize configuration
    config = ModelConfig()
    
    # Load processed data
    data_path = config.get_full_path('processed_data_file')
    print(f"Loading data from {data_path}")
    try:
        data = xr.open_dataset(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize data processor
    processor = TrainingDataProcessor(config.get_data_params())
    
    # Process data
    print("Processing data...")
    try:
        X, y = processor.process(data)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # Initialize and train model
    print("Training model...")
    model = BARTModel(config.get_model_params())
    try:
        model.train(X, y)
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    # Evaluate model
    print("Evaluating model...")
    try:
        metrics = model.evaluate(X, y)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    except Exception as e:
        print(f"Error evaluating model: {e}")
    
    # Save model
    model_path = config.get_full_path('model_dir') / config.paths['model_file']
    print(f"\nSaving model to {model_path}")
    try:
        model.save(str(model_path))
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 