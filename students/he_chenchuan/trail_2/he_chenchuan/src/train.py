import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import pickle
import xarray as xr

from data_processor import HorseRaceDataProcessor
from bart_model import HorseRaceBART

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HorseRaceTrainer:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        n_trees: int = 200,
        n_chains: int = 4,
        n_samples: int = 1000,
        n_tune: int = 1000,
        random_seed: int = 42
    ):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.data_processor = HorseRaceDataProcessor()
        self.model = HorseRaceBART(
            n_trees=n_trees,
            n_chains=n_chains,
            n_samples=n_samples,
            n_tune=n_tune,
            random_seed=random_seed
        )

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate the horse racing data."""
        print(f"Loading data from {self.data_path}")
        X, y = self.data_processor.load_training_data(self.data_path)
        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the BART model on the prepared data."""
        print("Training BART model")
        self.model.fit(X, y)
        
        # Calculate and log feature importance
        feature_importance = self.model.get_feature_importance()
        print("Feature importance:")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")

    def save_model(self) -> None:
        """Save the trained model and metadata."""
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters and trace
        model_file = self.model_path / "model_params.pkl"
        model_data = {
            'n_trees': self.model.n_trees,
            'n_chains': self.model.n_chains,
            'n_samples': self.model.n_samples,
            'n_tune': self.model.n_tune,
            'random_seed': self.model.random_seed,
            'trace': self.model.trace
        }
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model parameters and trace saved to {model_file}")
        
        # Save feature names
        feature_names = (
            self.data_processor.race_features +
            self.data_processor.recent_performance_features +
            self.data_processor.jockey_trainer_features
        )
        feature_file = self.model_path / "feature_names.pkl"
        with open(feature_file, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"Feature names saved to {feature_file}")

def main():
    # Initialize trainer
    trainer = HorseRaceTrainer(
        data_path='students/fleischhacker_adam2/data/processed/processed_race_data_with_results.nc',
        model_path='students/he_chenchuan/models',
        n_trees=50,
        n_chains=4,
        n_samples=4000,
        n_tune=4000,
        random_seed=42
    )
    
    # Load and process data
    X, y = trainer.load_data()
    
    # Train model
    trainer.train_model(X, y)
    
    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main() 