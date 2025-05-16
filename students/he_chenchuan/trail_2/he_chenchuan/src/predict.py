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

class HorseRacePredictor:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        output_path: str
    ):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.data_processor = HorseRaceDataProcessor()
        self.model = None
        self.feature_names = None

    def load_model(self) -> None:
        """Load the trained model and feature names."""
        # Load model parameters
        model_file = self.model_path / "model_params.pkl"
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model parameters loaded from {model_file}")
        
        # Load feature names
        feature_file = self.model_path / "feature_names.pkl"
        with open(feature_file, 'rb') as f:
            self.feature_names = pickle.load(f)
        print(f"Feature names loaded from {feature_file}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        
        # Create model instance with loaded parameters
        self.model = HorseRaceBART(
            n_trees=model_data['n_trees'],
            n_chains=model_data['n_chains'],
            n_samples=model_data['n_samples'],
            n_tune=model_data['n_tune'],
            random_seed=model_data['random_seed']
        )
        
        # Create dummy data to build the model structure
        dummy_X = np.ones((1, len(self.feature_names)))
        dummy_y = np.ones(1)
        
        print(f"Dummy X shape: {dummy_X.shape}")
        print(f"Dummy y shape: {dummy_y.shape}")
        
        # Build the model structure
        self.model.build_model(dummy_X, dummy_y)
        
        # Load and validate the trace
        if 'trace' not in model_data:
            raise ValueError("No trace found in model parameters")
            
        self.model.trace = model_data['trace']
        
        # Validate the trace
        if not hasattr(self.model.trace, 'posterior'):
            raise ValueError("Invalid trace: missing posterior distribution")
            
        # Log trace information for debugging
        print(f"Trace loaded with {len(self.model.trace.posterior.chain)} chains")
        print(f"Number of samples per chain: {len(self.model.trace.posterior.draw)}")
        
        # Validate trace dimensions
        if 'mu' not in self.model.trace.posterior:
            raise ValueError("Invalid trace: missing 'mu' parameter")
            
        mu_shape = self.model.trace.posterior['mu'].shape
        print(f"Trace 'mu' parameter shape: {mu_shape}")
        
        if len(mu_shape) != 3:  # Should be (chain, draw, obs)
            raise ValueError(f"Invalid trace 'mu' shape: {mu_shape}")
            
        # Rebuild the model with the correct data structure
        self.model.model = None  # Clear the old model
        self.model.build_model(dummy_X, dummy_y)  # Rebuild with dummy data

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and validate the horse racing data."""
        print(f"Loading data from {self.data_path}")
        X, race_ids, horse_names = self.data_processor.load_testing_data(self.data_path)
        print(f"Loaded data shape: {X.shape}")
        print(f"Number of races: {len(np.unique(race_ids))}")
        print(f"Number of horses: {len(horse_names)}")
        return X, race_ids, horse_names

    def predict_races(self, X: pd.DataFrame, race_ids: np.ndarray, horse_names: np.ndarray) -> Dict[str, List[str]]:
        """Generate predictions for all races in the dataset."""
        try:
            # Make predictions
            predictions, uncertainty = self.model.predict(X, return_uncertainty=True)
            
            # Ensure predictions are the right shape
            if len(predictions) != len(horse_names):
                raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of horses ({len(horse_names)})")
            
            # Create results DataFrame
            results = pd.DataFrame({
                'race': race_ids,
                'horse_name': horse_names,
                'predicted_lengths_back': predictions.flatten() if predictions.ndim > 1 else predictions,
                'prediction_lower': uncertainty[:, 0].flatten() if uncertainty is not None else None,
                'prediction_upper': uncertainty[:, 1].flatten() if uncertainty is not None else None
            })
            
            # Save predictions to CSV
            output_dir = Path('students/he_chenchuan/predictions')
            output_dir.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_dir / 'predictions.csv', index=False)
            
            # Get top 3 horses for each race
            race_predictions = {}
            for race_id in np.unique(race_ids):
                race_mask = race_ids == race_id
                race_results = results[race_mask].sort_values('predicted_lengths_back')
                top_3_horses = race_results['horse_name'].head(3).tolist()
                race_predictions[str(race_id)] = top_3_horses
            
            return race_predictions
            
        except Exception as e:
            raise ValueError(f"Error generating predictions: {str(e)}")

def main():
    # Initialize predictor
    predictor = HorseRacePredictor(
        data_path='students/fleischhacker_adam2/data/CDX0426_filtered.csv',
        model_path='students/he_chenchuan/models',
        output_path='students/he_chenchuan/predictions'
    )
    
    # Load model
    predictor.load_model()
    
    # Load and process data
    X, race_ids, horse_names = predictor.load_data()
    
    # Generate predictions
    predictions = predictor.predict_races(X, race_ids, horse_names)
    
    # Print sample predictions
    print("\nSample predictions:")
    for race_id, top_3 in list(predictions.items())[:5]:
        print(f"Race {race_id}: Top 3 horses {top_3}")

if __name__ == "__main__":
    main() 