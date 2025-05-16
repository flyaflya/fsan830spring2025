import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

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
        output_path: str,
        n_trees: int = 200,
        n_chains: int = 4,
        n_samples: int = 1000,
        n_tune: int = 1000,
        random_seed: int = 42
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.data_processor = HorseRaceDataProcessor()
        self.model = HorseRaceBART(
            n_trees=n_trees,
            n_chains=n_chains,
            n_samples=n_samples,
            n_tune=n_tune,
            random_seed=random_seed
        )

    def load_data(self) -> pd.DataFrame:
        """Load and validate the horse racing data."""
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Validate required columns
        required_columns = [
            'race_id', 'lengths_back_finish',
            *self.data_processor.race_features,
            *self.data_processor.horse_features,
            *self.data_processor.jockey_trainer_features
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df

    def train_model(self, df: pd.DataFrame) -> None:
        """Train the BART model on the prepared data."""
        print("Preparing features for training")
        X, y = self.data_processor.prepare_features(df)
        
        print("Training BART model")
        self.model.fit(X, y)
        
        # Calculate and log feature importance
        feature_importance = self.model.get_feature_importance()
        print("Feature importance:")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")

    def predict_races(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Generate predictions for all races in the dataset."""
        print("Preparing features for prediction")
        X, _ = self.data_processor.prepare_features(df)
        
        print("Generating predictions")
        predictions, uncertainty = self.model.predict(X, return_uncertainty=True)
        
        # Get top 3 predictions for each race
        race_predictions = self.data_processor.get_race_predictions(
            predictions,
            df['race_id'].values
        )
        
        # Save predictions with uncertainty
        results_df = pd.DataFrame({
            'race_id': df['race_id'],
            'horse_id': df['horse_id'],
            'predicted_lengths_back': predictions,
            'prediction_lower': uncertainty[:, 0],
            'prediction_upper': uncertainty[:, 1]
        })
        
        output_file = self.output_path / 'predictions.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return race_predictions

    def evaluate_predictions(
        self,
        predictions: Dict[str, List[int]],
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate the model's predictions."""
        metrics = {}
        
        # Calculate accuracy of top 3 predictions
        correct_predictions = 0
        total_races = len(predictions)
        
        for race_id, pred_horses in predictions.items():
            race_data = df[df['race_id'] == race_id]
            actual_top_3 = race_data.nsmallest(3, 'lengths_back_finish')['horse_id'].tolist()
            
            # Count how many predicted horses were in actual top 3
            correct_predictions += len(set(pred_horses) & set(actual_top_3))
        
        metrics['top_3_accuracy'] = correct_predictions / (total_races * 3)
        
        # Calculate mean absolute error of length predictions
        mae = np.mean(np.abs(
            df['predicted_lengths_back'] - df['lengths_back_finish']
        ))
        metrics['mae'] = mae
        
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics

def main():
    # Initialize predictor
    predictor = HorseRacePredictor(
        data_path='data/horse_races.csv',
        output_path='results',
        n_trees=200,
        n_chains=4,
        n_samples=1000,
        n_tune=1000,
        random_seed=42
    )
    
    # Load and process data
    df = predictor.load_data()
    
    # Train model
    predictor.train_model(df)
    
    # Generate predictions
    predictions = predictor.predict_races(df)
    
    # Evaluate results
    metrics = predictor.evaluate_predictions(predictions, df)

if __name__ == "__main__":
    main() 