from pathlib import Path

class ModelConfig:
    """Configuration for the BART model and data processing."""
    
    def __init__(self):
        # Model parameters
        self.model_params = {
            'n_draws': 1000,      # Number of posterior draws
            'n_tune': 1000,       # Number of tuning steps
            'n_chains': 2,        # Number of MCMC chains
            'random_seed': 42     # Random seed for reproducibility
        }
        
        # Data processing parameters
        self.data_params = {
            'missing_value_strategy': 'mean',  # Strategy for handling missing values
            'max_past_races': 5,              # Maximum number of past races to consider
            'recent_form_weights': [0.4, 0.3, 0.2, 0.1, 0.0]  # Weights for recent form calculation
        }
        
        # Get the project root directory (fsan830spring2025)
        # Start from the current file and go up to the project root
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent.parent.parent
        
        # File paths (relative to project root)
        self.paths = {
            'data_dir': Path('students/he_chenchuan/data/processed'),
            'model_dir': Path('students/he_chenchuan/train_results/models'),
            'prediction_dir': Path('students/he_chenchuan/predictions'),
            'processed_data_file': Path('students/he_chenchuan/data/processed/processed_race_data.nc'),
            'model_file': Path('bart_race_model.pkl')
        }
        
        # Create directories if they don't exist
        for path in self.paths.values():
            if not path.suffix in ['.nc', '.pkl']:
                (self.project_root / path).mkdir(parents=True, exist_ok=True)
    
    def get_model_params(self):
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_params
    
    def get_data_params(self):
        """Get data processing parameters.
        
        Returns:
            Dictionary of data processing parameters
        """
        return self.data_params
    
    def get_paths(self):
        """Get file paths.
        
        Returns:
            Dictionary of Path objects
        """
        return self.paths
    
    def get_full_path(self, key):
        """Get full path for a given key.
        
        Args:
            key: Key in the paths dictionary
            
        Returns:
            Path object for the given key
        """
        return self.project_root / self.paths[key] 