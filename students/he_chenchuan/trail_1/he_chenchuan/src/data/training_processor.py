import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .base_processor import BaseDataProcessor

class TrainingDataProcessor(BaseDataProcessor):
    """Data processor for training data."""
    
    def __init__(self, config):
        """Initialize the training data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.required_dims = ['race', 'starter', 'past_race']
        self.required_vars = [
            'distance_f', 'purse', 'surface',
            'horse', 'jockey', 'trainer', 'program_number',
            'recent_finish_pos', 'recent_lengths_back_finish',
            'recent_lengths_back_last_call', 'recent_last_call_pos',
            'recent_distance', 'recent_purse', 'recent_start_pos',
            'recent_num_starters', 'recent_trainer', 'recent_jockey',
            'recent_surface'
        ]
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def validate(self, data):
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        # Check if data is xarray Dataset
        if not isinstance(data, xr.Dataset):
            raise ValueError("Data must be an xarray Dataset")
            
        # Check required dimensions
        self._check_required_dimensions(data, self.required_dims)
        
        # Check required variables
        self._check_required_variables(data, self.required_vars)
        
        return True
    
    def process(self, data):
        """Process training data.
        
        Args:
            data: Raw input data as xarray Dataset
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        # Validate input data
        self.validate(data)
        
        # Handle missing values
        data = self._handle_missing_values(data, self.config.get('missing_value_strategy', 'mean'))
        
        # Process categorical variables
        data = self._process_categorical_variables(data)
        
        # Create derived features
        data = self._create_derived_features(data)
        
        # Prepare feature matrix and target vector
        X, y = self._prepare_features_and_target(data)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def _process_categorical_variables(self, data):
        """Process categorical variables in the dataset.
        
        Args:
            data: xarray Dataset
            
        Returns:
            Processed dataset
        """
        categorical_vars = [
            'horse', 'jockey', 'trainer', 'program_number',
            'recent_trainer', 'recent_jockey', 'recent_surface'
        ]
        
        for var in categorical_vars:
            if var in data.data_vars:
                # Create encoder if not exists
                if var not in self.encoders:
                    self.encoders[var] = LabelEncoder()
                
                # Get unique values and fit encoder
                values = data[var].values.flatten()
                unique_values = np.unique(values[~pd.isna(values)])
                self.encoders[var].fit(unique_values)
                
                # Transform values
                encoded_values = self.encoders[var].transform(values)
                data[var] = xr.DataArray(
                    encoded_values.reshape(data[var].shape),
                    dims=data[var].dims
                )
        
        return data
    
    def _create_derived_features(self, data):
        """Create derived features from the dataset.
        
        Args:
            data: xarray Dataset
            
        Returns:
            Dataset with additional derived features
        """
        # Average finish position in past races
        data['avg_finish_pos'] = data['recent_finish_pos'].mean(dim='past_race')
        
        # Consistency (standard deviation of finish positions)
        data['finish_pos_std'] = data['recent_finish_pos'].std(dim='past_race')
        
        # Recent form (weighted average of finish positions)
        weights = np.array([0.4, 0.3, 0.2, 0.1, 0.0])  # Weights for past races
        data['recent_form'] = (data['recent_finish_pos'] * weights).sum(dim='past_race')
        
        # Class jump (difference in purse between current and last race)
        data['class_jump'] = data['purse'] - data['recent_purse'].sel(past_race=0)
        
        # Distance change from previous race
        data['distance_change'] = data['distance_f'] - data['recent_distance'].sel(past_race=0)
        
        # Normalized finish position
        data['norm_finish_pos'] = data['recent_finish_pos'] / data['recent_num_starters']
        
        # Speed figure proxy
        data['speed_figure'] = (
            data['recent_lengths_back_finish'] * 
            data['recent_distance'] / 
            data['recent_num_starters']
        )
        
        return data
    
    def _prepare_features_and_target(self, data):
        """Prepare feature matrix and target vector.
        
        Args:
            data: xarray Dataset
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        # Define feature sets
        race_features = ['distance_f', 'purse']
        recent_features = [
            'recent_finish_pos', 'recent_lengths_back_finish',
            'recent_lengths_back_last_call', 'recent_last_call_pos',
            'recent_distance', 'recent_purse', 'recent_start_pos',
            'recent_num_starters'
        ]
        derived_features = [
            'avg_finish_pos', 'finish_pos_std', 'recent_form',
            'class_jump', 'distance_change', 'norm_finish_pos',
            'speed_figure'
        ]
        
        # Combine all features
        all_features = race_features + recent_features + derived_features
        
        # Create feature matrix
        X = np.column_stack([
            data[feature].values.flatten()
            for feature in all_features
        ])
        
        # Create target vector (finish position in most recent race)
        y = data['recent_finish_pos'].sel(past_race=0).values.flatten()
        
        # Store feature names
        self.feature_names = all_features
        
        return X, y
    
    def get_feature_names(self):
        """Get the names of features used in processing.
        
        Returns:
            List of feature names
        """
        return self.feature_names
    
    def get_target_name(self):
        """Get the name of the target variable.
        
        Returns:
            Name of the target variable
        """
        return 'recent_finish_pos' 