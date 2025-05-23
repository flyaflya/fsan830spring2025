import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HorseRaceDataProcessor:
    def __init__(self):
        # Define feature groups
        self.race_features = [
            'surface_code', 'distance', 'purse', 'horse_name'
        ]
        
        self.recent_performance_features = [
            'recentFinishPosition1', 'recentSurfaceCode1',
            'recentPostPosition1'
        ]
        
        self.jockey_trainer_features = [
            'jockey_name', 'trainer_name'
        ]
        
        # Combine all features
        self.all_features = (
            self.race_features +
            self.recent_performance_features +
            self.jockey_trainer_features
        )
        
        # Define required variables for validation
        self.required_vars = [
            'distance', 'purse', 'surface_code', 'recentFinishPosition1',
            'recentSurfaceCode1', 'recentPostPosition1',
            'jockey_name', 'trainer_name', 'horse_name'
        ]

    def load_training_data(self, filepath):
        """Load and preprocess training data."""
        try:
            # Load the dataset
            ds = xr.open_dataset(filepath)
            
            # Convert to pandas DataFrame
            df = ds.to_dataframe().reset_index()
            
            # Map variable names to match our expected format
            df['surface_code'] = df['surface']  # Map surface to surface_code
            df['distance'] = df['distance_f']   # Map distance_f to distance
            df['horse_name'] = df['horse']      # Map horse to horse_name
            
            # Map recent performance features to match our expected format
            df['recentFinishPosition1'] = df['recent_finish_pos']
            df['recentSurfaceCode1'] = df['recent_surface']
            df['recentPostPosition1'] = df['recent_start_pos']
            
            # Map jockey and trainer features
            df['jockey_name'] = df['jockey']
            df['trainer_name'] = df['trainer']
            
            # Handle missing values in recent performance features
            recent_cols = [
                'recentFinishPosition1', 'recentSurfaceCode1',
                'recentPostPosition1'
            ]
            for col in recent_cols:
                if col in df.columns:
                    # Fill missing values with median for numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    # Fill missing values with mode for categorical columns
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
            
            # Calculate derived features
            df = self.calculate_derived_features(df)
            
            # Calculate relative metrics
            df = self.calculate_relative_metrics(df)
            
            # Calculate jockey and trainer statistics
            df = self.calculate_jockey_trainer_stats(df)
            
            # Create concatenated field for trainer_name_jockey_name
            df['trainer_name_jockey_name'] = df['trainer_name'] + '_' + df['jockey_name']
            
            # Prepare features for training
            X = self.prepare_training_features(df)
            y = df['finish_position']
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"Error loading training data: {str(e)}")

    def load_testing_data(self, filepath):
        """Load and process testing data."""
        try:
            # create a dataframe from the csv file
            df = pd.read_csv(filepath)
            
            # Create race identifier from track_code and race_number
            df['race'] = df['track_code'].astype(str) + '_' + df['race_number'].astype(str)
            
            # Store horse names in a separate column and use it as starter identifier
            df['starter'] = df['horse_name']
            
            # Ensure we only have one entry per horse per race
            df = df.drop_duplicates(subset=['race', 'horse_name'], keep='first')
            
            # Map recent performance features to match training data format
            df['recentFinishPosition1'] = df['recentFinishPosition1']
            df['recentSurfaceCode1'] = df['recentSurfaceCode1']
            df['recentPostPosition1'] = df['recentPostPosition1']
            df['recent_purse'] = df['purse']  # Use current race purse as recent purse for testing
            
            # Map race features (these should already exist but we ensure they're properly named)
            df['surface_code'] = df['surface_code']
            df['distance'] = df['distance']
            df['purse'] = df['purse']
            
            # Map jockey and trainer features
            df['jockey_name'] = df['jockey_name'].fillna('unknown')
            df['trainer_name'] = df['trainer_name']
            
            # Calculate derived features
            df = self.calculate_derived_features(df)
            
            # Calculate relative metrics
            df = self.calculate_relative_metrics(df)
            
            # Calculate jockey and trainer statistics
            df = self.calculate_jockey_trainer_stats(df)
            
            # Create concatenated field for trainer_name_jockey_name
            df['trainer_name_jockey_name'] = df['trainer_name'] + '_' + df['jockey_name']
            
            # Prepare features for prediction
            X = self.prepare_training_features(df)
            
            return X, df['race'], df['horse_name']
            
        except Exception as e:
            raise ValueError(f"Error loading testing data: {str(e)}")

    def calculate_derived_features(self, df):
        """Calculate derived features from raw data."""
        # Calculate field size using horse_name instead of program_number
        df['field_size'] = df.groupby('race')['horse_name'].transform('count')
        
        return df

    def calculate_relative_metrics(self, df):
        """Calculate relative performance metrics."""
        # Determine which column names to use based on what's available
        finish_col = 'recentFinishPosition1' if 'recentFinishPosition1' in df.columns else 'recent_finish_pos'
        purse_col = 'purse' if 'purse' in df.columns else 'recent_purse'
        start_col = 'recentPostPosition1' if 'recentPostPosition1' in df.columns else 'recent_start_pos'
        
        # Calculate field averages
        field_avgs = df.groupby('race').agg({
            finish_col: 'mean',
            purse_col: 'mean',
            start_col: 'mean'
        }).reset_index()
        
        # Merge back to main dataframe
        df = df.merge(field_avgs, on='race', suffixes=('', '_field_avg'))
        
        # Calculate relative metrics
        df['relative_finish'] = df[finish_col] / df[f'{finish_col}_field_avg']
        df['relative_purse'] = df[purse_col] / df[f'{purse_col}_field_avg']
        df['relative_start'] = df[start_col] / df[f'{start_col}_field_avg']
        
        return df

    def calculate_jockey_trainer_stats(self, df, is_training=True):
        """Calculate jockey and trainer statistics."""
        if is_training and 'finish_position' in df.columns:
            # Calculate jockey stats
            jockey_stats = df.groupby('jockey_name').agg({
                'finish_position': ['count', 'mean']
            }).reset_index()
            jockey_stats.columns = ['jockey_name', 'jockey_starts', 'jockey_avg_finish']
            df = df.merge(jockey_stats, on='jockey_name', how='left')
            
            # Calculate trainer stats
            trainer_stats = df.groupby('trainer_name').agg({
                'finish_position': ['count', 'mean']
            }).reset_index()
            trainer_stats.columns = ['trainer_name', 'trainer_starts', 'trainer_avg_finish']
            df = df.merge(trainer_stats, on='trainer_name', how='left')
        else:
            # For testing data or when finish_position is not available, use default values
            df['jockey_starts'] = 0
            df['jockey_avg_finish'] = 0
            df['trainer_starts'] = 0
            df['trainer_avg_finish'] = 0
        
        # Add derived statistics to feature list
        self.all_features.extend([
            'jockey_starts', 'jockey_avg_finish',
            'trainer_starts', 'trainer_avg_finish'
        ])
        
        return df

    def prepare_training_features(self, df):
        """Prepare features for training."""
        # Select features
        X = df[self.all_features].copy()
        
        # Handle categorical variables
        categorical_cols = ['surface_code', 'recentSurfaceCode1', 'jockey_name', 'trainer_name', 'horse_name']
        
        # Replace empty strings with NaN
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].replace('', np.nan)
        
        # Fill NaN with mode for categorical columns
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0])
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols)
        
        # Convert to numeric, replacing any remaining non-numeric values with NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Fill any remaining NaN with median
        X = X.fillna(X.median())
        
        # Verify no NaN values remain
        if X.isna().any().any():
            raise ValueError("NaN values remain in the processed features")
        
        return X

    def prepare_testing_features(self, df):
        """Prepare features for testing."""
        print(f"\nPreparing testing features:")
        print(f"Input DataFrame shape: {df.shape}")
        
        # Store index columns and horse names
        index_cols = ['race', 'starter']
        horse_names = df['horse_name'].copy()
        
        # Create a copy of the index columns before processing
        index_df = df[index_cols].copy()
        print(f"Index DataFrame shape: {index_df.shape}")
        
        # Select features while preserving the multi-index
        X = df[self.all_features].copy()
        print(f"Selected features shape: {X.shape}")
        print(f"Selected features: {X.columns.tolist()}")
        
        # Handle categorical variables
        categorical_cols = ['surface_code', 'recentSurfaceCode1', 'jockey_name', 'trainer_name']
        print(f"Categorical columns: {categorical_cols}")
        
        # Replace empty strings with NaN and handle missing values
        for col in categorical_cols:
            if col in X.columns:
                # Replace empty strings with NaN
                X[col] = X[col].replace('', np.nan)
                print(f"Empty strings in {col}: {X[col].isna().sum()}")
                
                # Fill NaN with mode, handling the case where all values are NaN
                if X[col].isna().all():
                    X[col] = 'unknown'  # Use 'unknown' if all values are NaN
                else:
                    mode_value = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
                    X[col] = X[col].fillna(mode_value)
                print(f"NaN values in {col} after filling: {X[col].isna().sum()}")
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols)
        print(f"Shape after one-hot encoding: {X.shape}")
        
        # Ensure all training features are present
        missing_features = set(self.all_features) - set(X.columns)
        if missing_features:
            print(f"Adding missing features: {missing_features}")
            for feature in missing_features:
                X[feature] = 0  # Add missing features with zeros
        
        # Convert to numeric, replacing any remaining non-numeric values with NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        print(f"NaN values after numeric conversion: {X.isna().sum().sum()}")
        
        # Fill any remaining NaN with median
        X = X.fillna(X.median())
        print(f"NaN values after median filling: {X.isna().sum().sum()}")
        
        # Verify no NaN values remain
        if X.isna().any().any():
            raise ValueError("NaN values remain in the processed features")
        
        # Add back the index columns and horse names
        X = pd.concat([X, index_df], axis=1)
        X['horse_name'] = horse_names  # Add back the original horse names
        print(f"Shape after adding index columns: {X.shape}")
        
        # Set the index using race and starter columns
        X = X.set_index(index_cols)
        print(f"Final shape: {X.shape}")
        print(f"Index names: {X.index.names}")
        
        return X

    def validate_data_structure(self, ds, is_training=True):
        """Validate that the dataset has the required structure.
        
        Args:
            ds: xarray Dataset to validate
            is_training: Boolean indicating if this is training data (default: True)
        """
        # Check dimensions
        required_dims = ['race', 'starter']
        for dim in required_dims:
            if dim not in ds.dims:
                raise ValueError(f"Missing required dimension: {dim}")
        
        if is_training:
            # For training data, check all required variables
            for var in self.required_vars:
                if var not in ds.data_vars:
                    raise ValueError(f"Missing required variable: {var}")
                if ds[var].isnull().any():
                    raise ValueError(f"Required variable {var} contains missing values")
        else:
            # For testing data, only check essential variables
            essential_vars = ['distance', 'purse', 'jockey_name', 'trainer_name', 'horse_name']
            for var in essential_vars:
                if var not in ds.data_vars:
                    raise ValueError(f"Missing essential variable: {var}")
                if ds[var].isnull().any():
                    raise ValueError(f"Essential variable {var} contains missing values")

    def get_race_predictions(self, predictions: np.ndarray, race_ids: np.ndarray) -> Dict[str, List[int]]:
        """Convert length predictions to top 3 horses for each race."""
        race_predictions = {}
        
        # Ensure predictions is 1D array
        predictions = predictions.flatten()
        
        # Get unique race IDs
        unique_races = np.unique(race_ids)
        
        for race_id in unique_races:
            # Get indices for this race
            race_indices = np.where(race_ids == race_id)[0]
            
            # Get predictions for this race
            race_preds = predictions[race_indices]
            
            # Get indices of top 3 horses (smallest lengths back)
            top_3_indices = np.argsort(race_preds)[:3]
            
            # Convert to horse names instead of program numbers
            race_predictions[str(race_id)] = race_indices[top_3_indices].tolist()
            
        return race_predictions 