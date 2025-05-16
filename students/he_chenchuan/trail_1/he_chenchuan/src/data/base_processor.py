from abc import ABC, abstractmethod
import xarray as xr
import numpy as np

class BaseDataProcessor(ABC):
    """Abstract base class for all data processors in the system."""
    
    @abstractmethod
    def process(self, data):
        """Process raw data into model-ready format.
        
        Args:
            data: Raw input data (can be xarray Dataset, pandas DataFrame, or other format)
            
        Returns:
            Processed data ready for model training/prediction
        """
        pass
    
    @abstractmethod
    def validate(self, data):
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
            
        Raises:
            ValueError: If data is invalid
        """
        pass
    
    @abstractmethod
    def get_feature_names(self):
        """Get the names of features used in processing.
        
        Returns:
            List of feature names
        """
        pass
    
    @abstractmethod
    def get_target_name(self):
        """Get the name of the target variable.
        
        Returns:
            Name of the target variable
        """
        pass
    
    def _check_required_dimensions(self, data, required_dims):
        """Check if data has required dimensions.
        
        Args:
            data: xarray Dataset to check
            required_dims: List of required dimension names
            
        Raises:
            ValueError: If any required dimension is missing
        """
        if not isinstance(data, xr.Dataset):
            raise ValueError("Data must be an xarray Dataset")
            
        missing_dims = [dim for dim in required_dims if dim not in data.dims]
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")
    
    def _check_required_variables(self, data, required_vars):
        """Check if data has required variables.
        
        Args:
            data: xarray Dataset to check
            required_vars: List of required variable names
            
        Raises:
            ValueError: If any required variable is missing
        """
        if not isinstance(data, xr.Dataset):
            raise ValueError("Data must be an xarray Dataset")
            
        missing_vars = [var for var in required_vars if var not in data.data_vars]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
    
    def _handle_missing_values(self, data, strategy='mean'):
        """Handle missing values in the dataset.
        
        Args:
            data: xarray Dataset
            strategy: Strategy for handling missing values ('mean', 'median', 'zero')
            
        Returns:
            Dataset with missing values handled
        """
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'zero':
            return data.fillna(0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}") 