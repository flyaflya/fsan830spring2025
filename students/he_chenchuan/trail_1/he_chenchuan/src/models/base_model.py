from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models in the system."""
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model on the given data.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """Save model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass 