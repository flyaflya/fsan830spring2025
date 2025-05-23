import os
import pickle
import numpy as np
import pymc as pm
import pymc_bart as pmb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from .base_model import BaseModel

class BARTModel(BaseModel):
    """BART (Bayesian Additive Regression Trees) model for horse race prediction."""
    
    def __init__(self, config):
        """Initialize the BART model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.trace = None
        self.feature_names = None
        self.scaler = None
        self.encoders = {}
        
    def train(self, X, y, **kwargs):
        """Train the BART model.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
        """
        print("Training BART model...")
        
        with pm.Model() as model:
            # Create shared variable for data
            X_shared = pm.Data('X', X)
            
            # Define BART model
            mu = pmb.BART('mu', X_shared, y)
            sigma = pm.HalfNormal('sigma', sigma=1.0)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample from posterior
            print("Sampling from posterior...")
            self.trace = pm.sample(
                draws=self.config.get('n_draws', 1000),
                tune=self.config.get('n_tune', 1000),
                chains=self.config.get('n_chains', 2),
                return_inferencedata=True
            )
            
            self.model = model
            print("Training completed.")
            
        return self
    
    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        print("Generating predictions...")
        
        with self.model:
            # Set the data for prediction
            pm.set_data({'X': X})
            
            # Generate posterior predictions
            print("Sampling from posterior predictive...")
            ppc = pm.sample_posterior_predictive(self.trace)
            
        # Extract predictions from posterior
        try:
            if hasattr(ppc, "posterior_predictive") and "y_obs" in ppc.posterior_predictive:
                y_obs_ppc = ppc.posterior_predictive["y_obs"].values
                y_pred = y_obs_ppc.mean(axis=(0, 1))
            elif "mu" in self.trace.posterior:
                mu_samples = self.trace.posterior["mu"].values
                y_pred = mu_samples.mean(axis=(0, 1))
            else:
                raise ValueError("Could not extract predictions from model.")
                
        except Exception as e:
            print(f"Error extracting predictions: {str(e)}")
            raise
            
        return y_pred
    
    def save(self, path):
        """Save the model and related data to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None or self.trace is None:
            raise ValueError("No trained model to save.")
            
        model_data = {
            'trace': self.trace,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'config': self.config
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model data
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model and related data from disk.
        
        Args:
            path: Path to load the model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.trace = model_data['trace']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        self.encoders = model_data['encoders']
        self.config = model_data['config']
        
        print(f"Model loaded from {path}")
    
    def evaluate(self, X, y):
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Generate predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        spearman_corr, _ = spearmanr(y, y_pred)
        
        metrics = {
            'mse': mse,
            'r2': r2,
            'spearman_correlation': spearman_corr
        }
        
        return metrics 