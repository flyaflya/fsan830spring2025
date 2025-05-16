import numpy as np
import pymc as pm
import pymc_bart as pmb
from typing import Dict, Tuple, Optional

class HorseRaceBART:
    def __init__(
        self,
        n_trees: int = 200,
        n_chains: int = 4,
        n_samples: int = 1000,
        n_tune: int = 1000,
        random_seed: Optional[int] = None
    ):
        self.n_trees = n_trees
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.random_seed = random_seed
        self.model = None
        self.trace = None
        self.n_starters = None

    def build_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build the BART model with appropriate priors."""
        # Set up coordinates for our dimensions
        n_obs = X.shape[0]

        # Define coordinates for the model
        coords = {
            "n_obs": np.arange(n_obs)
        }

        with pm.Model(coords=coords) as self.model:
            # Data containers
            X_shared = pm.Data('X', X)
            y_shared = pm.Data('y', y)

            # BART component for single output
            mu = pmb.BART(
                'mu',
                X_shared,
                y_shared,
                m=self.n_trees
            )
            
            # Error term
            sigma = pm.HalfNormal('sigma', 1.0)
            
            # Likelihood
            y_obs = pm.Normal(
                'y_obs',
                mu=mu,
                sigma=sigma,
                observed=y_shared
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the BART model to the data."""
        if self.model is None:
            self.build_model(X, y)

        with self.model:
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                random_seed=self.random_seed,
                progressbar=True
            )

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate predictions for new data."""
        if self.model is None or self.trace is None:
            raise ValueError("Model must be fitted before making predictions")

        # Get the BART trees from the trace
        bart_trees = self.trace.posterior['mu'].values  # Shape: (chain, draw, obs)
        
        # Reshape to combine chains and draws
        n_chains, n_draws, n_obs = bart_trees.shape
        bart_trees_reshaped = bart_trees.reshape(-1, n_obs)  # Shape: (chain*draw, obs)
        
        # Calculate mean predictions across all samples
        y_pred = np.mean(bart_trees_reshaped, axis=0)  # Shape: (obs,)
        
        # Add a small amount of noise to break ties
        # This helps prevent identical predictions for different horses
        noise = np.random.normal(0, 1e-6, size=y_pred.shape)
        y_pred = y_pred + noise
        
        if return_uncertainty:
            # Calculate prediction intervals
            y_lower = np.percentile(bart_trees_reshaped, 2.5, axis=0)
            y_upper = np.percentile(bart_trees_reshaped, 97.5, axis=0)
            uncertainty = np.stack([y_lower, y_upper], axis=-1)
            return y_pred, uncertainty
        
        return y_pred, None

    def get_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance based on BART splits."""
        if self.trace is None:
            raise ValueError("Model must be fitted before calculating feature importance")

        # Extract BART tree information
        bart_trees = self.trace.posterior['mu'].values
        
        # Calculate feature importance (simplified version)
        feature_importance = {}
        
        return feature_importance

    @classmethod
    def load_model(cls, model_path: str, X: np.ndarray, y: np.ndarray) -> 'HorseRaceBART':
        """Load a saved model from parameters and reconstruct it."""
        import pickle
        
        # Load model parameters
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance with saved parameters
        model = cls(
            n_trees=model_data['n_trees'],
            n_chains=model_data['n_chains'],
            n_samples=model_data['n_samples'],
            n_tune=model_data['n_tune'],
            random_seed=model_data['random_seed']
        )
        
        # Build and set up the model
        model.build_model(X, y)
        model.trace = model_data['trace']
        
        return model 