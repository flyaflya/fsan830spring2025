import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.step_methods import PGBART
from sklearn.impute import SimpleImputer

# Assuming X_scaled and y are your training data
# Handle NaN values in X_scaled
imputer = SimpleImputer(strategy='mean')
X_scaled_clean = imputer.fit_transform(X_scaled)
# Ensure we have a regular numpy array, not a masked array
X_scaled_clean = np.asarray(X_scaled_clean, dtype=np.float64)

# Define the model
with pm.Model() as model:
    # Define the data containers with clean data
    X = pm.Data('X', X_scaled_clean)  # Create data container for features
    Y = pm.Data('Y', y)               # Create data container for target
    
    # Define the BART model
    mu = pmb.BART("mu", X=X, Y=Y, m=10)  # m = number of trees
    p = pm.Deterministic("p", pm.math.sigmoid(mu))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=Y)

    step = PGBART()

    # Sample from the model
    trace = pm.sample(
        draws=200,
        tune=200,
        chains=4,
        step=step
    )

# For predictions with new data
# First clean the test data using the same imputer
X_test_clean = imputer.transform(X_test)
# Ensure we have a regular numpy array, not a masked array
X_test_clean = np.asarray(X_test_clean, dtype=np.float64)

with model:
    # Update the data for predictions
    pm.set_data({'X': X_test_clean})  # Use cleaned test data
    ppc = pm.sample_posterior_predictive(trace, var_names=['p'])

print("Model sampling completed successfully")
print(f"Posterior predictive shape: {ppc.posterior_predictive['p'].shape}") 