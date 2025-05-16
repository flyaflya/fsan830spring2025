import xarray as xr
import numpy as np
import pymc_bart as pmb
import pymc as pm

if __name__ == "__main__":
    # Load the dataset
    input_file = 'housing_data.nc'
    ds = xr.open_dataset(input_file)

    # Select predictors and target
    y = ds['price'].values
    X = np.column_stack([
        ds['sqft'].values,
        ds['bedrooms'].values,
        ds['age'].values,
        ds['location_score'].values
    ])

    # Fit BART model with PyMC-BART
    with pm.Model() as model:
        X_shared = pm.Data('X', X)
        y_shared = pm.Data('y', y)
        mu = pmb.BART('mu', X_shared, y_shared)
        sigma = pm.HalfNormal('sigma', sigma=1.0)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_shared)
        trace = pm.sample(1000, tune=1000, cores=1, random_seed=42, progressbar=True)

    # Posterior predictive mean
    with model:
        pm.set_data({'X': X})
        ppc = pm.sample_posterior_predictive(trace)
        print(f"Posterior predictive groups: {ppc.groups()}")

    # Extract BART predictions from InferenceData object
    if hasattr(ppc, "posterior_predictive") and "y_obs" in ppc.posterior_predictive:
        y_obs_ppc = ppc.posterior_predictive["y_obs"].values  # shape: (chain, draw, sample)
        bart_pred = y_obs_ppc.mean(axis=(0, 1))
    else:
        raise KeyError("Could not find 'y_obs' in posterior_predictive group of InferenceData.")

    # Add predictions to dataset
    out_ds = ds.copy()
    out_ds['bart_predicted_price'] = (('sample',), bart_pred)

    # Save to new NetCDF file
    output_file = 'housing_data_with_bart_predictions.nc'
    out_ds.to_netcdf(output_file)

    print(f'BART predictions saved to {output_file}')
