import numpy as np
import pymc3 as pm
import pandas as pd
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data.drop('Time', axis=1, inplace=True)
# Prepare your data
X = data.drop('AQI', axis=1)
y = data['AQI']

# Define the Bayesian model
with pm.Model() as model:
    # Priors for regression coefficients
    beta = pm.Normal('beta', mu=0, sd=1, shape=X.shape[1])

    # Linear regression
    mu = pm.math.dot(X, beta)

    # Likelihood (assuming normal distribution for the target variable)
    sigma = pm.HalfNormal('sigma', sd=1)
    y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)

    # Run the Bayesian inference
    trace = pm.sample(2000, tune=1000)

# Predict on new data
X_new = ...  # New input features
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=1000)
    y_pred = post_pred['y_obs'].mean(axis=0)
