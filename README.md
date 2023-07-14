# UNDERSTANDING FBPROPHET

# Facebook Prophet: Time Series Forecasting

This README provides a step-by-step guide on using Facebook Prophet for time series forecasting. The code snippets provided demonstrate various aspects of the Prophet library, such as data loading, model instantiation, training, prediction, visualization, cross-validation, and performance evaluation.

## Installation

Before using Prophet, ensure that you have Python installed on your system. You can install Prophet and its dependencies using the following commands:

`pip install prophet`

`pip install pystan`


# Load the time series data
`df = pd.read_csv('example_wp_log_peyton_manning.csv')`

# View the first 10 rows of the data
`df.head(10)`

# View the last 10 rows of the data
`df.tail(10)`

# Get the dimensions of the data (rows, columns)
`df.shape`

# Obtain information about the data (data types, missing values, etc.)
`df.info()`

# Generate descriptive statistics of the data
`df.describe()`

# THE MODEL

# Instantiate the model
`model = Prophet()`

# Train the model
`model.fit(df)`

# Generate future timestamps using the ds column
# Use the make_future_dataframe method to generate future ds values
`future = model.make_future_dataframe(periods=365)`

# View the last 5 rows of the future dataframe
`future.tail(5)`

# Make predictions
`future = model.predict(future)`

# View the first 5 rows of the predicted values
`future.head(5)`

# View the last 3 rows of the predicted values with uncertainty intervals
`future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)`

# VISUALIZATION

# Plot the predicted values
`figure_1 = model.plot(future)`

# Plot the trend and components of the future dataframe (weekly and yearly)
`figure_2 = model.plot_components(future)`

# CROSS VALIDATION

`from prophet.diagnostics import cross_validation`

# Perform cross-validation
`cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')`

# View the first 5 rows of the cross-validation results
`cv.head(5)`

# PERFORMANCE METRICS

`from prophet.diagnostics import performance_metrics`

# Calculate performance metrics
`pm = performance_metrics(cv)`

# View the first 5 rows of the performance metrics
`pm.head(5)`

# VISUALIZING CROSS VALIDATION PERFORMANCE METRICS

`from prophet.plot import plot_cross_validation_metric`

# Plot a specific performance metric (e.g., SMAPE)
`figure_3 = plot_cross_validation_metric(cv, metric='smape')`

# HYPERPARAMETER TUNING

For detailed information on hyperparameter tuning in Prophet, refer to the official Prophet documentation. The documentation covers topics such as choosing the number of Fourier terms, adjusting trend flexibility, modifying seasonality prior scale, and diagnosing overfitting and underfitting.

By following the steps outlined in this README and referring to the official documentation, you can effectively utilize Facebook Prophet for time series forecasting and optimize your model's performance for your specific dataset.
