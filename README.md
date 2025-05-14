# Real Estate Price Prediction

This project applies various regression techniques to estimate the median value of apartment complexes using features such as age, number of rooms, bedrooms, inhabitants, and apartments. It was developed as part of an Artificial Intelligence lab at UTM.

## Objective

To compare the performance of several regression models and determine which best predicts real estate prices based on structured housing data.

## Technologies & Libraries

- Python 3
- NumPy, Pandas
- scikit-learn (Linear Regression, Ridge, Lasso, ElasticNet, Random Forest)
- Matplotlib, Seaborn

## Features

- Data cleaning and outlier removal
- Correlation heatmap of input features
- Training and evaluation of:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Random Forest Regressor
- Price prediction for a new apartment

## Results

- Linear/Ridge/ElasticNet models performed similarly with R² ~ 0.20
- Random Forest achieved the best performance with R² ~ 0.37
- Suggests underlying non-linear relationships in the data

## Example

```python
new_apartment = [[25, 3000, 500, 1500, 200]]
predicted_price = model.predict(new_apartment)
