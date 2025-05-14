# 1. Outlier Removal with IQR
# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
# df = df[~((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR))).any(axis=1)]

linear_regression_model = LinearRegression()
df['age_category'] = pd.cut(df['complexAge'], bins=[0, 10, 20, 30, 40, 50, np.inf],
                            labels=['0-10', '11-20', '21-30', '31-40', '41-50', '50+'])

# Evaluate the model within each age category
for category in df['age_category'].unique():
    # Filter the data for the current age category
    df_category = df[df['age_category'] == category]
    X_category = df_category.drop(['medianComplexValue', 'age_category'], axis=1)
    y_category = df_category['medianComplexValue']

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_category, y_category, test_size=0.2, random_state=42)

    # Train the model
    linear_regression_model.fit(X_train, y_train)

    # Make predictions
    y_pred = linear_regression_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = linear_regression_model.score(X_test, y_test)

    print(f'Age Category: {category}, Test MSE: {mse}, Test R2: {r2}')

################################################################################################
# Încarcă setul de date
df = pd.read_csv('path_to_your_file.csv', header=None, usecols=[2, 3, 4, 5, 6, 8],
                 names=['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'medianComplexValue'])

# Prepară variabilele X și y pentru antrenare
X = df[['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr']]
y = df['medianComplexValue']

# Împarte setul de date în set de antrenare și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inițializează modelul de regresie liniară
linear_regression_model = LinearRegression()

# Antrenează modelul pe setul de antrenare
linear_regression_model.fit(X_train, y_train)

# Evaluează modelul pe setul de test
y_pred = linear_regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')

# Pentru a prezice prețul unui apartament nou, va trebui să folosim caracteristicile cunoscute ale apartamentului
# Exemplu de caracteristici pentru un apartament nou
new_apartment_features = np.array([[complexAge_new, totalRooms_new, totalBedrooms_new, complexInhabitants_new, apartmentsNr_new]])

# Prezice prețul folosind modelul antrenat
predicted_price = linear_regression_model.predict(new_apartment_features)
print(f'Predicted price for the new apartment: {predicted_price[0]}')

###################################################

# First method:
# df['rooms_per_bedroom'] = df['totalRooms'] / df['totalBedrooms']
#
# # Prepare the features and target variable
# X = df.drop(['medianComplexValue', 'totalRooms', 'totalBedrooms'], axis=1)
# y = df['medianComplexValue']
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 3. Cross-Validation: Using k-fold cross-validation on the Linear Regression model
# linear_regression_model = LinearRegression()
# scores = cross_val_score(linear_regression_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
#
# # Training and evaluating the model
# linear_regression_model.fit(X_train, y_train)
# y_pred = linear_regression_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"Cross-validated MSE: {-np.mean(scores)}")
# print(f"Test MSE: {mse}")
# print(f"Test R2: {r2}")
# Create age categories

#############################################################
# Second method:
# from sklearn.preprocessing import PolynomialFeatures
#
# # Create a PolynomialFeatures object with degree 2
# poly = PolynomialFeatures(degree=2)
#
# # Apply polynomial transformation to the features
# X_poly = poly.fit_transform(df_no_outliers.drop('medianComplexValue', axis=1))
#
# # Now, X_poly can be used to train the model
# from sklearn.linear_model import RidgeCV
#
# # Prepare the features and target variable from the DataFrame without outliers
# X = df_no_outliers.drop('medianComplexValue', axis=1)
# y = df_no_outliers['medianComplexValue']
#
# # Define a range of alpha values to test
# alphas = np.logspace(-4, 4, 50)
#
# # Initialize RidgeCV with the alpha values and 5-fold cross-validation
# ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
#
# # Fit the model
# ridge_cv.fit(X, y)
#
# # The best alpha and corresponding test score
# best_alpha = ridge_cv.alpha_
# best_score = -ridge_cv.best_score_
#
# print(f"Best alpha: {best_alpha}")
# print(f"Best cross-validated MSE: {best_score}")


# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Assuming linear interpolation for missing values as a simple approach
# df = df.interpolate()
# Assuming that an outlier is defined as a value that is beyond 3 standard deviations from the mean.
# from scipy import stats
#
# # Calculate the z-score for each data point
# z_scores = np.abs(stats.zscore(df))
# outlier_rows = np.where(z_scores > 3)[0]
#
# # Drop rows with outliers
# df_no_outliers = df.drop(outlier_rows, axis=0)

# Features and target variable
# X = df.drop('medianComplexValue', axis=1)
# y = df['medianComplexValue']

# Splitting the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Initialize and train the linear regression model
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
#
# # Predictions
# y_pred = lr_model.predict(X_test)
#
# # Performance evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R-squared (R2): {r2}")
# #
# import matplotlib.pyplot as plt
#
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Median Complex Values')
# plt.ylabel('Predicted Median Complex Values')
# plt.title('Actual vs. Predicted')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# plt.show()
#
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
#
# # Re-train with Ridge regularization
# ridge_model = Ridge(alpha=1.0)
# ridge_model.fit(X_train, y_train)
# ridge_predictions = ridge_model.predict(X_test)
# ridge_mse = mean_squared_error(y_test, ridge_predictions)
# ridge_r2 = r2_score(y_test, ridge_predictions)
#
# # Re-train with Lasso regularization
# lasso_model = Lasso(alpha=0.1)
# lasso_model.fit(X_train, y_train)
# lasso_predictions = lasso_model.predict(X_test)
# lasso_mse = mean_squared_error(y_test, lasso_predictions)
# lasso_r2 = r2_score(y_test, lasso_predictions)
#
# # Re-train with Elastic Net regularization
# elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net_model.fit(X_train, y_train)
# elastic_net_predictions = elastic_net_model.predict(X_test)
# elastic_net_mse = mean_squared_error(y_test, elastic_net_predictions)
# elastic_net_r2 = r2_score(y_test, elastic_net_predictions)
#
# # Print the performance
# print(f"Ridge Regression MSE: {ridge_mse}, R2: {ridge_r2}")
# print(f"Lasso Regression MSE: {lasso_mse}, R2: {lasso_r2}")
# print(f"Elastic Net Regression MSE: {elastic_net_mse}, R2: {elastic_net_r2}")
#
# # Compare with the original Linear Regression
# print(f"Linear Regression MSE: {mse}, R2: {r2}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
# # Assuming df is already defined and preprocessed as per your code.
#
# # Features and target variable
# X = df.drop('medianComplexValue', axis=1)
# y = df['medianComplexValue']
#
# # Splitting the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define a pipeline that first scales the features and then trains the model
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('model', LinearRegression())
# ])
#
# # Train the pipeline
# pipeline.fit(X_train, y_train)
#
# # Predictions
# y_pred = pipeline.predict(X_test)
#
# # Performance evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# # Grid search for hyperparameter tuning on Ridge Regression
# ridge_params = {'model': [Ridge()],
#                 'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
# grid_ridge = GridSearchCV(pipeline, ridge_params, cv=5, scoring='neg_mean_squared_error')
# grid_ridge.fit(X_train, y_train)
#
# # Best Ridge model
# best_ridge_pipeline = grid_ridge.best_estimator_
# ridge_predictions = best_ridge_pipeline.predict(X_test)
# ridge_mse = mean_squared_error(y_test, ridge_predictions)
# ridge_r2 = r2_score(y_test, ridge_predictions)
#
# # Similarly, we can perform grid search for Lasso and Elastic Net as well
# # ...
#
# # Print the performance
# print(f"Linear Regression MSE: {mse}, R2: {r2}")
# print(f"Best Ridge Regression MSE: {ridge_mse}, R2: {ridge_r2}")
#
# # Visualization of the best model's performance
# plt.scatter(y_test, ridge_predictions)
# plt.xlabel('Actual Median Complex Values')
# plt.ylabel('Predicted Median Complex Values')
# plt.title('Actual vs. Predicted - Best Ridge Model')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# plt.show()