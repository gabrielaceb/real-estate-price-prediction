import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Citirea și preprocesarea datelor
columns = [2, 3, 4, 5, 6, 8]
column_names = ['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'medianComplexValue']
df = pd.read_csv('apartmentComplexData.txt', header=None, usecols=columns, names=column_names)

# Eliminarea valorilor aberante (outliers)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR))).any(axis=1)]

# Matricea de corelație (opțional)
# sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# plt.title('Heatmap of Correlation Matrix')
# plt.show()

# Separarea în variabile independente și dependentă
X = df.drop('medianComplexValue', axis=1)
y = df['medianComplexValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: Regressie Liniară
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

print("Linear Regression:")
print("Score:", linear_model.score(X_test, y_test))
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("MAE:", mean_absolute_error(y_test, y_pred_linear))
print("R2:", r2_score(y_test, y_pred_linear))

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 2: Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
print("Ridge Score:", ridge_model.score(X_test_scaled, y_test))

# Model 3: Lasso Regression
lasso_model = Lasso(alpha=100)
lasso_model.fit(X_train, y_train)
print("Lasso Score:", lasso_model.score(X_test, y_test))

# Model 4: Elastic Net
elastic_model = ElasticNet(alpha=1, l1_ratio=0.9)
elastic_model.fit(X_train, y_train)
y_pred_elastic = elastic_model.predict(X_test)
print("Elastic Net Score:", elastic_model.score(X_test, y_test))

# Model 5: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Score:", rf_model.score(X_test, y_test))

# Vizualizare comparativă a predicțiilor
plt.figure(figsize=(14, 7))

def plot_prediction(y_test, y_pred, title, pos):
    plt.subplot(2, 2, pos)
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

plot_prediction(y_test, y_pred_linear, 'Linear Regression', 1)
plot_prediction(y_test, y_pred_ridge, 'Ridge Regression', 2)
plot_prediction(y_test, y_pred_elastic, 'Elastic Net', 3)
plot_prediction(y_test, y_pred_rf, 'Random Forest', 4)

plt.tight_layout()
plt.show()
