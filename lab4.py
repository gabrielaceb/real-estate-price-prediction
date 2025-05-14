import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# importare si curatare date
columns = [2, 3, 4, 5, 6, 8]
column_names = ['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'medianComplexValue']

data = pd.read_csv('apartmentComplexData.txt', header=None, usecols=columns, names=column_names)

print(data.describe())
print(data.isnull().sum())

# antrenarea model regresie liniara
X = data.drop('medianComplexValue', axis=1)
y = data['medianComplexValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# prezicerea pretului
apartment_data = [[20, 4000, 400, 1000, 250]]
predicted_price = lr_model.predict(apartment_data)
print("Predicted price for the new apartment: ", predicted_price[0])

# antrenarea altor modele
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

elastic_net_model = ElasticNet(alpha=1, l1_ratio=0.9)
elastic_net_model.fit(X_train, y_train)

# testarea modelelor
y_pred = lr_model.predict(X_test)
score = lr_model.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("LR score:", score)
print("MSE: ", mse)
print("MAE: ", mae)
print("R-squared: ", r2)

lasso_score = lasso_model.score(X_test, y_test)
print("Lasso score: ", lasso_score)

ridge_score = ridge_model.score(X_test, y_test)
print("Ridge score: ", lasso_score)

elastic_net_score = elastic_net_model.score(X_test, y_test)
print("Elastic Net score: ", elastic_net_score)

