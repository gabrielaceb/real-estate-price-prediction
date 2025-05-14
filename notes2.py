import os; os.system('cls');
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

scaler_standard = StandardScaler()
scaler_min_max = MinMaxScaler()

def load_data():
    df = pd.read_csv('set_raw.csv')
    df = df.dropna()
    return df

def clean_data(df: pd.DataFrame):
    investigated_cols = ['totalRooms', 'totalBedrooms',  'complexInhibitants', 'apartmentsNr', 'medianComplexValue']
    for col in investigated_cols[:-1]:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR # * 1.5
        upper_bound = Q3 + IQR # * 1.5
        # print(f'{col}: Q1={Q1}, Q3={Q3}, IQR={IQR}, lower_bound={lower_bound}, upper_bound={upper_bound}')
        df: pd.DataFrame = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    df = df.drop(columns=set(df.columns) - set(investigated_cols))
    df.to_csv('set_filtered.csv', index=False)
    return df

def read_training_data():
    return pd.read_csv('set_filtered.csv')

def standartize_data(data: pd.DataFrame):
    return pd.DataFrame(scaler_standard.fit_transform(data), columns=data.columns)

def normalize_data(data: pd.DataFrame):
    return pd.DataFrame(scaler_min_max.fit_transform(data), columns=data.columns)

def train_standartizers_to_trainNum_features(data: pd.DataFrame):
    standardized_data = pd.DataFrame(scaler_standard.fit_transform(data), columns=data.columns)
    normalized_data = pd.DataFrame(scaler_min_max.fit_transform(standardized_data), columns=data.columns)
    return standardized_data, normalized_data
    
def preprare_Data_for_training(data: pd.DataFrame):
    _, data = train_standartizers_to_trainNum_features(data)
    X = data.drop(columns='medianComplexValue')
    y = data['medianComplexValue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_model(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model

def train_lasso_model(X_train, y_train):
    model = Lasso()
    model.fit(X_train, y_train)
    return model

def train_elastic_net_model(X_train, y_train):
    model = ElasticNet()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LinearRegression, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score = model.score(y_test, y_pred)
    return mse, score

def use_model(model: LinearRegression, to_predict: pd.DataFrame):
    standartized_data, normalized_data = train_standartizers_to_trainNum_features(to_predict)
    predictions = model.predict(normalized_data)
    return predictions, standartized_data, normalized_data

def plot_predictions(predictions, real_values):
    plt.plot(predictions, real_values, 'ro')
    plt.xlabel('Predictions')
    plt.ylabel('Real values')
    plt.title('Predictions vs Real values')
    plt.show()
    
def get_human_readable_predictions(predictions: list, training_data: pd.DataFrame, standartized_data: pd.DataFrame, normalized_data: pd.DataFrame):
    """ Training data here is used to retrain scalers for 5 features """
    train_standartizers_to_trainNum_features(training_data)
    normalized_data['medianComplexValue'] = predictions
    unNormalized = pd.DataFrame(scaler_min_max.inverse_transform(normalized_data), columns=training_data.columns)
    standartized_data['medianComplexValue'] = unNormalized['medianComplexValue']
    human_readable_data = pd.DataFrame(scaler_standard.inverse_transform(standartized_data), columns=standartized_data.columns)
    return human_readable_data['medianComplexValue'].values.tolist()

def get_to_predict_data():
    return pd.DataFrame(data = {'totalRooms':[2491.0, 1966.0, 880.0, 1387.0, 1111.0],
                   'totalBedrooms':[474.0, 347.0, 129.0, 341.0, 111.0],
                   'complexInhibitants':[1098.0, 793.0, 322.0, 1074.0, 366.0],
                   'apartmentsNr':[468.0, 331.0, 126.0, 304.0, 114.0]})

def main():
    initial_data = load_data()
    initial_data = clean_data(initial_data)
    training_data = read_training_data()
    X_train, X_test, y_train, y_test = preprare_Data_for_training(training_data)
    
    linearModel = train_linear_model(X_train, y_train)
    mse, score = evaluate_model(linearModel, X_test, y_test)
    print(f'\nLINEAR mse = {mse}. Score = {score}\n')
    
    ridgeModel = train_ridge_model(X_train, y_train)
    mse, score = evaluate_model(ridgeModel, X_test, y_test)
    print(f'\nRIDGE mse = {mse}. Score = {score}\n')

    lassModel = train_lasso_model(X_train, y_train)
    mse, score = evaluate_model(lassModel, X_test, y_test)
    print(f'\nLasso mse = {mse}. Score = {score}\n')

    elasticNetMODEL = train_elastic_net_model(X_train, y_train)
    mse, score = evaluate_model(elasticNetMODEL, X_test, y_test)
    print(f'\nElasticNet mse = {mse}. Score = {score}\n\n')

    data_for_prediction = get_to_predict_data()
    predictions, standartized_data, normalized_data = use_model(ridgeModel, data_for_prediction)
    print(f'Predictions: {predictions}')
    human_readable_predictions = get_human_readable_predictions(predictions, training_data, standartized_data, normalized_data)
    print(f'\nHuman Readable Predictions: {human_readable_predictions}')
    # plot_predictions(human_readable_predictions, df['medianComplexValue'])
    
if __name__ == '__main__':
    main()