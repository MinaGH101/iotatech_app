import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from statsmodels.tsa.seasonal import STL
import pywt
from sqlalchemy import create_engine
from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer

    

def data_preparation1(data, temp_params):
    data1 = data.replace(0, np.nan)
    data2 = data1.dropna(axis=1, how='all')
    d = data2.copy()

    X = d.iloc[: , 13:]
    X = X[temp_params]

    return X, d

def data_preparation2(X, y_col, d, ts):
    imputer = KNNImputer(n_neighbors=3)
    X = pd.DataFrame(imputer.fit_transform(X), columns= X.columns)
    # X_s = StandardScaler(with_mean=False).fit_transform(X)
    # X = pd.DataFrame(X_s, columns=X.columns)
    y = d[y_col]
    y = y.interpolate(method='linear', limit_direction='both')
    Fe_total = d['Fe_total']
    # X['Fe_total'] = Fe_total

    X_train = X.iloc[:ts]
    X_test = X.iloc[ts:]
    y_train = y.iloc[:ts]
    y_test = y.iloc[ts:]

    return X_train, y_train, X_test, y_test, Fe_total, X, y

def processing(y_test, y_pred, minn, maxx, ws, d):

    if ws:
        y_test = savgol_filter(y_test, ws, d)
        # y_pred = savgol_filter(y_pred, ws, d)

    y_test1 = list(y_test)
    y_pred = list(y_pred)
    for i, v in enumerate(y_test1):
        if v < minn:
            y_test1[i] = minn
        if v > maxx:
            y_test1[i] = maxx

    for i, v in enumerate(y_pred):
        if v < minn:
            y_pred[i] = minn
        if v > maxx:
            y_pred[i] = maxx
    # y_pred = remove_outliers(data=pd.DataFrame(y_pred, columns=['%Metallization']), ws=5, factor=0.8, method='median')
    # y_pred = y_pred.replace(np.nan, 91)
    return y_test1, y_pred