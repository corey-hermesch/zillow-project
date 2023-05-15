import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import wrangle as w
import evaluate as ev

import warnings
warnings.filterwarnings('ignore')

# copying function from Misty to help get metrics for comparing models
def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def get_baseline_train_val_metrics(y_train, y_validate):
    """
    This function will
    - accept y_train and y_validate
    - calculate a baseline which is the mean of y_train
    - return RMSE and R^2 scores for train and validate
    """
    y_train_baseline = np.repeat(y_train.mean(), len(y_train))
    y_val_baseline = np.repeat(y_train.mean(), len(y_validate))
    
    RMSE_train, R2_train = metrics_reg(y_train, y_train_baseline)
    RMSE_val, R2_val = metrics_reg(y_validate, y_val_baseline)
    
    return RMSE_train, R2_train, RMSE_val, R2_val


def get_ols_train_val_metrics(X_train, y_train, X_val, y_val):
    """
    This function will 
    - accept X_train, y_train, X_val, y_val)
    - it will build an OLS model on whatever was sent in X_train
        - note same number of columns should be in X_train and X_val
    - it will fit the model on X_train
    - it will make predictions and return RMSE and R^2 scores for the predictions for train and val
    - returns RMSE_train, R2_train, RMSE_val, R2_val
    """
    
    # make the thing
    lr = LinearRegression()
    
    # fit the thing
    lr.fit(X_train, y_train)
    
    # use the thing
    y_pred_train = lr.predict(X_train)
    y_pred_val = lr.predict(X_val)
    
    RMSE_train, R2_train = metrics_reg(y_train, y_pred_train)
    RMSE_val, R2_val = metrics_reg(y_val, y_pred_val)
    
    return RMSE_train, R2_train, RMSE_val, R2_val

def get_lassolars_train_val_metrics(X_train, y_train, X_val, y_val, alpha=1):
    """
    This function will 
    - accept X_train, y_train, X_val, y_val)
    - it will build a LassoLars model on whatever was sent in X_train (alpha=1 default)
        - note same number of columns should be in X_train and X_val
    - it will fit the model on X_train
    - it will make predictions and return RMSE and R^2 scores for the predictions for train and val
    - returns RMSE_train, R2_train, RMSE_val, R2_val
    """
    
    # make the thing
    lasso = LassoLars(alpha=alpha)
    
    # fit the thing
    lasso.fit(X_train, y_train)
    
    # print coefficients (during test)
#     print(pd.Series(lasso.coef_, index=lasso.feature_names_in_))
    
    # use the thing
    y_pred_train = lasso.predict(X_train)
    y_pred_val = lasso.predict(X_val)
    
    RMSE_train, R2_train = metrics_reg(y_train, y_pred_train)
    RMSE_val, R2_val = metrics_reg(y_val, y_pred_val)
    
    return RMSE_train, R2_train, RMSE_val, R2_val

def get_polynomial_train_val_metrics(X_train, y_train, X_val, y_val, degrees=2):
    """
    This function will 
    - accept X_train, y_train, X_val, y_val)
    - it will build polynomial features with degrees=2 default value, then test on a LinearRegression model
        - note same number of columns should be in X_train and X_val
    - it will fit the model on X_train
    - it will make predictions and return RMSE and R^2 scores for the predictions for train and val
    - returns RMSE_train, R2_train, RMSE_val, R2_val
    """
    
    # make the "thing" so I can get new polynomial features to send in to a different thing
    pf = PolynomialFeatures(degree=2)

    #fit the thing
    pf.fit(X_train)

    # use the thing to make new feature values
    X_train_degree2 = pf.transform(X_train)
    X_val_degree2 = pf.transform(X_val)

    # make it
    pr = LinearRegression()

    # fit it
    pr.fit(X_train_degree2, y_train)

    # use it
    y_pred_train = pr.predict(X_train_degree2)
    y_pred_val = pr.predict(X_val_degree2)
    
    RMSE_train, R2_train = metrics_reg(y_train, y_pred_train)
    RMSE_val, R2_val = metrics_reg(y_val, y_pred_val)
    
    return RMSE_train, R2_train, RMSE_val, R2_val

def get_glm_train_val_metrics(X_train, y_train, X_val, y_val, power=0, alpha=0):
    """
    This function will 
    - accept X_train, y_train, X_val, y_val)
    - it will build Generalized Linea Model power=0, alpha=0 default value
        - note same number of columns should be in X_train and X_val
    - it will fit the model on X_train
    - it will make predictions and return RMSE and R^2 scores for the predictions for train and val
    - returns RMSE_train, R2_train, RMSE_val, R2_val
    """
    
    # make it
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit it
    glm.fit(X_train, y_train)

    # use it
    y_pred_train = glm.predict(X_train)
    y_pred_val = glm.predict(X_val)
    
    RMSE_train, R2_train = metrics_reg(y_train, y_pred_train)
    RMSE_val, R2_val = metrics_reg(y_val, y_pred_val)
    
    return RMSE_train, R2_train, RMSE_val, R2_val