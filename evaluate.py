# IMPORTS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures

import wrangle as w

import warnings
warnings.filterwarnings('ignore')

# FUNCTIONS

# make a function to plot x, y, target mean, yhat
def get_feature_target_plot(df):
    """
    This function will
    - accept a dataframe with columns x, y, yhat (should be all continuous variables)
        - x = independent variable
        - y = target variable
        - yhat = predicted y value from some model
    - plots data (scatterplot), baseline (mean of target), and predicted line (yhat)
    """
    # store the column names for later
    cols = df.columns
    
    # I did this because in the notebook resetting the columns, even inside the function messed it up
    new_df = df.copy()
    new_df.columns=['x','y','yhat']

    # store the mean for use in the plot code below
    baseline = new_df.y.mean()
    
    # make the plot
    plt.scatter(new_df.x, new_df.y)
    plt.axhline(baseline, ls=':')
    plt.plot(new_df.x, new_df.yhat)
    plt.title('OLS linear model')
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()

# make a function to print out regression model metrics
def get_reg_model_metrics(y, yhat):
    """
    This function will
    - accept two series y, yhat (target, predicted target values)
    - print these values for the model and for the baseline (mean of y)
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - SSE: Sum of Squared Errors
    - prints these values for model compared to baseline
        - ESS: Explained Sum of Squares
        - TSS: Total Sum of Squares
        - R2: R-squared; the ratio of the Explained Sum of Squares (ESS) to the Total Sum of Squares (TSS)
            - aka the Explained Variance aka the fraction of the variance error explained by model
            - an R2 value closer to 1.0 is better; closer to 0.0 is worse
        """
    # put the two series into a dataframe and add the necessary columns
    df = pd.concat([y, yhat], axis=1)
    df.columns=['y','yhat']
    df['yhat_baseline'] = df.y.mean()
    df['residuals'] = df.yhat - df.y
    df['residuals_baseline'] = df.yhat_baseline - df.y
    
    # calculate model metrics
    MSE_model = mean_squared_error(df.y, df.yhat)
    RMSE_model = MSE_model ** .5
    SSE_model = MSE_model * len(df)
    
    # calculate baseline metrics
    MSE_baseline = mean_squared_error(df.y, df.yhat_baseline)
    RMSE_baseline = MSE_baseline ** .5
    SSE_baseline = MSE_baseline * len(df)
    
    # calculate the comparison metrics to get R^2
    ESS = sum((df.yhat - df.y.mean())**2)
    TSS = ESS + SSE_model
    R2 = ESS/TSS
    
    # put it all into a dataframe for printing
    results_df = pd.DataFrame({'metric': ['MSE', 'RMSE', 'SSE'],
                               'model_error': [MSE_model, RMSE_model, SSE_model],
                               'baseline_error': [MSE_baseline, RMSE_baseline, SSE_baseline]})
    display(results_df)
    print(f'ESS = {ESS}')
    print(f'TSS = {TSS}')
    print()
    print(f'R^2 (ESS/TSS) = {R2}')

# make a function to plot residuals
def plot_residuals(y, yhat):
    """
    This function will
    - accept two series: y, yhat (should be all continuous variables)
        - y = target variable
        - yhat = predicted y value from some model
    - prints 2 subplots, first is the model residuals, second is the baseline residuals
    """
    # put the two series into a dataframe and add the necessary columns
    df = pd.concat([y, yhat], axis=1)
    df.columns=['y','yhat']
    df['yhat_baseline'] = df.y.mean()
    df['residuals'] = df.yhat - df.y
    df['residuals_baseline'] = df.yhat_baseline - df.y
    
    # make two subplots for display side-by-side: Model Residuals and Baseline Residuals
    plt.figure(figsize=(16,7))
    plt.subplot(1,2,1)
    plt.scatter(df.y, df.residuals)
    plt.axhline(0, ls=':', color='red')
    plt.title('Model Residuals')
    plt.xlabel('y')
    plt.ylabel('yhat - y')
    
    plt.subplot(1,2,2)
    plt.scatter(df.y, df.residuals_baseline)
    plt.axhline(0, ls=':', color='red')
    plt.title('Baseline Residuals')
    plt.xlabel('y')
    plt.ylabel('yhat_baseline - y')
    
    plt.show()

# make a function to get regression error metrics for the model (for exercise instructions)
def regression_errors(y, yhat):
    """
    This function will
    - accpet two series: y, yhat (target value, predicted target value)
    - returns 
        - SSE: Sum of Squared Errors
        - ESS: Explained Sum of Squares
        - TSS: Total Sum of Squares
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
    """
    # put the two series into a dataframe and add yhat_baseline
    df = pd.concat([y, yhat], axis=1)
    df.columns=['y','yhat']
    df['yhat_baseline'] = df.y.mean()

    # calculate model metrics
    MSE_model = mean_squared_error(df.y, df.yhat)
    RMSE_model = MSE_model ** .5
    SSE_model = MSE_model * len(df)
    
    # calculate comparison metrics between the model and the baseline
    ESS = sum((df.yhat - df.y.mean())**2)
    TSS = ESS + SSE_model
    R2 = ESS/TSS
    
    return SSE_model, ESS, TSS, MSE_model, RMSE_model

# make a function to get baseline error metrics (for exercise instructions)
def baseline_mean_errors(y):
    """
    This function will
    - accpet one series: y (target value)
    - returns (for baseline model of the mean of y)
        - SSE: Sum of Squared Errors
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
    """
    # make a dataframe of y and add the yhat_baseline column
    df = pd.DataFrame(y)
    df['yhat_baseline'] = y.mean()
    df.columns=['y','yhat_baseline']
    
    # calculate baseline metrics
    MSE_baseline = mean_squared_error(df.y, df.yhat_baseline)
    RMSE_baseline = MSE_baseline ** .5
    SSE_baseline = MSE_baseline * len(df)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

# make a function to ask whether the model beats baseline
def better_than_baseline(y, yhat):
    """
    This function will
    - accpet two series: y, yhat (target value, predicted target value)
    - returns True if yhat beats the baseline of the mean of y, False otherwise
    """
    # get metrics from the above functions
    SSE_model, ESS, TSS, MSE_model, RMSE_model = regression_errors(y, yhat)
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    # it really doesn't matter which metric we use for this test,
    #   as long as it's the same metric from the model and from the baseline    
    if (RMSE_model < RMSE_baseline): 
        return True
    else:
        return False

# defining a function to plot visualization of how well to the top model (Polynomial Regression) performed
def plot_poly_residuals(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    This function will
    - accept X_train_scaled, y_train, X_test_scaled, y_test
    - build a Polynomial Regression model with degrees=2
    - call plot_residuals to visualize how well the model predicted the values
    """
    # make the "thing" so I can get new polynomial features to send in to a different thing
    pf = PolynomialFeatures(degree=2)

    #fit the thing
    pf.fit(X_train_scaled)

    # use the thing to make new feature values
    X_train_degree2 = pf.transform(X_train_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)

    # make the linear regression model to be used with polynomial features
    pr = LinearRegression()

    # fit it on train
    pr.fit(X_train_degree2, y_train)

    # use it on test to predict test target
    y_pred = pr.predict(X_test_degree2)
    
    # make y_pred a Series fro plotting
    y_pred = pd.Series(y_pred)
    
    # plot residuals
    plot_residuals(y_test, y_pred)
    
    return

