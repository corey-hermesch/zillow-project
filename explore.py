# IMPORTS
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

# FUNCTIONS
def plot_variable_pairs(train, cols):
    """
    This function is specific to zillow. It will
    - accept the train dataframe
    - accpt cols: the continuous columns to visualize with the target at the end
    - only look at a sample of 1000 to keep the run time reasonable
    - do sns.lmplot for the target variable vs each feature
    """

    sample = train.sample(10000, random_state=42)

    for i, col in enumerate(cols[:-1]):
        sns.lmplot(data=sample, x=col, y=cols[-1])
        plt.title(f'{cols[-1]} vs {col}')
        plt.show()

def plot_categorical_and_continuous_vars(train, cols_contin, cols_cat):
    """
    This function will
    - plot 3 plots (boxen, violin, and box) for each categorical variable vs each continuous variable
    - accepts a dataframe (train), a list of continuous column names (cols_contin),
      and a list of categorical column names (cols_cat)
    - prints all the plots
    - returns nothing
    """
    # set sample to something that will run in a reasonable amount of time
    sample = train.sample(10000, random_state=42)

    for cat in cols_cat:

        for col in cols_contin:

            sns.boxenplot(data=sample, x=cat, y=col)
            plt.title(f'{cat} vs. {col}, boxen')
            plt.show()

            sns.violinplot(data=sample, x=cat, y=col)
            plt.title(f'{cat} vs. {col}, violin')
            plt.show()

            sns.boxplot(data=sample, x=cat, y=col)
            plt.title(f'{cat} vs. {col}, boxplot')
            plt.show()

# define a function to return what the kbest features are
def get_kbest_multi (X_train_scaled, y_train):
    """
    This function will
    - accept X_train_scaled, a dataframe with scaled columns ready to check for which of those columns (features) 
        will be most useful to predict the values in y_train which is the target
    - return a dataframe with results of top features, iterating over k = 1 to number of columns
    """
    # Initialize col_list to capture rank-ordered columns (features)
    col_list = []
    
    # loop through checking for k best columns where k = 1 - n (number of columns)
    n = len(X_train_scaled.columns.to_list())
    for i in range(0, n):
 
        # make the thing and fit the thing
        f_selector = SelectKBest(f_regression, k=i+1)
        f_selector.fit(X_train_scaled, y_train)

        # get the mask so we know which are the k-best features
        feature_mask = f_selector.get_support()
        
        # code to add the next best feature to col_list
        for c in X_train_scaled[X_train_scaled.columns[feature_mask]].columns:
            if c not in col_list:
                col_list = col_list + [c]
    
    # make and return dataframe with results
    rank = list(range(1,len(col_list)+1))
    scores = f_selector.scores_
    scores = sorted(scores, reverse=True)
    results_df = pd.DataFrame({'Feature':col_list, 
                               'KBest Rank': rank, 
                               'KBest Scores': scores})
    return results_df

# define a function to get the RFE (Recursive Feature Engineering) best features
#  NOTE for later: See Amanda's code. it was shorter and probably better than this
def get_rfe_multi (X_train_scaled, y_train):
    """
    This function will
    - accept X_train_scaled, a dataframe with scaled columns ready to check for which of those columns (features) will be most useful to predict the values in y_train which is the target
    - return a dataframe with results of top features, iterating over k = 1 to number of columns
    """
 
    # initialize LinearRegression model
    lr = LinearRegression()

    # make the thing and fit the thing
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X_train_scaled, y_train)

    feature_rank = rfe.ranking_
    results_df = pd.DataFrame({'Feature': X_train_scaled.columns,
                                   'RFE Rank': feature_rank})
    return results_df.sort_values('RFE Rank')

# function from our exercise to select the kbest feature(s)
def select_kbest (X_train_scaled, y_train, k):
    """
    This function will
    - accept 
        -- X_train_scaled, a dataframe with scaled columns ready to check for which of those columns (features) 
        will be most useful to predict the values in y_train which is the target
        -- y_train, the target series
        -- k, the number of top features to return
    - makes and fits a SelectKBest object to evaluate which features are the best
    - returns a list with the column names of the top k columns (features)
    """
     
    # make the thing and fit the thing
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X_train_scaled, y_train)

    # get the mask so we know which are the k-best features
    feature_mask = f_selector.get_support()
        
    # get the columns associated with the top k features
    results_list = X_train_scaled[X_train_scaled.columns[feature_mask]].columns.to_list()

    return results_list

# function from our exercise to select the RFE-best feature(s)
def select_rfe (X_train_scaled, y_train, k):
    """
    This function will
    - accept 
        -- X_train_scaled, a dataframe with scaled columns ready to check for which of those columns (features) 
        will be most useful to predict the values in y_train which is the target
        -- y_train, the target series
        -- k, the number of top features to return
    - makes and fits a RFE (Recursive Feature Elimination) object with a LinearRegressiion model
        to evaluate which features are the best
    - returns a list with the column names of the top k columns (features)
    """
    
    # initialize LinearRegression model
    lr = LinearRegression()
    
    # make the thing:
    #  create RFE (Recursive Feature Elimination) object
    #  indicating lr model and number of features to select = k
    rfe = RFE(lr, n_features_to_select=k)
    
    # fit the thing:
    rfe.fit(X_train_scaled, y_train)
    
    # make a mask to select columns
    feature_mask = rfe.support_
    
    results_list = X_train_scaled[X_train_scaled.columns[feature_mask]].columns.to_list()
    
    return results_list