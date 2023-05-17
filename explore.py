# IMPORTS
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

# FUNCTIONS
#defining a function for the final_report notebook to print out a nice visual of the target: property_value
def get_target_plot (df, target='property_value'):
    """
    This function will 
    - accept a dataframe and a target column which is continuous
    - make two subplots of the target: a boxplot and a histplot with the mean vertical line plotted in red
    - the function was initially coded for the Zillow project, thus the target is defaulted to 'property_value' 
    - returns nothing
    - can be made more pretty with some callouts and possibly putting another vertical line in for median
    """
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.boxplot(df[target])
    plt.title('Boxplot')
    plt.xticks(ticks=[0,200_000,400_000,600_000,800_000,1_000_000,1_200_000,1_400_000]
               ,labels=['0', '200', '400', '600', '800', '1,000', '1,200', '1,400'])
    plt.xlabel('Property Value in thousands of dollars')
    plt.axvline(x=df[target].mean(), color='r')
    # plt.text(412_000,-0.45, 'Mean', fontsize=10, color='red')
    plt.annotate('Mean', xy=(412_000, -0.35), xytext=(700_000, -0.32), arrowprops={'facecolor': 'red'})
    
    plt.subplot(1,2,2)
    sns.histplot(df[target])
    plt.title('Histogram')
    plt.xticks(ticks=[0,200_000,400_000,600_000,800_000,1_000_000,1_200_000,1_400_000]
               ,labels=['0', '200', '400', '600', '800', '1,000', '1,200', '1,400'])
    plt.xlabel('Property Value in thousands of dollars')
    plt.axvline(x=df[target].mean(), color='r')

    plt.suptitle(f'Distribution of property_value (Mean: ${int(df[target].mean()/1000)}K)', fontweight='bold')

    plt.show()
    
# defining a function for the final_report to plot two continuous variables (generaly y is the target: property_value)
def get_reg_plot(df, x, y='property_value'):
    """
    This function will
    - accept a dataframe and the x and y column names to plot
    - plot a regplot
    - return nothing
    """
    sns.regplot(data=df, x=x, y=y, line_kws={'lw': 1, 'color': 'red'})
    plt.show()
    return

# defining a function to plot a categorica variable vs the target which is continuous    
def get_box_plot(df, x, y='property_value'):
    """
    This function will
    - accept a dataframe and the x and y column names to plot
    - plot a boxplot
    - return nothing
    """
    sns.boxplot(data=df, x='has_pool', y='property_value')
    plt.show()
    return

# defining a function to print t, p from a pearsonr test    
def get_pearsonr(df, x, y='property_value'):
    """
    This function will 
    - accept a dataframe, and two columns: x and y
    - run a pearsonr stats test
    - print out the t, p statistics
    
    """
    t, p = pearsonr(df[x], df[y])
    print (f't = {t}')
    print (f'p = {p}')
    return

# defining a function to print t, p from a t test, f_oneway  
def get_has_pool_f_oneway(df, target='property_value'):
    """
    This function is specific to the has_pool column in the zillow database. It will 
    - accept a dataframe, and the target (default to 'property_value')
    - run a f_oneway stats test on the property_values of has_pool == 1 vs has_pool == 0
    - print out the t, p statistics
    
    """
    y1 = df[df.has_pool == 1][target]
    y2 = df[df.has_pool == 0][target]
    t, p = f_oneway(y1, y2)
    print (f't = {t}')
    print (f'p = {p}')
    return

# defining a function to help plot continuous variable pairs in the explore phase (pairplot is nice for this, too)  
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
    return

# defining a function to plot categorical vs continuous variables in the explore phase        
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
    return

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