### IMPORTS 

# Imports
from env import host, user, password
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

### FUNCTIONS 

def get_db_url(db_name, user=user, host=host, password=password):
    '''
    get_db_url accepts a database name, username, hostname, password 
    and returns a url connection string formatted to work with codeup's 
    sql database.
    Default values from env.py are provided for user, host, and password.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# generic function to get a sql pull into a dataframe
def get_mysql_data(sql_query, database):
    """
    This function will:
    - take in a sql query and a database (both strings)
    - create a connection url to mySQL database
    - return a df of the given query, connection_url combo
    """
    url = get_db_url(database)
    return pd.read_sql(sql_query, url)    

def get_csv_export_url(g_sheet_url):
    '''
    This function will
    - take in a string that is a url of a google sheet
      of the form "https://docs.google.com ... /edit#gid=12345..."
    - return a string that can be used with pd.read_csv
    '''
    csv_url = g_sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return csv_url

# defining a function to get zillow data from either a cached csv or the Codeup MySQL server
def get_zillow_data(sql_query= """
                        SELECT parcelid, bathroomcnt, bedroomcnt, poolcnt
                                , calculatedfinishedsquarefeet, lotsizesquarefeet, yearbuilt
                                , fips, regionidcity, regionidzip
                                , latitude, longitude, rawcensustractandblock, censustractandblock, taxvaluedollarcnt
                        FROM properties_2017
                        JOIN propertylandusetype USING (propertylandusetypeid)
                        JOIN (
                            SELECT parcelid, logerror, maxdates.transactiondate
                            FROM (
                                SELECT parcelid, MAX(transactiondate) as transactiondate
                                FROM predictions_2017
                                GROUP BY parcelid ) as maxdates
                            JOIN predictions_2017 using (parcelid)
                            WHERE predictions_2017.transactiondate = maxdates.transactiondate
                            ) AS target_transactions USING (parcelid)
                        WHERE (propertylandusedesc = 'Single Family Residential')
                    """
                    , filename="zillow.csv"):
    
    """
    This function will:
    -input 2 strings: sql_query, filename 
        default query selects specific columns from zillow database that I determined to be useful
        default filename "zillow.csv"
    - check the current directory for filename (csv) existence
      - return df from that filename if it exists
    - If csv doesn't exist:
      - create a df of the sql_query
      - write df to csv
      - return that df
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        print ("csv file found and read")
        return df
    else:
        url = get_db_url('zillow')
        df = pd.read_sql(sql_query, url)
        df.to_csv(filename, index=False)
        print ("csv file not found, data read from sql query, csv created")
        return df

# defining overall function to acquire and clean up zillow data
def wrangle_zillow():
    """
    This function will acquire and prep specific column data from zillow database. 
    Specifically it will:
    - 1. get data from database
    - 2. rename the columns to something more useful
    - 3. change has_pool nulls to 0
    - 3. drop the remaining nulls
    - 4. change the datatype to int for all numeric columns except bathrooms
    - 5. drop outliers (property_value > $1.5M and lotsize_sqft > 60000
    
    - returns a df that looks like this:
        - property_value - int (property_value is the target)
        - bathrooms - float
        - bedrooms - int
        - has_pool - int (0 or 1)
        - squarefeet - int
        - year - int
        - county - string
        - dummy columns for later modeling
            - county_Orange - int
            - county_Ventura - int
    """
    # first get the data from csv or sql
    df = get_zillow_data()
    
    #rename columns to something less unwieldy
    df.columns = ['parcelid', 'bathrooms', 'bedrooms', 'has_pool', 'squarefeet', 'lotsize_sqft', 'year'
                  , 'county', 'cityid', 'zip', 'lat', 'long', 'rawcensus', 'censustract_block', 'property_value']
    
    # has_pool has many nulls; I am making the assumption these properties do not have pools
    df.has_pool = df.has_pool.fillna(0)
    
    # decided to drop the rest of the nulls since it was < 3% of data
    df = df.dropna()
    
    # most numeric columns can/should be integers; exception was bathrooms which I left as a float
    intcolumns = ['bedrooms', 'has_pool', 'squarefeet', 'lotsize_sqft', 'year', 'county'
                  , 'cityid', 'zip', 'censustract_block', 'property_value']
    for col in intcolumns:
        df[col] = df[col].astype(int)
    
    # county really should be a categorical, since it represents a county in California
    df.county = df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})

    # if these are even useful, they will be useful in the format degrees.decimal_degrees
    df.lat = df.lat / 1_000_000
    df.long = df.long / 1_000_000
    
    # for now, I will keep only these columns. Maybe the others will have use later
    keep = ['property_value', 'bathrooms', 'bedrooms', 'has_pool', 'squarefeet', 'lotsize_sqft', 'year', 'county']
    df = df[keep]
    
    # After univariate visualization, I want to purge a few outliers
    # in total, 4371 rows were cut out of 52320; ~8.4% of the data
    df = df[df.property_value <= 1_500_000]
    df = df[df.lotsize_sqft <= 60_000]
    
    # make dummy columns for the categorical column, 'county'
    dummy_df = pd.get_dummies(df[['county']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


def split_function(df, target_var=''):
    """
    This function will
    - take in a dataframe (df)
    - option to accept a target_var in string format
    - split the dataframe into 3 data frames: train (60%), validate (20%), test (20%)
    -   while stratifying on the target_var (if present)
    - And finally return the three dataframes in order: train, validate, test
    """
    if len(target_var)>0:
        train, test = train_test_split(df, random_state=42, test_size=.2, stratify=df[target_var])
        train, validate = train_test_split(train, random_state=42, test_size=.25, stratify=train[target_var])
    else:
        train, test = train_test_split(df, random_state=42, test_size=.2)
        train, validate = train_test_split(train, random_state=42, test_size=.25)        
        
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

# copying this function to assist with choosing what type of scaler to use in the future
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    """
    This function will
    - plot some charts of data before scaling next to data after scaling
    - returns nothing
    - example function call:
    
        # call function with minmax
        visualize_scaler(scaler=MinMaxScaler(), 
                         df=train, 
                         columns_to_scale=to_scale, 
                         bins=50)
    """
    #create subplot structure
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(12,12))

    #copy the df for scaling
    df_scaled = df.copy()
    
    #fit and transform the df
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    #plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()

# defining a function to get scaled data using MinMaxScaler
def get_minmax_scaled (train, validate, test, columns_to_scale):
    """ 
    This function will
    - accept train, validate, test, and which columns are to be scaled
    - makes minmax scaler, fits scaler on train columns
    - returns 3 scaled dataframes; one for train/validate/test
    """
    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # make and fit minmax scaler
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    # use the thing
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    return train_scaled, validate_scaled, test_scaled