# Zillow Project
 
# Project Description
 
Zillow is a well known real estate marketing company. Users of the Zillow app are typically shopping for homes. In addition to property value, Zillow's database holds a wide variety of information useful to app users such as square feet, number of bathrooms, number of bedrooms, etc. An accurate property value is crucial for Zillow and its users because this is often the first filter in any user search. Predicting property value is challenging given changing markets and varying property features.
 
# Project Goal
 
* Discover how various property features relate to property value
* Use features to develop a machine learning model to predict property value
* This information will further our understanding of what features are associated with property value. In addition, building a model to accurately predict home value will help build and maintain trust with users that Zillow has the most accurate valuation of all competitors, and a home-buyer's first stop when shopping for a home.
 
# Initial Thoughts
 
My initial hypothesis is that square feet, number of bathrooms, number of bedrooms, year built, county, lot size, and whether a property has a pool or not will be related to property value.
 
# The Plan
 
* Aquire data from Codeup's SQL server
 
* Prepare data
   * Remove unneccessary columns
   * Ensure remaining columns are in the correct format
   * Search for and handle any null values
   * Search for and handle outliers
 
* Explore data in search of drivers of upsets
   * Answer the following initial questions
       * What is the average property value?
       * What features affect property value?
      
* Develop a Model to predict property value
   * Use features identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Type | Definition |
|:--------|:-----|:-----------|
|property_value (target)|integer|actual property value|
|bathrooms|float|number of bathrooms|
|bedrooms|integer|number of bedrooms|
|has_pool|integer|0 for no pool, 1 if the propert has a pool|
|squarefeet|integer|square footage of the interior of the home|
|lotsize_sqft|integer|square footage of the lot|
|year|integer|year the home was built|
|county|string|county where the property lies (LA, Orange, Ventura)
|Additional Features|Encoded and values for categorical data and scaled versions continuous data|
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from Codeup SQL server via code in this repo
3) Run notebook.
 
# Takeaways and Conclusions
* The average property value in this dataset is $298,893
* Features affecting property value (in rank order)
    1) squarefeet:   higher squarefeet        -> higher property_value
    2) bathrooms :   more bathrooms           -> higher property_value
    3) bedrooms  :   more bedrooms            -> higher property_value
    4) year      :   higher year (newer home) -> higher property value
    5) has_pool  :   properties WITH a pool   -> higher property value
    6) county    :   Orange/Ventura           -> higher property value
    7) lotsize_sqft: higher lotsize_sqft      -> higher property value

* The final model outperformed the baseline:
    * Baseline Root Mean Squared Error (RMSE) / R^2 : $298,893, 0.0
    * Model    Root Mean Squared Error (RMSE) / R^2 : $242,448, 0.33
    * Note: a lower RMSE and a higher R^2 is desirable.
    
# Recommendations
* Given the RMSE value is still so high, I would not use this model as the primary source of estimating home value
* That said, there are clear correlations between the features in my model and property value; it just needs a bit more refinement
    * In particular, incorporate location data
    * If possible, gather more data: for example school district, specific school minor children would attend, and performance metrics of those schools