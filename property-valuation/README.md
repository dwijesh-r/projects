# Statistical-Modeling-for-Property-Valuation

This repository has the code for the project which estimates the pricing of places in Cook County.

Aim: Utilize existing data about House Pricing to create a regression model to predict the prices for a new data of housing in the same county.

Software Used: R

Approach:

Step 1: **Data Cleaning**

Our data set was quite huge with about 60 independent variables, of which I intended to use around 20 highly explainable variables. In the process I removed the columns with high number of outliers and null vaules. Additionally, quite a few columns were categorical in nature, which ever column was categorical was converted into factored-levels there by given a numerical value of importance to make the data crunchable for regression analysis.

Step 2: **Variable Selection**

Post data cleaning, I wanted to reduce the number of variables needed for regression analysis. Lasso Regression seemed like a good approach to see the best set of variables. Which resulted in approximately 50 variables still being there. In the later step each column's distribution was plotted to check the distribution to understand if a particular column is skewed. All the columns which were skewed were ignored in the regression analysis.

Step 3: **Modelling**

Once the variable selection was completed, a linear regression model was designed with the variables to render a model which would suggest the explainability of outcome using these variables. The model rendered an R-Squared value around 80% which was a good enough explainability percentage, considering the fact that higher R-Squared Value might result in overfitted model.

Post validation of the explainability, multiple models have been designed; like Linear, Lasso, Random Forest, LightGBM. Of these models LightGBM was extremely good but was computationally quite heavy taking approximately 10 hours to create the model, following the LightGBM was Random Forest which rendered a lower MSE within the remaining 3 models.

Step 4: **Prediction**

The data of the new dataset was parsed through the same data cleaning process as before. Post which the cleaned data was sent into the Random Forest Model to give an estimated prediction value. Additionally, there were a few data points which lacked the sufficient amount of data to be parsed into the model they have been imputed with the mean of the net outcomes.

All the code is available in the repository.
