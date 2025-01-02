# Title: FinalProject_FIN550
# Group Name: Decision Three
# Authors: Dwijesh Reddy NallappaReddy, Nikhil Reddy Satti, Rushali Naikoti

# Time to run the whole file: 35-40 mins 
# --------------------------------------------------


#load libraries

library(randomForest)
library(glmnet)
library(ggplot2)
library(tidyverse)
library(caret)
library(lightgbm)
library(corrplot)
library(scales)

# Data Extraction 

#load the data
df <- read.csv("historic_property_data.csv")

df <- df %>%
  select(
    char_age, char_air, char_attic_type, char_bldg_sf, char_bsmt,
    char_bsmt_fin, char_ext_wall, char_gar1_att, char_gar1_size, char_rooms,
    econ_midincome, econ_tax_rate, geo_black_perc, geo_fips,
    geo_other_perc, geo_property_zip, geo_school_elem_district, geo_tract_pop, geo_white_perc, meta_certified_est_bldg,
    meta_certified_est_land, sale_price
  )


# first six rows
head(df)

# number of rows with missing values 
sum(is.na(df))

# remove rows with any missing values 
df <- na.omit(df)

# dimension 
dim(df)

# number of missing values 
sum(is.na(df))

#Data-Preprocessing

#Converting Categorical Variables into Factored Levels

df$char_air <- as.factor(df$char_air)
df$char_attic_type <- as.factor(df$char_attic_type)
df$char_bsmt <- as.factor(df$char_bsmt)
df$char_bsmt_fin <- as.factor(df$char_bsmt_fin)
df$char_ext_wall <- as.factor(df$char_ext_wall)
df$char_gar1_att <- as.factor(df$char_gar1_att)
df$char_gar1_size <- as.factor(df$char_gar1_size)
df$geo_fips <- as.factor(df$geo_fips)
df$geo_property_zip <- as.factor(df$geo_property_zip)
df$geo_school_elem_district <- as.factor(df$geo_school_elem_district)



# Create a duplicate dataframe to track factored as numeric
df_dataset <- df

#Making the factored variables into numeric
df_dataset$char_air <- as.numeric(df$char_air)
df_dataset$char_attic_type <- as.numeric(df$char_attic_type)
df_dataset$char_bsmt <- as.numeric(df$char_bsmt)
df_dataset$char_bsmt_fin <- as.numeric(df$char_bsmt_fin)
df_dataset$char_ext_wall <- as.numeric(df$char_ext_wall)
df_dataset$char_gar1_att <- as.numeric(df$char_gar1_att)
df_dataset$char_gar1_size <- as.numeric(df$char_gar1_size)
df_dataset$geo_fips <- as.numeric(df$geo_fips)
df_dataset$geo_property_zip <- as.numeric(df$geo_property_zip)
df_dataset$geo_school_elem_district <- as.numeric(df$geo_school_elem_district)

# Creating Correlation Matrix

correlation_matrix <- cor(df_dataset)

filtered_matrix <- correlation_matrix
filtered_matrix[abs(filtered_matrix) <= 0.5] <- 0

# Plot the correlation matrix
corrplot(filtered_matrix, method = "color")

#Normalizing all the columns

numeric_cols <- c("char_age", "char_air", "char_attic_type", "char_bldg_sf",
                  "char_bsmt", "char_bsmt_fin", "char_ext_wall", "char_gar1_att",
                  "char_gar1_size", "char_rooms", "econ_midincome", "econ_tax_rate",
                  "geo_black_perc", "geo_fips", "geo_other_perc","geo_property_zip", "geo_school_elem_district",
                  "geo_tract_pop", "geo_white_perc", "meta_certified_est_bldg", "meta_certified_est_land")

df_dataset[numeric_cols] <- lapply(df_dataset[numeric_cols], function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
})

set.seed(6)  # Set seed for reproducibility
index <- createDataPartition(df_dataset$sale_price, p = 0.8, list = FALSE)

# Create train and test datasets
train_data <- df_dataset[index, ]
test_data <- df_dataset[-index, ]

# Create the formula for the model
formula_nn <- as.formula("sale_price ~ char_age + char_air + char_attic_type + char_bldg_sf + char_bsmt + char_bsmt_fin + char_ext_wall + char_gar1_att + char_gar1_size + char_rooms + econ_midincome + econ_tax_rate + geo_black_perc + geo_fips + geo_other_perc + geo_property_zip + geo_school_elem_district + geo_tract_pop + geo_white_perc + meta_certified_est_bldg + meta_certified_est_land")

#Linear Regression Model

# Defining the linear model
linear_model <- lm(formula_nn , data = train_data)


# Display the summary of the linear regression model
summary(linear_model)

#Prediction of values and MSE calculation of the linear model


#TestData Prediction
predictions_test_linear <- predict(linear_model, newdata = test_data)
mse_test_linear <- mean((test_data$sale_price - predictions_test_linear)^2)

#TrainingData Prediction
predictions_train_linear <- predict(linear_model, newdata = train_data)
mse_train_linear <- mean((train_data$sale_price - predictions_train_linear)^2)

#Checking how accurate model is compared to training data
accuracy_linear = (mse_test_linear-mse_train_linear)/mse_train_linear

# Lasso Regression Model
x <- model.matrix(formula_nn, data = train_data)
y <- train_data$sale_price
lasso_model <- cv.glmnet(x, y)
lambda.small <- min(lasso_model$lambda)
lambda.best <- lasso_model$lambda.min
lambda.large <- max(lasso_model$lambda)

lambda_seq <- seq(lambda.small, lambda.large, length = 100)

# Calculate CV MSE for lambda.large, lambda.small, and lambda.best
cv_mse_large <- cv.glmnet(x, y, alpha = 1, lambda = lambda_seq)$cvm[which(lambda_seq == lambda.large)]
cv_mse_small <- cv.glmnet(x, y, alpha = 1, lambda = lambda_seq)$cvm[which(lambda_seq == lambda.small)]
cv_mse_best <- cv.glmnet(x, y, alpha = 1)$cvm

# Fit LASSO model across the range of lambda values
lasso_coefficients <- glmnet(x, y, alpha = 1, lambda = lambda_seq)

# Plotting the coefficient paths
plot(lasso_coefficients, xvar = "lambda", label = TRUE)
title("Coefficient Paths for LASSO Regression")

#Prediction of values and MSE calculation of the lasso regression model


#TestData Prediction
predictions_test_lasso <-  predict(lasso_model, s = lambda.best, newx = x)
mse_test_lasso <- mean((test_data$sale_price - predictions_test_lasso)^2)

#TrainingData Prediction
predictions_train_lasso <- predict(lasso_model, s = lambda.best, newx = x)
mse_train_lasso <- mean((train_data$sale_price - predictions_train_lasso)^2)

#Checking how accurate model is compared to training data
accuracy_lasso = (mse_test_lasso-mse_train_lasso)/mse_train_lasso

#Random Forest Model
random_forest_model <- randomForest(formula_nn, data = train_data, mtry = 21, importance =TRUE)
# variable importance 
importance(random_forest_model)

# variable importance plot
varImpPlot(random_forest_model)

#Prediction of values and MSE calculation of the random forest model

#TestData Prediction
predictions_test_randomforest <- predict(random_forest_model, newdata = test_data)
mse_test_randomforest <- mean((test_data$sale_price - predictions_test_randomforest)^2)

#TrainingData Prediction
predictions_train_randomforest <- predict(random_forest_model, newdata = train_data)
mse_train_randomforest <- mean((train_data$sale_price - predictions_train_randomforest)^2)

#Checking how accurate model is compared to training data
accuracy_randomforest = (mse_test_randomforest-mse_train_randomforest)/mse_train_randomforest

#Loading prediction data

prediction_df <- read.csv("predict_property_data.csv")

prediction_df <- prediction_df %>%
  select(
    char_age, char_air, char_attic_type, char_bldg_sf, char_bsmt,
    char_bsmt_fin, char_ext_wall, char_gar1_att, char_gar1_size, char_rooms,
    econ_midincome, econ_tax_rate, geo_black_perc, geo_fips,
    geo_other_perc, geo_property_zip, geo_school_elem_district, geo_tract_pop, geo_white_perc, meta_certified_est_bldg,
    meta_certified_est_land, pid
  )

#Data Clean-up
# number of rows with missing values 
sum(is.na(prediction_df))
# remove rows with any missing values 
prediction_df <- na.omit(prediction_df)
# number of missing values 
sum(is.na(prediction_df))

#Data Pre-processing

prediction_df$char_air <- as.factor(prediction_df$char_air)
prediction_df$char_attic_type <- as.factor(prediction_df$char_attic_type)
prediction_df$char_bsmt <- as.factor(prediction_df$char_bsmt)
prediction_df$char_bsmt_fin <- as.factor(prediction_df$char_bsmt_fin)
prediction_df$char_ext_wall <- as.factor(prediction_df$char_ext_wall)
prediction_df$char_gar1_att <- as.factor(prediction_df$char_gar1_att)
prediction_df$char_gar1_size <- as.factor(prediction_df$char_gar1_size)
prediction_df$geo_fips <- as.factor(prediction_df$geo_fips)
prediction_df$geo_property_zip <- as.factor(prediction_df$geo_property_zip)
prediction_df$geo_school_elem_district <- as.factor(prediction_df$geo_school_elem_district)

prediction_df$char_air <- as.numeric(prediction_df$char_air)
prediction_df$char_attic_type <- as.numeric(prediction_df$char_attic_type)
prediction_df$char_bsmt <- as.numeric(prediction_df$char_bsmt)
prediction_df$char_bsmt_fin <- as.numeric(prediction_df$char_bsmt_fin)
prediction_df$char_ext_wall <- as.numeric(prediction_df$char_ext_wall)
prediction_df$char_gar1_att <- as.numeric(prediction_df$char_gar1_att)
prediction_df$char_gar1_size <- as.numeric(prediction_df$char_gar1_size)
prediction_df$geo_fips <- as.numeric(prediction_df$geo_fips)
prediction_df$geo_property_zip <- as.numeric(prediction_df$geo_property_zip)
prediction_df$geo_school_elem_district <- as.numeric(prediction_df$geo_school_elem_district)

#Normalizing the data
prediction_df[numeric_cols] <- lapply(prediction_df[numeric_cols], function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
})

#Predicting the sale prices

prediction_data <- prediction_df %>%
  select(
    char_age, char_air, char_attic_type, char_bldg_sf, char_bsmt,
    char_bsmt_fin, char_ext_wall, char_gar1_att, char_gar1_size, char_rooms,
    econ_midincome, econ_tax_rate, geo_black_perc, geo_fips,
    geo_other_perc, geo_property_zip, geo_school_elem_district, geo_tract_pop, geo_white_perc, meta_certified_est_bldg,
    meta_certified_est_land
  )


prediction_df$assessed_value <- predict(random_forest_model, newdata = prediction_data)

#Saving the data

prediction_data <- prediction_df %>%
  select(pid,assessed_value
  )

summary(prediction_data$assessed_value)


#Modifying the missing NA records
numbers_list <- seq(1, 10000)
not_present_values <- numbers_list[!numbers_list %in% prediction_data$pid]

missing_records <- data.frame(
  pid = not_present_values,
  assessed_value = rep(mean(prediction_data$assessed_value, na.rm = TRUE), length(not_present_values))
)

prediction_data <- rbind(prediction_data, missing_records)

prediction_data <- prediction_data[order(prediction_data$pid), ]


# Use write.csv() to write the data frame to a CSV file
write.csv(prediction_data, "assessed_value.csv", row.names = FALSE)

# Plot for Asssessed Values Distribution

class(prediction_data$assessed_value)

hist(
  prediction_data$assessed_value,
  main = "Frequency Distribution of Predicted Values",
  xlab = "Predicted Values in Dollars",
  ylab = "Frequency",
  col = "skyblue",    
  border = "black",  
  breaks = 100        
)

