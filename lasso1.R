# Loading required R packages
library(tidyverse)
library(caret)
library(glmnet)

# Preparing the data

# Load the data and remove NAs
data("PimaIndiansDiabetes2", package = "mlbench")
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)
# Inspect the data
sample_n(PimaIndiansDiabetes2, 3)
# Split the data into training and test set
set.seed(123)
training.samples <- PimaIndiansDiabetes2$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- PimaIndiansDiabetes2[training.samples, ]
test.data <- PimaIndiansDiabetes2[-training.samples, ]

# Computing penalized logistic regression
#     Additionnal data preparation

# Dumy code categorical predictor variables
x <- model.matrix(diabetes~., train.data)[,-1]
# Convert the outcome (class) to a numerical variable
y <- ifelse(train.data$diabetes == "pos", 1, 0)

# R functions
# We’ll use the R function glmnet() [glmnet package] for computing penalized logistic regression.
# The simplified format is as follow:

glmnet(x, y, family = "binomial", alpha = 1, lambda = NULL)


# x: matrix of predictor variables
# y: the response or outcome variable, which is a binary variable.
# family: the response type. Use “binomial” for a binary outcome variable
# alpha: the elasticnet mixing parameter. Allowed values include:
#   “1”: for lasso regression
# “0”: for ridge regression
# a value between 0 and 1 (say 0.3) for elastic net regression.
# lamba: a numeric value defining the amount of shrinkage. Should be specify by analyst.

# In penalized regression, you need to specify a constant lambda to adjust 
# the amount of the coefficient shrinkage. The best lambda for your data, 
# can be defined as the lambda that minimize the cross-validation prediction error rate. 
# This can be determined automatically using the function cv.glmnet().


# Quick start R code

library(glmnet)
# Find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
# Fit the final model on the training data
model <- glmnet(x, y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
# Display regression coefficients
coef(model)
# Make predictions on the test data
x.test <- model.matrix(diabetes ~., test.data)[,-1]
probabilities <- model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy
observed.classes <- test.data$diabetes
mean(predicted.classes == observed.classes)


# Compute lasso regression

library(glmnet)
set.seed(123)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)

cv.lasso$lambda.min

cv.lasso$lambda.1se

coef(cv.lasso, cv.lasso$lambda.min)

coef(cv.lasso, cv.lasso$lambda.1se)

# Compute the final lasso model:
#  Compute the final model using lambda.min:

# Final model with lambda.min
lasso.model <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.min)
# Make prediction on test data
x.test <- model.matrix(diabetes ~., test.data)[,-1]
probabilities <- lasso.model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy
observed.classes <- test.data$diabetes
mean(predicted.classes == observed.classes)


# Final model with lambda.1se
lasso.model <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.1se)
# Make prediction on test data
x.test <- model.matrix(diabetes ~., test.data)[,-1]
probabilities <- lasso.model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy rate
observed.classes <- test.data$diabetes
mean(predicted.classes == observed.classes)


# Compute the full logistic model


# Fit the model
full.model <- glm(diabetes ~., data = train.data, family = binomial)
# Make predictions
probabilities <- full.model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy
observed.classes <- test.data$diabetes
mean(predicted.classes == observed.classes)


# Discussion
# This chapter described how to compute penalized logistic regression model in R. Here, we focused on lasso model, but you can also fit the ridge regression by using alpha = 0 in the glmnet() function. For elastic net regression, you need to choose a value of alpha somewhere between 0 and 1. This can be done automatically using the caret package. See Chapter @ref(penalized-regression).
# 
# Our analysis demonstrated that the lasso regression, using lambda.min as the best lambda, results to simpler model without compromising much the model performance on the test data when compared to the full logistic model.
# 
# The model accuracy that we have obtained with lambda.1se is a bit less than what we got with the more complex model using all predictor variables (n = 8) or using lambda.min in the lasso regression. Even with lambda.1se, the obtained accuracy remains good enough in addition to the resulting model simplicity.
# 
# This means that the simpler model obtained with lasso regression does at least as good a job fitting the information in the data as the more complicated one. According to the bias-variance trade-off, all things equal, simpler model should be always preferred because it is less likely to overfit the training data.
# 
# For variable selection, an alternative to the penalized logistic regression techniques is the stepwise logistic regression described in the Chapter @ref(stepwise-logistic-regression).
# 
# 
# 

Referencia

http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/