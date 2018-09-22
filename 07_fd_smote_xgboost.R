library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caret)
install.packages("DMwR")
library(DMwR) # smote
library(xgboost)
library(Matrix)
library(reshape) #melt
#install.packages("pROC")
library(pROC) # AUC

ccdata <- read.csv("data/creditcard.csv")

# Checking the data. Variables are normalized, except Amount and Time. Remove Time variable, maybe normalize Amount if it gets too much importance. Very imbalanced dataset.

table(ccdata$Class)
ccdata$Time <- NULL

# Split data
# Split into train, cv and test.

set.seed(1900)
inTrain <- createDataPartition(y = ccdata$Class, p = .6, list = F)
train <- ccdata[inTrain,]
testcv <- ccdata[-inTrain,]
inTest <- createDataPartition(y = testcv$Class, p = .5, list = F)
test <- testcv[inTest,]
cv <- testcv[-inTest,]
train$Class <- as.factor(train$Class)
rm(inTrain, inTest, testcv)

# SMOTE
i <- grep("Class", colnames(train)) # Get index Class column
train_smote <- SMOTE(Class ~ ., as.data.frame(train), perc.over = 20000, perc.under=100)

table(train_smote$Class)

# Prepare XGboost
# Prepare data for XGBoost and set parameters. Use AUC as evaluation metric, as accuracy does not make sense for such a imbalanced dataset.

# Back to numeric
train$Class <- as.numeric(levels(train$Class))[train$Class]
train_smote$Class <- as.numeric(levels(train_smote$Class))[train_smote$Class]

# As Matrix
train <- Matrix(as.matrix(train), sparse = TRUE)
train_smote <- Matrix(as.matrix(train_smote), sparse = TRUE)
test <- Matrix(as.matrix(test), sparse = TRUE)
cv <- Matrix(as.matrix(cv), sparse = TRUE)

# Create XGB Matrices
train_xgb <- xgb.DMatrix(data = train[,-i], label = train[,i])
train_smote_xgb <- xgb.DMatrix(data = train_smote[,-i], label = train_smote[,i])
test_xgb <- xgb.DMatrix(data = test[,-i], label = test[,i])
cv_xgb <- xgb.DMatrix(data = cv[,-i], label = cv[,i])

# Watchlist
watchlist <- list(train  = train_xgb, cv = cv_xgb)

# set parameters:
parameters <- list(
    # General Parameters
    booster            = "gbtree",          
    silent             = 0,                 
    # Booster Parameters
    eta                = 0.3,               
    gamma              = 0,                 
    max_depth          = 6,                 
    min_child_weight   = 1,                 
    subsample          = 1,                 
    colsample_bytree   = 1,                 
    colsample_bylevel  = 1,                 
    lambda             = 1,                 
    alpha              = 0,                 
    # Task Parameters
    objective          = "binary:logistic",   
    eval_metric        = "auc",
    seed               = 1900               
)

# Train model

# Train the model with the parameters set above and nrounds = 25 (increasing nrounds does not improve the model anymore). Plots show increasing train and cv AUC in the beginning and stagnating at later rounds as expected.

# Original
xgb.model <- xgb.train(
    parameters, 
    train_xgb, 
    nrounds = 25, 
    watchlist)

#Plot:
melted <- melt(
    xgb.model$evaluation_log, 
    id.vars="iter"
)
ggplot(
    data=melted, 
    aes(x=iter, 
        y=value, 
        group=variable, 
        color = variable
        )) + 
    geom_line()
# Smote
xgb_smote.model <- xgb.train(
    parameters, 
    train_smote_xgb, 
    nrounds = 25, 
    watchlist)

#Plot:
melted <- melt(
    xgb_smote.model$evaluation_log, 
    id.vars="iter"
)
ggplot(
    data=melted, 
    aes(x=iter, 
        y=value, 
        group=variable, 
        color = variable)) + 
    geom_line()

# Predict
# Set threshold and predict with test set.

# Threshold
q <-  0.5

# Original
xgb.predict <- predict(xgb.model, test, validate_features = FALSE)
xgb.predictboolean <- ifelse(xgb.predict >= q,1,0)  
roc <- roc(test[,i], predict(xgb.model, test, type = "prob"))
xgb.cm <- confusionMatrix(xgb.predictboolean, test[,i])
xgb.cm$table
print(paste("AUC of XGBoost is:",roc$auc))
print(paste("F1 of XGBoost is:", xgb.cm$byClass["F1"]))
xgb.cm$byClass

# SMOTE
roc_smote <- roc(test[,i], predict(xgb_smote.model, test, type = "prob"))
xgb_smote.predict <- predict(xgb_smote.model, test)
xgb_smote.predictboolean <- ifelse(xgb_smote.predict >= q,1,0)  
xgb_smote.cm <- confusionMatrix(xgb_smote.predictboolean, test[,i])
xgb_smote.cm$table
print(paste("AUC of SMOTE XGBoost is:",roc_smote$auc))
print(paste("F1 of SMOTE XGBoost is:", xgb_smote.cm$byClass["F1"]))
xgb_smote.cm$byClass

# Conclusion so far
# The first XGBoost model is working pretty well, with both an high AUC and F1-score. So far using SMOTE does not improve this model (AUC: 0.989 vs 0.981). Although it increases the number of True Positives, the number of False Positives increase way more when using SMOTE. Hence the lower F1-score for the SMOTE model.
# If I want to allow more false negatives, maybe the best option is to lower the threshold a bit for the original model.