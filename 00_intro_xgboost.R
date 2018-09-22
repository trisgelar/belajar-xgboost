install.packages("xgboost")
install.packages("DiagrammeR")
library(xgboost)
library(DiagrammeR)

# predict whether a mushroom can be eaten or not
# Mushroom data is cited from UCI Machine Learning Repository. @Bache Lichman:2013.

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test

# split with caret in the real world. https://topepo.github.io/caret/data-splitting.html

str(train)
summary(train$data)
dim(train$data)
head(train$data)
class(train$data)[1] # sparse matrix
class(train$label)

bst_sparse <- xgboost(
    data = train$data,
    label = train$label,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    objective = "binary:logistic"
)

# objective = "binary:logistic": we will train a binary classification model ;
# max.deph = 2: the trees won’t be deep, because our case is very simple ;
# nthread = 2: the number of cpu threads we are going to use;
# nrounds = 2: there will be two passes on the data, the second one will enhance the model by further reducing the difference between ground truth and prediction.

# Alternative bst_sparse
bst_dense <- xgboost(
    data = as.matrix(train$data),
    label = train$label,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    objective = "binary:logistic"
)

# Another alternative
d_train <- xgb.DMatrix(data = train$data, label = train$label)
bst_dmatrix <- xgboost(
    data = d_train,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    obejctive = "binary:logistic"
)

# verbose option

# verbose = 1, print evaluation metric
bst_dmatrix_verbose_1 <- xgboost(
    data = d_train,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    obejctive = "binary:logistic",
    verbose = 1
)

# verbose = 2, print evaluation metric
bst <- xgboost(
    data = d_train,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    obejctive = "binary:logistic",
    verbose = 2
)

# prediction
pred <- predict(bst, test$data)
print(head(pred))
# 0.08585283 0.94223028 0.08585283 0.08585283 0.02808681 0.94223028 not {0,1}

# Transform to binary
prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error", err))

# Advance features

## Dataset Preparation
# For the following advanced features, we need to put data in xgb.DMatrix as explained above.

d_train <- xgb.DMatrix(data = train$data, label = train$label)
d_test <- xgb.DMatrix(data = test$data, label = test$label)

# Measure learning progress with xgb.train
# having too many rounds lead to an overfitting

watchlist <- list(train = d_train, test = d_test)
bst <- xgb.train(
    data = d_train,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    watchlist = watchlist,
    eval.metric = "error",
    eval.metric = "logloss",
    objective = "binary:logistic"
)

# Linear Boosting
lin_bst <- xgb.train(
    data = d_train,
    booster = "gblinear",
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = 2,
    watchlist = watchlist,
    eval.metric = "error",
    eval.metric = "logloss",
    objective = "binary:logistic"
)

# Manipulating xgb.DMatrix
xgb.DMatrix.save(d_train, "data/d_train.buffer")
d_train2 <- xgb.DMatrix("data/d_train.buffer")

label = getinfo(d_test, "label")
pred <- predict(bst, d_test)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))


# View feature importance/influence

importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

xgb.dump(bst, with.stats = T)
xgb.plot.tree(model = bst)

# Save and load models
#if you provide a path to fname parameter you can save the trees to your hard drive.

xgb.save(bst, "data/xgboost.model")

# pilot XGBoost from caret package, save the model as a R binary vector
rawVec <- xgb.save.raw(bst)
print(class(rawVec))
bst <- xgb.load(rawVec)


rm(list = ls())