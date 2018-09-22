library(xgboost)
library(ggplot2)
library(reshape2)
library(Ecdat)

data = Housing
N = nrow(data)

# = divided price by 10000 to use smaller values of gamma = #
y = data$price/10000
x = data[,-1]
# = Transform categorical variables into dummies = #
for(i in 1:ncol(x)){
  if(is.factor(x[,i])){
    x[,i] = ifelse(x[,i] == "yes",1,0)
  }
}
x = as.matrix(x)
# = select train and test indexes = #
set.seed(1)
train=sample(1:N,456)
test=setdiff(1:N,train)
# = min_child_weights candidates = #
mcw=c(1,10,100,400)
# = gamma candidates = #
gamma=c(0.1,1,10,100)
# = train and test data = #
xtrain = x[train,]
ytrain = y[train]
xtest = x[test,]
ytest = y[test]

# min_child_weight
# The code below runs the boosting for the four min_child_weights in the mcw vector. You can see the convergence and the test RMSE for each setup below the code. The results show a significant change in the model as we move on the min_child_weight. Values of 1 and 10 seems to overfit the data a little while and a value of 400 ignores a lot of information and returns a poor predictive model. The best solution in this case was for min_child_weight = 100.

set.seed(1)
conv_mcw = matrix(NA,500,length(mcw))
pred_mcw = matrix(NA,length(test), length(mcw))
colnames(conv_mcw) = colnames(pred_mcw) = mcw
for(i in 1:length(mcw)){
  params = list(eta = 0.1, colsample_bylevel=2/3,
              subsample = 1, max_depth = 6,
              min_child_weight = mcw[i], gamma = 0)
  xgb = xgboost(xtrain, label = ytrain, nrounds = 500, params = params)
  conv_mcw[,i] = xgb$evaluation_log$train_rmse
  pred_mcw[,i] = predict(xgb, xtest)
}
 
conv_mcw = data.frame(iter=1:500, conv_mcw)
conv_mcw = melt(conv_mcw, id.vars = "iter")
ggplot(data = conv_mcw) + geom_line(aes(x = iter, y = value, color = variable))

(RMSE_mcw = sqrt(colMeans((ytest-pred_mcw)^2)))

# gamma
# The results for gamma are similar to the min_child_weight. The models in the middle (gamma = 1 and gamma = 10) are superior in terms of predictive accuracy. Unfortunately, the convergence plot does not give us any clue on which model is the best. We have to test the model in a test sample or in a cross-validation scheme to select the most accurate. Both the min_child_weight and the gamma are applying shrinkage to the trees by limiting their size, bu they look at different measures to do so.

set.seed(1)
conv_gamma = matrix(NA,500,length(gamma))
pred_gamma = matrix(NA,length(test), length(gamma))
colnames(conv_gamma) = colnames(pred_gamma) = gamma
for(i in 1:length(gamma)){
  params = list(eta = 0.1, colsample_bylevel=2/3,
              subsample = 1, max_depth = 6, min_child_weight = 1,
              gamma = gamma[i])
  xgb = xgboost(xtrain, label = ytrain, nrounds = 500, params = params)
  conv_gamma[,i] = xgb$evaluation_log$train_rmse
  pred_gamma[,i] = predict(xgb, xtest)
}
 
conv_gamma = data.frame(iter=1:500, conv_gamma)
conv_gamma = melt(conv_gamma, id.vars = "iter")
ggplot(data = conv_gamma) + geom_line(aes(x = iter, y = value, color = variable))

(RMSE_gamma = sqrt(colMeans((ytest-pred_gamma)^2)))

rm(list = ls())