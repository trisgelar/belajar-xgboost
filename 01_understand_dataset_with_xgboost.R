library(xgboost)
library(Matrix)
library(data.table)
library(vcd)

# Preparation of the dataset
# Numeric VS categorical variables, xgboost manages only numeric vector
# if categorical / factor convert to numeric
# transform data.frame (dense = few zeroes in the matrix) to very sparse matrix (sparse = lots of zero in the matrix) of numeric
# [one-hot-encoding](http://en.wikipedia.org/wiki/One-hot)

data(Arthritis)
df <- data.table(Arthritis, keep.rownames = F)

str(df)
# 2 columns have factor type, one has ordinal type.
# ordinal variable :
# can take a limited number of values (like factor)
# these values are ordered (unlike factor).
# ex: Marked > Some > None

head(df)
dim(df)
summary(df)

# Creation of new features based on old ones
# Grouping age per 10 years
head(df[,AgeDiscret := as.factor(round(Age/10,0))])

# Random split age in two groups simplification()
head(df[,AgeCat:= as.factor(ifelse(Age > 30, "Old", "Young"))])

# Cleaning data {remove Id}
df[,ID:=NULL]

# list column Treatment values
levels(df[,Treatment])

# One-hot encoding {transform the categorical data to dummy variables}
# categorical value to {0,1}
sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
head(sparse_matrix)

# Create the output numeric vector (not as a sparse Matrix): 
# (output -> True/False)
output_vector = df[,Improved] == "Marked"
head(output_vector)

bst <- xgboost(
    data = sparse_matrix,
    label = output_vector,
    max.depth = 4,
    eta = 1,
    nthread = 2,
    nrounds = 10,
    eval.metric = "error",
    eval.metric = "logloss",
    objective = "binary:logistic",
)

# You can see some train-error: 0.XXXXX lines followed by a number. It decreases. Each line shows how well the model explains your data. Lower is better.


bst <- xgboost(
    data = sparse_matrix,
    label = output_vector,
    max.depth = 4,
    eta = 1,
    nthread = 2,
    nrounds = 4,
    eval.metric = "error",
    eval.metric = "logloss",
    objective = "binary:logistic",
)

# Feature importance
# Build the feature importance data.table

str(sparse_matrix)

importance <- xgb.importance(
    feature_names = sparse_matrix@Dimnames[[2]], 
    model = bst
)
head(importance)

importanceRaw <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst, data = sparse_matrix, label = output_vector)

# Cleaning for better display
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]

head(importanceClean)

##             Feature        Split       Gain RealCover RealCover %
## 1: TreatmentPlacebo -1.00136e-05 0.28575061         7   0.2500000
## 2:              Age         61.5 0.16374034        12   0.4285714
## 3:              Age           39 0.08705750         8   0.2857143
## 4:              Age         57.5 0.06947553        11   0.3928571
## 5:          SexMale -1.00136e-05 0.04874405         4   0.1428571
## 6:              Age         53.5 0.04620627        10   0.3571429

# In xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst,  :
#   xgb.importance: parameters 'data', 'label' and 'target' are deprecated

# Cleaning for better display
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]
head(importanceClean)

xgb.plot.importance(importance_matrix = importanceRaw)

c2 <- chisq.test(df$Age, output_vector)
print(c2)

# Pearson correlation between Age and illness disappearing is 35.48.

c2 <- chisq.test(df$AgeDiscret, output_vector)
print(c2)

# Our first simplification of Age gives a Pearson correlation is 8.26.

c2 <- chisq.test(df$AgeCat, output_vector)
print(c2)

# As you can see, in general destroying information by simplifying it won’t improve your model. Chi2 just demonstrates that.
# But in more complex cases, creating a new feature based on existing one which makes link with the outcome more obvious may help the algorithm and improve the model.
# The case studied here is not enough complex to show that. Check Kaggle website for some challenging datasets. However it’s almost always worse when you add some arbitrary rules.
# Moreover, you can notice that even if we have added some not useful new features highly correlated with other features, the boosting tree algorithm have been able to choose the best one, which in this case is the Age.
# Linear models may not be that smart in this scenario.