library(dplyr)
library(ggplot2)
# install.packages("GGally")
library(GGally)
library(xgboost)


d_train <- readr::read_csv(
    file  = "data/monster_train.csv",
    col_names = TRUE)

d_test <- readr::read_csv("data/monster_test.csv")

glimpse(d_train)

# Graphical Data Analysis

ggpairs(d_train, 
    columns = , c("bone_length", "rotting_flesh", "hair_length", "has_soul"),
    mapping = ggplot2::aes(color = type),
    lower = list(continuous = wrap("density", alpha = 0.5), combo = "box"),
    # upper = list(continuous = wrap("points", alpha = 0.3), combo = wrap("dot", alpha = 0.4)),
    title = "Monsters!",
    axisLabels = "show")

# First look at the density plots on the diagonals which are “sorta” histograms for the associated variable (read off the column or row, they are the same). Helpfully they lie between zero and one, no need to normalise, thanks Kaggle! The lower panel visualises all interactions and may indicate which ones are useful for distinguishing between Monster classes (read off from the y axis and the x axis to identify the interaction being plotted). We can even quantify these correlations as shown in the top panel for those of us who prefer numbers to graphics (“Boo!” said the Ghost). The corporeal monsters appear more similar than their ethereal counterpart. Eyeballing this it looks like linear discriminant analysis (LDA) might perform well (Spoiler alert; using default mlr LDA outperforms default XGBoost).

# Dummy variables
train_tedious <- d_train %>%
    mutate(color = as.factor(color), type = as.factor(type)) %>%
    mutate(
        black = if_else(color == "black", 1, 0),
        blood = if_else(color == "blood", 1, 0),
        blue = if_else(color == "blue", 1, 0),
        clear = if_else(color == "clear", 1, 0),
        green = if_else(color == "green", 1, 0),
        white = if_else(color == "white", 1, 0)
    )

# alternative use caret::dummyVars

# XGBoost compatible

label <- as.integer(train_tedious$type) - 1 #  range from 0 to number of classes
# Ghost, 1 Ghoul, 2 Goblin, 3
 
dat <- as.matrix(select(train_tedious, -id, -color, -type),
    nrow = dim(train_tedious)[1], ncol = dim(train_tedious)[2] - 3,
    byrow = FALSE)
dm_train <- xgb.DMatrix(data = dat, label = label)

bst_DMatrix <- xgboost(
    data = dm_train,
    nthread = 2, nround = 5,
    objective = "multi:softmax",
    num_class = 3, seed = 1337)

importance_matrix <- xgb.importance(
    names(
        dplyr::select(
            train_tedious, 
            -id, 
            -color, 
            -type)
    ),
    model = bst_DMatrix)

xgb.plot.importance(importance_matrix)

# Testing the model

d_test <- d_test %>%
    mutate(color = as.factor(color)) %>%
    mutate(black = if_else(color == "black", 1, 0),
    blood = if_else(color == "blood", 1, 0),
    blue = if_else(color == "blue", 1, 0),
    clear = if_else(color == "clear", 1, 0),
    green = if_else(color == "green", 1, 0),
    white = if_else(color == "white", 1, 0)
)
# make test
dat_test <- as.matrix(
    select(d_test, -id, -color),
    nrow = dim(train)[1], 
    ncol = dim(train)[2] - 3,
    byrow = FALSE)

dtest <- xgb.DMatrix(data = dat_test)

# predict

pred <- predict(bst_DMatrix, dtest)

# Convert for Kaggle Submission
## Write for submission to Kaggle
testid <- as_tibble(d_test$id) %>%
    rename(id = value)

submission <- pred %>%  #  $class
    as_tibble() %>%
    rename(type = value) %>%
    bind_cols(testid) %>%
    select(id, type) %>%
    mutate(type = if_else(type == 0, "Ghost", 
    false = if_else(type == 1, "Ghoul", "Goblin"))) %>%
    mutate(type = as.factor(type))
# write_csv(x = submission, path = "submission_xgboost.csv")

devtools::session_info()
rm(list = ls())