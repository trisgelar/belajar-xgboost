if (!require("outbreaks")) install.packages("outbreaks")
library(outbreaks)
library(tidyr)
library(plyr)
library(ggplot2)
library(gridExtra)
library(grid)
library(dplyr)
install.packages("mice")
library(mice)
library(caret)
library(xgboost)
install.packages("Ckmeans.1d.dp")
library(Ckmeans.1d.dp)

fluH7N9_china_2013_backup <- fluH7N9_china_2013 # back up original dataset in case something goes awry along the way
# convert ? to NAs
fluH7N9_china_2013$age[which(fluH7N9_china_2013$age == "?")] <- NA
# create a new column with case ID
fluH7N9_china_2013$case.ID <- paste("case", fluH7N9_china_2013$case.ID, sep = "_")
head(fluH7N9_china_2013)
str(fluH7N9_china_2013)

fluH7N9_china_2013_gather <- fluH7N9_china_2013 %>%
    gather(
        Group, 
        Date, 
        date_of_onset:date_of_outcome)

# rearrange group order
fluH7N9_china_2013_gather$Group <- factor(
    fluH7N9_china_2013_gather$Group, 
    levels = c("date_of_onset", 
                "date_of_hospitalisation", 
                "date_of_outcome"))

# rename groups
fluH7N9_china_2013_gather$Group <- mapvalues(
    fluH7N9_china_2013_gather$Group, 
    from = c("date_of_onset", 
            "date_of_hospitalisation", 
            "date_of_outcome"), 
    to = c("Date of onset", 
            "Date of hospitalisation", 
            "Date of outcome"))
# renaming provinces
fluH7N9_china_2013_gather$province <- mapvalues(
    fluH7N9_china_2013_gather$province,
    from = c("Anhui", 
            "Beijing", 
            "Fujian", 
            "Guangdong", 
            "Hebei", 
            "Henan", 
            "Hunan", 
            "Jiangxi", 
            "Shandong", 
            "Taiwan"), 
    to = rep("Other", 10))

# add a level for unknown gender
levels(fluH7N9_china_2013_gather$gender) <- c(levels(
    fluH7N9_china_2013_gather$gender
    ), "unknown")
    fluH7N9_china_2013_gather$gender[is.na(fluH7N9_china_2013_gather$gender)] <- "unknown"

# rearrange province order so that Other is the last
fluH7N9_china_2013_gather$province <- factor(
    fluH7N9_china_2013_gather$province, 
    levels = c("Jiangsu",  
                "Shanghai", 
                "Zhejiang", 
                "Other"))

# convert age to numeric
fluH7N9_china_2013_gather$age <- as.numeric(as.character(
    fluH7N9_china_2013_gather$age))

my_theme <- function(base_size = 12, base_family = "sans"){
    theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
        axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5),
        axis.title = element_text(size = 14),
        panel.grid.major = element_line(color = "grey"),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "aliceblue"),
        strip.background = element_rect(fill = "lightgrey", color = "grey", size = 1),
        strip.text = element_text(face = "bold", size = 12, color = "black"),
        legend.position = "bottom",
        legend.justification = "top", 
        legend.box = "horizontal",
        legend.box.background = element_rect(colour = "grey50"),
        legend.background = element_blank(),
        panel.border = element_rect(color = "grey", fill = NA, size = 0.5)
    )
}



# This plot shows the dates of onset, hospitalisation and outcome (if known) of each data point. Outcome is marked by color and age shown on the y-axis. Gender is marked by point shape.

# The density distribution of date by age for the cases seems to indicate that older people died more frequently in the Jiangsu and Zhejiang province than in Shanghai and in other provinces.

# When we look at the distribution of points along the time axis, it suggests that their might be a positive correlation between the likelihood of death and an early onset or early outcome.

# I also want to know how many cases there are for each gender and province and compare the genders’ age distribution.

fluH7N9_china_2013_gather_2 <- fluH7N9_china_2013_gather[, -4] %>%
    gather(group_2, value, gender:province)

fluH7N9_china_2013_gather_2$value <- mapvalues(
    fluH7N9_china_2013_gather_2$value, 
    from = c("m", "f", "unknown", "Other"),
    to = c("Male", "Female", "Unknown gender", "Other province"))

fluH7N9_china_2013_gather_2$value <- factor(
    fluH7N9_china_2013_gather_2$value,
    levels = c("Female", 
                "Male", 
                "Unknown gender", 
                "Jiangsu", 
                "Shanghai", 
                "Zhejiang", 
                "Other province"))

p1 <- ggplot(
    data = fluH7N9_china_2013_gather_2, 
    aes(x = value, 
        fill = outcome, 
        color = outcome)) +
    geom_bar(position = "dodge", alpha = 0.7, size = 1) +
    my_theme() +
    scale_fill_brewer(palette="Set1", na.value = "grey50") +
    scale_color_brewer(palette="Set1", na.value = "grey50") +
    labs(
        color = "",
        fill = "",
        x = "",
        y = "Count",
        title = "2013 Influenza A H7N9 cases in China",
        subtitle = "Gender and Province numbers of flu cases",
        caption = ""
)

p2 <- ggplot(
    data = fluH7N9_china_2013_gather, 
    aes(x = age, fill = outcome, color = outcome)) +
    geom_density(alpha = 0.3, size = 1) +
    geom_rug() +
    scale_color_brewer(palette="Set1", na.value = "grey50") +
    scale_fill_brewer(palette="Set1", na.value = "grey50") +
    my_theme() +
    labs(
        color = "",
        fill = "",
        x = "Age",
        y = "Density",
        title = "",
        subtitle = "Age distribution of flu cases",
        caption = ""
)
grid.arrange(p1, p2, ncol = 2)

# In the dataset, there are more male than female cases and correspondingly, we see more deaths, recoveries and unknown outcomes in men than in women. This is potentially a problem later on for modeling because the inherent likelihoods for outcome are not directly comparable between the sexes.

# Most unknown outcomes were recorded in Zhejiang. Similarly to gender, we don’t have an equal distribution of data points across provinces either.

# When we look at the age distribution it is obvious that people who died tended to be slightly older than those who recovered. The density curve of unknown outcomes is more similar to that of death than of recovery, suggesting that among these people there might have been more deaths than recoveries.

# And lastly, I want to plot how many days passed between onset, hospitalisation and outcome for each case.

ggplot(
    data = fluH7N9_china_2013_gather, 
    aes(x = Date, 
        y = age, 
        color = outcome)) +
geom_point(aes(shape = gender), size = 1.5, alpha = 0.6) +
geom_path(aes(group = case.ID)) +
facet_wrap( ~ province, ncol = 2) +
my_theme() +
scale_shape_manual(values = c(15, 16, 17)) +
scale_color_brewer(palette="Set1", na.value = "grey50") +
scale_fill_brewer(palette="Set1") +
labs(
    color = "Outcome",
    shape = "Gender",
    x = "Date in 2013",
    y = "Age",
    title = "2013 Influenza A H7N9 cases in China",
    subtitle = "Dataset from 'outbreaks' package (Kucharski et al. 2014)",
    caption = "\nTime from onset of flu to outcome."
)

# Features
# In Machine Learning-speak features are the variables used for model training. Using the right features dramatically influences the accuracy of the model.

# Because we don’t have many features, I am keeping age as it is, but I am also generating new features:

# from the date information I am calculating the days between onset and outcome and between onset and hospitalisation
# I am converting gender into numeric values with 1 for female and 0 for male
# similarly, I am converting provinces to binary classifiers (yes == 1, no == 0) for Shanghai, Zhejiang, Jiangsu and other provinces
# the same binary classification is given for whether a case was hospitalised, and whether they had an early onset or early outcome (earlier than the median date)

# preparing the data frame for modeling
dataset <- fluH7N9_china_2013 %>%
    mutate(
        hospital = as.factor(ifelse(is.na(date_of_hospitalisation), 0, 1)),
        gender_f = as.factor(ifelse(gender == "f", 1, 0)),
        province_Jiangsu = as.factor(ifelse(province == "Jiangsu", 1, 0)),
        province_Shanghai = as.factor(ifelse(province == "Shanghai", 1, 0)),
        province_Zhejiang = as.factor(ifelse(province == "Zhejiang", 1, 0)),
        province_other = as.factor(ifelse(province == "Zhejiang" | province == "Jiangsu" | province == "Shanghai", 0, 1)),
        days_onset_to_outcome = as.numeric(
            as.character(
                gsub(" days", "",
                as.Date(as.character(date_of_outcome), 
                format = "%Y-%m-%d") - 
                as.Date(as.character(date_of_onset), 
                format = "%Y-%m-%d")))),
        days_onset_to_hospital = as.numeric(
            as.character(gsub(" days", "",
            as.Date(as.character(
                date_of_hospitalisation), 
                format = "%Y-%m-%d") - 
                as.Date(as.character(date_of_onset), format = "%Y-%m-%d")))),
                age = as.numeric(as.character(age)),
        early_onset = as.factor(ifelse(date_of_onset < summary(fluH7N9_china_2013$date_of_onset)[[3]], 1, 0)),
        early_outcome = as.factor(ifelse(date_of_outcome < summary(
            fluH7N9_china_2013$date_of_outcome)[[3]], 1, 0))) %>%
                subset(select = -c(2:4, 6, 8))

rownames(dataset) <- dataset$case.ID
dataset <- dataset[, -1]
head(dataset)

# Imputing missing values
# When looking at the dataset I created for modeling, it is obvious that we have quite a few missing values.

# The missing values from the outcome column are what I want to predict but for the rest I would either have to remove the entire row from the data or impute the missing information. I decided to try the latter with the mice package.


dataset_impute <- mice(dataset[, -1],  print = FALSE)
dataset_impute

dataset_complete <- merge(
    dataset[, 1, drop = FALSE], 
    mice::complete(dataset_impute, 1), 
    by = "row.names", 
    all = TRUE
)
rownames(dataset_complete) <- dataset_complete$Row.names
dataset_complete <- dataset_complete[, -1]

summary(dataset$outcome)

train_index <- which(is.na(dataset_complete$outcome))
train_data <- dataset_complete[-train_index, ]
test_data  <- dataset_complete[train_index, -1]

set.seed(27)
val_index <- createDataPartition(train_data$outcome, p = 0.7, list=FALSE)
val_train_data <- train_data[val_index, ]
val_test_data  <- train_data[-val_index, ]
val_train_X <- val_train_data[,-1]
val_test_X <- val_test_data[,-1]

# Decision trees
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

set.seed(27)
fit <- rpart(
    outcome ~ .,
    data = train_data,
    method = "class",
    control = rpart.control(
        xval = 10, 
        minbucket = 2, 
        cp = 0), 
        parms = list(split = "information"))
fancyRpartPlot(fit)

# Feature Plot
# Before I start actually building models, I want to check whether the distribution of feature values is comparable between training, validation and test datasets.

# prepare for plotting
dataset_complete_gather <- dataset_complete %>%
    mutate(
        set = ifelse(rownames(dataset_complete) %in% rownames(test_data), 
        "Test Data", ifelse(rownames(dataset_complete) %in% rownames(val_train_data), "Validation Train Data", ifelse(rownames(dataset_complete) %in% rownames(val_test_data), "Validation Test Data", "NA"))), case_ID = rownames(.)) %>% gather(group, value, age:early_outcome)

dataset_complete_gather$group <- mapvalues(
    dataset_complete_gather$group, 
    from = c("age", "hospital", "gender_f", "province_Jiangsu", "province_Shanghai", "province_Zhejiang", "province_other", "days_onset_to_outcome", "days_onset_to_hospital", "early_onset", "early_outcome" ), 
    to = c("Age", "Hospital", "Female", "Jiangsu", "Shanghai", "Zhejiang", "Other province", "Days onset to outcome", "Days onset to hospital", "Early onset", "Early outcome" ))

ggplot(data = dataset_complete_gather, aes(x = as.numeric(value), fill = outcome, color = outcome)) +
    geom_density(alpha = 0.2) +
    geom_rug() +
    scale_color_brewer(palette="Set1", na.value = "grey50") +
    scale_fill_brewer(palette="Set1", na.value = "grey50") +
    my_theme() +
    facet_wrap(set ~ group, ncol = 11, scales = "free") +
    labs(
        x = "Value",
        y = "Density",
        title = "2013 Influenza A H7N9 cases in China",
        subtitle = "Features for classifying outcome",
        caption = "\nDensity distribution of all features used for classification of flu outcome."
)

ggplot(subset(dataset_complete_gather, group == "Age" | group == "Days onset to hospital" | group == "Days onset to outcome"), 
    aes(x=outcome, y=as.numeric(value), fill=set)) + geom_boxplot() +
    my_theme() +
    scale_fill_brewer(palette="Set1", type = "div ") +
    facet_wrap( ~ group, ncol = 3, scales = "free") +
    labs(
        fill = "",
        x = "Outcome",
        y = "Value",
        title = "2013 Influenza A H7N9 cases in China",
        subtitle = "Features for classifying outcome",
        caption = "\nBoxplot of the features age, days from onset to hospitalisation and days from onset to outcome."
)

matrix_train <- apply(val_train_X, 2, function(x) as.numeric(as.character(x)))
outcome_death_train <- ifelse(val_train_data$outcome == "Death", 1, 0)

matrix_test <- apply(val_test_X, 2, function(x) as.numeric(as.character(x)))
outcome_death_test <- ifelse(val_test_data$outcome == "Death", 1, 0)

xgb_train_matrix <- xgb.DMatrix(data = as.matrix(matrix_train), label = outcome_death_train)
xgb_test_matrix <- xgb.DMatrix(data = as.matrix(matrix_test), label = outcome_death_test)

watchlist <- list(train = xgb_train_matrix, test = xgb_test_matrix)
label <- getinfo(xgb_test_matrix, "label")

param <- list("objective" = "binary:logistic")

xgb.cv(
    param = param, 
    data = xgb_train_matrix, 
    nfold = 3,
    label = getinfo(xgb_train_matrix, "label"),
    nrounds = 5)

# Training with gbtree
# gbtree is the default booster for xgb.train.

bst_1 <- xgb.train(
    data = xgb_train_matrix, 
    label = getinfo(xgb_train_matrix, "label"),
    max.depth = 2, 
    eta = 1, 
    nthread = 4, 
    nround = 50, # number of trees used for model building
    watchlist = watchlist, 
    objective = "binary:logistic")

features = colnames(matrix_train)
importance_matrix_1 <- xgb.importance(features, model = bst_1)
print(importance_matrix_1)

xgb.ggplot.importance(importance_matrix_1) +
    theme_minimal()

pred_1 <- predict(bst_1, xgb_test_matrix)
result_1 <- data.frame(
    case_ID = rownames(val_test_data),
    outcome = val_test_data$outcome, 
    label = label, 
    prediction_p_death = round(pred_1, digits = 2),
    prediction = as.integer(pred_1 > 0.5),
    prediction_eval = ifelse(as.integer(pred_1 > 0.5) != label, "wrong", "correct"))
result_1

err <- as.numeric(sum(as.integer(pred_1 > 0.5) != label))/length(label)
print(paste("test-error =", round(err, digits = 2)))

bst_2 <- xgb.train(
    data = xgb_train_matrix, 
    booster = "gblinear", 
    label = getinfo(xgb_train_matrix, "label"),
    max.depth = 2, 
    eta = 1, 
    nthread = 4, 
    nround = 50, # number of trees used for model building
    watchlist = watchlist, 
    objective = "binary:logistic")

features = colnames(matrix_train)
importance_matrix_2 <- xgb.importance(features, model = bst_2)
print(importance_matrix_2)

xgb.ggplot.importance(importance_matrix_2) +
    theme_minimal()

pred_2 <- predict(bst_2, xgb_test_matrix)

result_2 <- data.frame(
    case_ID = rownames(val_test_data),
    outcome = val_test_data$outcome, 
    label = label, 
    prediction_p_death = round(pred_2, digits = 2),
    prediction = as.integer(pred_2 > 0.5),
    prediction_eval = ifelse(as.integer(pred_2 > 0.5) != label, "wrong", "correct"))
result_2

err <- as.numeric(sum(as.integer(pred_2 > 0.5) != label))/length(label)
print(paste("test-error =", round(err, digits = 2)))

# caret
# Extreme gradient boosting is also implemented in the caret package. Caret also provides options for preprocessing, of which I will compare a few.

set.seed(27)
model_xgb_null <-train(
    outcome ~ .,
    data=val_train_data,
    method="xgbTree",
    preProcess = NULL,
    trControl = trainControl(
        method = "repeatedcv", 
        number = 5, 
        repeats = 10, 
        verboseIter = FALSE))

confusionMatrix(predict(model_xgb_null, val_test_data[, -1]), val_test_data$outcome)

# Scaling and centering
# With this method the column variables are centered (subtracting the column mean from each value in a column) and standardized (dividing by the column standard deviation).

set.seed(27)
model_xgb_sc_cen <-train(
    outcome ~ .,
    data=val_train_data,
    method="xgbTree",
    preProcess = c("scale", "center"),
    trControl = trainControl(
        method = "repeatedcv", 
        number = 5, 
        repeats = 10, 
        verboseIter = FALSE))

confusionMatrix(predict(model_xgb_sc_cen, val_test_data[, -1]), val_test_data$outcome)

# Scaling and centering did not improve the predictions.
pred_3 <- predict(model_xgb_sc_cen, val_test_data[, -1])
pred_3b <- round(predict(model_xgb_sc_cen, val_test_data[, -1], type="prob"), digits = 2)

result_3 <- data.frame(
    case_ID = rownames(val_test_data),
    outcome = val_test_data$outcome, 
    label = label, 
    prediction = pred_3,
    pred_3b)

result_3$prediction_eval <- ifelse(result_3$prediction != result_3$outcome, "wrong", "correct")
result_3

# Box-Cox transformation
# The Box-Cox power transformation is used to normalize data.

set.seed(27)
model_xgb_BoxCox <-train(
    outcome ~ .,
    data=val_train_data,
    method="xgbTree",
    preProcess = "BoxCox",
    trControl = trainControl(
        method = "repeatedcv", 
        number = 5, 
        repeats = 10, 
        verboseIter = FALSE))

confusionMatrix(predict(model_xgb_BoxCox, val_test_data[, -1]), val_test_data$outcome)

# Box-Cox transformation did not improve the predictions either.
pred_4 <- predict(model_xgb_BoxCox, val_test_data[, -1])
pred_4b <- round(predict(model_xgb_BoxCox, val_test_data[, -1], type="prob"), digits = 2)

result_4 <- data.frame(
    case_ID = rownames(val_test_data),
    outcome = val_test_data$outcome, 
    label = label, 
    prediction = pred_4,
    pred_4b)
result_4$prediction_eval <- ifelse(result_4$prediction != result_4$outcome, "wrong", "correct")
result_4

# Principal Component Analysis (PCA)
# PCA is used for dimensionality reduction. When applied as a preprocessing method the number of features are reduced by using the eigenvectors of the covariance matrix.

set.seed(27)
model_xgb_pca <-train(outcome ~ .,
    data=val_train_data,
    method="xgbTree",
    preProcess = "pca",
    trControl = trainControl(method = "repeatedcv", number = 5, repeats = 10, verboseIter = FALSE))
confusionMatrix(predict(model_xgb_pca, val_test_data[, -1]), val_test_data$outcome)

# PCA did improve the predictions slightly.
pred_5 <- predict(model_xgb_pca, val_test_data[, -1])
pred_5b <- round(predict(model_xgb_pca, val_test_data[, -1], type="prob"), digits = 2)

result_5 <- data.frame(case_ID = rownames(val_test_data),
    outcome = val_test_data$outcome, 
    label = label, 
    prediction = pred_5,
    pred_5b)
result_5$prediction_eval <- ifelse(result_5$prediction != result_5$outcome, "wrong", "correct")
result_5

# Median imputation
set.seed(27)
model_xgb_medianImpute <-train(outcome ~ .,
    data=val_train_data,
    method="xgbTree",
    preProcess = "medianImpute",
    trControl = trainControl(method = "repeatedcv", number = 5, repeats = 10, verboseIter = FALSE))

confusionMatrix(predict(model_xgb_medianImpute, val_test_data[, -1]), val_test_data$outcome)

# Median imputation did not improve the predictions either.

pred_6 <- predict(model_xgb_medianImpute, val_test_data[, -1])
pred_6b <- round(predict(model_xgb_medianImpute, val_test_data[, -1], type="prob"), digits = 2)

result_6 <- data.frame(case_ID = rownames(val_test_data),
    outcome = val_test_data$outcome, 
    label = label, 
    prediction = pred_6,
    pred_6b)
result_6$prediction_eval <- ifelse(result_6$prediction != result_6$outcome, "wrong", "correct")
result_6

# Comparison of extreme gradient boosting models
# Combining results
library(dplyr)
result <- left_join(result_1[, c(1, 2, 6)], result_2[, c(1, 6)], by = "case_ID")
result <- left_join(result, result_3[, c(1, 7)], by = "case_ID")
result <- left_join(result, result_4[, c(1, 7)], by = "case_ID")
result <- left_join(result, result_5[, c(1, 7)], by = "case_ID")
result <- left_join(result, result_6[, c(1, 7)], by = "case_ID")
colnames(result)[-c(1:2)] <- c("pred_xgboost_gbtree", "pred_xgboost_gblinear", "model_xgb_sc_cen", "model_xgb_BoxCox", "pred_xgbTree_pca", "model_xgb_medianImpute")

round(sum(result$pred_xgboost_gbtree == "correct")/nrow(result), digits = 2)
round(sum(result$pred_xgboost_gblinear == "correct")/nrow(result), digits = 2)
round(sum(result$model_xgb_sc_cen == "correct")/nrow(result), digits = 2)
round(sum(result$model_xgb_BoxCox == "correct")/nrow(result), digits = 2)
round(sum(result$pred_xgbTree_pca == "correct")/nrow(result), digits = 2)
round(sum(result$model_xgb_medianImpute == "correct")/nrow(result), digits = 2)

set.seed(27)
model_xgb_pca <-train(outcome ~ .,
    data = train_data,
    method = "xgbTree",
    preProcess = "pca",
    trControl = trainControl(method = "repeatedcv", number = 5, repeats = 10, verboseIter = FALSE))

pred <- predict(model_xgb_pca, test_data)
predb <- round(predict(model_xgb_pca, test_data, type="prob"), digits = 2)

result <- data.frame(case_ID = rownames(test_data),
    prediction = pred,
    predb)
result$predicted_outcome <- ifelse(result$Death > 0.7, "Death",
                ifelse(result$Recover > 0.7, "Recover", "uncertain"))
result

results <- data.frame(randomForest = predict(model_rf, newdata = test_data, type="prob"),
glmnet = predict(model_glmnet, newdata = test_data, type="prob"),
kknn = predict(model_kknn, newdata = test_data, type="prob"),
pda = predict(model_pda, newdata = test_data, type="prob"),
slda = predict(model_slda, newdata = test_data, type="prob"),
pam = predict(model_pam, newdata = test_data, type="prob"),
C5.0Tree = predict(model_C5.0Tree, newdata = test_data, type="prob"),
pls = predict(model_pls, newdata = test_data, type="prob"))

results$sum_Death <- rowSums(results[, grep("Death", colnames(results))])
results$sum_Recover <- rowSums(results[, grep("Recover", colnames(results))])
results$log2_ratio <- log2(results$sum_Recover/results$sum_Death)
results$predicted_outcome <- ifelse(results$log2_ratio > 1.5, "Recover", ifelse(results$log2_ratio < -1.5, "Death", "uncertain"))
results[, -c(1:16)]

results <- results_last_week


result <- merge(result[, c(1, 5)], results_last_week[, ncol(results_last_week), drop = FALSE], by.x = "case_ID", by.y = "row.names")
colnames(result)[2:3] <- c("predicted_outcome_xgboost", "predicted_outcome_last_week")

results_combined <- merge(result, fluH7N9_china_2013[which(fluH7N9_china_2013$case.ID %in% result$case_ID), ], 
    by.x = "case_ID", by.y = "case.ID")
results_combined <- results_combined[, -c(6,7)]

library(tidyr)
results_combined_gather <- results_combined %>%
    gather(group_dates, date, date.of.onset:date.of.hospitalisation)

results_combined_gather$group_dates <- factor(results_combined_gather$group_dates, levels = c("date.of.onset", "date.of.hospitalisation"))

results_combined_gather$group_dates <- mapvalues(
    results_combined_gather$group_dates, 
    from = c("date.of.onset", "date.of.hospitalisation"), 
    to = c("Date of onset", "Date of hospitalisation"))

results_combined_gather$gender <- mapvalues(
    results_combined_gather$gender, 
    from = c("f", "m"), 
    to = c("Female", "Male"))

levels(results_combined_gather$gender) <- c(levels(
    results_combined_gather$gender), "unknown")

results_combined_gather$gender[is.na(results_combined_gather$gender)] <- "unknown"

results_combined_gather <- results_combined_gather %>%
    gather(group_pred, prediction, predicted_outcome_xgboost:predicted_outcome_last_week)

results_combined_gather$group_pred <- mapvalues(
    results_combined_gather$group_pred, 
    from = c("predicted_outcome_xgboost", "predicted_outcome_last_week"), 
    to = c("Predicted outcome from XGBoost", "Predicted outcome from last week"))
my_theme <- function(base_size = 12, base_family = "sans"){
    theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
        axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5),
        axis.title = element_text(size = 14),
        panel.grid.major = element_line(color = "grey"),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "aliceblue"),
        strip.background = element_rect(fill = "lightgrey", color = "grey", size = 1),
        strip.text = element_text(face = "bold", size = 12, color = "black"),
        legend.position = "bottom",
        legend.justification = "top", 
        legend.box = "horizontal",
        legend.box.background = element_rect(colour = "grey50"),
        legend.background = element_blank(),
        panel.border = element_rect(color = "grey", fill = NA, size = 0.5),
        panel.spacing = unit(1, "lines")
    )
}

results_combined_gather$province <- mapvalues(
    results_combined_gather$province, 
    from = c("Anhui", "Beijing", "Fujian", "Guangdong", "Hebei", "Henan", "Hunan", "Jiangxi", "Shandong", "Taiwan"), 
    to = rep("Other", 10))

levels(results_combined_gather$gender) <- c(levels(results_combined_gather$gender), "unknown")
results_combined_gather$gender[is.na(results_combined_gather$gender)] <- "unknown"

results_combined_gather$province <- factor(results_combined_gather$province, levels = c("Jiangsu",  "Shanghai", "Zhejiang", "Other"))

ggplot(data = subset(results_combined_gather, group_dates == "Date of onset"), aes(x = date, y = as.numeric(age), fill = prediction)) +
    stat_density2d(aes(alpha = ..level..), geom = "polygon") +
    geom_jitter(aes(color = prediction, shape = gender), size = 2) +
    geom_rug(aes(color = prediction)) +
    labs(
        fill = "Predicted outcome",
        color = "Predicted outcome",
        alpha = "Level",
        shape = "Gender",
        x = "Date of onset in 2013",
        y = "Age",
        title = "2013 Influenza A H7N9 cases in China",
        subtitle = "Predicted outcome of cases with unknown outcome",
        caption = ""
    ) +
    facet_grid(group_pred ~ province) +
    my_theme() +
    scale_shape_manual(values = c(15, 16, 17)) +
    scale_color_brewer(palette="Set1", na.value = "grey50") +
    scale_fill_brewer(palette="Set1")