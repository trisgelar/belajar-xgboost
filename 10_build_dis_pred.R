# The dataset

# Sample ID (code number)
# Clump thickness
# Uniformity of cell size
# Uniformity of cell shape
# Marginal adhesion
# Single epithelial cell size
# Number of bare nuclei
# Bland chromatin
# Number of normal nuclei
# Mitosis
# Classes, i.e. diagnosis

bc_data <- read.table(
    "data/breast-cancer-wisconsin.data.txt",
    header = FALSE,
    sep = ",")
colnames(bc_data) <- c("sample_code_number",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitosis",
    "classes")

bc_data$classes <- ifelse(
    bc_data$classes == "2",
    "benign",
    ifelse(
        bc_data$classes == "4",
        "malignant", NA)
)

# Missing data
bc_data[bc_data == "?"] <- NA
# how many NAs are in the data
length(which(is.na(bc_data)))

# how many samples would we loose, if we removed them?
nrow(bc_data)

nrow(bc_data[is.na(bc_data), ])

# Missing values are imputed with the mice package.
library(mice)

bc_data[,2:10] <- apply(
    bc_data[, 2:10], 
    2, 
    function(x) as.numeric(as.character(x))
)
dataset_impute <- mice(
    bc_data[, 2:10],  
    print = FALSE
)
bc_data <- cbind(
    bc_data[, 11, drop = FALSE], 
    mice::complete(dataset_impute, 1)
)

bc_data$classes <- as.factor(bc_data$classes)

# how many benign and malignant cases are there?
summary(bc_data$classes)

# Data exploration
library(ggplot2)

ggplot(bc_data, 
    aes(x = classes, fill = classes)) +
    geom_bar()

# We can see that our data is unbalanced. For simplicity’s sake, I am not going to go into how to deal with this, here. But I added a short post about dealing with unbalanced datasets using caret for you to check out.

# Response variable for regression
ggplot(bc_data, 
    aes(x = clump_thickness)) +
    geom_histogram(bins = 10)

# Principal Component Analysis
library(pcaGoPromoter)
library(ellipse)

# perform pca and extract scores
pcaOutput <- pca(
    t(bc_data[, -1]), 
    printDropped = FALSE, 
    scale = TRUE, 
    center = TRUE
)
pcaOutput2 <- as.data.frame(pcaOutput$scores)

# define groups for plotting
pcaOutput2$groups <- bc_data$classes

centroids <- aggregate(
    cbind(PC1, PC2) ~ groups, 
    pcaOutput2, 
    mean
)

conf.rgn  <- do.call(
    rbind,
    lapply(
        unique(pcaOutput2$groups),
        function(t)
    data.frame(groups = as.character(t),
        ellipse(
            cov(
                pcaOutput2[pcaOutput2$groups == t,
                1:2]
            ),
            centre = as.matrix(
                centroids[centroids$groups == t, 2:3]),
            level = 0.95
        ),
    stringsAsFactors = FALSE))
)

ggplot(
    data = pcaOutput2,
    aes(
        x = PC1,
        y = PC2,
        group = groups,
        color = groups)
    ) +
    geom_polygon(
        data = conf.rgn,
        aes(fill = groups),
        alpha = 0.2) +
    geom_point(
        size = 2,
        alpha = 0.6) +
    scale_color_brewer(
        palette = "Set1") +
    labs(color = "",
        fill = "",
        x = paste0(
            "PC1: ",
            round(
                pcaOutput$pov[1],
                digits = 2) * 100, "% variance"),
        y = paste0(
            "PC2: ",
            round(
                pcaOutput$pov[2],
                digits = 2) * 100, "% variance")
    )

library(tidyr)

gather(
    bc_data,
    x,
    y,
    clump_thickness:mitosis) %>%
    ggplot(
        aes(x = y, color = classes, fill = classes)) +
        geom_density(alpha = 0.3) +
        facet_wrap( ~ x, scales = "free", ncol = 3)

# Machine Learning packages for R
# configure multicore
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

library(caret)

set.seed(42)
index <- createDataPartition(bc_data$classes, p = 0.7, list = FALSE)
train_data <- bc_data[index, ]
test_data  <- bc_data[-index, ]

library(dplyr)

rbind(
    data.frame(group = "train", train_data),
    data.frame(group = "test", test_data)) %>%
    gather(x, y, clump_thickness:mitosis) %>%
    ggplot(aes(x = y, color = group, fill = group)) +
    geom_density(alpha = 0.3) +
    facet_wrap( ~ x, scales = "free", ncol = 3)

# Regression
set.seed(42)
model_glm <- caret::train(
    clump_thickness ~ .,
    data = train_data,
    method = "glm",
    preProcess = c("scale", "center"),
    trControl = trainControl(
        method = "repeatedcv",
        number = 10,
        repeats = 10,
        savePredictions = TRUE,
        verboseIter = FALSE))

model_glm

predictions <- predict(model_glm, test_data)
# model_glm$finalModel$linear.predictors == model_glm$finalModel$fitted.values
data.frame(
    residuals = resid(model_glm),
    predictors = model_glm$finalModel$linear.predictors) %>%
    ggplot(aes(x = predictors, y = residuals)) +
    geom_jitter() +
    geom_smooth(method = "lm")

# y == train_data$clump_thickness
data.frame(residuals = resid(model_glm),
    y = model_glm$finalModel$y) %>%
    ggplot(aes(x = y, y = residuals)) +
    geom_jitter() +
    geom_smooth(method = "lm")

data.frame(actual = test_data$clump_thickness,
    predicted = predictions) %>%
    ggplot(aes(x = actual, y = predicted)) +
    geom_jitter() +
    geom_smooth(method = "lm")

# Classification
# Decision trees

library(rpart)
library(rpart.plot)

set.seed(42)
fit <- rpart(
    classes ~ .,
    data = train_data,
    method = "class",
    control = rpart.control(
        xval = 10,
        minbucket = 2,
        cp = 0),
    parms = list(split = "information"))

rpart.plot(fit, extra = 100)

# Random Forests
set.seed(42)
model_rf <- caret::train(
    classes ~ .,
    data = train_data,
    method = "rf",
    preProcess = c("scale", "center"),
    trControl = trainControl(
        method = "repeatedcv",
        number = 10,
        repeats = 10,
        savePredictions = TRUE,
        verboseIter = FALSE))

# When you specify savePredictions = TRUE, you can access the cross-validation resuls with model_rf$pred.

model_rf$finalModel$confusion

# Feature Importance
imp <- model_rf$finalModel$importance
imp[order(imp, decreasing = TRUE), ]

importance <- varImp(model_rf, scale = TRUE)
plot(importance)

# predicting test data
confusionMatrix(predict(model_rf, test_data), test_data$classes)
results <- data.frame(
    actual = test_data$classes,
    predict(model_rf, test_data, type = "prob")
)

results$prediction <- ifelse(
    results$benign > 0.5, "benign",
    ifelse(results$malignant > 0.5, "malignant", NA))

results$correct <- ifelse(results$actual == results$prediction, TRUE, FALSE)

ggplot(results, aes(x = prediction, fill = correct)) +
    geom_bar(position = "dodge")

ggplot(
    results, 
    aes(x = prediction, 
        y = benign, 
        color = correct, 
        shape = correct)) +
    geom_jitter(size = 3, alpha = 0.6)

# Extreme gradient boosting trees
# Extreme gradient boosting (XGBoost) is a faster and improved implementation of gradient boosting for supervised learning.

set.seed(42)
model_xgb <- caret::train(
    classes ~ .,
    data = train_data,
    method = "xgbTree",
    preProcess = c("scale", "center"),
    trControl = trainControl(
        method = "repeatedcv",
        number = 10,
        repeats = 10,
        savePredictions = TRUE,
        verboseIter = FALSE))

# Feature Importance
importance <- varImp(model_xgb, scale = TRUE)
plot(importance)

confusionMatrix(
    predict(
        model_xgb, test_data
    ), test_data$classes)

results <- data.frame(
    actual = test_data$classes,
    predict(model_xgb, test_data, type = "prob"))

results$prediction <- ifelse(
    results$benign > 0.5, "benign",
    ifelse(results$malignant > 0.5, "malignant", NA))

results$correct <- ifelse(results$actual == results$prediction, TRUE, FALSE)

ggplot(results, aes(x = prediction, fill = correct)) +
    geom_bar(position = "dodge")

ggplot(
    results, 
    aes(
        x = prediction, 
        y = benign, 
        color = correct, 
        shape = correct)) +
    geom_jitter(size = 3, alpha = 0.6)

# Feature Selection
# Performing feature selection on the whole dataset would lead to prediction bias, we therefore need to run the whole modeling process on the training data alone!

# Correlation
# Correlations between all features are calculated and visualised with the corrplot package. I am then removing all features with a correlation higher than 0.7, keeping the feature with the lower mean.

library(corrplot)

# calculate correlation matrix
corMatMy <- cor(train_data[, -1])
corrplot(corMatMy, order = "hclust")

#Apply correlation filter at 0.70,
highlyCor <- colnames(
    train_data[, -1])[findCorrelation(
        corMatMy, 
        cutoff = 0.7, 
        verbose = TRUE)]

highlyCor
train_data_cor <- train_data[, which(!colnames(train_data) %in% highlyCor)]

# Recursive Feature Elimination (RFE)
# Another way to choose features is with Recursive Feature Elimination. RFE uses a Random Forest algorithm to test combinations of features and rate each with an accuracy score. The combination with the highest score is usually preferential.

set.seed(7)
results_rfe <- rfe(
    x = train_data[, -1],
    y = train_data$classes,
    sizes = c(1:9),
    rfeControl = rfeControl(
        functions = rfFuncs, 
        method = "cv", 
        number = 10)
)

predictors(results_rfe)
train_data_rfe <- train_data[, c(1, which(
    colnames(train_data) %in% predictors(
        results_rfe)))]

# Genetic Algorithm (GA)

# The Genetic Algorithm (GA) has been developed based on evolutionary principles of natural selection: It aims to optimize a population of individuals with a given set of genotypes by modeling selection over time. In each generation (i.e. iteration), each individual’s fitness is calculated based on their genotypes. Then, the fittest individuals are chosen to produce the next generation. This subsequent generation of individuals will have genotypes resulting from (re-) combinations of the parental alleles. These new genotypes will again determine each individual’s fitness. This selection process is iterated for a specified number of generations and (ideally) leads to fixation of the fittest alleles in the gene pool.

# This concept of optimization can be applied to non-evolutionary models as well, like feature selection processes in machine learning.

set.seed(27)
model_ga <- gafs(
    x = train_data[, -1],
    y = train_data$classes,
    iters = 10, # generations of algorithm
    popSize = 10, # population size for each generation
    levels = c("malignant", "benign"),
    gafsControl = gafsControl(
        functions = rfGA, # Assess fitness with RF
        method = "cv",    # 10 fold cross validation
        genParallel = TRUE, # Use parallel programming
        allowParallel = TRUE))

plot(model_ga) # Plot mean fitness (AUC) by generation
train_data_ga <- train_data[, c(
    1, 
    which(colnames(train_data) %in% model_ga$ga$final))
]

# Grid search with caret
# Automatic Grid
set.seed(42)
model_rf_tune_auto <- caret::train(
    classes ~ .,
    data = train_data,
    method = "rf",
    preProcess = c("scale", "center"),
    trControl = trainControl(
        method = "repeatedcv",
        number = 10,
        repeats = 10,
        savePredictions = TRUE,
        verboseIter = FALSE,
        search = "random"),
        tuneLength = 15)

model_rf_tune_auto
plot(model_rf_tune_auto)

# Manual Grid
# mtry: Number of variables randomly sampled as candidates at each split.

set.seed(42)
grid <- expand.grid(mtry = c(1:10))

model_rf_tune_man <- caret::train(
    classes ~ .,
    data = train_data,
    method = "rf",
    preProcess = c("scale", "center"),
    trControl = trainControl(
        method = "repeatedcv",
        number = 10,
        repeats = 10,
        savePredictions = TRUE,
        verboseIter = FALSE,
        search = "random"),
        tuneGrid = grid)

model_rf_tune_man
plot(model_rf_tune_man)

# Grid search with h2o
# The R package h2o provides a convenient interface to H2O, which is an open-source machine learning and deep learning platform. H2O distributes a wide range of common machine learning algorithms for classification, regression and deep learning.

library(h2o)
h2o.init(nthreads = -1)

bc_data_hf <- as.h2o(bc_data)

h2o.describe(bc_data_hf) %>%
    gather(x, y, Zeros:Sigma) %>%
    mutate(group = 
        ifelse(
            x %in% c("Min", "Max", "Mean"), "min, mean, max",
            ifelse(x %in% c("NegInf", "PosInf"), "Inf", "sigma, zeros"))) %>%
            ggplot(aes(x = Label, y = as.numeric(y), color = x)) +
            geom_point(size = 4, alpha = 0.6) +
            scale_color_brewer(palette = "Set1") +
            theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
            facet_grid(group ~ ., scales = "free") +
            labs(x = "Feature",
                y = "Value",
                color = "")

library(reshape2) # for melting

bc_data_hf[, 1] <- h2o.asfactor(bc_data_hf[, 1])

cor <- h2o.cor(bc_data_hf)
rownames(cor) <- colnames(cor)

melt(cor) %>%
    mutate(Var2 = rep(rownames(cor), nrow(cor))) %>%
    mutate(Var2 = factor(Var2, levels = colnames(cor))) %>%
    mutate(variable = factor(variable, levels = colnames(cor))) %>%
    ggplot(aes(x = variable, y = Var2, fill = value)) + 
        geom_tile(width = 0.9, height = 0.9) +
        scale_fill_gradient2(low = "white", high = "red", name = "Cor.") +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
        labs(x = "", 
            y = "")

# Training, validation and test data
splits <- h2o.splitFrame(
    bc_data_hf,
    ratios = c(0.7, 0.15),
    seed = 1)

train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

response <- "classes"
features <- setdiff(colnames(train), response)

summary(train$classes, exact_quantiles = TRUE)
summary(valid$classes, exact_quantiles = TRUE)
summary(test$classes, exact_quantiles = TRUE)

pca <- h2o.prcomp(
    training_frame = train,
    x = features,
    validation_frame = valid,
    transform = "NORMALIZE",
    impute_missing = TRUE,
    k = 3,
    seed = 42)

eigenvec <- as.data.frame(pca@model$eigenvectors)
eigenvec$label <- features

library(ggrepel)
ggplot(eigenvec, aes(x = pc1, y = pc2, label = label)) +
    geom_point(color = "navy", alpha = 0.7) +
    geom_text_repel()

# Classification
# Random Forest

hyper_params <- list(
                     ntrees = c(25, 50, 75, 100),
                     max_depth = c(10, 20, 30),
                     min_rows = c(1, 3, 5)
                     )

search_criteria <- list(
                        strategy = "RandomDiscrete", 
                        max_models = 50,
                        max_runtime_secs = 360,
                        stopping_rounds = 5,          
                        stopping_metric = "AUC",      
                        stopping_tolerance = 0.0005,
                        seed = 42
                        )

rf_grid <- h2o.grid(algorithm = "randomForest", # h2o.randomForest, 
                                                # alternatively h2o.gbm 
                                                # for Gradient boosting trees
                    x = features,
                    y = response,
                    grid_id = "rf_grid",
                    training_frame = train,
                    validation_frame = valid,
                    nfolds = 25,                           
                    fold_assignment = "Stratified",
                    hyper_params = hyper_params,
                    search_criteria = search_criteria,
                    seed = 42
                    )

# performance metrics where smaller is better -> order with decreasing = FALSE
sort_options_1 <- c("mean_per_class_error", "mse", "err", "logloss")

for (sort_by_1 in sort_options_1) {
    grid <- h2o.getGrid("rf_grid", sort_by = sort_by_1, decreasing = FALSE)
    model_ids <- grid@model_ids
    best_model <- h2o.getModel(model_ids[[1]])
    h2o.saveModel(best_model, path="models", force = TRUE)
}

# performance metrics where bigger is better -> order with decreasing = TRUE
sort_options_2 <- c("auc", "precision", "accuracy", "recall", "specificity")

for (sort_by_2 in sort_options_2) {
    grid <- h2o.getGrid("rf_grid", sort_by = sort_by_2, decreasing = TRUE)
    model_ids <- grid@model_ids
    best_model <- h2o.getModel(model_ids[[1]])
    h2o.saveModel(best_model, path = "models", force = TRUE)
}

files <- list.files(path = "models")
rf_models <- files[grep("rf_grid_model", files)]

for (model_id in rf_models) {
    path <- paste0("data/models", model_id)
    best_model <- h2o.loadModel(path)
    mse_auc_test <- data.frame(
        model_id = model_id, 
        mse = h2o.mse(h2o.performance(best_model, test)),
        auc = h2o.auc(h2o.performance(best_model, test)))
    if (model_id == rf_models[[1]]) {
        mse_auc_test_comb <- mse_auc_test
    } else {
        mse_auc_test_comb <- rbind(mse_auc_test_comb, mse_auc_test)
    }
}

mse_auc_test_comb %>%
    gather(x, y, mse:auc) %>%
    ggplot(aes(x = model_id, y = y, fill = model_id)) +
        facet_grid(x ~ ., scales = "free") +
        geom_bar(stat = "identity", alpha = 0.8, position = "dodge") +
        scale_fill_brewer(palette = "Set1") +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
            plot.margin = unit(c(0.5, 0, 0, 1.5), "cm")) +
        labs(x = "", y = "value", fill = "")

for (model_id in rf_models) {
    best_model <- h2o.getModel(model_id)
    finalRf_predictions <- data.frame(
        model_id = rep(best_model@model_id,
                        nrow(test)),
                        actual = as.vector(test$classes),
                            as.data.frame(
                                h2o.predict(
                                    object = best_model,
                                    newdata = test))
    )

    finalRf_predictions$accurate <- ifelse(
        finalRf_predictions$actual == finalRf_predictions$predict,
        "yes", "no")

    finalRf_predictions$predict_stringent <- ifelse(
        finalRf_predictions$benign > 0.8,
        "benign",
        ifelse(finalRf_predictions$malignant > 0.8, "malignant", "uncertain"))

    finalRf_predictions$accurate_stringent <- ifelse(
        finalRf_predictions$actual == finalRf_predictions$predict_stringent, "yes",
        ifelse(finalRf_predictions$predict_stringent == "uncertain", "na", "no"))

    if (model_id == rf_models[[1]]) {
        finalRf_predictions_comb <- finalRf_predictions
    } else {
        finalRf_predictions_comb <- rbind(
        finalRf_predictions_comb, finalRf_predictions)
    }
}

finalRf_predictions_comb %>%
    ggplot(aes(x = actual, fill = accurate)) +
        geom_bar(position = "dodge") +
        scale_fill_brewer(palette = "Set1") +
        facet_wrap(~ model_id, ncol = 3) +
        labs(fill = "Were\npredictions\naccurate?",
            title = "Default predictions")

finalRf_predictions_comb %>%
    subset(accurate_stringent != "na") %>%
    ggplot(aes(x = actual, fill = accurate_stringent)) +
        geom_bar(position = "dodge") +
        scale_fill_brewer(palette = "Set1") +
        facet_wrap(~ model_id, ncol = 3) +
        labs(fill = "Were\npredictions\naccurate?",
            title = "Stringent predictions")

rf_model <- h2o.loadModel("models/rf_grid_model_6")
h2o.varimp_plot(rf_model)
h2o.varimp(rf_model)

h2o.mean_per_class_error(rf_model, train = TRUE, valid = TRUE, xval = TRUE)
h2o.confusionMatrix(rf_model, valid = TRUE)

plot(rf_model,
    timestep = "number_of_trees",
    metric = "classification_error")

plot(rf_model,
    timestep = "number_of_trees",
    metric = "logloss")

plot(rf_model,
    timestep = "number_of_trees",
    metric = "AUC")

plot(rf_model,
    timestep = "number_of_trees",
    metric = "rmse")

h2o.auc(rf_model, train = TRUE)
h2o.auc(rf_model, valid = TRUE)
h2o.auc(rf_model, xval = TRUE)

perf <- h2o.performance(rf_model, test)
perf

plot(perf)
h2o.logloss(perf)
h2o.mse(perf)
h2o.auc(perf)
head(h2o.metric(perf))

finalRf_predictions <- data.frame(
    actual = as.vector(test$classes),
    as.data.frame(h2o.predict(object = rf_model,
    newdata = test)))


finalRf_predictions$accurate <- ifelse(
    finalRf_predictions$actual == finalRf_predictions$predict, "yes", "no")

finalRf_predictions$predict_stringent <- ifelse(
    finalRf_predictions$benign > 0.8, "benign",
    ifelse(finalRf_predictions$malignant  > 0.8, "malignant", "uncertain"))

finalRf_predictions$accurate_stringent <- ifelse(
    finalRf_predictions$actual == finalRf_predictions$predict_stringent, "yes",
    ifelse(finalRf_predictions$predict_stringent == "uncertain", "na", "no"))

finalRf_predictions %>%
    group_by(actual, predict) %>%
    dplyr::summarise(n = n())

finalRf_predictions %>%
    group_by(actual, predict_stringent) %>%
    dplyr::summarise(n = n())


finalRf_predictions %>%
    ggplot(aes(x = actual, fill = accurate)) +
        geom_bar(position = "dodge") +
        scale_fill_brewer(palette = "Set1") +
        labs(fill = "Were\npredictions\naccurate?",
            title = "Default predictions")

finalRf_predictions %>%
    subset(accurate_stringent != "na") %>%
    ggplot(aes(x = actual, fill = accurate_stringent)) +
        geom_bar(position = "dodge") +
        scale_fill_brewer(palette = "Set1") +
        labs(fill = "Were\npredictions\naccurate?",
            title = "Stringent predictions")

df <- finalRf_predictions[, c(1, 3, 4)]
thresholds <- seq(from = 0, to = 1, by = 0.1)
prop_table <- data.frame(
    threshold = thresholds, 
    prop_true_b = NA, 
    prop_true_m = NA)

for (threshold in thresholds) {
    pred <- ifelse(df$benign > threshold, "benign", "malignant")
    pred_t <- ifelse(pred == df$actual, TRUE, FALSE)
    group <- data.frame(df, "pred" = pred_t) %>%
    group_by(actual, pred) %>%
    dplyr::summarise(n = n())
    group_b <- filter(group, actual == "benign")
    prop_b <- sum(filter(group_b, pred == TRUE)$n) / sum(group_b$n)
    prop_table[prop_table$threshold == threshold, "prop_true_b"] <- prop_b
    group_m <- filter(group, actual == "malignant")
    prop_m <- sum(filter(group_m, pred == TRUE)$n) / sum(group_m$n)
    prop_table[prop_table$threshold == threshold, "prop_true_m"] <- prop_m
}

prop_table %>%
    gather(x, y, prop_true_b:prop_true_m) %>%
        ggplot(aes(x = threshold, y = y, color = x)) +
        geom_point() +
        geom_line() +
        scale_color_brewer(palette = "Set1") +
        labs(
            y = "proportion of true predictions",
            color = "b: benign cases\nm: malignant cases")

h2o.shutdown()



sessionInfo()
rm(list=ls())