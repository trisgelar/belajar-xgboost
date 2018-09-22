# For data manipulation and tidying
library(dplyr)

# For data visualizations
library(ggplot2)
# install.packages("fpc")
library(fpc)

# For modeling and predictions
library(caret)
# install.packages("glmnet")
library(glmnet)
# install.packages("ranger")
library(ranger)
# install.packages("e1071")
library(e1071)
# install.packages("clValid")
library(clValid)

d_train <- read.csv('data/monster_train.csv', header = TRUE, stringsAsFactors = FALSE)
d_train$Dataset <- "train"
d_test <- read.csv('data/monster_test.csv', header = TRUE, stringsAsFactors = FALSE)
d_test$Dataset <- "test"
full <- bind_rows(d_train, d_test)


str(full)
summary(full)

We have 8 variables currently:

# * **ID** : Appears to be the identification number of the monster in question
# * **Bone Length** : Average length of the bones in the creature, normalized to 0 - 1
# * **Rotting Flesh** : Percentage of flesh on the creature that is rotting
# * **Hair Length** : Average length of the hair on the creature, normalized from 0 - 1
# * **Has Soul** : The percentage of a soul present in the creature
# * **Color** : The color of the creature
# * **Type** : The category of the creature (i.e. ghoul, goblin or ghost)
# * **Dataset** : The column I added when importing data indicating whether the observation was part of the original training or test set

# It seems like a few of these variables would serve better as factors, rather than character strings, so I'll take care of that. 

factor_variables <- c('id', 'color', 'type', 'Dataset')
full[factor_variables] <- lapply(full[factor_variables], function(x) as.factor(x))

# Data Exploration

train_2 <- full[full$Dataset == 'train', ]

# Distribution of Continuous Variables by Creature Type{.tabset}

# Bone Length
ggplot(train_2, 
    aes(x = type, 
        y = bone_length, 
        fill = type)) + 
    geom_boxplot() +
    guides(fill = FALSE) + 
    xlab("Creature") + 
    ylab("Bone Length") +
    scale_fill_manual(values = c("#D55E00", "#0072B2", "#009E73"))

#### Rotting Flesh

ggplot(train_2, 
    aes(x = type, 
        y = rotting_flesh, 
        fill = type)) + 
    geom_boxplot() +
    guides(fill = FALSE) + 
    xlab("Creature") + 
    ylab("Percentage of Rotting Flesh") + 
    scale_fill_manual(values = c("#D55E00", "#0072B2", "#009E73"))


#### Hair Length

ggplot(train_2, 
    aes(x = type, 
        y = hair_length, 
        fill = type)) + 
    geom_boxplot() +
    guides(fill = FALSE) + 
    xlab("Creature") + 
    ylab("Hair Length") + 
    scale_fill_manual(values = c("#D55E00", "#0072B2", "#009E73"))


#### Soul

ggplot(train_2, 
    aes(x = type, 
        y = has_soul, 
        fill = type)) + 
    geom_boxplot() +
    guides(fill = FALSE) + 
    xlab("Creature") + 
    ylab("Percentage of Soul Present") + 
    scale_fill_manual(values = c("#D55E00", "#0072B2", "#009E73"))


### Distribution of Color by Creature Type{.tabset}
#### Ghost

ghost_color <- train_2 %>%
        filter(type == 'Ghost') %>%
        group_by(color) %>%
        summarise(count = n())
ggplot(ghost_color,
    aes(x = color,
        y = count,
        fill = color)) +
    geom_bar(stat = "identity") + 
    xlab("Color") + 
    ylab("Number of Observations") +
    scale_fill_manual(values = c("Black", "#D55E00", "#0072B2", "#F0E442", "#009E73", "#999999")) + 
    theme(panel.grid.minor = element_blank()) + 
    ylim(0, 50) + 
    guides(fill = FALSE)


#### Ghoul

ghoul_color <- train_2 %>%
        filter(type == 'Ghoul') %>%
        group_by(color) %>%
        summarise(count = n())
ggplot(ghoul_color,
    aes(x = color,
        y = count,
        fill = color)) +
    geom_bar(stat = "identity") + 
    xlab("Color") + 
    ylab("Number of Observations") +
    scale_fill_manual(values = c("Black", "#D55E00", "#0072B2", "#F0E442", "#009E73", "#999999")) + 
    theme(panel.grid.minor = element_blank()) + 
    ylim(0, 50) + 
    guides(fill = FALSE)


#### Goblin
goblin_color <- train_2 %>%
    filter(type == 'Goblin') %>%
    group_by(color) %>%
    summarise(count = n())
ggplot(goblin_color,
    aes(x = color,
        y = count,
        fill = color)) +
    geom_bar(stat = "identity") + 
    xlab("Color") + 
    ylab("Number of Observations") +
    scale_fill_manual(values = c("Black", "#D55E00", "#0072B2", "#F0E442", "#009E73", "#999999")) + 
    theme(panel.grid.minor = element_blank()) + 
    ylim(0, 50) + 
    guides(fill = FALSE)

pairs(full[,2:5], 
    col = full$type, 
    labels = c("Bone Length", "Rotting Flesh", "Hair Length", "Soul"))

full <- full %>%
    mutate(hair_soul = hair_length * has_soul)

full_1 <- full %>%
    filter(!is.na(type))

ggplot(full_1, 
    aes(x = type, 
        y = hair_soul, 
        fill = type)) + 
    geom_boxplot() +
    guides(fill = FALSE) + 
    xlab("Creature") + 
    ylab("Combination of Hair/Soul") + 
    scale_fill_manual(values = c("#D55E00", "#0072B2", "#009E73"))

full <- full %>%
    mutate(bone_flesh = bone_length * rotting_flesh,
        bone_hair = bone_length * hair_length,
        bone_soul = bone_length * has_soul,
        flesh_hair = rotting_flesh * hair_length,
        flesh_soul = rotting_flesh * has_soul)

summary(full)

### Cluster Without Categorical Variables
# Set the seed
set.seed(100)

# Extract creature labels and remove column from dataset
creature_labels <- full$type
full2 <- full
full2$type <- NULL

# Remove categorical variables (id, color, and dataset) from dataset
full2$id <- NULL
full2$color <- NULL
full2$Dataset <- NULL

# Perform k-means clustering with 3 clusters, repeat 30 times
creature_km_1 <- kmeans(full2, 3, nstart = 30)

plotcluster(full2, creature_km_1$cluster)

dunn_ckm_1 <- dunn(clusters = creature_km_1$cluster, Data = full2)
dunn_ckm_1

table(creature_km_1$cluster, creature_labels)

train_complete <- full[full$Dataset == 'train', ]
test_complete <- full[full$Dataset == 'test', ]

# Because I plan on using the `caret` package for all of my modeling, I'm going to generate a standard `trainControl` so that those tuning parameters remain consistent throughout the various models.

### Creating trainControl
# I will create a system that will perform 20 repeats of a 10-Fold cross-validation of the data. 
myControl <- trainControl(
    method = "cv", 
    number = 10,
    repeats = 20, 
    verboseIter = TRUE)


### Random Forest Modeling
set.seed(10)

rf_model <- train(
    type ~ bone_length + rotting_flesh + hair_length + has_soul + color + hair_soul + bone_flesh + bone_hair + 
        bone_soul + flesh_hair + flesh_soul,
    tuneLength = 3,
    data = train_complete, 
    method = "ranger", 
    trControl = myControl,
    importance = 'impurity'
)

# Creating a Variable Importance variable
vimp <- varImp(rf_model)

# Plotting "vimp"
ggplot(vimp, 
    top = dim(vimp$importance)[1])

# Huh.  Our "hair_soul" variable seems to be the most important to this model and our other interactions rank pretty highly.  I suppose we can hold on to them for now.  Color, on the other hand, hardly plays into this.  Let's try removing it from a second random forest model.  
set.seed(10)

rf_model_2 <- train(
    type ~ bone_length + rotting_flesh + hair_length + has_soul + hair_soul + bone_flesh + bone_hair + 
        bone_soul + flesh_hair + flesh_soul,
    tuneLength = 3,
    data = train_complete, 
    method = "ranger", 
    trControl = myControl,
    importance = 'impurity'
)

### GLMnet Modeling

# I'm going to follow the random forest model up with a glmnet model, also from the `caret` package.
set.seed(10)

glm_model <- train(
    type ~ bone_length + rotting_flesh + hair_length + has_soul + color + hair_soul + bone_flesh + bone_hair + 
        bone_soul + flesh_hair + flesh_soul, 
    method = "glmnet",
    tuneGrid = expand.grid(alpha = 0:1,
      lambda = seq(0.0001, 1, length = 20)),
    data = train_complete,
    trControl = myControl
)

set.seed(10)

glm_model_2 <- train(
    type ~ bone_length + rotting_flesh + hair_length + has_soul + hair_soul + bone_flesh + bone_hair + 
        bone_soul + flesh_hair + flesh_soul, 
    method = "glmnet",
    tuneGrid = expand.grid(alpha = 0:1,
      lambda = seq(0.0001, 1, length = 20)),
    data = train_complete,
    trControl = myControl
)

### Comparing model fit
# Create a list of models
models <- list(rf = rf_model, rf2 = rf_model_2, glmnet = glm_model, glmnet2 = glm_model_2)

# Resample the models
resampled <- resamples(models)

# Generate a summary
summary(resampled)

# Plot the differences between model fits
dotplot(resampled, metric = "Accuracy")

## Predicting Creature Identity
test_complete <- test_complete %>%
    arrange(id)

my_prediction <- predict(glm_model_2, test_complete)

# my_solution_GGG_03 <- data.frame(id = test_complete$id, Type = my_prediction)
# write.csv(my_solution_GGG_03, file = "my_solution_GGG_03.csv", row.names = FALSE)