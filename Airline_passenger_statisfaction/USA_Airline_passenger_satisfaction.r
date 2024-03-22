original_data <- read.csv("train.csv")

#showing first 10 rows of the data ........
head(original_data,10)

#dimension of the data .........
dim(original_data)

#considering only required variables .........................
variables_of_interest <- original_data[,-c(1,2)]
head(variables_of_interest,10)

#structure of the dataset ......................
str(variables_of_interest)

#summary statistics of the dataset .......................
summary(variables_of_interest)

#locating the missing values ................

table(is.na(variables_of_interest))

apply(is.na(variables_of_interest), 2, which) 

#changing all the 'character' variables to 'factor' ........................

variables_of_interest[sapply(variables_of_interest, is.character)] <-                        
      lapply(variables_of_interest[sapply(variables_of_interest, is.character)], as.factor) 

str(variables_of_interest)

#changing some 'integer' variables to factors ..........................

for (i in c(7:20)) {
variables_of_interest[,i] <- as.factor(variables_of_interest[,i])    
}

head(variables_of_interest,10)


#impution NA values using the median................... 

median_arrival_delay <- median(variables_of_interest$Arrival.Delay.in.Minutes, na.rm = TRUE)
median_arrival_delay

variables_of_interest[is.na(variables_of_interest$Arrival.Delay.in.Minutes),
                      "Arrival.Delay.in.Minutes"] <- median_arrival_delay

#checking the proportionality of 'satisfied' and 'nutral or dissatisfied' i the dataset ............
library("tidyverse")

variables_of_interest %>% 
    group_by(satisfaction) %>% 
    summarise(frequency = n()) %>% 
    mutate(proportion = frequency/nrow(variables_of_interest))

#loading requried libraries ...............
library('ggplot2')
library('randomForest')
library('cowplot')


#splitting train data and test data ..........................

library('caTools')
set.seed(99) 
sample = sample.split(variables_of_interest$satisfaction, SplitRatio = .80)
train = subset(variables_of_interest, sample == TRUE)
test  = subset(variables_of_interest, sample == FALSE)
# head(train,10)
# head(test,10)

#random forest model ...........................

rf_model <- randomForest(satisfaction ~., data = train, importance = TRUE)
rf_model

head(rf_model$err.rate)
dim(oob_error_data)

#OOB error rate vs tree ..............................

oob_error_data <- data.frame(
    Trees = rep(1:nrow(rf_model$err.rate),times = 3),
    Type = rep(c("OOB","Neutral or Dissatisfied","Satisfied"),each = nrow(rf_model$err.rate)),
    Error = c(rf_model$err.rate[,"OOB"], 
              rf_model$err.rate[,"neutral or dissatisfied"], 
              rf_model$err.rate[,"satisfied"]))
head(oob_error_data)
dim(oob_error_data)

# options(repr.plot.width = 20, repr.plot.height = 10)
ggplot(data = oob_error_data, aes(x = Trees, y = Error))+
    geom_line(aes(color = Type))+
    theme(text = element_text(size = 25), element_line(size = 0.8))

#varifying whether 500 trees are optimum number of trees for our random forest model or not ...............

rf_model_2 <- randomForest(satisfaction ~., data = train, ntree = 1000, importance = TRUE)
rf_model_2



#varifying whether 4 splits at each node is optimum or not ..................

# Errors <- vector(length = 10)
# for (i in 1:10){
#     temp_model <- randomForest(satisfaction ~., data = train, mtry = i, ntree = 500)
#     Errors[i] <- temp_model$err.rate[nrow(temp_model$err.rate),1] 
# }
Errors

#random forest on test data ...............

rf_model_test <- randomForest(satisfaction ~., data = test, ntree = 500, importance = TRUE)

#predictions on testing dataset .........................................

rf_predictions <- predict(rf_model_test)
table(rf_predictions)

#confusion matrix and model accuracy ...............

library("caret")
confusion_matrix_test <- confusionMatrix(predict(rf_model_test), test$satisfaction, positive = "satisfied")
confusion_matrix_test

#ploting confusion matrix ...............

# install.packages("cvms")
library("cvms")


table_test = as_tibble(confusion_matrix_test$table)
colnames(table_test)=c('Target','Prediction','N')

# options(repr.plot.width = 10, repr.plot.height = 5)
plot_confusion_matrix(table_test)


# install.packages("e1071")
# library("e1071")
# nb_model = train(train[,-23],train$satisfaction,'nb',trControl=trainControl(method='cv',number=10))

nb_model

#naive bayes model on testing dataset .........

nb_model_test = train(test[,-23],test$satisfaction,'nb',trControl=trainControl(method='cv',number=10))

nb_confusion_matrix_test <- confusionMatrix(predict(nb_model_test), test$satisfaction, positive = "satisfied")


nb_confusion_matrix_test

#visualizing the confusion matrix ...............
library("cvms")


nb_table_test = as_tibble(nb_confusion_matrix_test$table)
colnames(nb_table_test)=c('Target','Prediction','N')

# options(repr.plot.width = 10, repr.plot.height = 5)
plot_confusion_matrix(nb_table_test)
