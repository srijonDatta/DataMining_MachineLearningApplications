data <- read.csv("onlinefraud.csv")

#showing first 10 rows of the dataset ............
head(data,10)

dim(data)            #dimenssion of the dataset

DATA <- data[,-c(1,4,7,11)]             #removing the first and last column named "step","isFlagedFraud"
head(DATA,10)               #reading first 10 observations in the dataset

str(DATA)               #structure of the dataset

summary(DATA)            #summary of the dataset

dim(DATA)

names(DATA)             #variable names in the dataset

library("tidyverse")

new_DATA <-
DATA %>% 
rename('transaction_type'='type',                   #renaming variables for better understandings 
       'transaction_amount'='amount',
       'balance_before_transaction_at_origin'='oldbalanceOrg',
       'balance_after_transaction_at_origin'='newbalanceOrig',
       'balance_before_transaction_at_destination'='oldbalanceDest',
       'balance_after_transaction_at_destination'='newbalanceDest') %>% 
mutate(isFraud = as.factor(isFraud)) %>%                                       #converting 'isFraud' variable from integer
mutate(transaction_type = as.factor(transaction_type))                                                                          # to factor

head(new_DATA,10)



data.frame(colSums(is.na(new_DATA)))        #checking for missing values

#Checking our target variable...

new_DATA$isFraud %>%       #our target variable is binary in nature and it has two factors, namely, '0' and '1'
    levels()


#Checking the proportions of '0's and '1's in our target variable...

new_DATA %>%                         
    group_by(isFraud) %>% 
    summarise(freq = n()) %>% 
    mutate(prop = freq/nrow(new_DATA))

#Conductiong Stratified Sampling ...

# install.packages("rsample")
library(rsample)

set.seed(100)
stratify_sample <- initial_split(new_DATA, proportions = 0.75, strata = isFraud)

stratify_sample      #checking the spliting

#Creating training dataset ...
stratify_train <- training(stratify_sample)
nrow(stratify_train)       #no of rows in training dataset

#Creating testing dataset ...
stratify_test <- testing(stratify_sample)
nrow(stratify_test)       #no of rows in training dataset

stratify_train 


temp_train <- filter(stratify_train,isFraud == 0)        #filtering out for "isFraud = 0"
head(temp_train,10)
dim(temp_train)


set.seed(100)

reduced_training_dataset <- 
        temp_train[sample(nrow(temp_train), 2000), ]   #taking 2000 samples randomly from 785561 rows

reduced_training_dataset
dim(reduced_training_dataset)

Temp_train <- filter(stratify_train,isFraud == 1)      #filtering out for "isFraud = 1"
dim(Temp_train)



down_sample_training <- rbind(reduced_training_dataset,Temp_train)         #training data after downsampling  
down_sample_training

stratify_test             

temp_test <- filter(stratify_test,isFraud == 0)     #filtering out for "isFraud = 0"
temp_test


set.seed(100)

reduced_testing_dataset <- 
        temp_test[sample(nrow(temp_test), 666), ]     #taking 666 samples randomly from 261872 rows,.. 
                                                      #..maintaining the proportion like training set 
reduced_testing_dataset

Temp_test <- filter(stratify_test,isFraud == 1)

down_sample_testing <- rbind(reduced_testing_dataset,Temp_test)
down_sample_testing

#Cross-checking the proportionality of '0' and '1' for the train data...

down_sample_training %>% 
    group_by(isFraud) %>% 
    summarise(frequency = n()) %>% 
    mutate(proportion = frequency/nrow(down_sample_training))   

#Cross-checking the proportionality of '0' and '1' for the test data ...

down_sample_testing %>% 
    group_by(isFraud) %>% 
    summarise(frequency = n()) %>% 
    mutate(proportion = frequency/nrow(down_sample_testing)) 

# install.packages("FSelector")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("data.tree")

library(FSelector)
library(rpart)
library(rpart.plot)
library(data.tree)

#decision tree modelling on train dataset ...
decision_tree <- rpart(isFraud ~. , data = down_sample_training)



summary(decision_tree)

#model performance and accuracy on test data .................... 
library("caret")

decisiontree_model_predictions <- predict(decision_tree, down_sample_testing, type = "class")
table(decisiontree_model_predictions)
table(down_sample_testing$isFraud)

decisiontree_confusion_matrix_test <- confusionMatrix(decisiontree_model_predictions, down_sample_testing$isFraud, 
                                                     positive = "1")
decisiontree_confusion_matrix_test

#visialising confusion matrix ..........................

library("cvms")


decisiontree_table_test = as_tibble(decisiontree_confusion_matrix_test$table)
colnames(decisiontree_table_test)=c('Target','Prediction','N')

# options(repr.plot.width = 15, repr.plot.height = 5)         #to resize chart in notebook 
plot_confusion_matrix(decisiontree_table_test)

#visualizing the tree using rpart.plot function ...................

# options(repr.plot.width = 10, repr.plot.height = 8)
rpart.plot(decision_tree, # middle graph
type=4,
extra=101, 
box.palette="GnBu",
branch.lty=3, 
shadow.col="gray", 
nn=TRUE
)

#ROC curve ..........................................
# install.packages("ROCR")

# options(repr.plot.width = 5, repr.plot.height = 5)

library("ROCR")
pred <- prediction(predict(decision_tree, down_sample_testing, type="prob")[, 2], down_sample_testing$isFraud)

plot(performance(pred, "tpr", "fpr"),main = "ROC curve", colorize = T)
abline(0, 1, lty = 2)



#area under ROC curve ...................................
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values[[1]]

#renaming all the variables and levels of the factor variables following the norm of "caret" package.

demo <- down_sample_training %>% 
    rename('transactiontype'='transaction_type',                    
       'transactionamount'='transaction_amount',
       'balancebeforetransactionatorigin'='balance_before_transaction_at_origin',
       'balanceaftertransactionatorigin'='balance_after_transaction_at_origin',
       'balancebeforetransactionatdestination'='balance_before_transaction_at_destination',
       'balanceaftertransactionatdestination'='balance_after_transaction_at_destination',
       'isfraud'='isFraud')

levels(demo$transactiontype) <- c('CashIn','CashOut','Debit','Payment','Transfer')      #In "caret" package, while..
levels(demo$isfraud) <- c('NotFraud','Fraud')                                    #..xgboost modeling, the internal..  
                                                                 #..algorithm tries to change the factor levels..
head(demo)             #..to x0,x1,..etc, So it doesnot allow any scecial characters such as('_','.'etc) or numbers   

#tunning parameters for the model ....................
library("caret")

parameter_tune <- expand.grid(
    nrounds = 1500,          #number of trees
    max_depth = c(3,6),             #depth for each tree
    eta = 0.3,                     #learning rate (generally, c(0.025,0.05,0.1,0.3) are preferable)
    gamma = 1,                      #pruning -> should be tuned, i.e c(0, 0.05,0.1,0.5,0.7,0.9,1.0)
    colsample_bytree = c(0.8,1),           #generally, c(0.4,0.6,0.8,1) -> subsample ratio of columns for tree
    min_child_weight = 1,           #generally, c(1,2,3) -> the larger, the more censervative the model is..
    subsample = 1)                  #used to prevent overfitting 



#3 fold cross validation .......

train_control <- trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE)
    

#xgboost model on training data ...................................

library("caret")
xgboost_model <- train(isfraud ~., data = demo, method = "xgbTree", 
                           trControl = train_control, tuneGrid = parameter_tune )

#model summary ......................
xgboost_model

##renaming all the variables and levels of the factor variables following the norm of "caret" package.

demo_test <- down_sample_testing %>% 
    rename('transactiontype'='transaction_type',                   
       'transactionamount'='transaction_amount',
       'balancebeforetransactionatorigin'='balance_before_transaction_at_origin',
       'balanceaftertransactionatorigin'='balance_after_transaction_at_origin',
       'balancebeforetransactionatdestination'='balance_before_transaction_at_destination',
       'balanceaftertransactionatdestination'='balance_after_transaction_at_destination',
       'isfraud'='isFraud')

levels(demo_test$transactiontype) <- c('CashIn','CashOut','Debit','Payment','Transfer')
levels(demo_test$isfraud) <- c('NotFraud','Fraud') 

head(demo_test)   

#model performance and accuracy on test data .................... 

xgboost_model_predictions <- predict(xgboost_model,demo_test)

xgboost_confusion_matrix_test <- confusionMatrix(xgboost_model_predictions, demo_test$isfraud)
xgboost_confusion_matrix_test

#visualizing confusion matrix ..................

# install.packages("cvms")
library("cvms")


xgboost_table_test = as_tibble(xgboost_confusion_matrix_test$table)
colnames(xgboost_table_test)=c('Target','Prediction','N')

# options(repr.plot.width = 15, repr.plot.height = 5)
plot_confusion_matrix(xgboost_table_test)

#ROC curve ..........................................

# options(repr.plot.width = 5, repr.plot.height = 5)

library("ROCR")

pred_xgb <- prediction(predict(xgboost_model,demo_test, type="prob")[, 1], demo_test$isfraud)

plot(performance(pred_xgb, "tpr", "fpr"),main = "ROC curve", colorize = T)
abline(0, 1, lty = 2)


#area under ROC curve ...................................
xgb.auc.perf = performance(pred_xgb, measure = "auc")
xgb.auc.perf@y.values[[1]]

