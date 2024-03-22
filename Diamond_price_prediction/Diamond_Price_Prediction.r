original_data <- read.csv("diamonds.csv")

head(original_data)

dim(original_data)

summary(original_data)

############# Things to do before model fitting ######################

#First column, "X", has minimum value 1 and maximim value 53940, which is the total no of datapoints in each variable.
#so, this column is the unique id number and we will omit this from our dataset before model fitting.

##Third column,Fourth column & Fifth column, namely "cut","color" & "clarty" are in character variables. 
##Need to change those to factors.

###Need to change last 3 column variable names to "length","width", and "depth".

####we can notice that in last 3 columns, have 0 as a minimum value which does not make any sense. 
####Because if any one of those 3 variable values become zero, then that will imply dimension of a particular diamond
####reduced to 2D from 3D. So we need to drop those entire row or we can impute those values with the median values
####of the corresponding column.

#droping first column ..........
DATA <- original_data[,-1]
head(DATA,10)

#changing character type variables to factors ..................
DATA$cut <- as.factor(DATA$cut)
DATA$color <- as.factor(DATA$color)
DATA$clarity <- as.factor(DATA$clarity)
head(DATA)

#checking different levels ...........
levels(DATA$cut)
levels(DATA$color)
levels(DATA$clarity)

#changing the names of last 3 variables ...............
library(tidyverse)

DATA<-DATA %>% 
        rename('percentage_depth' = 'depth', 
               'length' = 'x',
               'width' = 'y',
               'depth' = 'z')
head(DATA)

#filtering out only non zero data points in last 3 columns ..........

DATA <-DATA %>% 
        filter(length != 0 & width != 0 & depth != 0)

summary(DATA)

#checking for any missing values in the dataset ............
data.frame(colSums(is.na(DATA)))

#checking the correlation of all the numeric variables with the target variable ("price") .............

variables_list <- names(DATA)          #listing the variables
variables_list
length(variables_list)             #total number of variables in the dataset 

correlations <- data.frame(cor(DATA[,c(variables_list[c(1,5:10)])]))    #checking the correlations between "price" 
# correlations                                                         #and all other numeric variables

correlations_price_others<- correlations[4,-4]                                   
correlations_price_others

# dropping the factor type variables ....
##dropping the variable "percentage_depth" from the dataset since it has low correlation with "price" ......

model_data <- DATA[,c(1,6:10)]
head(model_data)

variables_in_model_data <- names(model_data)
variables_in_model_data

#plotting "price" vs all other independent variables in model_data ................

# par(mfrow = c(2,2))             #creating 2X2 plots

corr.title <- function(x, y)                          #creating a correlation function
{ paste('Correlation r=', round(cor(x,y),2), sep='')} 

for(i in c(1,6,8:10)){                         #scatter plots of "price" against other variables stored in variables_list
plot(DATA[,7] ~ DATA[,i],xlab = variables_list[i],ylab = "Price",ylim = c(0,max(DATA[,7])),
     main=corr.title(DATA[,7], DATA[,i]),col = "#2471A3")
abline(lm(DATA[,7] ~ DATA[,i]),col = "red")    
}

#boxplot of all the 

library(ggplot2)

ggplot(model_data) +
  aes(x = "boxplot of 'carat'", y = carat ) +
  geom_boxplot(fill = "#2471A3") +
  theme_minimal()

ggplot(model_data) +
  aes(x = "boxplot of 'table'", y = table) +
  geom_boxplot(fill = "#2471A3") +
  theme_minimal()

ggplot(model_data) +
  aes(x = "boxplot of 'length'", y = length ) +
  geom_boxplot(fill = "#2471A3") +
  theme_minimal()

ggplot(model_data) +
  aes(x = "boxplot of 'width'", y = width ) +
  geom_boxplot(fill = "#2471A3") +
  theme_minimal()

ggplot(model_data) +
  aes(x = "boxplot of 'depth'", y = depth ) +
  geom_boxplot(fill = "#2471A3") +
  theme_minimal()

ggplot(model_data) +
  aes(x = "boxplot of 'price'", y = price ) +
  geom_boxplot(fill = "#2471A3") +
  theme_minimal()

#detecting outliers using IQR method ..................

is_outlier <- function(x, na.rm = FALSE)                   #creating a function to detect outlier using IQR method.
{                                                          #in the function 'x' is a vector quantity
  qs = quantile(x, probs = c(0.25, 0.75), na.rm = na.rm)

  lowerq <- qs[1]
  upperq <- qs[2]
  iqr = upperq - lowerq 

  extreme_threshold_upper = (iqr * 3) + upperq
  extreme_threshold_lower = lowerq - (iqr * 3)
    
  x > extreme_threshold_upper | x < extreme_threshold_lower         # Return logical vector 
}

#detecting the number of outliers using the above defined function ...............................

for (i in 1:6){                                          #here number of datapoints under TRUE represent the Outliers 
    cat("Number of Outliers in variable ", variables_in_model_data[i]," :")
    print(table(is_outlier(model_data[,i])))
}

#removing the Outliers from the model_data .........................................

remove_outliers <- function(df, cols = names(df)) {       #creating a function to remove the outliers from the dataset
  for (col in cols) {
    cat("Removing outliers in column: ", col, " \n")
    df <- df[!is_outlier(df[[col]]),]                     #filtering those rows only where no outliers are there 
  }
  df
}

outlier_free_data <- remove_outliers(model_data,variables_in_model_data)      #finally outlier free dataset

tail(outlier_free_data,20)                                     #showing last 20 rows in the outlier free data
dim(model_data)                                    #before removing outliers we had 53920 rows 
dim(outlier_free_data)                             #after removing outliers we have 53713 rows.. so in total 
                                                   #(53920-53713)= 207 outliers have been removed     

#splitting the dataset into Train and Test Dataset ............................

library("caTools")

set.seed(100)   #  setting the seed to generate always the same random numbers 

sample = sample.split(outlier_free_data,SplitRatio = 0.8) # splits the data in the ratio mentioned in SplitRatio. 
# sample                                 


train_data =subset(outlier_free_data,sample ==TRUE)#creates a training dataset named train with rows which are marked as TRUE
dim(train_data)
# train_data

test_data=subset(outlier_free_data, sample==FALSE)
dim(test_data)
# test_data


#checking the distribution of the target variable ("price") ............................

# install.packages("gridExtra")
library(gridExtra)


library(ggplot2)

plot1<- ggplot(outlier_free_data, aes(x=price))+
              geom_histogram(aes(y=..density..),color="darkblue", fill="#2471A3")+ 
              geom_density(alpha=.2, fill="white")+
              geom_vline(aes(xintercept=mean(price)),
                        color="red", linetype="dashed", size=1)+ ggtitle("            Histogram 
(before log transformation)")
    

plot2 <- ggplot(outlier_free_data, aes(x= log(price)))+
  geom_histogram(aes(y=..density..),color="darkblue", fill="#2471A3")+ 
  geom_density(alpha=.2, fill="white")+
  geom_vline(aes(xintercept=mean(log(price))),
            color="red", linetype="dashed", size=1)+ ggtitle("            Histogram 
(after log transformation)")

# grid.arrange(plot1, plot2, ncol=2, nrow = 1)                     #substitue of 'par(mfrow)'...



#Multiple Linear regression Model on train data.....................................

mlm <- lm(log(price) ~ carat+table+length+width+depth, data = train_data)

summary(mlm)

#visualising the model .......

# par(mfrow = c(2,2))
plot(mlm,col = "#2471A3")

##Multiple Linear regression Model on train data.....................................

mlm_test <- lm(log(price) ~ carat+table+length+width+depth, data = test_data)

summary(mlm_test)

#MSE (Mean Square Error) of the model on test data.....................

predicted_values <- mlm_test$fitted.values
# head(predicted_values,10)
# length(predicted_values)

observed_vs_predicted <- data.frame(observed = log(test_data$price), predicted = predicted_values)
# head(observed_vs_predicted,10)

MSE_table <- data.frame(observed=log(test_data$price), predicted=predicted_values, 
                        error=observed_vs_predicted$observed - observed_vs_predicted$predicted,
                        squared_error=(observed_vs_predicted$observed - observed_vs_predicted$predicted)^2)


head(MSE_table,10)

MSE = mean(MSE_table$squared_error)
cat("MSE for the test data =",MSE,"\n")


#MAPE (MEan Absolute Percentage Error) Measure and Accuracy of the model .............

mape_measure=mean(abs(observed_vs_predicted$observed - observed_vs_predicted$predicted)/observed_vs_predicted$observed)
accuracy=1-mape_measure
cat("Model Accuracy is :",accuracy)

#checking the assumptions for the Multiple Linear Regression Model .................

#NCV test (test for Homoscadasticity):

library(car)
print("NCV test (test for Homoscadasticity) :")
ncvTest(mlm)

#VIF test (test for Auto Correlation):
library(leaps)
print("VIF test (test for Auto Correlation):")
vif(mlm)

#durbinWatson test (test for independence of errors):
print("durbinWatson test (test for independence of errors):")
durbinWatsonTest(mlm)


##colclusion: this model does not satisfy any one of the prior requisite assumptions....
##therefore multiple linear regression model will not be suitable for this dataset though the model accuracy and
##MSE is very convincing  
