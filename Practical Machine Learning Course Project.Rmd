---
title: "Practical Machine Learning Project"
output:
  html_document:
    df_print: paged
  pdf_document: default
---
# Brief
This project is about predicting human movements. Wearable devices are getting common usage. In our data set users wore some devices and performed to do 5 different movements, and these accelerometers recorded changes in belt, arm, forearm and dumbell. The main idea is to investigate if we can determine which movement has been performed by these records. 
# Downloading training and test sets:
```{r}
options(warn=-1)
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
```     

```{r}
download.file(url, destfile = paste(getwd(),"/training.csv",sep = ""))
```

```{r}
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```         

```{r}
download.file(testurl, destfile = paste(getwd(),"/testing.csv",sep = ""))

```
```{r}
training <- read.csv(paste(getwd(),"/training.csv",sep = ""))
testing <- read.csv(paste(getwd(),"/testing.csv",sep = ""))
dim(training);dim(testing)
```
# Preparing the data
```{r}
head(training)
str(training$classe)
```
Since our target variable is classe and it is a categorical variable we should make it a factor variable. 
```{r}
training$classe <- as.factor(training$classe)

```
Let's have a look which columns have NA values. 
```{r}
colSums(is.na(training))[colSums(is.na(training))>100]
```
There are missing values in some columns in training and test sets. We will drop them from both sets. So we are going to drop merge of empty coulmns in the training and test set.   
```{r}
empty_columns <- colnames(training)[colSums(is.na(training))>100]
training = training[,!(names(training) %in% empty_columns)]
testing = testing[,!(names(testing) %in% empty_columns)]
empty_chars <- colnames(training)[sapply(training,function(x) table(as.character(x) =="")["TRUE"])>100]
empty_chars <- empty_chars[!(is.na(empty_chars))]
training = training[,!(names(training) %in% empty_chars)]
testing = testing[,!(names(testing) %in% empty_chars)]
dim(training); dim(testing)
```
```{r}
head(training,2);head(testing,2)
```
Now we have a clean data. Let's do some exploring. Let's have a look at some plots.
# Exploring the data
```{r}
library(caret)
featurePlot(x=training[,c(7,11,14,16)], y=training$classe, plot="pairs")
```

In this plot we can see classes are piled. Sure we can look at any plot of combination of different covariants and seek for a relation. Let's try another one but a regression line added on it:
```{r}
q <- qplot(roll_belt, pitch_belt, data=training, color=classe)
q + geom_smooth(method="lm", formula = y ~ x)
```

In this plot we can see some classe categories are piled on different places in the plot. So there are some ways to distinguish the classes but relations between them still not kind of a regression problem. And those variables are positive numbers. Are the left of the covariants has similar ranges with this ? 
What is the range of the data excluding user_name, new_window and classe column?
```{r}
sapply(training[,-c(2,5,6,60)], range)
```
We have negative and zero values as we expected. And we have some date-time information. This might can cause it to consider as a time-series forecasting but there is no value to forecast. In each row classe variable is defined. So we can't use previous knowledge to forecast the classe because every movement is also classified in classe. Let's have an ensembling approach for this data. We have training data but our test data has no classe variable. So we should create a new training and test sets from training data. And we will remove date-time information because classification is not relevant to date-time information. Also user_name, x and new_window is irrelevant. As a starting model we are going to take %5 train set and %95 test set.

# Model Selecting
```{r}
set.seed(323)
inTrain <- createDataPartition(y = training$classe, p = 0.05, list = F)
train <- training[-c(1,2,3,4,5,6)][inTrain,]
tester <- training[-c(1,2,3,4,5,6)][-inTrain,]
```

Now since we are going to make ensembling models, we should choose some classifying algorithms and measure accuracy and decide which is better and finally tuning hyperparameters of that model. Sure we can try to tune every model we try, but it will take a serious time. So we will use Random Forest, RPart, Gradient Boosting Machine, Support Vector Machine, Naive Bayes and linear discriminant analysis (lda) as classifier methods. And our measurement will be accuracy which is calculated as TP+TN/TP+TN+FP+FN. We will use "gbm" package for gradient boosting machine and "e1071" package for support vector machine and naive bayes classifier and "randomForest" package for random forest classifier.
We will train models with those methods and make predictions and save accuracy in an array named accuracy_rate.
```{r}
library(gbm)
library(e1071)
library(randomForest)
method = c("Random Forest","RPart", "Gradient Boosting", "Support Vector Machine", "Naive Bayes", "Lda")
accuracy_rate = c()
accuracy <- function(pred, test){
  sum(diag(table(pred, test)))/sum(table(pred, test))
}

set.seed(1252)
model_rf <- train(classe~., method = "rf", data = train)
pred_rf <- predict(model_rf, tester)
accuracy_rate <- append(accuracy_rate, accuracy(pred_rf, tester$classe))
model_rpart <- train(classe~., method = "rpart", data = train)
pred_rp <- predict(model_rpart, tester)
accuracy_rate <- append(accuracy_rate, accuracy(pred_rp, tester$classe))
model_gbm <- train(classe~., method = "gbm", data = train,verbose=FALSE)
pred_gbm <- predict(model_gbm, tester)
accuracy_rate <- append(accuracy_rate, accuracy(pred_gbm, tester$classe))
model_sv <- svm(classe~.,  data = train)
pred_sv <- predict(model_sv, tester)
accuracy_rate <- append(accuracy_rate, accuracy(pred_sv, tester$classe))
model_nb <- naiveBayes(classe~.,  data = train, laplace=1)
pred_nb <- predict(model_nb, tester)
accuracy_rate <- append(accuracy_rate, accuracy(pred_nb, tester$classe))
model_lda <- train(classe~.,method="lda",  data = train)
pred_lda <- predict(model_lda, tester)
accuracy_rate <- append(accuracy_rate, accuracy(pred_lda, tester$classe))
                                                                      
```
We have trained models and deployed accuracy rates. And let's have a look which method worked well for this data:
```{r}
df <- data.frame(method, accuracy_rate)
df[order(accuracy_rate,decreasing = T),]
```
Random Forest has the best accuracy overall. But gradient boosting has also good results. As we mentioned before we used set.seed(323) to choose train set in particular randomness. Algorithms work well in different kinds of data. What if our random train set was better for Random Forest algorithm and when train data become larger our model will have faulty results? Let's compare these two models with different train sets.  
```{r}
set.seed(101)
inTrain <- createDataPartition(y = training$classe, p = 0.05, list = F)
train <- training[-c(1,2,3,4,5,6)][inTrain,]
tester <- training[-c(1,2,3,4,5,6)][-inTrain,]
set.seed(152)
second_model_rf <- randomForest(classe~., data = train)
second_pred_rf <- predict(second_model_rf, tester)
second_model_gbm <- train(classe~., method = "gbm", data = train,verbose=FALSE)
second_pred_gbm <- predict(second_model_gbm, tester)
paste("Random Forest accuracy:", accuracy(second_pred_rf, tester$classe))
paste("Gradient Boosting Machine accuracy:", accuracy(second_pred_gbm, tester$classe))
```
Accuracy is still pretty good for both. But random forest has slightly better results. Let's have a look at confusion matrix for random forest model:
```{r}
library(cvms)
library(tibble)
library(ggimage)
library(rsvg)
tbl <- tibble("Test"=tester$classe, "Prediction"=second_pred_rf)
conf_mat <- confusion_matrix(targets = tbl$Test, predictions = tbl$Prediction)
plot_confusion_matrix(conf_mat$`Confusion Matrix`[[1]])
```

# Tuning the hyperparameters
Random Forest is a better algorithm for our data. We are going to tune hyper parameters. For this we will create trainControl and expand.grid options. This method will create different size and proportion of samples and train the model with all options. But for the final model training set will be larger. 
So let's create a bigger train set and make a deeper training in random forest model:
```{r}
set.seed(323)
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = F)
train <- training[-c(1,2,3,4,5,6)][inTrain,]
tester <- training[-c(1,2,3,4,5,6)][-inTrain,]
control <- trainControl(method="repeatedcv", number=5, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry = (1:60)) 
set.seed(1252)
tuned_model <- train(classe~.,method ="rf", data = train,tuneGrid= tunegrid, trControl=control)
pred_rf <- predict(tuned_model, tester)
confusionMatrix(pred_rf, tester$classe)
```
Let's look at our model's properties:
```{r}
print(tuned_model$finalModel)
```
Confussion Matrix with other metrics:
```{r}
confusionMatrix(pred_rf, tester$classe)
```

Now let's plot our tuned model:
```{r}
plot(tuned_model$finalModel, main="Tuned Random Forest Model")
```
```{r}
varImpPlot(tuned_model$finalModel, main = " Variable Importance For Tuned Random Forest Model")
```

# Final Test
From the start of this project we have downloaded the test data. Let's make our final predictions with this tuned random forest model:
```{r}
final_pred <- predict(tuned_model, testing)
final_pred
write.csv(array(final_pred), paste(getwd(),"/predictions.csv",sep = ""), row.names = F)
```
# Conclusion
We have predicted movements by accelerometers records using random forest method with high accuracy. Although we didn't tune the Gradient Boosting Machine method, we still can assume that this data could be modelled by gbm method too.  

