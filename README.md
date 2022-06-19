# Coursera Practical Machine Learning Course Project
Project aims to predict human movements by wearable accelerometers records.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

# Dependencies
Codes are written in R language (v 4.1.2)
Libraries that used:
1. caret--------v 6.0.92 
2. gbm----------v 2.1.8 
3. e1071--------v 1.7.9
4. randomForest-v 4.7.1.1
5. cvms---------v 1.3.3
6. tibble-------v 3.1.6
7. ggimage------v 0.3.1
8. rsvg---------v 2.3.1
Codes can work with previous versions of packages but it wasn't tested for that.

Here let me note a warning:
The codes in this project has a tuning paramether section which requires a lot of calculations made by computer that might cause a damage on your computer. So use it wisely.
