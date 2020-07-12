

## Load packages
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not
pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071", "partykit", "rpart", "randomForest", "MASS", "xgboost", "readxl", "ggplot2") #Check, and if needed install the necessary packages

#load files
credit_Data_train <-read_excel("C:\\Users\\panes\\Desktop\\Jaspal\\Queens MMA\\MMA 867 - Predictive Modelling\\A3\\MMA867 A3 -- credit data.xlsx") # Load the datafile to R
credit_Data_predict <-read_excel("C:\\Users\\panes\\Desktop\\Jaspal\\Queens MMA\\MMA 867 - Predictive Modelling\\A3\\MMA867 A3 -- new applications.xlsx") # Load the datafile to R
credit_Data_predict$default_0 <- NA

#delete
credit_Data_predict2 <-read_excel("C:\\Users\\panes\\Desktop\\Jaspal\\Queens MMA\\MMA 867 - Predictive Modelling\\A3\\MMA867 A3 -- new applications2.xlsx") # Load the datafile to R


credit_Data <- rbind(credit_Data_train,credit_Data_predict)
str(credit_Data)
summary(credit_Data)

#__________________________________________________________________________________
###############################
# Feature Engineering/Data Wrangling
###############################
#__________________________________________________________________________________


# below FE code didn't really help with AUC
# credit_Data <- transform(credit_Data,RISK=ifelse((credit_Data$EDUCATION==3|credit_Data$EDUCATION==4|credit_Data$EDUCATION==5|credit_Data$EDUCATION==6|credit_Data$EDUCATION==0)&(credit_Data$AGE<=35)&(credit_Data$MARRIAGE==0|credit_Data$MARRIAGE==2|credit_Data$MARRIAGE==3),1,0))
# credit_Data$RISK <- as.factor(credit_Data$RISK)

# credit_Data$Avg_Balance <- (((credit_Data$BILL_AMT1-credit_Data$PAY_AMT1)+(credit_Data$BILL_AMT2-credit_Data$PAY_AMT2)+(credit_Data$BILL_AMT3-credit_Data$PAY_AMT3)+
#    (credit_Data$BILL_AMT4-credit_Data$PAY_AMT4)+(credit_Data$BILL_AMT5-credit_Data$PAY_AMT5)+(credit_Data$BILL_AMT6-credit_Data$PAY_AMT6))/6)/credit_Data$LIMIT_BAL


# adding surrogate variables for missing values in Education and Marriage columns
credit_Data$EDUCATION_srg <- as.factor(ifelse(credit_Data$EDUCATION==0, 1,ifelse(credit_Data$EDUCATION==5,1,ifelse(credit_Data$EDUCATION==6,1,0))))
credit_Data$MARRIAGE_srg <- as.factor(ifelse(credit_Data$MARRIAGE==0, 1,0))

# re-assigning values for Marriage and Education where values are 0 (missing?) & grouping based on data dictionary
credit_Data[credit_Data$MARRIAGE==0,][,"MARRIAGE"] <- 3
credit_Data[credit_Data$EDUCATION==0,][,"EDUCATION"] <- 4
credit_Data[credit_Data$EDUCATION==5,][,"EDUCATION"] <- 4
credit_Data[credit_Data$EDUCATION==6,][,"EDUCATION"] <- 4

# converting vars to factors to ensure they work better with models being used
credit_Data$SEX <- as.factor(credit_Data$SEX)
credit_Data$EDUCATION <- as.factor(credit_Data$EDUCATION)
credit_Data$MARRIAGE <- as.factor(credit_Data$MARRIAGE)
credit_Data$PAY_1 <- as.factor(credit_Data$PAY_1)
credit_Data$PAY_2 <- as.factor(credit_Data$PAY_2)
credit_Data$PAY_3 <- as.factor(credit_Data$PAY_3)
credit_Data$PAY_4 <- as.factor(credit_Data$PAY_4)
credit_Data$PAY_5 <- as.factor(credit_Data$PAY_5)
credit_Data$PAY_6 <- as.factor(credit_Data$PAY_6)
credit_Data$default_0 <- as.factor(credit_Data$default_0)

# taking log of LIMIT_BAL and AGE fields to help normalize as these were slightly skewed
credit_Data$LIMIT_BAL <- log(credit_Data$LIMIT_BAL)
credit_Data$AGE <- log(credit_Data$AGE)


# Splitting up datasets
credit_Data_train <- credit_Data[!is.na(credit_Data$default_0),]
credit_Data_predict <- credit_Data[is.na(credit_Data$default_0),-25]

summary(credit_Data_train)
summary(credit_Data_predict)

set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime 4800
inTrain <- createDataPartition(y = credit_Data_train$default_0,
                               p = 19199/24000, list = FALSE)
training <- credit_Data_train[ inTrain,]
testing <- credit_Data_train[ -inTrain,]

training <- training[,-1]
testing <- testing[,-1]


#__________________________________________________________________________________
##############################
#testing with a variety of models to see which one gives the best fit
###############################
#__________________________________________________________________________________



#__________________________________________________________________________________
####LOGISTIC REGRESSION
#__________________________________________________________________________________


model_logistic<-glm(default_0~ LIMIT_BAL+SEX+EDUCATION+MARRIAGE+AGE+PAY_1+
                      PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+
                      BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6+EDUCATION_srg+MARRIAGE_srg+SEX*MARRIAGE+MARRIAGE*EDUCATION+SEX*AGE, data=training, family="binomial"(link="logit"))

summary(model_logistic) 


model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 


###Finding predicitons: probabilities and classification
logistic_probabilities<-predict(model_logistic,newdata=testing,type="response") #Predict probabilities


logistic_classification<-rep("1",4800)
logistic_classification[logistic_probabilities<0.2]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
logistic_classification<-as.factor(logistic_classification)

###Confusion matrix  
confusionMatrix(logistic_classification,testing$default_0,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, testing$default_0)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") #Create AUC data
logistic_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
logistic_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value


#__________________________________________________________________________________
### CTREE 
#__________________________________________________________________________________


ctree_tree<-ctree(default_0~.,data=training) #Run ctree on training data
plot(ctree_tree, gp = gpar(fontsize = 8)) #Plotting the tree (adjust fontsize if needed)

ctree_probabilities<-predict(ctree_tree,newdata=testing,type="prob") #Predict probabilities
ctree_classification<-rep("1",4800)
ctree_classification[ctree_probabilities[,2]<0.2]="0" #Predict classification using 0.6073 threshold. Why 0.6073 - that's the average probability of being retained in the data. An alternative code: logistic_classification <- as.integer(logistic_probabilities > mean(testing$Retained.in.2012. == "1"))
ctree_classification<-as.factor(ctree_classification)

###Confusion matrix  
confusionMatrix(ctree_classification,testing$default_0,positive = "1")

####ROC Curve
ctree_probabilities_testing <-predict(ctree_tree,newdata=testing,type = "prob") #Predict probabilities
ctree_pred_testing <- prediction(ctree_probabilities_testing[,2], testing$default_0) #Calculate errors
ctree_ROC_testing <- performance(ctree_pred_testing,"tpr","fpr") #Create ROC curve data
plot(ctree_ROC_testing) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(ctree_pred_testing,"auc") #Create AUC data
ctree_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
ctree_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(ctree_probabilities[,2],  testing$default_0, cumulative = TRUE, n.buckets = 10) # Plot Lift chart


#__________________________________________________________________________________
###
### RPART
#__________________________________________________________________________________
###

# The rpart method has an important "complexity parameter", cp, which determines how big the tree is.  

CART_cp = rpart.control(cp = 0.0005) #set cp to a small number to "grow" a large tree

rpart_tree<-rpart(default_0~.,data=training, method="class", control=CART_cp) #"Grow" a tree on training data

prunned_rpart_tree<-prune(rpart_tree, cp=0.003) #Prun the tree. Play with cp to see how the resultant tree changes
plot(as.party(prunned_rpart_tree), type = "extended",gp = gpar(fontsize = 7)) #Plotting the tree (adjust fontsize if needed)

# Understand the relationship between the cross-validated error, size of the tree and cp.
plotcp(rpart_tree) # Use printcp(rpart_tree) to print the values. As a rule of thumb pick up the largest cp which does not give a substantial drop in error

rpart_prediction_class<-predict(prunned_rpart_tree,newdata=testing, type="class") #Predict classification (for confusion matrix)
confusionMatrix(rpart_prediction_class,testing$default_0,positive = "1") #Display confusion matrix

rpart_probabilities_testing <-predict(prunned_rpart_tree,newdata=testing,type = "prob") #Predict probabilities
rpart_pred_testing <- prediction(rpart_probabilities_testing[,2], testing$default_0) #Calculate errors
rpart_ROC_testing <- performance(rpart_pred_testing,"tpr","fpr") #Create ROC curve data
plot(rpart_ROC_testing) #Plot ROC curve

auc.tmp <- performance(rpart_pred_testing,"auc") #Create AUC data
rpart_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
rpart_auc_testing #Display AUC value

plotLift(rpart_prediction_class,  testing$default_0, cumulative = TRUE, n.buckets = 10) # Plot Lift chart




#__________________________________________________________________________________
###
### Random Forest
###
#__________________________________________________________________________________


model_forest <- randomForest(default_0~ ., data=training, 
                             type="classification",
                             importance=TRUE,
                             ntree = 2500,           # hyperparameter: number of trees in the forest
                             mtry = 50,             # hyperparameter: number of random columns to grow each tree
                             nodesize = 50,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 50,         # hyperparameter: maximum number of leafs of a tree
                             cutoff = c(.5, 0.5)   # hyperparameter: how the voting works; (0.5, 0.5) means majority vote
) 

plot(model_forest)  # plots error as a function of number of trees in the forest; use print(model_forest) to print the values on the plot

varImpPlot(model_forest) # plots variable importances; use importance(model_forest) to print the values


###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") #Predict probabilities -- an array with 2 columns: for not retained (class 0) and for retained (class 1)
forest_classification<-rep("1",4800)
forest_classification[forest_probabilities[,2]<0.2]="0" #Predict classification using 0.5 threshold. Why 0.5 and not 0.6073? Use the same as in cutoff above
forest_classification<-as.factor(forest_classification)

confusionMatrix(forest_classification,testing$default_0, positive="1") #Display confusion matrix. Note, confusion matrix actually displays a better accuracy with threshold of 50%

#There is also a "shortcut" forest_prediction<-predict(model_forest,newdata=testing, type="response") 
#But it by default uses threshold of 50%: actually works better (more accuracy) on this data


####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$default_0) #Calculate errors
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(forest_ROC) #Plot ROC curve

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") #Create AUC data
forest_AUC <- as.numeric(AUC.tmp@y.values) #Calculate AUC
forest_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(forest_probabilities[,2],  testing$default_0, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_forest <- performance(forest_ROC_prediction,"lift","rpp")
plot(Lift_forest)



#__________________________________________________________________________________
##SVM
#__________________________________________________________________________________

model_svm <- svm(default_0 ~., data=training, probability=TRUE)
summary(model_svm)

svm_probabilities<-attr(predict(model_svm,newdata=testing, probability=TRUE), "prob")
svm_prediction<-svm_probabilities[,1]

svm_classification<-rep("1",4800)
svm_classification[svm_prediction<0.2]="0" 
svm_classification<-as.factor(svm_classification)
confusionMatrix(svm_classification,testing$default_0,positive = "1")

####ROC Curve
svm_ROC_prediction <- prediction(svm_prediction, testing$default_0) #Calculate errors
svm_ROC_testing <- performance(svm_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(svm_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(svm_ROC_prediction,"auc") #Create AUC data
svm_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
svm_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value



#__________________________________________________________________________________
###
### Gradient Boosting Machines (XGboost)
#__________________________________________________________________________________
###

# As many other methods, xgboost requires matrices as inputs
# But creating them per the two lines below is a BAD idea: may lead to different levels

# x_train <-model.matrix(Retained.in.2012.~ ., data = training)
# x_test <-model.matrix(Retained.in.2012.~ ., data = testing)


credit_Data_train<- credit_Data_train[,-1]

STCdata_A_matrix <- model.matrix(default_0~ ., data = credit_Data_train)[,-1]

x_train <- STCdata_A_matrix[ inTrain,]
x_test <- STCdata_A_matrix[ -inTrain,]

y_train <-training$default_0
y_test <-testing$default_0

model_XGboost<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 8,  # hyperparameter: size of a tree in each boosting iteration
                       nround=20,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic"
)

XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="response") #Predict classification (for confusion matrix)
confusionMatrix(as.factor(ifelse(XGboost_prediction>.2,1,0)),y_test,positive="1") #Display confusion matrix

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)





##Testing Ensemble Method on Different models to help maximize the AUC/accuracy
#testProb <- (logistic_probabilities+XGboost_prediction+ctree_probabilities_testing[,1]+forest_probabilities[,1]+svm_prediction)/5
#take average 
testProb <- (logistic_probabilities+XGboost_prediction)/2
names(testProb) <- NULL


####Temporary Code 
cm <-confusionMatrix(as.factor(ifelse(testProb>.20,1,0)),y_test,positive="1") #Display confusion matrix
cm

final_pred <- as.factor(ifelse(testProb>.20,1,0))
####ROC Curve
XGboost_ROC_prediction <- prediction(testProb, y_test) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_predict, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

TN_test <- cm[2][[1]][[1]]
FN_test <- cm[2][[1]][[3]]

expected_profit_test <- 1500*TN_test - 5000*FN_test
expected_profit_test



##______________________________________________________________________________________________________________
#################################______________________________________________________________________________________________________________
###FINAL MODEL 
#################################__________________________________________________________________________________________________________________________________
#################################______________________________________________________________________________________________________________

# retrain model on full dataset & on best performing models: Logistic Regression, CTREE, XGBoost, Random Forest
#, we are leveraging ensembling (weighted average of predictions for models chosen) to maximize overall profit:

#__________________________________________________________________________________
## xgBoost Model:
#__________________________________________________________________________________

credit_Data <- credit_Data[,-1]
credit_Data$default_0[is.na(credit_Data$default_0)] <- '1'


STCdata_A_matrix <- model.matrix(default_0~ ., data = credit_Data)[,-1]

x_train <- STCdata_A_matrix[ 1:24000,]
x_predict <- STCdata_A_matrix[24001:25000,]


y_train <-credit_Data_train$default_0
y_predict <-as.factor(credit_Data_predict2$default_0)

model_XGboost_final<-xgboost(data = data.matrix(x_train), 
                       label = as.numeric(as.character(y_train)), 
                       eta = 0.2,       # hyperparameter: learning rate 
                       max_depth = 8,  # hyperparameter: size of a tree in each boosting iteration
                       nround=100,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic"
)

XGboost_prediction_final<-predict(model_XGboost_final,newdata=x_predict, type="response") #Predict classification (for confusion matrix)




#__________________________________________________________________________________
####LOGISTIC REGRESSION
#__________________________________________________________________________________

model_logistic_final<-glm(default_0~ LIMIT_BAL+SEX+EDUCATION+MARRIAGE+AGE+PAY_1+
                      PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+
                      BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6+EDUCATION_srg+MARRIAGE_srg+SEX*MARRIAGE+MARRIAGE*EDUCATION+SEX*AGE+SEX*MARRIAGE*EDUCATION+LIMIT_BAL*AGE, data=credit_Data_train, family="binomial"(link="logit"))

summary(model_logistic_final) 


model_logistic_stepwiseAIC<-stepAIC(model_logistic_final,direction = c("both"),trace = 1) #AIC stepwise
summary(model_logistic_stepwiseAIC) 


###Finding predicitons: probabilities and classification
logistic_probabilities_final<-predict(model_logistic,newdata=credit_Data_predict,type="response") #Predict probabilities


#__________________________________________________________________________________
### CTREE 
#__________________________________________________________________________________

credit_Data_train <- credit_Data_train[,-1]
ctree_treeF<-ctree(default_0~.,data=credit_Data_train) #Run ctree on training data
plot(ctree_treeF, gp = gpar(fontsize = 8)) #Plotting the tree (adjust fontsize if needed)

ctree_probabilities<-predict(ctree_treeF,newdata=credit_Data_predict,type="prob") #Predict probabilities



#__________________________________________________________________________________
####ENSEMBLE --->> take average of Logistic and XGBoost models
#__________________________________________________________________________________

names(logistic_probabilities_final) <- NULL
#take average 
finalProb <- (logistic_probabilities_final+XGboost_prediction_final)/2



#delete after
####Temporary Code 
cm_final <-confusionMatrix(as.factor(ifelse(finalProb>.20,1,0)),y_predict,positive="1") #Display confusion matrix
cm_final
####ROC Curve
XGboost_ROC_prediction <- prediction(finalProb, y_predict) #Calculate errors
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, y_predict, cumulative = TRUE, n.buckets = 10) # Plot Lift chart

### An alternative way is to plot a Lift curve not by buckets, but on all data points
Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)

TN <- cm_final[2][[1]][[1]]
FN <- cm_final[2][[1]][[3]]

expected_profit <- 1500*TN - 5000*FN
expected_profit












