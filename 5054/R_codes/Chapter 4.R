
################### The Stock Market Data: This data set consists of percentage returns for the S&P 500 stock index over 1, 250 days, 
################### from the beginning of 2001 until the end of 2005. For each date, we have recorded the percentage returns for each 
################### of the five previous trading days, Lag1 through Lag5. We have also recorded Volume (the number of shares traded on
################### the previous day, in billions), Today (the percentage return on the date in question) and Direction (whether the 
################### market was Up or Down on this date).

library(ISLR)                                            ## load the library
names(Smarket)                                           ## show the features of Stock Market data
Smarket[1:10,]                                           ## show the first 10 rows of the data
cor(Smarket)                                             ## calculate the correlation between features
cor(Smarket[,-9])                                        ## calculate the correlation between features without the 9th feature
attach(Smarket)    ## attach to the search path of R
plot(Volume)                                             ## plot the Volume feature


##################  Default Data
library(ISLR)
Default[1:10,]

################# plot
pty<-rep(1,10000)
pty[which(Default$default=='Yes')]=3
pcl<-rep('blue',10000)
pcl[which(Default$default=='Yes')]='red'
plot(Default$balance,Default$income,pch=pty,col=pcl,xlab="Balance",ylab="Income")


################## BoxPlot
boxplot(balance~default,data=Default,col=c("blue","red"))
boxplot(income~default,data=Default,col=c("blue","red"))



##################  Logistic regression
glmod1<-glm(default~balance, data=Default,family=binomial)
summary(glmod1)

##################  Logistic regression
glmod2<-glm(default~student, data=Default,family=binomial)
summary(glmod2)

#################  prediction
xnew<-data.frame(student='yes',balance=1000,income=99)
pred1<-predict(glmod1,xnew,type="response")
pred1



#################  prediction
xnew<-data.frame(student='Yes',balance=1000,income=99)
pred2<-predict(glmod2,xnew,type="response")
pred2



##################  Logistic regression
glmod3<-glm(default~., data=Default,family=binomial)
summary(glmod3)


#################### For student and non-student
DefaultY<-Default[which(Default$student=='Yes'),]
glmod4<-glm(default~balance, data=DefaultY,family=binomial)
summary(glmod4)


#################### For student and non-student
DefaultN<-Default[which(Default$student=='No'),]
glmod5<-glm(default~balance, data=DefaultN,family=binomial)
summary(glmod5)


####################  Output probability
DefaultT<-data.frame(balance=seq(500,2500,by=1))
pred4<-predict(glmod4,DefaultT,type="response")
pred5<-predict(glmod5,DefaultT,type="response")
plot(DefaultT$balance,pred4,main="Predicted Probability of Default for Students and non-Students",
     xlab="Balance",ylab="Probability of Default",type="l", lwd=3,col="red")
lines(DefaultT$balance,pred5,lwd=3,col="blue")
legend("topleft", 
       legend = c("Student", "non-Student"), 
       col = c("red","blue"),
       lty=c(1,1)
       )


##################################LDA on Default data
library(MASS)                                                          ## load library
lda.fit<-lda(default~.,data=Default)
lda.train<-predict(lda.fit,Default)
table(lda.train$class,Default$default)

### change the default probability to 0.2
lda.fit.pred1<-rep("No",10000)
lda.fit.pred1[which(lda.train$posterior[,2]>0.2)]<-"Yes"
table(lda.fit.pred1,Default$default)



### change the default probability to 0.1
lda.fit.pred2<-rep("No",10000)
lda.fit.pred2[which(lda.train$posterior[,2]>0.1)]<-"Yes"
table(lda.fit.pred2,Default$default)



### ROC curve for LDA
library(ROCR)
lda.fit.pred3<-prediction(lda.train$posterior[,2],Default$default)
perf<-performance(lda.fit.pred3,"tpr","fpr")
plot(perf,colorize=TRUE,lwd=4)



#################### Test Error of LDA
train.ids<-sample(seq(1,10000),9000)
lda.fit.mod2<-lda(default~.,data=Default,subset=train.ids)
lda.fit.pred<-predict(lda.fit,Default[-train.ids,])           ## threshold=0.5 by default
table(lda.fit.pred$class,Default[-train.ids,]$default)



#################### Test Error of LDA
train.ids<-sample(seq(1,10000),9000)
lda.fit.mod2<-lda(default~.,data=Default,subset=train.ids)
lda.fit.pred<-predict(lda.fit,Default[-train.ids,])           ## threshold=0.5 by default
pred4<-rep("No",1000)
pred4[which(lda.fit.pred$posterior[,2]>0.2)]<-"Yes"           ##  Set threshold to 0.2
table(lda.fit.pred$class,Default[-train.ids,]$default)



################# QDA
qda.fit.mod1<-qda(default~.,data=Default)  
qda.fit.mod1
qda.fit.pred1<-predict(qda.fit.mod1,Default)
table(qda.fit.pred1$class,Default$default)



################## QDA ROC cruve
library(ROCR)
qda.fit.pred<-prediction(qda.fit.pred1$posterior[,2],Default$default)
perf<-performance(qda.fit.pred,"tpr","fpr")
plot(perf,colorize=TRUE,lwd=4)


#################  AUC for LDA
train.ids<-sample(seq(1,10000),9000)
lda.fit.mod2<-lda(default~.,data=Default,subset=train.ids)
Default_test<-Default[-train.ids,]
lda.fit.pred<-predict(lda.fit.mod2,Default_test) 
lda.fit.pred3<-prediction(lda.fit.pred$posterior[,2],Default_test$default)
perf<-performance(lda.fit.pred3,"auc")
perf@y.values[[1]]




#################  AUC for QDA
qda.fit.mod2<-qda(default~.,data=Default,subset=train.ids)
Default_test<-Default[-train.ids,]
qda.fit.pred<-predict(qda.fit.mod2,Default_test) 
qda.fit.pred3<-prediction(qda.fit.pred$posterior[,2],Default_test$default)
perf<-performance(qda.fit.pred3,"auc")
perf@y.values[[1]]




#################   ROC and AUC for Logistic regression
glm.fit.mod2<-glm(default~.,data=Default,subset=train.ids,family=binomial)
Default_test<-Default[-train.ids,]
glm.fit.pred<-predict(glm.fit.mod2,Default_test,"response") 
glm.fit.pred3<-prediction(glm.fit.pred,Default_test$default)
perf<-performance(glm.fit.pred3,measure = "tpr", x.measure = "fpr")
plot(perf,colorize=TRUE,lwd=4,main="ROC on Test Data by Logistic Regression")



#################   ROC and AUC for Logistic regression
glm.fit.mod2<-glm(default~.,data=Default,subset=train.ids,family=binomial)
Default_test<-Default[-train.ids,]
glm.fit.pred<-predict(glm.fit.mod2,Default_test,"response") 
glm.fit.pred3<-prediction(glm.fit.pred,Default_test$default)
perf<-performance(glm.fit.pred3, "auc")
perf@y.values[[1]]



#################  Test error of KNN on Default Data
Default$student<-as.integer(Default$student)   ## convert the categorical variable to integer
Default_train<-Default[train.ids,]
Default_test<-Default[-train.ids,]
knn.fit.mod2<-class::knn(Default_train[,2:4],Default_test[,2:4],Default_train$default,k=5)  ###k=5
table(knn.fit.mod2, Default_test$default)



################################  Plot TestError Versus K
TestErr<-rep(0,10)
for (K in seq(1,10)){
        Default_train<-Default[train.ids,]
        Default_test<-Default[-train.ids,]
        knn.fit.mod2<-class::knn(Default_train[,2:4],Default_test[,2:4],Default_train$default,k=K)  ###k=5
        X<-table(knn.fit.mod2, Default_test$default)
        TestErr[K]<-(X[1,2]+X[2,1])/1000
}
plot(1:10,TestErr,lwd=3,type="l",col="red",xlab="K",ylab="Test Error")






################### Logistic regression on the Stock Market Data to predict "Direction" using Lag1 through Lag5 and Volume. 
################### The glm() function fits generalized glm() linear models, a class of models that includes logistic regression. 
################### The syntax generalized of the glm() function is similar to that of lm(), except that we must pass in linear model 
################### the argument family=binomial in order to tell R to run a logistic regression rather than some other type of 
################### generalized linear model.
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=Smarket ,family=binomial)           ## fit a logistic regression model on Smarket      
summary(glm.fit)                                                                                 ## show the results of logistic regression
coef(glm.fit)                                                                                    ## access only the coefficients
summary(glm.fit)$coef                                                                            ## access only the coefficients and its statistics
summary(glm.fit)$coef[,4]                                                                        ## access only the p-values of coefficients
glm.probs=predict(glm.fit,type="response")                                                       ## predict the probability of direction, since no test data is input, it calucaltes the probabilities on training data       
glm.probs[1:10]
contrasts(Direction)                                                                             ## checks whether R use "up" or "down" as 1. 
glm.pred=rep("Down",1250)         ## creates a vector of 1,250 Down elements
glm.pred[glm.probs >.5]="Up"      ## transforms to Up all of the elements for which the predicted probability of a market increase exceeds 0.5
table(glm.pred,Direction)         ## produce a confusion matrix in order to determine how many observations were correctly or incorrectly classified.
mean(glm.pred==Direction )        ## calculate the fraction of of days which the predictions are correct (basically, the accuracy on training data)                               

########### Now we still apply Logistic regression on Stock Market Data, but we use 2001-2004 data as training, and 2005 data as test.
train=(Year<2005)                      ## the row indices for training 
Smarket.2005=Smarket[!train,]          ## the 2005 data points as test data
dim(Smarket.2005)                      ## show the size of test data
Direction.2005=Direction[!train]       ## the direction in test data
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=Smarket ,family=binomial,subset=train)      ## fit Logistic regression on training data
glm.probs=predict(glm.fit,Smarket.2005,type="response")                                                  ## prediction on the test data
glm.pred=rep("Down",252)
glm.pred[glm.probs >.5]="Up"
table(glm.pred,Direction.2005)    ## produce a confusion matrix to determine how many observations were correctly or incorrectly classified.
mean(glm.pred==Direction.2005)    ## accuracy on testing data
mean(glm.pred!=Direction.2005)    ## test error

########### Now we apply Logistic regression on Stock Market Data using Lag1 and Lag2,  we use 2001-2004 data as training, and 2005 data as test.
glm.fit=glm(Direction~Lag1+Lag2,data=Smarket ,family=binomial, subset=train)
glm.probs=predict(glm.fit,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs >.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)     ## accuracy on testing data

#################   ROC and AUC for Logistic regression
glm.probs<-predict(glm.fit,Smarket.2005,"response") 
glm.fit.pred<-prediction(glm.probs,Direction.2005)
perf1<-performance(glm.fit.pred,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)
perf<-performance(glm.fit.pred, "auc")
perf@y.values[[1]]



################### Now, we perform LDA on the Smarket data to predict direction. Similar as above, we use 2001-2004 as train data and 2005 as test data.
################### We just use Lag1 and Lag2 as the predictors.
library(MASS)                                                          ## load library
lda.fit=lda(Direction~Lag1+Lag2,data=Smarket ,subset=train)            ## you can use this only when Smarket is attached to R search path
lda.fit                                                                ## show the results
plot(lda.fit)
lda.pred=predict(lda.fit, Smarket.2005)                                ## LDA prediction on test data
names(lda.pred)
lda.class=lda.pred$class                                               ## obtain the predicted classes on test data
table(lda.class ,Direction.2005)                                       ## confusion matrix
mean(lda.class==Direction.2005)                                        ## accuracy on test data
lda.pred$posterior                                                     ## show the calculated probabilities on test data

################# ROC and AUC for LDA
lda.fit.pred<-prediction(lda.pred$posterior[,2],Direction.2005)
perf1<-performance(lda.fit.pred,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)
perf<-performance(lda.fit.pred,"auc")
perf@y.values[[1]]




################### Now, we perform QDA on the Smarket data to predict direction. Similar as above, we use 2001-2004 as train data and 2005 as test data.
################### We just use Lag1 and Lag2 as the predictors.
qda.fit=qda(Direction~Lag1+Lag2,data=Smarket ,subset=train)           ## fit a quaratic discriminant analysis model on training data
qda.fit                                                               ## show the results of QDA
qda.class=predict(qda.fit,Smarket.2005)$class                         ## prediction on test data
table(qda.class ,Direction.2005)                                      ## confusion matrix
mean(qda.class==Direction.2005)                                       ## accuracy on test data
################# ROC and AUC for LDA
qda.pred=predict(qda.fit,Smarket.2005)
qda.fit.pred<-prediction(qda.pred$posterior[,2],Direction.2005)
perf1<-performance(qda.fit.pred,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)
perf<-performance(qda.fit.pred,"auc")
perf@y.values[[1]]



################### Now, we perform KNN on the Smarket data to predict direction. Similar as above, we use 2001-2004 as train data and 2005 as test data.
################### We just use Lag1 and Lag2 as the predictors.
library(class)                                 ## load library
train.X=cbind(Lag1 ,Lag2)[train ,]             ## construct train data
test.X=cbind(Lag1,Lag2)[!train,]               ## construct test data
train.Direction =Direction [train]             ## construct test outcomes
set.seed(1)                      ## We set a random seed before we apply knn() because if several observations are tied as nearest neighbors, then R will randomly break the tie. Therefore, a seed must be set in order to ensure reproducibil- ity of results.
knn.pred=knn(train.X,test.X,train.Direction ,k=1)    ## train KNN with k=1 on train data and predict on test data
table(knn.pred,Direction.2005)                ## confusion matrix
mean(knn.pred==Direction.2005)   
knn.pred=knn(train.X,test.X,train.Direction ,k=3)    ## train KNN with k=3 on train data and predict on test data
table(knn.pred,Direction.2005)                ## confusion matrix
mean(knn.pred==Direction.2005)  






