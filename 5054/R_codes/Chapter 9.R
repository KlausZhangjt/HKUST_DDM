################### 
###                              R codes for Chapter 9: Support Vector Machines
###################

## We use the e1071 library in R to demonstrate the support vector classifier and the SVM. 
## Another option is the LiblineaR library, which is useful for very large linear problems.

## The e1071 library contains implementations for a number of statistical learning methods. 
## In particular, the svm() function can be used to fit a support vector classifier when the argument kernel="linear" is used. 
## A cost argument allows us to specify the cost of a violation to the margin. When the cost argument is small, 
## then the mar- gins will be wide and many support vectors will be on the margin or will violate the margin. 
## When the cost argument is large, then the margins will be narrow and there will be few support vectors on the margin or violating the margin.


###################  On the Heart Data
Heart<-read.csv("Heart.csv",header=TRUE)
Heart<-na.omit(Heart)
Heart[1:10,]
Heart$AHD<-as.factor(Heart$AHD)
train<-sample(297,207)

library(e1071)
svmfit=svm(AHD~., data=Heart[train,], kernel="linear", cost=10,scale=FALSE)          ## Support Vector Classifier
summary(svmfit)
svmfit$index
### prediction on test data
ypred=predict(svmfit ,Heart[-train,])
table(predict=ypred, truth=Heart[-train,]$AHD)
sum(ypred==Heart[-train,]$AHD)/90


####################### Use a different cost value
svmfit=svm(AHD~., data=Heart[train,], kernel="linear", cost=5,scale=TRUE)          ## Support Vector Classifier
### prediction on test data
ypred=predict(svmfit ,Heart[-train,])
table(predict=ypred, truth=Heart[-train,]$AHD)
sum(ypred==Heart[-train,]$AHD)/90



####################### Use CV to choose the best cost value
set.seed(1)
tune.out=tune(svm, AHD~.,data=Heart[train,],kernel="linear",scale=TRUE,ranges=list(cost=c(1,5,10,15,20,25,30)))
bestmodel<-tune.out$best.model


#####################################  Support Vector Machine on Heart Data
svmfit=svm(AHD~., data=Heart[train,], kernel="radial", gamma=1,cost =1)             ###### radial kernel with gamma = 1
summary(svmfit)
svmfit$index
### prediction on test data
ypred=predict(svmfit ,Heart[-train,])
table(predict=ypred, truth=Heart[-train,]$AHD)
sum(ypred==Heart[-train,]$AHD)/90


####################################  Try different gamma values
svmfit=svm(AHD~., data=Heart[train,], kernel="radial", gamma=0.1,cost =1)             ###### radial kernel with gamma = 1
### prediction on test data
ypred=predict(svmfit ,Heart[-train,])
table(predict=ypred, truth=Heart[-train,]$AHD)
sum(ypred==Heart[-train,]$AHD)/90



####################################  ROC curves on training dataset:  LDA and Support Vector Classifier
library(MASS)                                                          ## load library
lda.fit<-lda(AHD~.,data=Heart[train,])
svmfit=svm(AHD~., data=Heart[train,], kernel="linear", cost=5,scale=TRUE)          ## Support Vector Classifier
library(ROCR)
lda.train<-predict(lda.fit,Heart[train,])
svmfit.train<-predict(svmfit,Heart[train,],decision.values=T)
lda.fit.pred<-prediction(lda.train$posterior[,2],Heart[train,]$AHD)
svmfit.pred<-prediction(-attributes(svmfit.train)$decision.values, Heart[train,]$AHD)
perf<-performance(lda.fit.pred,"tpr","fpr")
perfsvm<-performance(svmfit.pred,"tpr","fpr")
plot(perf,col="red",lwd=4)
plot(perfsvm,col="blue",lwd=4,add=TRUE)

legend("bottomright", 
       legend = c("LDA", "Support Vector Classifier"), 
       col = c("red","blue"),
       lty=c(1,1)
)



####################################  ROC curves on training dataset:  LDA and Support Vector Classifier
library(ROCR)
svmfit.1=svm(AHD~., data=Heart[train,], kernel="linear", cost =5) 
svmfit.train.1<-predict(svmfit.1,Heart[train,],decision.values=T)
svmfit.pred.1<-prediction(-attributes(svmfit.train.1)$decision.values, Heart[train,]$AHD)
perfsvm.1<-performance(svmfit.pred.1,"tpr","fpr")

svmfit.2=svm(AHD~., data=Heart[train,], kernel="radial",gamma=0.1,cost =1) 
svmfit.train.2<-predict(svmfit.2,Heart[train,],decision.values=T)
svmfit.pred.2<-prediction(-attributes(svmfit.train.2)$decision.values, Heart[train,]$AHD)
perfsvm.2<-performance(svmfit.pred.2,"tpr","fpr")

svmfit.3=svm(AHD~., data=Heart[train,], kernel="radial",gamma=0.01,cost =1) 
svmfit.train.3<-predict(svmfit.3,Heart[train,],decision.values=T)
svmfit.pred.3<-prediction(-attributes(svmfit.train.3)$decision.values, Heart[train,]$AHD)
perfsvm.3<-performance(svmfit.pred.3,"tpr","fpr")

svmfit.4=svm(AHD~., data=Heart[train,], kernel="radial",gamma=0.001,cost =1) 
svmfit.train.4<-predict(svmfit.4,Heart[train,],decision.values=T)
svmfit.pred.4<-prediction(-attributes(svmfit.train.4)$decision.values, Heart[train,]$AHD)
perfsvm.4<-performance(svmfit.pred.4,"tpr","fpr")

plot(perfsvm.1,col=1,lwd=4)
plot(perfsvm.2,col=2,lwd=4,add=TRUE)
plot(perfsvm.3,col=3,lwd=4,add=TRUE)
plot(perfsvm.4,col=4,lwd=4,add=TRUE)

legend("bottomright", 
       legend = c("Support Vector Classifier","SVM1","SVM2","SVM3"), 
       col = c(1,2,3,4),
       lty=c(1,1,1,1),
       lwd=c(4,4,4,4)
)









### prediction on test data
ypred=predict(svmfit ,Heart[-train,])
























#############################################################################################################################
######   Support Vector Classifier
#############################################################################################################################

## We now use the svm() function to fit the support vector classifier for a given value of the cost parameter. 
## Here we demonstrate the use of this function on a two-dimensional example so that we can plot the resulting decision boundary. 
set.seed (1)
x=matrix(rnorm(20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
plot(x, col=(3-y))


## They are not. Next, we fit the support vector classifier. Note that in order for the svm() function to perform 
## classification (as opposed to SVM-based regression), we must encode the response as a factor variable. 
## We now create a data frame with the response coded as a factor
dat=data.frame(x=x, y=as.factor(y))
library(e1071)
svmfit=svm(y~., data=dat, kernel="linear", cost=10,scale=FALSE)
##The argument scale=FALSE tells the svm() function not to scale each feature to have mean zero or standard deviation one;
## depending on the application, one might prefer to use scale=TRUE.


## now plot the support vector classifier obtained:
plot(svmfit , dat)


## The support vectors are plotted as crosses and the remaining observations are plotted as circles; 
## we see here that there are seven support vectors. We can determine their identities as follows:
svmfit$index
summary(svmfit)

## What if we instead used a smaller value of the cost parameter?
svmfit=svm(y~., data=dat, kernel="linear", cost=0.1, scale=FALSE)
plot(svmfit , dat)
svmfit$index
## Now that a smaller value of the cost parameter is being used, we obtain a larger number of support vectors,
## because the margin is now wider. Unfor- tunately, the svm() function does not explicitly output the coefficients 
## of the linear decision boundary obtained when the support vector classifier is fit, nor does it output the width of the margin.



## The e1071 library includes a built-in function, tune(), to perform cross- validation. By default, tune() 
## performs ten-fold cross-validation on a set of models of interest. In order to use this function, we pass in 
## relevant information about the set of models that are under consideration. The following command indicates 
## that we want to compare SVMs with a linear kernel, using a range of values of the cost parameter.
set.seed(1)
tune.out=tune(svm,y~.,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)


## We see that cost=0.1 results in the lowest cross-validation error rate. The tune() function stores the best 
## model obtained, which can be accessed as follows:
bestmod=tune.out$best.model 
summary(bestmod)


## The predict() function can be used to predict the class label on a set of test observations, at any given 
## value of the cost parameter. We begin by generating a test data set.
xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest [ ytest ==1 ,]= xtest [ ytest ==1 ,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))
## Now we predict the class labels of these test observations. Here we use the best model obtained through 
## cross-validation in order to make predictions.
ypred=predict(bestmod ,testdat)
table(predict=ypred, truth=testdat$y)


## Thus, with this value of cost, 19 of the test observations are correctly classified. What if we had instead 
## used cost=0.01?
svmfit=svm(y~., data=dat, kernel="linear", cost=.01, scale=FALSE)
ypred=predict(svmfit ,testdat)
table(predict=ypred, truth=testdat$y)


## Now consider a situation in which the two classes are linearly separable.
## Then we can find a separating hyperplane using the svm() function. We first further separate the two classes 
## in our simulated data so that they are linearly separable:
x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)
dat=data.frame(x=x,y=as.factor(y))
svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)
plot(svmfit , dat)


## We now try a smaller value of cost:
svmfit=svm(y~., data=dat, kernel="linear", cost=1)       
summary(svmfit)
plot(svmfit ,dat)




#############################################################################################################################
######   Support Vector Machine
#############################################################################################################################
## In order to fit an SVM using a non-linear kernel, we once again use the svm() function. However, now we use a 
## different value of the parameter kernel. To fit an SVM with a polynomial kernel we use kernel="polynomial", 
## and to fit an SVM with a radial kernel we use kernel="radial". In the former case we also use the degree 
## argument to specify a degree for the polynomial kernel

## generate data with non-linear boundary
set.seed(1)
x=matrix(rnorm(200*2), ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))

## Plotting the data makes it clear that the class boundary is indeed non- linear:
plot(x, col=y)


## The data is randomly split into training and testing groups. We then fit the training data using the svm() 
## function with a radial kernel and gamma = 1:
train=sample(200,100)
svmfit=svm(y~., data=dat[train,], kernel="radial", gamma=1,cost =1)
plot(svmfit , dat[train ,])


## The plot shows that the resulting SVM has a decidedly non-linear boundary. The summary() function can be 
## used to obtain some information about the SVM fit:
summary(svmfit)


## We can see from the figure that there are a fair number of training errors in this SVM fit. If we increase the 
## value of cost, we can reduce the number of training errors. However, this comes at the price of a more 
## irregular decision boundary that seems to be at risk of overfitting the data.
svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1, cost=1e5)
plot(svmfit ,dat[train ,])


## We can perform cross-validation using tune() to select the best choice of γ and cost for an SVM with a radial kernel:
set.seed(1)
tune.out=tune(svm, y~., data=dat[train,], kernel="radial",ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4) ))
summary(tune.out)


## Therefore, the best choice of parameters involves cost=1 and gamma=2. We can view the test set predictions for 
## this model by applying the predict() function to the data. Notice that to do this we subset the dataframe dat 
## using -train as an index set.
table(true=dat[-train,"y"], pred=predict(tune.out$best.model, newdata=dat[-train ,]))




#############################################################################################################################
######   ROC curves
#############################################################################################################################
## The ROCR package can be used to produce ROC curves such as those in Figures 9.10 and 9.11. We first write a 
## short function to plot an ROC curve given a vector containing a numerical score for each observation, pred, 
## and a vector containing the class label for each observation, truth.
library(ROCR) 
rocplot=function(pred, truth, ...){
  predob = prediction (pred, truth)
  perf = performance (predob , "tpr", "fpr") 
  plot(perf ,...)}


## In order to obtain the fitted values for a given SVM model fit, we use decision.values=TRUE when fitting svm().
## Then the predict() function will output the fitted values.
svmfit.opt=svm(y~., data=dat[train,], kernel="radial", gamma=2, cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values


## Now we can produce the ROC plot.
par(mfrow=c(1,2))
rocplot(fitted ,dat[train ,"y"],main="Training Data")


## SVM appears to be producing accurate predictions. By increasing gamma we can produce a more flexible fit and 
## generate further improvements in accuracy.
svmfit.flex=svm(y~., data=dat[train,], kernel="radial", gamma=50, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
rocplot(fitted ,dat[train ,"y"],add=T,col="red")


## However, these ROC curves are all on the training data. We are really more interested in the level of prediction
## accuracy on the test data. When we compute the ROC curves on the test data, the model with gamma = 2 appears 
## to provide the most accurate results.
fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],main="Test Data")
fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red")





#############################################################################################################################
######   SVM with Multiple Classes
#############################################################################################################################
## If the response is a factor containing more than two levels, then the svm() function will perform multi-class 
## classification using the one-versus-one ap- proach. We explore that setting here by generating a third class 
## of obser- vations.
set.seed(1)
x=rbind(x, matrix(rnorm(50*2), ncol=2))
y=c(y, rep(0,50))
x[y==0,2]=x[y==0,2]+2
dat=data.frame(x=x, y=as.factor(y))
par(mfrow=c(1,1))
plot(x,col=(y+1))


## We now fit an SVM to the data:
svmfit=svm(y~., data=dat, kernel="radial", cost=10, gamma=1) 
plot(svmfit , dat)





#############################################################################################################################
######   Application to Gene Expression Data
#############################################################################################################################
## We now examine the Khan data set, which consists of a number of tissue samples corresponding to four distinct 
## types of small round blue cell tu- mors. For each tissue sample, gene expression measurements are available. 
## The data set consists of training data, xtrain and ytrain, and testing data, xtest and ytest.
library(ISLR)
names(Khan)
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)


## This data set consists of expression measurements for 2,308 genes.
## The training and test sets consist of 63 and 20 observations respectively.
table(Khan$ytrain)
table(Khan$ytest)


## We will use a support vector approach to predict cancer subtype using gene expression measurements. 
## In this data set, there are a very large number of features relative to the number of observations. 
## This suggests that we should use a linear kernel, because the additional flexibility that will result from 
## using a polynomial or radial kernel is unnecessary.
dat=data.frame(x=Khan$xtrain , y=as.factor(Khan$ytrain ))
out=svm(y~., data=dat, kernel="linear",cost=10)
summary(out)



## We see that there are no training errors. In fact, this is not surprising, because the large number of 
## variables relative to the number of observations implies that it is easy to find hyperplanes that fully 
## separate the classes. We are most interested not in the support vector classifier’s performance on the 
## training observations, but rather its performance on the test observations.
dat.te=data.frame(x=Khan$xtest , y=as.factor(Khan$ytest))
pred.te=predict(out, newdata=dat.te)
table(pred.te, dat.te$y)
sum(pred.te==dat.te$y)/20



################################################### SVM
svmfit.2=svm(y~., data=dat, kernel="radial",gamma=10,cost=10)
dat.te=data.frame(x=Khan$xtest , y=as.factor(Khan$ytest))
pred.te=predict(svmfit.2, newdata=dat.te)
pred.tr=predict(svmfit.2,newdata=dat)
table(pred.te, dat.te$y)
sum(pred.tr==dat$y)/63      ##### train error
sum(pred.te==dat.te$y)/20   ##### test error






