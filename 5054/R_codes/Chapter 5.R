################### R codes for Chapter 5: Resampling Methods 


############################################  AUTO DATA
library(ISLR)
Auto[1:10,]

###################### Validation approach
n<-dim(Auto)[1]
trainid<-sample(1:n,n/2)
lmfit1<-lm(mpg~poly(horsepower,2),data=Auto)


set.seed(1997)                                              ## set the randomness seed for reproducibility in the future
train=sample(392, 196)                                   ## split for train and test
lm.fit=lm(mpg~horsepower ,data=Auto,subset=train)        ## train a linear model using the subset of Auto data
mean((mpg-predict(lm.fit,Auto))[-train]^2)               ## average prediction error on test data


######################################
###################    The Validation Set Approach
######################################
library(ISLR)                                            ## load the library
set.seed(1)                                              ## set the randomness seed for reproducibility in the future
train=sample(392, 196)                                   ## split for train and test
lm.fit=lm(mpg~horsepower ,data=Auto,subset=train)        ## train a linear model using the subset of Auto data

attach(Auto)
mean((mpg-predict(lm.fit,Auto))[-train]^2)               ## average prediction error on test data


set.seed(1)                                              ## set the randomness seed for reproducibility in the future
train=sample(392, 196) 
lm.fit2=lm(mpg~poly(horsepower ,2),data=Auto,subset=train)      ## quadratic regression
mean((mpg-predict(lm.fit2,Auto))[-train]^2)                     ## now we fit a polynomial regression and get the average prediction error


set.seed(1)                                              ## set the randomness seed for reproducibility in the future
train=sample(392, 196) 
lm.fit3=lm(mpg~poly(horsepower ,3),data=Auto,subset=train)      ## cubic regression
mean((mpg-predict(lm.fit3,Auto))[-train]^2)                     ## average prediction error



######## now we use a different randomness seed, which will produce different train and test sample. Repeat the procedure and test the prediction error
set.seed(2)                                              ## set the randomness seed for reproducibility in the future
train=sample(392, 196)                                   ## split for train and test
lm.fit=lm(mpg~horsepower ,data=Auto,subset=train)        ## train a linear model using the subset of Auto data
mean((mpg-predict(lm.fit,Auto))[-train]^2)               ## average prediction error on test data
lm.fit2=lm(mpg~poly(horsepower ,2),data=Auto,subset=train)      ## quadratic regression
mean((mpg-predict(lm.fit2,Auto))[-train]^2)                     ## now we fit a polynomial regression and get the average prediction error
lm.fit3=lm(mpg~poly(horsepower ,3),data=Auto,subset=train)      ## cubic regression
mean((mpg-predict(lm.fit3,Auto))[-train]^2)                     ## average prediction error


####################### draw the plot
valErr<-rep(0,10)
set.seed(1)                                              ## set the randomness seed for reproducibility in the future
train=sample(392, 196)                                   ## split for train and test
for(k in 1:10){
  lm.fit=lm(mpg~poly(horsepower ,k),data=Auto,subset=train)      ## cubic regression
  valErr[k]<-mean((mpg-predict(lm.fit,Auto))[-train]^2)                     ## average prediction error
}
plot(1:10,valErr,type='b',col='red',xlab='Degree of Polynomial',ylab='Valdiation Error')


####################### draw the plot
valErr<-matrix(rep(0,100),10,10)
for(j in 1:10){
  set.seed(j+2000)                                              ## set the randomness seed for reproducibility in the future
  train=sample(392, 196)                                   ## split for train and test
  for(k in 1:10){
    lm.fit=lm(mpg~poly(horsepower ,k),data=Auto,subset=train)      ## cubic regression
    valErr[j,k]<-mean((mpg-predict(lm.fit,Auto))[-train]^2)                     ## average prediction error
  }
}
matplot(t(valErr), type = "l",
        col = 1:10, lty = 1, lwd = 2)



######################################
###################    Leave-One-Out Cross Validation
######################################
## The LOOCV estimate can be automatically computed for any generalized linear model using the glm() and cv.glm() functions.
glm.fit=glm(mpg~horsepower ,data=Auto)       ## fit a linear regression model with "family=...", identical to lm(mpg~horsepower, data=Auto)
coef(glm.fit)                                ## check the coefficients

library(boot)                           ##The cv.glm() function is part of the boot library.
glm.fit=glm(mpg~horsepower ,data=Auto)
cv.err=cv.glm(Auto,glm.fit)             ## LOOCV estimate of prediction error
cv.err$delta                            ## The two numbers in the delta vector contain the cross-validation results.



### we repeate the procedure for increasingly complex polynomial fits. We use for() to automate the process.
cv.error=rep(0,10)
for (i in 1:10){
  glm.fit=glm(mpg~poly(horsepower ,i),data=Auto)
  cv.error[i]=cv.glm(Auto,glm.fit)$delta[1]}
cv.error
plot(1:10,cv.error,type='b',xlab="Degree of Polynomial",ylab="CV Error",main="LOOCV")



######################################
###################    K-fold Cross Validation
######################################
##  Notice that the computation time is much shorter than that of LOOCV.
set.seed(17)
cv.error.10=rep(0,10)
for (i in 1:10){
  glm.fit=glm(mpg~poly(horsepower ,i),data=Auto)
  cv.error.10[i]=cv.glm(Auto,glm.fit,K=10)$delta[1]     ## K=10 means 10-fold cross validation
}
cv.error.10   ## Based on the results, We still see little evidence that using cubic or higher-order polynomial terms leads to lower test error than simply using a quadratic fit.
plot(1:length(cv.error.10),cv.error.10,type='b',col='blue',xlab='Degree of Polynomial',ylab='CV Error',main='10-fold CV')




######################################
###################    K-fold Cross Validation
######################################
##  Notice that the computation time is much shorter than that of LOOCV.
cv.error.10=matrix(rep(0,100),10,10)
for(j in 1:10){
  set.seed(j+2020)
  for (i in 1:10){
    glm.fit=glm(mpg~poly(horsepower ,i),data=Auto)
    cv.error.10[i,j]=cv.glm(Auto,glm.fit,K=10)$delta[1]     ## K=10 means 10-fold cross validation
  }
}
matplot(cv.error.10, type = "l",col = 1:10, lty = 1, lwd = 2,xlab="Degree of Polynomial",ylab="CV Error",main="10-fold CV")




############################################################  Using CV to determine the best K in KNN
library(caret)
X<-data.frame(x=runif(80,min=-1,max=1))
y<-(X$x)^2+2+rnorm(40,mean=0,sd=0.1)
flds <- createFolds(1:80, k = 10, list = TRUE, returnTrain = FALSE)    ### random split into 10 folds
CVErr<-rep(0,10)
for(k in 1:10){
  err<-0
  for(fold in flds){
    trainx<-data.frame(x=X[-fold,])
    testx<-data.frame(x=X[fold,])
    knnmod <- knnreg(trainx, y[-fold],k=k)
    test_y=predict(knnmod,testx)
    err<-err+sum((test_y-y[fold])^2)
  }
  CVErr[k]<-err/80
}
plot(1:10,CVErr,type='b',xlab="K",ylab="CV Error",main="10-fold CV of KNN")



####################################################### logistic regression
###########  Generate Data
fakedat<-data.frame(x=runif(300,min=-1,max=1),y=runif(300,min=-1,max=1))
fakedat$class<-apply(as.matrix(fakedat),1,function(x) ifelse((x[1]+1)^2+(x[2]+1)^2 > 2.56,1,2))
plot(fakedat$x,fakedat$y,col=fakedat$class,xlab="x",ylab="y")
x<-seq(-1,0.6,by=0.002)
lines(x,sqrt(2.56-(x+1)^2)-1,col="red",lwd=3)

######logistic polynomial regression by 10-fold CV
set.seed(17)
cv.error.10=rep(0,10)
fakedat$class<-as.factor(fakedat$class)
for (i in 1:10){
  glm.fit=glm(class~poly(x ,i)+poly(y,i),data=fakedat,family=binomial)
  cv.error.10[i]=cv.glm(fakedat,glm.fit,K=10)$delta[1]     ## K=10 means 10-fold cross validation
}
cv.error.10   ## Based on the results, We still see little evidence that using cubic or higher-order polynomial terms leads to lower test error than simply using a quadratic fit.
plot(1:length(cv.error.10),cv.error.10,type='b',col='blue',xlab='Degree of Polynomial',ylab='CV Error',main='10-fold CV by Logistic Polynomial Regression')




######logistic polynomial regression by LOOCV
set.seed(17)
cv.error=rep(0,10)
fakedat$class<-as.factor(fakedat$class)
for (i in 1:10){
  glm.fit=glm(class~poly(x ,i)+poly(y,i),data=fakedat,family=binomial)
  cv.error[i]=cv.glm(fakedat,glm.fit)$delta[1]     ## K=10 means 10-fold cross validation
}
cv.error.10   ## Based on the results, We still see little evidence that using cubic or higher-order polynomial terms leads to lower test error than simply using a quadratic fit.
plot(1:length(cv.error),cv.error,type='b',col='blue',xlab='Degree of Polynomial',ylab='CV Error',main='LOOCV by Logistic Polynomial Regression')



######################################
###################    Bootstrap
######################################
## Performing a bootstrap analysis in R entails only two steps. First, we must create a function that computes the statistic of interest. 
## Second, we use the boot() function, which is part of the boot library, to perform the bootstrap by repeatedly sampling observations from the data set with replacement.

## The Portfolio data set in the ISLR package is described in Section 5.2.
## We first create a function, alpha.fn(), which takes as input the (X,Y) data as well as a vector indicating which observations 
## should be used to estimate alpha. The function then outputs the estimate for α based on the selected observations.
alpha.fn=function(data,index){
  X=data$X[index]
  Y=data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))}

alpha.fn(Portfolio ,1:100)

set.seed(1)
alpha.fn(Portfolio,sample(100,100,replace=T))  ## to get an estimate of alpha based on 1 bootstrap sample

boot(Portfolio ,alpha.fn,R=1000)   ## output the the bootstrap statistics based on 1000 bootstrap samples





######################################
###################    Use Bootstrap to Estimate the Accuracy of a Linear Regression Model
######################################

boot.fn=function(data,index){                                    ## create the function for bootstrap to use
  return(coef(lm(mpg~horsepower ,data=data,subset=index)))}
boot.fn(Auto, 1:392)


set.seed(1)
boot.fn(Auto,sample(392,392,replace=T))                          ## linear regression on one bootstrap sample
boot.fn(Auto,sample(392,392,replace=T))                          ## linear regression on another bootstrap sample

boot.res<-boot(Auto ,boot.fn ,1000)                                     ## output the the bootstrap statistics based on 1000 bootstrap samples
hist(boot.res$t[,1],xlab="Bootstrap Estimate of Intecept")
hist(boot.res$t[,2],xlab="Bootstrap Estimate of Coefficient")

summary(lm(mpg~horsepower ,data=Auto))$coef                   ## the summary of estimated std.err by linear regression method, compare it to bootstrap estimate



### Below we compute the bootstrap standard error estimates and the standard linear regression estimates that result from fitting the quadratic model to the data.
boot.fn=function(data,index)
  coefficients(lm(mpg~horsepower+I(horsepower^2),data=data,subset=index))

set.seed(1)
boot(Auto ,boot.fn ,1000)

