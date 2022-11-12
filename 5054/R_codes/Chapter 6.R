################### R codes for Chapter 6: Linear Model Selection and Regularization
###################

############################################  Credit Data
library(ISLR)
Credit[1:10,]

pairs(Credit[,c(12,6,5,7,2,3,4)], main="Scatterplot Matrix",pch=19)    ## ScatterPlot Matrix

library(leaps)
regfit.full<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=20,nvmax=10)             ## best subset selection, nbest is the number of subsets of each size to record

#### plot RSS for all models
full.sum<-summary(regfit.full)
plot(as.numeric(row.names(full.sum$which)),full.sum$rss,pch=19,col="grey",xlab="Number of Predictors",ylab="Residual Sum of Squares",main="Best Subsets Selection")
#### plot RSS for best models
regfit.full1<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=1,nvmax=10)
full.sum1<-summary(regfit.full1)
lines(1:10,full.sum1$rss,col="red",lwd=3)


regfit.full1<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=1,nvmax=11)
summary(regfit.full1)



#### plot Rsq for all models
full.sum<-summary(regfit.full)
plot(as.numeric(row.names(full.sum$which)),full.sum$rsq,pch=19,col="grey",xlab="Number of Predictors",ylab="Rsquared",main="Best Subsets Selection")
#### plot Rsq for best models
regfit.full1<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=1,nvmax=10)
full.sum1<-summary(regfit.full1)
lines(1:10,full.sum1$rsq,col="red",lwd=3)


#####################  Adjusted R2,  BIC and Mallow's Cp
names(full.sum)
names(full.sum1)


#### plot adjusted Rsq for all models
regfit.full<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=20,nvmax=10)             ## best subset selection, nbest is the number of subsets of each size to record
full.sum<-summary(regfit.full)
plot(as.numeric(row.names(full.sum$which)),full.sum$adjr2,pch=19,col="grey",xlab="Number of Predictors",ylab="Adjusted Rsquared",main="Best Subsets Selection")
#### plot adjusted Rsq for best models
regfit.full1<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=1,nvmax=10)
full.sum1<-summary(regfit.full1)
lines(1:10,full.sum1$adjr2,col="red",lwd=3)


#### plot BIC for all models
regfit.full<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=20,nvmax=10)             ## best subset selection, nbest is the number of subsets of each size to record
full.sum<-summary(regfit.full)
plot(as.numeric(row.names(full.sum$which)),full.sum$bic,pch=19,col="grey",xlab="Number of Predictors",ylab="BIC",main="Best Subsets Selection")
#### plot BIC for best models
regfit.full1<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=1,nvmax=10)
full.sum1<-summary(regfit.full1)
lines(1:10,full.sum1$bic,col="red",lwd=3)



#### plot Cp for all models
regfit.full<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=20,nvmax=10)             ## best subset selection, nbest is the number of subsets of each size to record
full.sum<-summary(regfit.full)
plot(as.numeric(row.names(full.sum$which)),full.sum$cp,pch=19,col="grey",xlab="Number of Predictors",ylab="Mallow's Cp",main="Best Subsets Selection")
#### plot Cp for best models
regfit.full1<-regsubsets(Balance~.,data=Credit,method="exhaustive",nbest=1,nvmax=10)
full.sum1<-summary(regfit.full1)
lines(1:10,full.sum1$cp,col="red",lwd=3)



##################### Forward Subset Selection
regfit.fwd<-regsubsets(Balance~.,data=Credit,method="forward",nvmax=11)             ## forward subset selection
summary(regfit.fwd)


fwd.sum<-summary(regfit.fwd)
par(mfrow=c(2,2))
plot(1:11,fwd.sum$rss,type="b",col=2,lwd=2,xlab="Number of Variables",ylab="Residual Sum of Squares",main="Forward Subset Selection")
plot(1:11,fwd.sum$adjr2,type="b",col=3,lwd=2,xlab="Number of Variables",ylab="Adjusted RSquares",main="Forward Subset Selection")
plot(1:11,fwd.sum$bic,type="b",col=4,lwd=2,xlab="Number of Variables",ylab="Adjusted BIC",main="Forward Subset Selection")
plot(1:11,fwd.sum$cp,type="b",col=5,lwd=2,xlab="Number of Variables",ylab="Mallows' Cp",main="Forward Subset Selection")



##################### Backward Subset Selection
regfit.bwd<-regsubsets(Balance~.,data=Credit,method="backward",nvmax=11)             ## backward subset selection
summary(regfit.bwd)


bwd.sum<-summary(regfit.bwd)
par(mfrow=c(2,2))
plot(1:11,bwd.sum$rss,type="b",col=2,lwd=2,xlab="Number of Variables",ylab="Residual Sum of Squares",main="Backward Subset Selection")
plot(1:11,bwd.sum$adjr2,type="b",col=3,lwd=2,xlab="Number of Variables",ylab="Adjusted RSquares",main="Backward Subset Selection")
plot(1:11,bwd.sum$bic,type="b",col=4,lwd=2,xlab="Number of Variables",ylab="Adjusted BIC",main="Backward Subset Selection")
plot(1:11,bwd.sum$cp,type="b",col=5,lwd=2,xlab="Number of Variables",ylab="Mallows' Cp",main="Backward Subset Selection")




#######################  10-fold CV Errors of Best Models by Forward Subset Selections
library(boot)
CV10.err<-rep(0,10)
for(p in 1:10){
  x<-which(summary(regfit.fwd)$which[p,])
  x<-as.numeric(x)
  x<-x[-1]-1
  dfname<-c("Balance",names(Credit)[x])
  newCred<-Credit[,dfname]
  
  glm.fit=glm(Balance~. ,data=newCred)
  cv.err=cv.glm(newCred,glm.fit,K=10)
  CV10.err[p]=cv.err$delta[1]
}
CV10.err
plot(1:10,CV10.err,type="b",lwd=2,col=2,xlab="Number of Variables",ylab="CV Error",main="10-fold CV")




############################################## Ridge Regression
x=model.matrix(Balance~.,Credit)[,-1]        ## set up the covariate matrix
y=Credit$Balance                             ## set up the response vector

##We will use the glmnet package in order to perform ridge regression and the lasso.
library(glmnet)
grid=10^seq(4,-2,length=100)                  ## set up the possible values for lambda (regularization parameter)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)      ## alpha=0 for ridge regression; alpha=1 for LASSO
names(ridge.mod)                               ## results contained in the summary

ridge.mod$lambda[50]                           ## the 50-th lambda value
coef(ridge.mod)[,50]                           ## coefficient for the 50-th model
sqrt(sum(coef(ridge.mod)[-1,50]^2))            ## l2 norm of the above coefficients

ridge.mod$lambda[60]                           ## the 60-th lambda value
coef(ridge.mod)[,60]                           ## coefficient for the 60-th model
sqrt(sum(coef(ridge.mod)[-1,60]^2))            ## l2 norm of the above coefficients

plot(sqrt(colSums(coef(ridge.mod)[-1,]^2)),xlab="index",ylab="l2 norm of coeff",type='b',col=2,main="Ridge Regression")   ## plot l2 norm


## plot solution path
matplot(t(coef(ridge.mod)[c(3,4,5,10),]), type = "l",col = c(1,3,5,7), lty = c(1,2,3,4), lwd = 2,xlab="index",main="Solution Path",ylab="beta")
legend("topleft", 
       legend = c("Income", "Limit","Rating","StudentYes"), 
       col = c(1,3,5,7),
       lty=c(1,2,3,4)
)



## We can use the predict() function for a number of purposes.
predict(ridge.mod,s=50,type="coefficients")[1:13,]      ##obtain the ridge regression coefficients for a new value of lambda, say 50:

set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]                                   ## Split into train and test

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh =1e-12)   ## ridge regression with different lambda values on train dataset
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])                           ## predict on test data with lambda=4
mean((ridge.pred-y.test)^2)                                               ## average test error


ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])                        ## predict on test data with lambda=10
mean((ridge.pred-y.test)^2)               



######## use cross validation to choose the best lambda. We can do this using the built-in cross-validation function, cv.glmnet().
######## By default, it implements 10-fold CV
set.seed(1)
cv.out=cv.glmnet(x[train ,],y[train],alpha=0)    ## the default number of folds =10
plot(cv.out)                                     ## plot CV error
bestlam=cv.out$lambda.min                        ## which lambda returns the smallest CV error
ridge.pred=predict(ridge.mod,s=bestlam ,newx=x[test,])     ## use best lambda to predict on test data
mean((ridge.pred-y.test)^2)


out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:13,]        ## use full dataset to fit the ridge regression model







###################################################################  LASSO
lasso.mod=glmnet(x[train ,],y[train],alpha=1,lambda=grid) ## fit lasso with many different lambda values
plot(lasso.mod)           ## plot the coefficients w.r.t. l1 norm 

set.seed(1)
cv.out=cv.glmnet(x[train ,],y[train],alpha=1)      ## CV errors by fitting LASSO on train dataset
plot(cv.out)                                       ## plot mean squared error w.r.t. values of lambda


bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])   ## prediction on the test data using best lambda values
mean((lasso.pred-y.test)^2)



## The lasso has a substantial advantage over ridge regression in that the resulting coefficient estimates are sparse. Here we see that 12 of the 19 coefficient estimates are exactly zero.
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:75,]
lasso.coef
lasso.coef[lasso.coef!=0]













#############################################################################################################################
######   Best Subset Selection
#############################################################################################################################
library(ISLR)
# fix(Hitters)                     ## use Hitters dataset
names(Hitters)                   ## covariate names of Hitters
dim(Hitters)                     ## dimension of Hitters
sum(is.na(Hitters$Salary))       ## how many missing values

Hitters=na.omit(Hitters)         ## omit the data containing missing values
dim(Hitters)
sum(is.na(Hitters$Salary))







#############   The regsubsets() function (part of the leaps library) performs best sub- set selection by identifying the best model that contains a given number of predictors, where best is quantified using RSS
library(leaps)
regfit.full=regsubsets(Salary~.,Hitters)             ## best subset selection
summary(regfit.full)                                 ## output the results, An asterisk indicates that a given variable is included in the corresponding model.
plot(1:8,summary(regfit.full)$adjr2,type='b')                 ## adjusted r-square
plot(1:8,regfit.full$rsq,type='b')

regfit.full=regsubsets(Salary~.,data=Hitters ,nvmax=19)   ## best subset selection using up to 19 variables
reg.summary=summary(regfit.full)                          ## summary of the above regression
names(reg.summary)                                        ## what is included in the summary

reg.summary$rsq                                           ## r square of each of 19 models

par(mfrow=c(1,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",type="b")                  ## plot Residual sum squares
plot(reg.summary$adjr2 ,xlab="Number of Variables ",ylab="Adjusted RSq",type="b")       ## plot adjusted R squared
which.max(reg.summary$adjr2)                                                            ## which model has largest adjusted R square
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)   ## mark this point on plot

par(mfrow=c(1,1))
plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type="l")                   ## plot Cp statistics
which.min(reg.summary$cp )
points(10,reg.summary$cp [10],col="red",cex=2,pch=20)                                   ## mark the point of minimal Cp statistics
which.min(reg.summary$bic )                                                             ## which model has minimal BIC statistics
plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",type="l")                  ## plot BIC
points(6,reg.summary$bic [6],col="red",cex=2,pch=20)

####### The regsubsets() function has a built-in plot() command which can be used to display the selected variables for the best model
par(mfrow=c(2,2))
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")

######  We can use the coef() function to see the coefficient estimates associated with this model.
coef(regfit.full ,6)





#############################################################################################################################
######   Forward and Backward Stepwise Selection
#############################################################################################################################

## We can also use the regsubsets() function to perform forward stepwise or backward stepwise selection, using the argument method="forward" or method="backward".
regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19, method ="forward")     ## forward stepwise selection
summary(regfit.fwd)
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method ="backward")     ## backward stepwise selection
summary(regfit.bwd)

which.max(summary(regfit.fwd)$adjr2)
which.max(summary(regfit.bwd)$adjr2)
which.max(summary(regfit.full)$adjr2)

### compare best subset seletction, forward and backward selections
coef(regfit.full ,11)    ## the result from best subset selection
coef(regfit.fwd ,11)     ## the result from forward subset selection
coef(regfit.bwd ,11)     ## the result from backward subset selection


#############################################################################################################################
######   Choosing Models using Valdiation Set Approach and Cross Validation
#############################################################################################################################
set.seed(1)
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)
test =(!train)                                             ## Split the data into a train and test set

regfit.best=regsubsets(Salary~.,data=Hitters[train,], nvmax =19)  ## fit the best subset selection on train dataset
test.mat=model.matrix(Salary~.,data=Hitters[test,])               ## to get the model matrix for test dataset

val.errors=rep(NA,19)                                             ## obtain the test error for each of 19 models
for(i in 1:19){
  coefi=coef(regfit.best,id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((Hitters$Salary[test]-pred)^2) }
val.errors
par(mfrow=c(1,1))
plot(val.errors,xlab="Number of Variables ",ylab="Validation error",type='b')
which.min(val.errors)                                             ## which has the smallest test error
coef(regfit.best ,10)                                             ## coefficients of the best model


######### To automate the above process
predict.regsubsets =function (object ,newdata ,id ,...){   ## define a prediction function for regsubsets(), object=models, newdata=testdata, id=the id-th model
   form=as.formula(object$call [[2]])    ## extract the formula from regsubsets
   mat=model.matrix(form,newdata)
   coefi=coef(object ,id=id)
   xvars=names(coefi)                    ## which variables are included here
   mat[,xvars]%*%coefi }                 ## predict by X*coeff

regfit.best=regsubsets(Salary~.,data=Hitters ,nvmax=19)        ## best subset selection on full dataset
coef(regfit.best,10)                                          ## the result is different from above because there uses only train dataset


######################### now use cross validation to choose best subset model
k=10
set.seed(1)
folds=sample(1:k,nrow(Hitters),replace=TRUE)                    ## randomly split into k folds
cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))     ## create a k*19 matrix storing the CV error

for(j in 1:k){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,], nvmax=19)
  for(i in 1:19){
    pred=predict(best.fit, Hitters[folds==j,],id=i)
    cv.errors[j,i]=mean((Hitters$Salary[folds==j]-pred)^2)
  }
}
                
mean.cv.errors=apply(cv.errors ,2,mean)                       ## column-wise mean of a matrix
mean.cv.errors
par(mfrow=c(1,1))
plot(mean.cv.errors ,type="b")                                ## plot CV errors of best subset selection
#We see that cross-validation selects an 11-variable model. We now perform best subset selection on the full data set in order to obtain the 11-variable model.

reg.best=regsubsets (Salary~.,data=Hitters , nvmax=19)
coef(reg.best,11)

  
#############################################################################################################################
######   Ridge regression
#############################################################################################################################
x=model.matrix(Salary~.,Hitters)[,-1]        ## set up the covariate matrix
y=Hitters$Salary                             ## set up the response vector

##We will use the glmnet package in order to perform ridge regression and the lasso.
library(glmnet)
grid=10^seq(10,-2,length=100)                  ## set up the possible values for lambda (regularization parameter)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)      ## alpha=0 for ridge regression; alpha=1 for LASSO
names(ridge.mod)                               ## results contained in the summary

ridge.mod$lambda[50]                           ## the 50-th lambda value
coef(ridge.mod)[,50]                           ## coefficient for the 50-th model
sqrt(sum(coef(ridge.mod)[-1,50]^2))            ## l2 norm of the above coefficients

ridge.mod$lambda[60]                           ## the 60-th lambda value
coef(ridge.mod)[,60]                           ## coefficient for the 60-th model
sqrt(sum(coef(ridge.mod)[-1,60]^2))            ## l2 norm of the above coefficients
plot(sqrt(colSums(coef(ridge.mod)[-1,]^2)),xlab="index",ylab="l2 norm of coeff",type='b')

## We can use the predict() function for a number of purposes.
predict(ridge.mod,s=50,type="coefficients")[1:20,]      ##obtain the ridge regression coefficients for a new value of lambda, say 50:

set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]                                   ## Split into train and test

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh =1e-12)   ## ridge regression with different lambda values on train dataset
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])                           ## predict on test data with lambda=4
mean((ridge.pred-y.test)^2)                                               ## average test error


ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])                        ## predict on test data with lambda=10
mean((ridge.pred-y.test)^2)                                               ## average test error


######## use cross validation to choose the best lambda. We can do this using the built-in cross-validation function, cv.glmnet().
######## By default, it implements 10-fold CV
set.seed(1)
cv.out=cv.glmnet(x[train ,],y[train],alpha=0)    ## the default number of folds =10
plot(cv.out)                                     ## plot CV error
bestlam=cv.out$lambda.min                        ## which lambda returns the smallest CV error
ridge.pred=predict(ridge.mod,s=bestlam ,newx=x[test,])     ## use best lambda to predict on test data
mean((ridge.pred-y.test)^2)


out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]        ## use full dataset to fit the ridge regression model



#############################################################################################################################
######   LASSO
#############################################################################################################################
lasso.mod=glmnet(x[train ,],y[train],alpha=1,lambda=grid) ## fit lasso with many different lambda values
plot(lasso.mod)           ## plot the coefficients w.r.t. l1 norm 

set.seed(1)
cv.out=cv.glmnet(x[train ,],y[train],alpha=1)      ## CV errors by fitting LASSO on train dataset
plot(cv.out)                                       ## plot mean squared error w.r.t. values of lambda
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])   ## prediction on the test data using best lambda values
mean((lasso.pred-y.test)^2)



## The lasso has a substantial advantage over ridge regression in that the resulting coefficient estimates are sparse. Here we see that 12 of the 19 coefficient estimates are exactly zero.
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]












  
  
  
  
  
  
  
  
  













