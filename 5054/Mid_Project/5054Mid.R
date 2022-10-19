########################## Midterm project - R code ##########################
library(boot) 
library(leaps)
library(glmnet)

######################## problem 1 ########################

## use python ##

## get results from R to compare
car<-read.csv("/Users/klaus_zhangjt/desktop/cars.csv")
car

### question 1 ###
cv.error=rep(0,10)
for (i in 1:10){
  glm.fit=glm(dist~poly(speed ,i),data=car)
  cv.error[i]=cv.glm(car,glm.fit)$delta[1]}
cv.error
plot(1:10,cv.error,type='b',xlab="Degree of Polynomial",ylab="CV Error",main="LOOCV")

### question 2 ###
set.seed(7)
cv.error.5=rep(0,10)
for (i in 1:10){
  glm.fit=glm(dist~poly(speed ,i),data=car)
  cv.error.5[i]=cv.glm(car,glm.fit,K=5)$delta[1]    ## K=5 means 5-fold cross validation
}
cv.error.5  
plot(1:length(cv.error.5),cv.error.5,type='b',col='blue',xlab='Degree of Polynomial',ylab='CV Error',main='5-Fold CV')

### question 2 ###
cv.error.5=matrix(rep(0,100),10,10)
for(j in 1:10){
  set.seed(j+2022)
  for (i in 1:10){
    glm.fit=glm(dist~poly(speed ,i),data=car)
    cv.error.5[i,j]=cv.glm(car,glm.fit,K=5)$delta[1]    
  }
}
matplot(cv.error.5, type = "l",col = 1:5, lty = 1, lwd = 2,xlab="Degree of Polynomial",ylab="CV Error",main="5-fold CV")




######################## problem 2 ########################

## use python ##

###########################################################




######################## problem 3 ########################
gpa<-read.csv("/Users/klaus_zhangjt/desktop/FirstYearGPA.csv")
gpa

### question 1 ###
# plot Rsq for best models
regfit.full1<-regsubsets(GPA~.,data=gpa,method="exhaustive",nbest=1,nvmax=8)
full.sum1<-summary(regfit.full1)
full.sum1

plot(full.sum1$rsq,pch=19,col="blue",xlab="Model size",ylab="R-square",main="Best subset selection")
lines(1:8,full.sum1$rsq,col="red",lwd=3)

full.sum1$rsq

plot(full.sum1$adjr2,pch=19,col="blue",xlab="Model size",ylab="Adjusted R-square",main="Adjusted R-square of 8 best models")
lines(1:8, full.sum1$adjr2,col="red",lwd=3)

full.sum1$adjr2


### question 2 ###
regfit.full1<-regsubsets(GPA~.,data=gpa,method="exhaustive",nbest=1,nvmax=8)

set.seed(66518)
CV.err<-rep(0,8)
for(p in 1:8){
  x<-which(summary(regfit.full1)$which[p,])
  x<-as.numeric(x)
  x<-x[-1]-1
  dfname<-c("GPA",names(gpa)[x])
  newCred<-gpa[,dfname]
  
  glm.fit=glm(GPA~. ,data=newCred)
  cv.err=cv.glm(newCred,glm.fit,K=5)
  CV.err[p]=cv.err$delta[1]
}
CV.err
plot(1:8,CV.err,type="b",lwd=2,col=2,xlab="Model size",ylab="CV Error",main="5-Fold CV")

l <- lm(GPA~HSGPA+SATV+HU+White, data=gpa)
summary(l)

### question 3 ###
regfit.fwd<-regsubsets(GPA~.,data=gpa,method="forward",nvmax=8) ## forward subset selection
sf <- summary(regfit.fwd)
sf

plot(sf$adjr2,pch=19,col="blue",xlab="Model size",ylab="Adjusted R-square",main="Forward stepwise selection ")
lines(1:8, sf$adjr2,col="red",lwd=3)

sf$adjr2
sf$bic
min(sf$bic)

### question 4 ####

set.seed(777)
CV.err1<-rep(0,8)
for(p in 1:8){
  x<-which(summary(regfit.fwd)$which[p,])
  x<-as.numeric(x)
  x<-x[-1]-1
  dfname<-c("GPA",names(gpa)[x])
  newCred<-gpa[,dfname]
  
  glm.fit=glm(GPA~. ,data=newCred)
  cv.err=cv.glm(newCred,glm.fit,K=5)
  CV.err1[p]=cv.err$delta[1]
}
CV.err1
plot(1:8,CV.err1,type="b",lwd=2,col=2,xlab="Model size",ylab="CV Error",main="5-Fold CV")

l1 <- lm(GPA~HSGPA+HU+White, data=gpa)
summary(l1)


######################## problem 4 ########################

train <- read.csv("/Users/klaus_zhangjt/desktop/diabetes_train.csv")
test <- read.csv("/Users/klaus_zhangjt/desktop/diabetes_test.csv")

library(glmnet)
grid=10^seq(4,-2,length=100) 
x=model.matrix(Y~.,train)[,-1]        
y=train$Y                            

lasso.mod=glmnet(x,y,alpha=1,lambda=grid) ## fit lasso with many different lambda values
plot(lasso.mod)           ## plot the coefficients w.r.t. l1 norm 


set.seed(7)
cv.out=cv.glmnet(x,y,alpha=1)   ## the default number of folds =10
plot(cv.out)  

bestlam=cv.out$lambda.min
bestlam

out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:75,]
lasso.coef
lasso.coef[lasso.coef!=0]

x1=model.matrix(Y~.,test)[,-1]        
y1=test$Y   
lasso.pred=predict(lasso.mod,s=bestlam,newx=x1)   ## prediction on the test data using best lambda values
mean((lasso.pred-y1)^2)



