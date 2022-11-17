########## 5054 HW2 ##########


#### Problem 1 ####

trees<-read.csv("/Users/klaus_zhangjt/desktop/trees.csv")
trees

#### 1 #### 
fit1=lm(Volume~Girth,data=trees)
fit2=lm(Volume~poly(Girth,2),data=trees)   
fit3=lm(Volume~poly(Girth,3),data=trees)
fit4=lm(Volume~poly(Girth,4),data=trees)

summary(fit1)
summary(fit2)
summary(fit3)
summary(fit4)

## prediction 
Girth<-trees$Girth
Volume<-trees$Volume
Girthlims=range(Girth)
Girth.grid=seq(from=Girthlims[1],to=Girthlims[2])
preds=predict(fit2, newdata=list(Girth=Girth.grid), se=TRUE)
se.bands=cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

par(mfrow=c(1,1))
plot(Girth, Volume, xlim=Girthlims, cex=.5, col="darkgrey")
title("Degree -2 Polynomial ")
lines(Girth.grid, preds$fit,lwd=2,col="blue")
matlines(Girth.grid,se.bands,lwd=1,col="blue",lty=3)

#### 2 #### 
fit5=glm(I(Volume>30)~poly(Girth,2),data=trees,family=binomial)
preds2=predict(fit5,newdata=list(Girth=Girth.grid),se=T)    

pfit=exp(preds2$fit)/(1+exp(preds2$fit))
se.bands.logit = cbind(preds2$fit+2*preds2$se.fit, preds2$fit-2*preds2$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
preds=predict(fit5,newdata=list(Girth=Girth.grid),type="response", se=T)

plot(Girth, I(Volume>30), xlim=Girthlims, type="n", ylim=c(0,1))
points(jitter(unlist(Girth)), I(Volume>30),cex=.5, pch="|",col =" darkgrey ")
lines(Girth.grid, pfit,lwd=2, col="blue")
matlines(Girth.grid, se.bands, lwd=1, col="blue", lty=3)

#### 3 #### 
library(splines)
fitc=lm(Volume~bs(Girth,knots=c(10,14,18),df=2),data=trees)
predc=predict(fitc,newdata=list(Girth=Girth.grid),se=T)
plot(Girth,Volume,col="gray",main="Square Spline on Selected Knots")
lines(Girth.grid,predc$fit,lwd=2,col="blue")
lines(Girth.grid,predc$fit+2*predc$se ,lty="dashed",col="red")
lines(Girth.grid,predc$fit-2*predc$se ,lty="dashed",col="red")

#### 4 #### 
library(splines)
plot(Girth,Volume,xlim=Girthlims ,cex=.5,col="darkgrey",main=" Smoothing Spline ")
fit2=smooth.spline(Girth,Volume,cv=TRUE)  # select the smoothness level by cross- validation;
fit2$df
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("3.9 DF"),col=c("blue"),lty=1,lwd=2,cex=.8)

#### 5 #### 
library(gam)
Height<-trees$Height
gam.m3=gam(Volume~s(Girth,4)+s(Height,5),data=trees)# fit a GAM model using smoothing splines
par(mfrow=c(1,2))
plot(gam.m3, se=TRUE,col="blue")        

      

#### Problem 2 ####

library(tree)
audit_train <- read.csv("/Users/klaus_zhangjt/desktop/audit_train.csv")
audit_test <- read.csv("/Users/klaus_zhangjt/desktop/audit_test.csv")
audit_train<-na.omit(audit_train)
audit_test<-na.omit(audit_test)
#### 1 #### 
audit_train$Risk<-as.factor(audit_train$Risk)
tree.audit_train=tree(Risk~., audit_train)
summary(tree.audit_train)
plot(tree.audit_train)
text(tree.audit_train,pretty=0)

p1 = predict(tree.audit_train,audit_test,type="class")
table(p1, audit_test$Risk) 

#### 2 #### 
set.seed (3)
cv.audit_train =cv.tree(tree.audit_train, FUN=prune.misclass )
names(cv.audit_train)
cv.audit_train

## We plot the error rate as a function of size.
par(mfrow=c(1,1))
plot(cv.audit_train$size ,cv.audit_train$dev ,type="b", xlab = 'Size', ylab = 'CV Error')

## We now apply the prune.misclass() function in order to prune the tree to obtain the five-node tree.
prune.audit_train=prune.misclass(tree.audit_train,best=5)
plot(prune.audit_train )
text(prune.audit_train,pretty=0)

## How well does this pruned tree perform on the test data set? Once again, we apply the predict() function.
tree.pred=predict(prune.audit_train,audit_test,type="class")
table(tree.pred ,audit_test$Risk)


#### 3 #### 

library(randomForest)
set.seed(7)
rf.audit_train=randomForest(Risk~.,data=audit_train,mtry=13,ntree=25,na.action=na.omit)   
rf.audit_train

# yhat.rf = predict(rf.audit_train ,newdata=audit_train[-train3,],type="class")
# table(yhat.rf,audit_train[-train3,]$Risk)


#### 4 #### 
set.seed(7)
rf.audittrain1=randomForest(Risk~.,data=audit_train, mtry=8,ntree=25,na.action=na.omit)   
rf.audittrain1
set.seed(7)
rf.audittrain2=randomForest(Risk~.,data=audit_train, mtry=12,ntree=25,na.action=na.omit)   
rf.audittrain2
set.seed(7)
rf.audittrain3=randomForest(Risk~.,data=audit_train, mtry=14,ntree=25,na.action=na.omit)   
rf.audittrain3
set.seed(7)
rf.audittrain4=randomForest(Risk~.,data=audit_train, mtry=16,ntree=25,na.action=na.omit)   
rf.audittrain4
set.seed(7)
rf.audittrain5=randomForest(Risk~.,data=audit_train, mtry=18,ntree=25,na.action=na.omit)   
rf.audittrain5


yhat_test1.rf = predict(rf.audittrain1,newdata=audit_test,type="class")
table(yhat_test1.rf,audit_test$Risk)


#### 5 #### 




