################### 
###                              R codes for Chapter 8: Tree-based Methods
###################

## The tree library is used to construct classification and regression trees.
library(tree)

##################################################################### Hitters Data
library(ISLR)
Hitters[1:10,]

########## Regression Tree
library(tree)
pHitters<-Hitters
pHitters$Salary<-log(pHitters$Salary)
tree.Hitters=tree(Salary~Years+Hits,pHitters)
summary(tree.Hitters)
## plot the tree
plot(tree.Hitters)
text(tree.Hitters ,pretty=0)


########## Regression Tree
tree.Hitters.fit1=prune.tree(tree.Hitters,best=3)    ### only keep 3 terminal nodes
summary(tree.Hitters.fit1)
## plot the tree
plot(tree.Hitters.fit1)
text(tree.Hitters.fit1,pretty=0)


########### Using More Predictors
tree.Hitters=tree(Salary~Years+Hits+RBI+Walks+PutOuts+Runs,pHitters)
summary(tree.Hitters)
## plot the tree
plot(tree.Hitters)
text(tree.Hitters ,pretty=0)


########### Using More Predictors
tree.Hitters=tree(Salary~Years+Hits+RBI+Walks+PutOuts+Runs,pHitters,control =tree.control(dim(Hitters)[1],mindev=0.05))
summary(tree.Hitters)
## plot the tree
plot(tree.Hitters)
text(tree.Hitters ,pretty=0)


########### Using More Predictors
tree.Hitters=tree(Salary~Years+Hits+RBI+Walks+PutOuts+Runs,pHitters,control =tree.control(dim(Hitters)[1],mindev=0.005))
summary(tree.Hitters)
## plot the tree
plot(tree.Hitters)
text(tree.Hitters ,pretty=0)




########### Using More Predictors
tree.Hitters=tree(Salary~Years+Hits+RBI+Walks+PutOuts+Runs,pHitters,split="gini",control =tree.control(dim(Hitters)[1],minsize=40))
summary(tree.Hitters)
## plot the tree
plot(tree.Hitters)
text(tree.Hitters ,pretty=0)





###################################################  Tree Pruning
tree.Hitters=tree(Salary~Years+Hits+RBI+Walks+PutOuts+Runs,pHitters,control =tree.control(dim(Hitters)[1],mindev=0.007))
summary(tree.Hitters)
## plot the tree
plot(tree.Hitters)
text(tree.Hitters ,pretty=0)

prune.Hitters=prune.tree(tree.Hitters ,best=7)
plot(prune.Hitters)
text(prune.Hitters ,pretty=0)


## Now we use the cv.tree() function to see whether pruning the tree will improve performance.
cv.Hitters=cv.tree(tree.Hitters)
plot(cv.Hitters$size ,cv.Hitters$dev ,type="b",xlab="Tree Size",ylab="Sum of Deviations")





#####################################################  Heart data
Heart<-read.csv("Heart.csv",header=TRUE)
Heart[1:10,]
Heart$AHD<-as.factor(Heart$AHD)
Heart$Thal<-as.factor(Heart$Thal)
Heart$ChestPain<-as.factor(Heart$ChestPain)
tree.Heart=tree(AHD~.,Heart)
summary(tree.Heart)
## plot the tree
plot(tree.Heart)
text(tree.Heart ,pretty=0)



#####################################################  Heart data
Heart<-read.csv("Heart.csv",header=TRUE)
Heart[1:10,]
Heart$AHD<-as.factor(Heart$AHD)
Heart$Thal<-as.factor(Heart$Thal)
Heart$ChestPain<-as.factor(Heart$ChestPain)
tree.Heart=tree(AHD~.,Heart,control =tree.control(dim(Hitters)[1],mindev=0.015))
summary(tree.Heart)
## plot the tree
plot(tree.Heart)
text(tree.Heart ,pretty=0)


cv.Heart =cv.tree(tree.Heart ,FUN=prune.misclass )
names(cv.Heart)
cv.Heart

plot(cv.Heart$size ,cv.Heart$dev ,type="b",xlab="Size",ylab="CV Error")






####################################### Bagging
library(randomForest)
set.seed(1)
Heart<-na.omit(Heart)
train=sample(dim(Heart)[1],250)
bag.Heart=randomForest(AHD~.,data=Heart,subset=train,mtry=13,ntree=50,na.action=na.omit)   ## using all 13 variables in each tree and 10 trees
bag.Heart

yhat.bag = predict(bag.Heart ,newdata=Heart[-train ,],type="class")
table(yhat.bag,Heart[-train,]$AHD)
length(which(yhat.bag==Heart[-train,]$AHD))/47

importance(bag.Heart)



#######################################  Random Forest
library(randomForest)
set.seed(1)
Heart<-na.omit(Heart)
train=sample(dim(Heart)[1],250)
rf.Heart=randomForest(AHD~.,data=Heart,subset=train,mtry=5,ntree=150,na.action=na.omit)   
rf.Heart

yhat.rf = predict(rf.Heart ,newdata=Heart[-train ,],type="class")
table(yhat.rf,Heart[-train,]$AHD)
length(which(yhat.rf==Heart[-train,]$AHD))/47



#####################################  Boosting Tree
library(gbm)
set.seed(1)
Heart<-read.csv("Heart.csv",header=TRUE)
Heart<-na.omit(Heart)
train=sample(dim(Heart)[1],250)
Heart[which(Heart$AHD=="Yes"),]$AHD<-1
Heart[which(Heart$AHD=="No"),]$AHD<-0
#Heart$AHD <- as.factor(Heart$AHD)
Heart$ChestPain<-as.factor(Heart$ChestPain)
Heart$Thal<-as.factor(Heart$Thal)

boost.Heart=gbm(AHD~.,data=Heart[train,],distribution = "bernoulli",n.trees=100, interaction.depth=3)
summary(boost.Heart)

## We now use the boosted model to predict medv on the test set:
yhat.boost=predict(boost.Heart,newdata=Heart[-train,], n.trees=100,type="response")
pred.boost=rep(0,47)
pred.boost[yhat.boost>0.5]=1
table(pred.boost,Heart[-train,]$AHD)
length(which(pred.boost==Heart[-train,]$AHD))/47





## We see that lstat and rm are by far the most important variables. We can also produce partial dependence plots for these two variables. 
## These plots illustrate the marginal effect of the selected variables on the response after integrating out the other variables. 
## In this case, as we might expect, median house prices are increasing with rm and decreasing with lstat.
par(mfrow=c(1,2)) 
plot(boost.boston ,i="rm") 
plot(boost.boston ,i="lstat")















#############################################################################################################################
######   Classification Trees
#############################################################################################################################
library(ISLR)
attach(Carseats)
High=as.factor(ifelse(Sales<=8,"No","Yes"))             ## turn into a binary classification problems

Carseats=data.frame(Carseats,High)          ##  we use the data.frame() function to merge High with the rest of the Carseats data
tree.carseats =tree(High~.-Sales,Carseats)  ## We now use the tree() function to fit a classification tree in order to predict High using all variables but Sales.
summary(tree.carseats)


plot(tree.carseats)                         ## plot tree structure
text(tree.carseats,pretty =0)


####  Split into train and test, test the performance
set.seed (2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats[-train ,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred ,High.test)               ## return the confusion matrix


### Next, we consider whether pruning the tree might lead to improved results. The function cv.tree() performs cross-validation in order 
### to determine the optimal level of tree complexity; cost complexity pruning is used in order to select a sequence of trees for 
### consideration. We use the argument FUN=prune.misclass in order to indicate that we want the classification error rate to guide the 
### cross-validation and pruning process, rather than the default for the cv.tree() function, which is deviance.
set.seed (3)
cv.carseats =cv.tree(tree.carseats ,FUN=prune.misclass )
names(cv.carseats)
cv.carseats




## We plot the error rate as a function of both size and k.
par(mfrow=c(1,2))
plot(cv.carseats$size ,cv.carseats$dev ,type="b")
plot(cv.carseats$k ,cv.carseats$dev ,type="b")


## We now apply the prune.misclass() function in order to prune the tree to obtain the nine-node tree.
prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats )
text(prune.carseats,pretty=0)


## How well does this pruned tree perform on the test data set? Once again, we apply the predict() function.
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred ,High.test)



## If we increase the value of best, we obtain a larger pruned tree with lower classification accuracy:
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats )
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred ,High.test)





#############################################################################################################################
######   Regression Trees
#############################################################################################################################

#Here we fit a regression tree to the Boston data set. First, we create a training set, and fit the tree to the training data.
library(MASS)
set.seed (1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston ,subset=train)
summary(tree.boston)

## plot the tree
plot(tree.boston)
text(tree.boston ,pretty=0)


## Now we use the cv.tree() function to see whether pruning the tree will improve performance.
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size ,cv.boston$dev ,type="b")


## In this case, the most complex tree is selected by cross-validation. However, if we wish to prune the tree, 
## we could do so as follows, using the prune.tree() function:
prune.boston=prune.tree(tree.boston ,best=5)
plot(prune.boston)
text(prune.boston ,pretty=0)


## In keeping with the cross-validation results, we use the unpruned tree to make predictions on the test set.
yhat=predict(tree.boston ,newdata=Boston[-train ,])
boston.test=Boston[-train ,"medv"]
plot(yhat,boston.test)
abline (0 ,1)
mean((yhat-boston.test)^2)






#############################################################################################################################
######   Bagging and Random Forests
#############################################################################################################################

## Here we apply bagging and random forests to the Boston data, using the randomForest package in R. The exact results obtained in this section may
## depend on the version of R and the version of the randomForest package installed on your computer. Recall that bagging is simply a special case of a random forest with m = p.
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance =TRUE)
bag.boston

## The argument mtry=13 indicates that all 13 predictors should be considered for each split of the tree—in other words, 
## that bagging should be done. How well does this bagged model perform on the test set?
yhat.bag = predict(bag.boston ,newdata=Boston[-train ,])
plot(yhat.bag, boston.test)
abline (0 ,1)
mean((yhat.bag-boston.test)^2)


## The test set MSE associated with the bagged regression tree is 13.16, almost half that obtained using an optimally-pruned single tree. 
## We could change the number of trees grown by randomForest() using the ntree argument:
bag.boston=randomForest(medv~.,data=Boston,subset=train, mtry=13,ntree=25)
yhat.bag = predict(bag.boston ,newdata=Boston[-train ,])
mean((yhat.bag-boston.test)^2)



## Growing a random forest proceeds in exactly the same way, except that we use a smaller value of the mtry argument. By default, randomForest() 
## uses p/3 variables when building a random forest of regression trees, 
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance =TRUE)
yhat.rf = predict(rf.boston ,newdata=Boston[-train ,])
mean((yhat.rf-boston.test)^2)

## Using the importance() function, we can view the importance of each importance() variable.
importance(rf.boston)





#############################################################################################################################
######   Boosting 
#############################################################################################################################

## Hereweusethegbmpackage,andwithinitthegbm()function,tofitboosted regression trees to the Boston data set. We run gbm() with the 
## option distribution="gaussian" since this is a regression problem; if it were a bi- nary classification problem, we would use 
## distribution="bernoulli". The argument n.trees=5000 indicates that we want 5000 trees, and the option interaction.depth=4 limits 
## the depth of each tree.
library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000, interaction.depth=4)
summary(boost.boston)


## We see that lstat and rm are by far the most important variables. We can also produce partial dependence plots for these two variables. 
## These plots illustrate the marginal effect of the selected variables on the response after integrating out the other variables. 
## In this case, as we might expect, median house prices are increasing with rm and decreasing with lstat.
par(mfrow=c(1,2)) 
plot(boost.boston ,i="rm") 
plot(boost.boston ,i="lstat")



## We now use the boosted model to predict medv on the test set:
yhat.boost=predict(boost.boston,newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost -boston.test)^2)


## use different lambda values
boost.boston=gbm(medv~.,data=Boston[train,],distribution= "gaussian",n.trees=5000, interaction.depth=4,shrinkage =0.2, verbose =F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost -boston.test)^2)











