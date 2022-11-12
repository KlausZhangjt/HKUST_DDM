##### R codes in Linear Regression from textbook ISLR
library(MASS)
library(ISLR)


##### Advertising Dataset
adv<-read.csv('Datasets/Advertising.csv',header=TRUE)
adv[1:10,]
# scatter plot to check the linearity
plot(Sales ~ TV, data = adv,
     xlab = "TV",
     ylab = "Sales",
     main = "Sales v.s. TV",
     pch  = 20,
     cex  = 2,
     col  = "black")

mod1<-lm(Sales~TV,data=adv)
summary(mod1)
abline(mod1, lwd = 5, col = "darkorange")

mod2<-lm(Sales~.,data=adv)
summary(mod2)

##ANOVA
mod1<-lm(Sales~TV,data=adv)
mod2<-lm(Sales~.,data=adv)
anova(mod1,mod2)


## conf. interval
mod1<-lm(Sales~TV,data=adv)
confint(mod1,level=0.95)

## pred. interval
mod1<-lm(Sales~TV,data=adv)
new_obs <- data.frame(TV=145.3,Radio=51.1,Newspaper=33.9)   ## create the new observation x0
predict(mod1,newdata=new_obs,interval="confidence",level=0.95)   ## output the prediction interval of mean response



################################## KNN regression
library(caret)
train_x<-data.frame(x=runif(40,min=-1,max=1))
train_y<-2*train_x$x+2+rnorm(40,mean=0,sd=0.3)
knnmod1 <- knnreg(train_x, train_y,k=3)
str(knnmod1)
## plot
test_x<-data.frame(x=seq(-1,1,0.001))
test_y=predict(knnmod1,test_x)
plot(train_x$x,train_y,pch  = 20,
     cex  = 2,
     col  = "black",
     main="KNN with k=3 to fit y=2x+2+eps",
     xlab="x",
     ylab="y")
abline(2,2,col="orange",lwd=4)
lines(test_x$x,test_y,col="blue",lwd=4)




################################## KNN regression
library(caret)
train_x<-data.frame(x=runif(80,min=-1,max=1))
train_y<-(train_x$x)^2+2+rnorm(40,mean=0,sd=0.3)
knnmod2 <- knnreg(train_x, train_y,k=6)
str(knnmod2)

## plot
test_x<-data.frame(x=seq(-1,1,0.001))
test_y=predict(knnmod2,test_x)
plot(train_x$x,train_y,pch  = 20,
     cex  = 2,
     col  = "black",
     main="KNN with k=3 to fit y=x^2+2+eps",
     xlab="x",
     ylab="y")
lines(test_x$x,(test_x$x^2)+2,col="orange",lwd=4)
lines(test_x$x,test_y,col="blue",lwd=4)















##### Simple Linear Regression on Boston Data
?Boston                                 ##to find out details of Boston data
Boston[1:10,]                           ##to list the first 10 rows of Boston data
fix(Boston)                             ##into user's workspace
lm.fit=lm(medv~lstat, data=Boston)      ## regress medv onto lstat 
summary(lm.fit)                         ## shows the results of linear regression
par(mfrow=c(2,2));plot(lm.fit)          ## plot the outputs of linear regression
class(lm.fit)                           ## what is the class of lm.fit?  It is 'lm', a linear model. 
names(lm.fit)                           ## find out what information is stored in a linear model
lm.fit$coefficients                     ## returns the estimated coefficients
lm.fit$residuals                        ## returns the residuals
confint(lm.fit)                         ## 95% confidence interval of coefficients
confint(lm.fit,level=0.99)              ## 99% confidence interval of coefficients
predict(lm.fit,data.frame(lstat=c(5,10,15)), interval ="confidence")        #### if lstat=5 or 10 or 15, what are the 95% confidence intervals of expected medv
predict(lm.fit,data.frame(lstat=c(5,10,15)), interval ="confidence",level=0.99)    #### if lstat=5 or 10 or 15, what are the 99% confidence intervals of expected medv
predict(lm.fit,data.frame(lstat=c(5,10,15)), interval ="prediction")        #### if lstat=5 or 10 or 15, what are the 95% prediction intervals of medv
predict(lm.fit,data.frame(lstat=c(5,10,15)), interval ="prediction",level=0.99)    #### if lstat=5 or 10 or 15, what are the 99% prediction intervals of medv
plot(Boston$lstat,Boston$medv,xlab='lstat',ylab='medv')     ## plot dot graph for two variables
plot(Boston$lstat,Boston$medv,xlab='lstat',ylab='medv',col='blue')     ## plot dot graph for two variables in blue color
plot(Boston$lstat,Boston$medv,xlab='lstat',ylab='medv',col='blue',pch="+")     ## plot dot graph for two variables in blue color with symbol '+'
abline(lm.fit)                          ## draw the fitted line on the dot graph
abline(lm.fit,col='red')                ## draw the fitted line on the dot graph in red color
abline(lm.fit,col='red',lwd=3)          ## draw the fitted line on the dot graph in red color with line width=3
par(mfrow=c(2,2))                       ## create a 2 x 2 grid of panels to show seveal graphs
plot(lm.fit)                            ## plot res-fitted, qq plot, etc. graphs
plot(predict(lm.fit), residuals(lm.fit)) ## it also plots res-fitted graph, i.e., the top-left graph if you run plot(lm.fit)
plot(predict(lm.fit), rstudent(lm.fit))  ## plots rstudent-fitted graph, restudent() returns the studentized residuals
plot(hatvalues(lm.fit))                  ## plot leverage statistics
which.max(hatvalues(lm.fit))             ## returns the index whose hatvalues are maximum




#############  Multiple Linear Regression on Boston data
lm.fit=lm(medv~lstat+age,data=Boston)                     ### regress medv onto lstat, age
summary(lm.fit)                                           ### summary of the results of linear regression
lm.fit=lm(medv~.,data=Boston)                             ### regress medv onto all other 13 variables
summary(lm.fit)                                           ### summary of the results of linear regression
summary(lm.fit)$r.sq                                      ### R square
summary(lm.fit)$sigma                                     ### RSE
library(car)                                              ### car package can calculate variance inflation factor
vif(lm.fit)                                               ### variance inflation factor
lm.fit1=lm(medv~.-age,data=Boston)                        ### Regress medv onto all variables except age
summary(lm.fit1)                                           ### summary of the results of linear regression






#############  The interaction terms of multiple Linear Regression on Boston data
lm.fit2=lm(medv~lstat*age,data=Boston)                      ### regress medv onto lstat, age, lstat*age
summary(lm.fit2)                                           ### summary of the results of linear regression
lm.fit3=lm(medv~lstat:age,data=Boston)                      ### regress medv onto lstat*age
summary(lm.fit3)                                           ### summary of the results of linear regression





###########    Non-linear transformations of predictors on Boston data
lm.fit2=lm(medv~lstat+I(lstat^2),data = Boston)                      ### regress medv onto lstat, lstat^2
summary(lm.fit2)                                           ### summary of the results of linear regression
lm.fit=lm(medv~lstat,data=Boston)                                      ### regress medv onto lstat
anova(lm.fit,lm.fit2)                                     ### analysis of variance table of two models
summary(lm(medv~log(rm),data=Boston))                     ### summary of regressing medv onto log(rm)



###########   Qualitative predictors on Carseats data
fix(Carseats)                                             ### into user's workspace
Carseats[1:10,]                                           ### first 10 rows, ShelveLoc, Urban, US are qualitative predictors
lm.fit=lm(Sales~.+Income:Advertising+Price:Age,data=Carseats)  ### regress Sales on all predictors + income*ad + price*age
summary(lm.fit)
attach(Carseats)                                          ### attach the database to R search path
contrasts(ShelveLoc)                                      ### R coded qualitative predictors into indicators
contrasts(US) 




















































