################### R codes for Chapter 7: Non-linear Regression
###################

library(ISLR)
attach(Wage)                                         ## use Wage dataset

plot(age,wage,col=8,xlab="age",ylab="wage",main="Wage versus Age")

#############################################################################################################################
######   Polynomial Regression and Step Functions
#############################################################################################################################
fit=lm(wage~poly(age,4),data=Wage)                   ## fit a degree 4 polynomial, The function returns a matrix whose columns are a basis of orthogonal polynomials
coef(summary(fit))                                   ## estimate of coefficients


fit2=lm(wage~poly(age,4,raw=T),data=Wage)            ## use raw ploynomial, age, age^2, age^3, age^4
coef(summary(fit2))

fit2a=lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)      ## another way to fit degree 4 polynomial, This simply creates the polynomial basis functions on the fly, taking care to protect terms like age^2 via the wrapper function I()
coef(fit2a)

fit2b=lm(wage~cbind(age,age^2,age^3,age^4),data=Wage)     ## This does the same more compactly, using the cbind() function for building a matrix from a collection of vectors; 

## prediction on all range of age, and confidence bands
agelims=range(age)
age.grid=seq(from=agelims[1],to=agelims[2])
preds=predict(fit,newdata=list(age=age.grid),se=TRUE)
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)

## now plot the above bands, Here the mar and oma arguments to par() allow us to control the margins of the plot, and the title() function creates a figure title that spans both subplots.
#par(mfrow=c(1,2),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
par(mfrow=c(1,2))
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
title("Degree -4 Polynomial ",outer=T)
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)


### Now, we fit different degree polynomials and apply ANOVA to compare these models
fit.1=lm(wage~age,data=Wage)
fit.2=lm(wage~poly(age,2),data=Wage)        ## Note that without using "raw=T", the method uses an orthonormal basis of polynomials
fit.3=lm(wage~poly(age,3),data=Wage)
fit.4=lm(wage~poly(age,4),data=Wage)
fit.5=lm(wage~poly(age,5),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5)

preds5=predict(fit.5,newdata=list(age=age.grid),se=TRUE)
se5.bands=cbind(preds5$fit+2*preds5$se.fit,preds5$fit-2*preds5$se.fit)
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
title("Degree -5 Polynomial ",outer=T)
lines(age.grid,preds5$fit,lwd=2,col="blue")
matlines(age.grid,se5.bands,lwd=1,col="blue",lty=3)

## coefficients of degree-5 polynomial
coef(summary(fit.5))


## incorporate other variables into the function
fit.1=lm(wage~education+age,data=Wage) 
fit.2=lm(wage~education+poly(age,2),data=Wage) 
fit.3=lm(wage~education+poly(age,3),data=Wage) 
anova(fit.1,fit.2,fit.3)



## fit a polynomial logistic regression model
fit=glm(I(wage>250)~poly(age,4),data=Wage,family=binomial)
preds=predict(fit,newdata=list(age=age.grid),se=T)              ## predict on all the age values
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))

preds=predict(fit,newdata=list(age=age.grid),type="response", se=T)

##### plot the confidence bands
plot(age,I(wage>250),xlim=agelims ,type="n",ylim=c(0,.2))
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col =" darkgrey ")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)


## In order to fit a step function, as discussed in Section 7.2, we use the cut() function.
table(cut(age,4))
fit=lm(wage~cut(age ,4),data=Wage)
coef(summary(fit))


## prediction on all range of age, and confidence bands
agelims=range(age)
age.grid=seq(from=agelims[1],to=agelims[2])
preds=predict(fit,newdata=list(age=age.grid),se=TRUE)
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)

plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey",main="Step Functions Fit and Confidence Bands")
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="red",lty=3)




############################################# 8 Cuts
table(cut(age,8))
fit=lm(wage~cut(age ,8),data=Wage)
coef(summary(fit))
## prediction on all range of age, and confidence bands
agelims=range(age)
age.grid=seq(from=agelims[1],to=agelims[2])
preds=predict(fit,newdata=list(age=age.grid),se=TRUE)
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey",main="Step Functions Fit and Confidence Bands")
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="red",lty=3)




## fit a stepfunction logistic regression model
fit=glm(I(wage>250)~cut(age,4),data=Wage,family=binomial)
preds=predict(fit,newdata=list(age=age.grid),se=T)              ## predict on all the age values
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
preds=predict(fit,newdata=list(age=age.grid),type="response", se=T)
##### plot the confidence bands
plot(age,I(wage>250),xlim=agelims ,type="n",ylim=c(0,.2),main="Step Function Logistic Model")
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col =" darkgrey ")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="red",lty=3)




## fit a stepfunction logistic regression model
fit=glm(I(wage>250)~cut(age,8),data=Wage,family=binomial)
preds=predict(fit,newdata=list(age=age.grid),se=T)              ## predict on all the age values
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
preds=predict(fit,newdata=list(age=age.grid),type="response", se=T)
##### plot the confidence bands
plot(age,I(wage>250),xlim=agelims ,type="n",ylim=c(0,.2),main="Step Function Logistic Model")
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col =" darkgrey ")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="red",lty=3)





################## piecewise cubic regression
Wage1<-Wage[which(Wage$age<50),]
Wage2<-Wage[which(Wage$age>=50),]
fit.1=lm(wage~poly(age,3),data=Wage1)
fit.2=lm(wage~poly(age,3),data=Wage2)
agelims=range(age)
preds.1=predict(fit.1,newdata=list(age=seq(from=agelims[1],to=50)),se=TRUE)
se.bands.1=cbind(preds.1$fit+2*preds.1$se.fit,preds.1$fit-2*preds.1$se.fit)
preds.2=predict(fit.2,newdata=list(age=seq(from=50,to=agelims[2])),se=TRUE)
se.bands.2=cbind(preds.2$fit+2*preds.2$se.fit,preds.2$fit-2*preds.2$se.fit)

plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey",main="PieceWise Cubic Regression")
lines(seq(from=agelims[1],to=50),preds.1$fit,lwd=2,col="blue")
matlines(seq(from=agelims[1],to=50),se.bands.1,lwd=1,col="blue",lty=3)
lines(seq(from=50,to=agelims[2]),preds.2$fit,lwd=2,col="red")
matlines(seq(from=50,to=agelims[2]),se.bands.2,lwd=1,col="red",lty=3)


############ Cubic Spline
library(splines)
fit=lm(wage~bs(age,knots=c(25,40,60),df=3),data=Wage)           ## fit a regression pline on selected knots, The bs() function generates the entire matrix of bs() basis functions 
                                                                ## for splines with the specified set of knots. Df=3 means a cubic spline
pred=predict(fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray",main="Cubic Spline on Selected Knots")
lines(age.grid,pred$fit,lwd=2,col="blue")
lines(age.grid,pred$fit+2*pred$se ,lty="dashed",col="red")
lines(age.grid,pred$fit-2*pred$se ,lty="dashed",col="red")



############ Cubic Spline
library(splines)
fit=lm(wage~bs(age,knots=c(10,20,30,40,50,60),df=3),data=Wage)           
pred=predict(fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray",main="Cubic Spline on Selected Knots")
lines(age.grid,pred$fit,lwd=2,col="blue")
lines(age.grid,pred$fit+2*pred$se ,lty="dashed",col="red")
lines(age.grid,pred$fit-2*pred$se ,lty="dashed",col="red")



############### Logistic Cubic Spline
fit=glm(I(wage>250)~bs(age,knots=c(25,40,60),df=3),data=Wage,family=binomial)
preds=predict(fit,newdata=list(age=age.grid),se=T)              ## predict on all the age values
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
preds=predict(fit,newdata=list(age=age.grid),type="response", se=T)
##### plot the confidence bands
plot(age,I(wage>250),xlim=agelims ,type="n",ylim=c(0,.2),main="Cubic Spline Logistic Model")
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col =" darkgrey ")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="red",lty=3)




############ Quartic Spline
library(splines)
fit=lm(wage~bs(age,knots=c(10,20,30,40,50,60),df=4),data=Wage)           
pred=predict(fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray",main="Quartic Spline on Selected Knots")
lines(age.grid,pred$fit,lwd=2,col="blue")
lines(age.grid,pred$fit+2*pred$se ,lty="dashed",col="red")
lines(age.grid,pred$fit-2*pred$se ,lty="dashed",col="red")



## In order to instead fit a natural spline, we use the ns() function.
fit2=lm(wage~ns(age,df=4),data=Wage)                 ## df is the number of knots, then the knots are chosen from quantiles of data
pred2=predict(fit2,newdata=list(age=age.grid),se=T) 
plot(age,wage,col="gray",main="Natural Spline")
lines(age.grid, pred2$fit,col="blue",lwd=2)
lines(age.grid,pred2$fit+2*pred2$se ,lty="dashed",col="red")
lines(age.grid,pred2$fit-2*pred2$se ,lty="dashed",col="red")


############### Logistic Natural Spline
fit=glm(I(wage>250)~ns(age,df=4),data=Wage,family=binomial)
preds=predict(fit,newdata=list(age=age.grid),se=T)              ## predict on all the age values
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
preds=predict(fit,newdata=list(age=age.grid),type="response", se=T)
##### plot the confidence bands
plot(age,I(wage>250),xlim=agelims ,type="n",ylim=c(0,.2),main="Natural Spline Logistic Model")
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col =" darkgrey ")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="red",lty=3)






#################  10-fold CV for Natural Spline
library(boot) 
library(splines)
cv.err<-rep(0,10)
for (k in 1:10){
  glm.fit=glm(wage~ns(age,df=k),data=Wage)  
  Cerr=cv.glm(Wage,glm.fit,K=10) 
  cv.err[k]<-Cerr$delta[1]
}
cv.err
plot(1:10,cv.err,type='b',col="red",xlab="Deg. of Freedom",ylab="10-fold CV Err")


################################## Compare NS and Polynomial Regression
fit1=lm(wage~poly(age,15),data=Wage)
fit2=lm(wage~ns(age,df=15),data=Wage)
agelims=range(age)
preds.1=predict(fit1,newdata=list(age=seq(from=agelims[1],to=agelims[2])),se=TRUE)
preds.2=predict(fit2,newdata=list(age=seq(from=agelims[1],to=agelims[2])),se=TRUE)
plot(age,wage,col="gray",main="Natural Spline v.s. Polynomial Regression")
lines(seq(from=agelims[1],to=agelims[2]), preds.1$fit,col="red",lwd=2)
lines(seq(from=agelims[1],to=agelims[2]), preds.2$fit,col="blue",lwd=2)
legend("topleft", 
       legend = c("Natural Spline with d.f.=15", "Degree 15 Polynomial"), 
       col = c("blue","red"),
       lty=c(1,1)
)




######################## Smoothing Spline
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey",main=" Smoothing Spline ")
fit=smooth.spline(age,wage,df=16)    ## we specified df=16. The function then determines which value of lambda leads to 16 degrees of freedom.
fit2=smooth.spline(age,wage,cv=TRUE)  ## we select the smoothness level by cross- validation;
fit2$df
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)



########################   10-fold CV of Smoothing Spline
library(caret)
X<-data.frame(age=Wage$age)
y<-Wage$wage
flds <- createFolds(1:3000, k = 10, list = TRUE, returnTrain = FALSE)    ### random split into 10 folds
CVErr<-rep(0,19)
for(k in 2:20){
  err<-0
  for(fold in flds){
    fit <- smooth.spline(X[-fold,], y[-fold],df=k)
    test_y=predict(fit,X[fold,])
    err<-err+sum((test_y$y-y[fold])^2)
  }
  CVErr[k-1]<-err/10
}
CVErr
plot(1:19,CVErr,type='b',xlab="Degrees of Freedom",ylab="CV Error",main="10-fold CV of Smoothing Spline")




## fit a local polynomial regression
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey",main="Local Regression")
fit=loess(wage~age,span=.2,data=Wage)           ## by default, degree=2.  Common choices are 1 or 2
fit2=loess(wage~age,span=.5,data=Wage)          ## by default, degree=2
age.grid=seq(from=agelims[1],to=agelims[2])
lines(age.grid,predict(fit,data.frame(age=age.grid)), col="red",lwd=2)
lines(age.grid,predict(fit2,data.frame(age=age.grid)), col="blue",lwd=2)
legend("topright",legend=c("Span=0.2","Span=0.5"), col=c("red","blue"),lty=1,lwd=2,cex=.8)




########################   10-fold CV of Smoothing Spline
library(caret)
X<-data.frame(age=Wage$age)
y<-Wage$wage
flds <- createFolds(1:3000, k = 10, list = TRUE, returnTrain = FALSE)    ### random split into 10 folds
CVErr<-rep(0,10)
for(k in 1:10){
  err<-0
  for(fold in flds){
    fit <- loess(wage~age,span=k*0.1,data=Wage[-fold,]) 
    test_y=predict(fit,Wage[fold,])
    err<-err+sum((test_y-Wage$wage[fold])^2)
  }
  CVErr[k]<-err/10
}
CVErr
plot((1:10)/10,CVErr,type='b',xlab="BandWidth",ylab="CV Error",main="10-fold CV of Local Polynomial")




######################### GAM for Wage Data
gam1=lm(wage~ns(year,4)+ns(age,5)+education ,data=Wage)       ## natrual spline
library(gam)
gam.m3=gam(wage~s(year,4)+s(age,5)+education ,data=Wage)      ## fit a GAM model, df=4 for year, df=5 for age using smoothing splines
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE,col="blue")        ## plot the results




######################### GAM for Credit Data
library(gam)
gam.m3=gam(Balance~s(Income,4)+s(Age,5)+Student+s(Limit,2)+Education ,data=Credit)      
par(mfrow=c(1,5))
plot(gam.m3, se=TRUE,col="blue")        ## plot the results
















#############################################################################################################################
######   Splines
#############################################################################################################################
library(splines)
fit=lm(wage~bs(age,knots=c(25,40,60)),data=Wage)           ## fit a regression pline on selected knots, The bs() function generates the entire matrix of bs() basis functions for splines with the specified set of knots.
pred=predict(fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray")
lines(age.grid,pred$fit,lwd=2)
lines(age.grid,pred$fit+2*pred$se ,lty="dashed")
lines(age.grid,pred$fit-2*pred$se ,lty="dashed")


## We could also use the df option to produce a spline with knots at uniform quantiles of the data
## The function bs() also has a degree argument, so we can fit splines of any degree, rather than the default degree of 3 (which yields a cubic spline).
dim(bs(age,knots=c(25,40,60)))
dim(bs(age,df=6))
attr(bs(age,df=6),"knots")


## In order to instead fit a natural spline, we use the ns() function.
fit2=lm(wage~ns(age,df=4),data=Wage)
pred2=predict(fit2,newdata=list(age=age.grid),se=T) 
lines(age.grid, pred2$fit,col="red",lwd=2)


## In order to fit a smoothing spline, we use the smooth.spline() function.
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
title (" Smoothing Spline ")
fit=smooth.spline(age,wage,df=16)    ## we specified df=16. The function then determines which value of lambda leads to 16 degrees of freedom.
fit2=smooth.spline(age,wage,cv=TRUE)  ## we select the smoothness level by cross- validation;
fit2$df
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)



## fit a local polynomial regression
plot(age,wage,xlim=agelims ,cex=.5,col="darkgrey")
title (" Local Regression ")
fit=loess(wage~age,span=.2,data=Wage)
fit2=loess(wage~age,span=.5,data=Wage)
lines(age.grid,predict(fit,data.frame(age=age.grid)), col="red",lwd=2)
lines(age.grid,predict(fit2,data.frame(age=age.grid)), col="blue",lwd=2)
legend("topright",legend=c("Span=0.2","Span=0.5"), col=c("red","blue"),lty=1,lwd=2,cex=.8)



#############################################################################################################################
######   Generalized Additive Models
#############################################################################################################################
gam1=lm(wage~ns(year,4)+ns(age,5)+education ,data=Wage)       ## natrual spline


library(gam)
gam.m3=gam(wage~s(year,4)+s(age,5)+education ,data=Wage)      ## fit a GAM model, df=4 for year, df=5 for age using smoothing splines
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE,col="blue")        ## plot the results


## The generic plot() function recognizes that gam.m3 is an object of class gam, and invokes the appropriate plot.gam()method.
## Conveniently,eventhough plot.gam() gam1 is not of class gam but rather of class lm, we can still use plot.gam()
## on it.
plot.Gam(gam1, se=TRUE, col="red")



#### We can perform a series of ANOVA tests in order to determine which of these three models is best: 
gam.m1=gam(wage~s(age ,5)+education ,data=Wage)
gam.m2=gam(wage~year+s(age ,5)+education ,data=Wage)
anova(gam.m1,gam.m2,gam.m3,test="F")
summary(gam.m3)



preds=predict(gam.m2,newdata=Wage)              ## predict on newdata














