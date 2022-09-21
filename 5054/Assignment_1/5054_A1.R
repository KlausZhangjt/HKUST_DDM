####### problem-1

data<-read.csv("/Users/klaus_zhangjt/1_new.csv")

Year <- data$Year
Adult.Mortality <- data$Adult.Mortality
infant.deaths <- data$infant.deaths
Alcohol<-data$Alcohol
percentage.expenditure<-data$percentage.expenditure
Hepatitis.B<-data$Hepatitis.B
Measles<-data$Measles
BMI<-data$BMI
under.five.deaths<-data$under.five.deaths
Polio<-data$Polio
Total.expenditure<-data$Total.expenditure
Diphtheria<-data$Diphtheria
HIV.AIDS<-data$HIV.AIDS
GDP<-data$GDP
Population<-data$Population
thinness..1.19.years<-data$thinness..1.19.years
thinness.5.9.years<-data$thinness.5.9.years
Income.composition.of.resources<-data$Income.composition.of.resources
Schooling<-data$Schooling
Status_Developed<-data$Status_Developed
Status_Developing<-data$Status_Developing
Life.expectancy<-data$Life.expectancy

df <- data.frame(
  Year,
  Adult.Mortality,
  infant.deaths,
  Alcohol,
  percentage.expenditure,
  Hepatitis.B,
  Measles,
  BMI,
  under.five.deaths,
  Polio,
  Total.expenditure,
  Diphtheria,
  HIV.AIDS,
  GDP,
  Population,
  thinness..1.19.years,
  thinness.5.9.years,
  Income.composition.of.resources,
  Schooling,
  Status_Developed,
  Status_Developing,
  Life.expectancy
)

model1 <- lm(data$Life.expectancy ~ data$Year + data$Adult.Mortality +
               data$infant.deaths + data$Alcohol + data$percentage.expenditure +
               data$Hepatitis.B + data$Measles + data$BMI + data$under.five.deaths +
               data$Polio + data$Total.expenditure + data$Diphtheria +
               data$HIV.AIDS + data$GDP + data$Population + 
               data$thinness..1.19.years + data$thinness.5.9.years + 
               data$Income.composition.of.resources + data$Schooling +
               data$Status_Developed + data$Status_Developing, data = df)
summary(model1)

model2 <- step(model1)
summary(model2)

model3 <- lm(Life.expectancy ~ Adult.Mortality + infant.deaths +
               BMI + under.five.deaths + HIV.AIDS + Schooling +
               Income.composition.of.resources, data = df)
summary(model3)

x1 <- data.frame(
  Adult.Mortality = 125,
  infant.deaths = 94,
  BMI = 55,
  under.five.deaths = 2,
  HIV.AIDS = 0.5,
  Schooling = 18,
  Income.composition.of.resources = 0.9
)

predict(model3, x1, interval="prediction",level=0.99)

########## problem-2

x_test_1 <-read.csv("/Users/klaus_zhangjt/x_test_1.csv")
x_train_1 <- read.csv("/Users/klaus_zhangjt/x_train_1.csv")
x_test_2 <- read.csv("/Users/klaus_zhangjt/x_test_2.csv")
x_train_2 <- read.csv("/Users/klaus_zhangjt/x_train_2.csv")
y_test <- read.csv("/Users/klaus_zhangjt/y_test.csv")
y_train <- read.csv("/Users/klaus_zhangjt/y_train.csv")

df2 <- data.frame(
  x_train_1,
  y_train
)

df3 <- data.frame(
  x_train_2,
  y_train
)

############## LR-full model
glmod1<-glm(Class~., data=df2,family=binomial)
summary(glmod1)

pred1<-predict(glmod1,x_test_1,type="response")
pred1

lr1 <-prediction(pred1, y_test$Class)
perf1<-performance(lr1,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)

perfa<-performance(lr1,"auc")
perfa@y.values[[1]]

glm.pred=rep("0",299)
glm.pred[pred1 >=.5]="1"

(table <- table(glm.pred, y_test$Class))
(accuracy <- sum(diag(table))/sum(table))

############## LR-smaller model
glmod2<-glm(Class~., data=df3,family=binomial)
summary(glmod2)

pred2<-predict(glmod2,x_test_2,type="response")
pred2

lr2 <-prediction(pred2, y_test$Class)
perf2<-performance(lr2,"tpr","fpr")
plot(perf2,colorize=TRUE,lwd=4)

perfa<-performance(lr2,"auc")
perfa@y.values[[1]]

glm.pred2=rep("0",299)
glm.pred2[pred2 >=.5]="1"

(table <- table(glm.pred2, y_test$Class))
(accuracy <- sum(diag(table))/sum(table))


############## LDA-full model
#install.packages("pROC") 
#install.packages("ROCR") 
library(pROC) 
library(ROCR)
library(MASS)

df2 <- data.frame(
  x_train_1,
  y_train
)

fit_lda1=lda(Class~.,data = df2)
fit_lda1

plot(fit_lda1)

pre_ldal <- predict(fit_lda1,x_test_1)
l1 <-prediction(pre_ldal$posterior[,2], y_test$Class)
perf1<-performance(l1,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)

perfa<-performance(l1,"auc")
perfa@y.values[[1]]

(table <- table(pre_ldal$class, y_test$Class))
(accuracy <- sum(diag(table))/sum(table))


############## LDA-smaller model
df3 <- data.frame(
  x_train_2,
  y_train
)

fit_lda2=lda(Class~.,data = df3)
fit_lda2

pre_lda2 <- predict(fit_lda2,x_test_2)
l2 <-prediction(pre_lda2$posterior[,2], y_test$Class)
perf1<-performance(l2,"tpr","fpr")
plot(perf1,colorize=TRUE,lwd=4)

perf2<-performance(l2,"auc")
perf2@y.values[[1]]

(table <- table(pre_lda2$class, y_test$Class))
(accuracy <- sum(diag(table))/sum(table))


############## QDA 
fit_qda <- qda(Class~.,data = df2)
fit_qda

qda.fit.pred1<-predict(fit_qda, x_test_1)
q <-prediction(qda.fit.pred1$posterior[,2], y_test$Class)
perf<-performance(q,"tpr","fpr")
plot(perf,colorize=TRUE,lwd=4)

perf1<-performance(q,"auc")
perf1@y.values[[1]]

(table <- table(qda.fit.pred1$class, y_test$Class))
(accuracy <- sum(diag(table))/sum(table))
