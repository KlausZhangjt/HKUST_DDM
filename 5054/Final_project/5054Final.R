######################  5054 Final Project ###################### 
library(randomForest)
library(caret)
library(gbm)
library(MASS)    
library(ISLR)
library(dplyr)
library(e1071)
library(ssc)
library(kernlab)

#################################### Part 1 ################################### 
####  data process
groups <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/newsgroups.txt")
documents <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/documents.txt")
wordlist <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/wordlist.txt")
names(groups) <- c("group")
dim(groups)
dim(documents)
summary(documents$V1)
summary(documents$V2)

mat1 <- matrix(0, nrow=16242, ncol=100)
for(i in 1:65451){
  mat1[documents$V1[i], documents$V2[i]] <- 1
}

X <- as.data.frame(mat1)
train <- cbind(groups, X)
col_names <- rep(0,101)
col_names[1] <- c("group")
for(i in 1:100){
  col_names[i+1] <- c(wordlist[i,])
}
names(train) <- col_names
factor_name <- names(train)

for(i in factor_name){
  train[,i] <- as.factor(train[,i])
}
test_list <- sample(dim(train)[1], 1624)
test <- train[test_list,]
train <- train[-test_list, ]

### 1. random forest
# parameter tuning - find the best mtry, with ntree=100
set.seed(7)
flds <- createFolds(1:14618, k = 5, list = TRUE, returnTrain = FALSE)
CVErr<-rep(0,25)
for(k in 1:25){
  err<-0
  for(fold in flds){
    set.seed(7+k)
    train_cv <- data.frame(train[-fold,])
    test_cv <- data.frame(train[fold,])
    rf = randomForest(group~.,data=train_cv, mtry=k, ntree=100, na.action=na.omit)  
    pred = predict(rf, newdata=test_cv, type="class")
    conf <- table(pred, test_cv$group)
    err <- err + 1 - sum(diag(conf))/sum(conf)
  }
  CVErr[k]<-err/5
}

plot(1:25,CVErr,type='b',col='blue', xlab="mtry(Number of predictors)",ylab="CV Error",main="5-Fold CV of Random Forest")
CVErr
min(CVErr)
which(CVErr==min(CVErr))

# parameter tuning - find the best ntree, with mtry=15
ntree_list <- seq(100, 1000, 100)
set.seed(17)
flds <- createFolds(1:14618, k = 5, list = TRUE, returnTrain = FALSE)
CVErr<-rep(0,10)
for(k in 1:10){
  err<-0
  for(fold in flds){
    set.seed(7+k)
    train_cv <- data.frame(train[-fold,])
    test_cv <- data.frame(train[fold,])
    rf = randomForest(group~.,data=train_cv, mtry=15, ntree=100*k, na.action=na.omit)  
    pred = predict(rf, newdata=test_cv, type="class")
    conf <- table(pred, test_cv$group)
    err <- err + 1 - sum(diag(conf))/sum(conf)
  }
  CVErr[k]<-err/5
}

plot(ntree_list,CVErr,type='b',col='blue', xlab="ntree(Number of trees)",ylab="CV Error",main="5-Fold CV of Random Forest")
CVErr
min(CVErr)
which(CVErr==min(CVErr))

# finish parameter tuning: mtry=15, ntree=800

rf = randomForest(group~.,data=train, mtry=15, ntree=800, na.action=na.omit)
pred=predict(rf,newdata=test, mtry=15, ntree=800, na.action=na.omit, type="class")
conf <- table(pred,test$group)
conf
1 - sum(diag(conf))/sum(conf)

important <- importance(rf)
important[order(-important),]
varImpPlot(rf, n.var=10, main = 'Top 10 - Importance Keywords')



### 2. boosting tree
# find best n.tree
ntree_list = c(100,200,300,400,500)
CVErr <- rep(0,5)
set.seed(7)
flds <- createFolds(1:14618, k = 5, list = TRUE, returnTrain = FALSE)
for(i in 1:5){
  err<-0
  for(fold in flds){
    set.seed(7+i)
    train_cv <- data.frame(train[-fold,])
    test_cv <- data.frame(train[fold,])
    boost = gbm(group~.,data=train_cv,distribution = "multinomial",shrinkage=0.1, n.trees=ntree_list[i], interaction.depth=1)
    pred = predict(boost,newdata=test_cv, type="response")
    yhat <- apply(pred, 1, which.max)
    conf <- table(yhat,test_cv$group)
    err <- err + 1 - sum(diag(conf))/sum(conf)
  }
  CVErr[i]<-err/5
}
plot(ntree_list,CVErr,type='b',col='blue', xlab="n.tree(total number of trees)",ylab="CV Error",main="5-Fold CV of boosting tree")
CVErr

# find best shrinkage
lr_list <- c(0.06, 0.08, 0.1, 0.12, 0.14)
CVErr <- rep(0,5)
for(i in 1:5){
  err<-0
  for(fold in flds){
    set.seed(17+i)
    train_cv <- data.frame(train[-fold,])
    test_cv <- data.frame(train[fold,])
    boost = gbm(group~.,data=train_cv,distribution = "multinomial",shrinkage=lr_list[i], n.trees=500, interaction.depth=1)
    pred = predict(boost,newdata=test_cv, type="response")
    yhat <- apply(pred, 1, which.max)
    conf <- table(yhat,test_cv$group)
    err <- err + 1 - sum(diag(conf))/sum(conf)
  }
  CVErr[i]<-err/5
}
plot(lr_list,CVErr,type='b',col='blue', xlab="shrinkage(learning rate)",ylab="CV Error",main="5-Fold CV of boosting tree")
CVErr

# finish tuning: “n.trees”=500, “shrinkage”=0.08, “interaction.depth”= 1
boost = gbm(group~.,data=train, distribution = "multinomial",n.trees=500,shrinkage=0.08,interaction.depth=1)
pred = predict(boost,newdata=test, type="response")
yhat <- apply(pred, 1, which.max)
conf <- table(yhat,test$group)
conf
1 - sum(diag(conf))/sum(conf)

### 4. LDA  
set.seed(7)
flds <- createFolds(1:14618, k = 5, list = TRUE, returnTrain = FALSE)
err<-0
for(fold in flds){
  train_cv <- data.frame(train[-fold,])
  test_cv <- data.frame(train[fold,])
  lda.fit<-lda(group~.,data=train_cv)
  pred<-predict(lda.fit,test_cv)
  conf <- table(pred$class,test_cv$group)
  err <- err + 1 - sum(diag(conf))/sum(conf)
}
err/5

lda.fit<-lda(group~.,data=train)
pred<-predict(lda.fit,test)
conf <- table(pred$class,test$group)
conf
1 - sum(diag(conf))/sum(conf)


### 5. QDA  
set.seed(17)
flds <- createFolds(1:14618, k = 5, list = TRUE, returnTrain = FALSE)
err<-0
for(fold in flds){
  train_cv <- data.frame(train[-fold,])
  test_cv <- data.frame(train[fold,])
  qda.fit<-qda(group~windows+god+christian+car+government+team,data=train_cv)
  pred<-predict(qda.fit,test_cv)
  conf <- table(pred$class,test_cv$group)
  err <- err + 1 - sum(diag(conf))/sum(conf)
}
err/5

qda.fit<-qda(group~windows+god+christian+car+government+team,data=train)
pred<-predict(qda.fit,test)
conf <- table(pred$class,test$group)
conf
1 - sum(diag(conf))/sum(conf)

#################################### Part 2 ################################### 
# get data
groups <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/newsgroups.txt")
documents <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/documents.txt")
wordlist <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/wordlist.txt")
names(groups) <- c("group")
mat1 <- matrix(0, nrow=16242, ncol=100)
for(i in 1:65451){
  mat1[documents$V1[i], documents$V2[i]] <- 1
}
X <- as.data.frame(mat1)

col_names <- rep(0,100)
for(i in 1:100){
  col_names[i] <- c(wordlist[i,])
}
names(X) <- col_names

# PCA
pr.out=prcomp(X, scale=TRUE)
pr.out$rotation
pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)
pve
cumsum(pve)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained ", ylim=c(0,1),type="b")
plot(cumsum(pve), xlab="Principal Component ", ylab=" Cumulative Proportion of Variance Explained ", ylim=c(0,1), type="b")
pr.out$x[,1:4]

# K-means Clustering
set.seed(7)
km.out=kmeans(pr.out$x[,1:4], 4, nstart=50)
km.out$cluster
plot(pr.out$x[,1:2], col=(km.out$cluster +1), main="K-Means Results based on PC1-PC4 with K=4", xlab="PC1", ylab="PC2", pch=20, cex=0.5)

# mis-clustering error rate
num1<-1
err1<-rep(0,24)
groups <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/newsgroups.txt")
for(i in 1:4){
  for(j in 1:4){
    for(h in 1:4){
      for(k in 1:4){
        if((i!=j)&(i!=h)&(i!=k)&(j!=h)&(j!=k)&(h!=k)){
          dt<-as.factor(groups$V1)
          dt[which(dt=='1')]<-i
          dt[which(dt=='2')] <-j
          dt[which(dt=='3')] <-h
          dt[which(dt=='4')] <-k
          err1[num1]<-0
          for(row in 1:16242){
            if(dt[row]!=km.out$cluster[row]){
              err1[num1]<-err1[num1]+1
            }
          }
          num1<-num1+1
        }
      }
    }
  }
}
min(err1)
min(err1)/16242


# K-means Clustering
set.seed(7)
km.out=kmeans(pr.out$x[,1:5], 4, nstart=50)
km.out$cluster
plot(pr.out$x[,1:2], col=(km.out$cluster +1), main="K-Means Results based on PC1-PC5 with K=4", xlab="PC1", ylab="PC2", pch=20, cex=0.5)

# mis-clustering error rate
num1<-1
groups <- read.table("/Users/klaus_zhangjt/desktop/20newsgroup/newsgroups.txt")
err1<-rep(0,24)
for(i in 1:4){
  for(j in 1:4){
    for(h in 1:4){
      for(k in 1:4){
        if((i!=j)&(i!=h)&(i!=k)&(j!=h)&(j!=k)&(h!=k)){
          dt<-as.factor(groups$V1)
          dt[which(dt=='1')]<-i
          dt[which(dt=='2')] <-j
          dt[which(dt=='3')] <-h
          dt[which(dt=='4')] <-k
          err1[num1]<-0
          for(row in 1:16242){
            if(dt[row]!=km.out$cluster[row]){
              err1[num1]<-err1[num1]+1
            }
          }
          num1<-num1+1
        }
      }
    }
  }
}
min(err1)
min(err1)/16242


#################################### Part 3 ################################### 
# get data ### 
train <- read.csv("/Users/klaus_zhangjt/desktop/MNIST/train_resized.csv")
test <- read.csv("/Users/klaus_zhangjt/desktop/MNIST/test_resized.csv")

### 1 ### 
train1 <- rbind(subset(train,label==3), subset(train,label==6))
test1 <- rbind(subset(test,label==3), subset(test,label==6))
train1$label <- as.factor(train1$label)
test1$label <- as.factor(test1$label)

cost_list <- seq(10, 200, 10)
CVerr <- rep(0, length(cost_list))
for(i in 1:length(cost_list)){
  svm1=svm(label~., data=train1, kernel="linear", cost=cost_list[i], scale=TRUE, cross=5)        
  CVerr[i] <- mean(100-svm1$accuracies)
}

plot(cost_list, CVerr, type='b',col='blue', xlab="cost",ylab="CV Error",main="5-Fold CV of SVM")
min(CVerr)

svm1=svm(label~., data=train1, kernel="linear", cost=190, scale=TRUE) 
pred <- predict(svm1, newdata = test1)
mean(pred!=test1$label)
confusionMatrix(pred, test1$label)


### 2 ### 
cost_list <- c(10, 50, 100, 150, 200)
gamma_list <- c(0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.015, 0.02, 0.025)
CVerr <- matrix(0, nrow = length(gamma_list), ncol = length(cost_list))
for(i in 1:length(gamma_list)){
  for(j in 1:length(cost_list)){
    svm2=svm(label~., data=train1, kernel="radial", gamma=gamma_list[i], 
             cost=cost_list[j], scale=TRUE, cross=5)
    CVerr[i,j] <- mean(100-svm2$accuracies)
  }
}
CVerr

svm2=svm(label~., data=train1, kernel="radial",gamma=0.003,cost=200,scale=TRUE)
pred <- predict(svm2, newdata = test1)
mean(pred!=test1$label)
confusionMatrix(pred, test1$label)


### 3 ### 
train2 <- train[train$label %in% c(1,2,5,8) , ]
test2 <- test[test$label %in% c(1,2,5,8) , ]
train2$label <- as.factor(train2$label)
test2$label <- as.factor(test2$label)

# 10:47
cost_list <- c(1, 10, 50, 70, 100)
CVerr <- rep(0, length(cost_list))
for(i in 1:length(cost_list)){
  svm3=svm(label~., data=train2, kernel="linear", cost=cost_list[i], scale=TRUE, cross=5)        
  CVerr[i] <- mean(100-svm3$accuracies)
}
plot(cost_list, CVerr, type='b',col='blue', xlab="cost",ylab="CV Error",main="5-Fold CV of SVM")
min(CVerr)
# 10:57

svm3=svm(label~., data=train2, kernel="linear", cost=1, scale=TRUE)
pred <- predict(svm3, newdata = test2)
mean(pred!=test2$label)
confusionMatrix(pred, test2$label)


### 4 ### 
train$label <- as.factor(train$label)
test$label <- as.factor(test$label)

cost_list <- c(10, 20, 30, 40)
gamma_list <- c(0.001, 0.003, 0.005, 0.007, 0.009, 0.01)
CVerr <- matrix(0, nrow = length(gamma_list), ncol = length(cost_list))
# begin time: 11:54
for(i in 1:length(gamma_list)){
  for(j in 1:length(cost_list)){
    svm4=svm(label~., data=train, kernel="radial", gamma=gamma_list[i], 
             cost=cost_list[j], scale=TRUE, cross=5)
    CVerr[i,j] <- mean(100-svm4$accuracies)
  }
}
CVerr
min(CVerr)
# end time: 13:34

svm4=svm(label~., data=train, kernel="radial", cost=20, gamma=0.005, scale=TRUE)
pred <- predict(svm4, newdata = test)
mean(pred!=test$label)
confusionMatrix(pred, test$label)


#################################### Part 5 ################################### 

# data process
test <- read.csv("/Users/klaus_zhangjt/desktop/CTG/CTG_Test.csv")
train_label <- read.csv("/Users/klaus_zhangjt/desktop/CTG/CTG_Train_labeled.csv")
train_unlabel <- read.csv("/Users/klaus_zhangjt/desktop/CTG/CTG_Train_unlabeled.csv")
train_label <- na.omit(train_label)
train_unlabel <- na.omit(train_unlabel)
test <- na.omit(test)
train_label$NSP <- as.factor(train_label$NSP)
test$NSP <- as.factor(test$NSP)
y1 <- train_label$NSP
y2 <- test$NSP

### supervised learning 1: random forest ###
set.seed(7)
flds <- createFolds(1:length(train_label), k = 5, list = TRUE, returnTrain = FALSE)
ntree_list <- c(10, 30, 50, 70, 90)
mtry_list <- c(1, 2, 3, 4, 5)
CVErr<-c()
for(k in 1:5){
  for(j in 1:5){
    err<-0
    for(fold in flds){
      fit<-randomForest(NSP~.,data=train_label[-fold,],
                        mtry=mtry_list[k], ntree=ntree_list[j])
      test_y=predict(fit,train_label[fold,])
      err<-err+1-length(which(test_y==y1))/length(y1)
    }
    CVErr<-c(CVErr,err/5)
  }
}
CVErr
min(CVErr)
which(CVErr==min(CVErr))

rf <- randomForest(NSP~.,data=train_label,mtry=3, ntree=10)
pred=predict(rf,newdata=test, type="class")
conf <- table(pred,test$NSP)
conf
1 - sum(diag(conf))/sum(conf)


### supervised learning 2: svm with linear kernel ###
cost_list <- seq(1, 10, 1)
CVerr <- rep(0, length(cost_list))
for(i in 1:length(cost_list)){
  svm1=svm(NSP~., data=train_label, kernel="linear", cost=cost_list[i], scale=TRUE, cross=5)        
  CVerr[i] <- mean(100-svm1$accuracies)
}
CVerr
min(CVerr)
which(CVerr==min(CVerr))

svm1=svm(NSP~., data=train_label, kernel="linear", cost=3, scale=TRUE)
pred <- predict(svm1, newdata = test)
mean(pred!=test$NSP)
confusionMatrix(pred, test$NSP)

### supervised learning 3: knn ###
flds <- createFolds(1:length(train_label), k=5, list=TRUE, returnTrain=FALSE)   
CVErr<-rep(0,10)
for(k in 1:10){
  err<-0
  for(fold in flds){
    trainx<-data.frame(train_label[-fold,])
    testx<-data.frame(train_label[fold,])
    knnmod <- knn3(NSP~., trainx, k=k)
    pre <- predict(knnmod, testx, type="class")
    t<-table(testx$NSP, pre)
    err<-err+1-sum(diag(t))/sum(t)
  }
  CVErr[k]<-err/5
}
CVErr
min(CVErr)
which(CVErr==min(CVErr))


model<-knn3(NSP~.,train_label,k=4)
pre <-  predict(model, test, type="class")
t<-table(test$NSP, pre)
1-sum(diag(t))/sum(t)
t

# 5.2
###  semi-supervised learning 1: semi-knn ###
train_unlabel$NSP<-0
trains <- rbind(train_label,train_unlabel)
x_train <- trains[,-21]
y_train <- trains[,21]
x_test <- test[,-21]
y_test <- test[,21]
y_train[which(y_train==0)]<-NA
x_test <- as.matrix(x_test)


semi1 <- selfTraining(x = x_train, y = y_train, learner = knn3,
                      learner.pars = list(k = 1), pred = "predict")
pred<-predict(semi1,x_test)
table(pred,y_test)
mean(pred!=y_test)

###  semi-supervised learning 2: semi-svm ###
learner <- e1071::svm
learner.pars <- list(type = "C-classification", kernel="radial", 
                     probability = TRUE, scale = TRUE)
pred <- function(m, x){
  r <- predict(m, x, probability = TRUE)
  prob <- attr(r, "probabilities")
  prob
}
semi2 <- selfTraining(x = x_train, y = y_train, 
                      learner = learner, 
                      learner.pars = learner.pars, 
                      pred = pred)
pred <- predict(semi2, x_test)
table(pred,y_test)
mean(pred!=y_test)

########################################################################

