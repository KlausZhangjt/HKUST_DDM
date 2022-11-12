
###  Father and Son's height dataset
father.son=read.table("http://www.math.wustl.edu/~jmding/math3200/pearson.dat")
names(father.son) <- c("father","son")
father.son[1:30,]
summary(father.son)
plot(father.son)



####  Wages Datasets
library(ISLR)
Wage[1:20,]
plot(Wage$age,Wage$wage)
plot(Wage$year,Wage$wage)
plot(Wage$education,Wage$wage)


#####  Spam Datasets
###  The first 48 variables contain the frequency of the variable name (e.g., business) in the e-mail. If the variable name starts with 
###  num (e.g., num650) the it indicates the frequency of the corresponding number (e.g., 650). The variables 49-54 indicate the frequency
###  of the characters ‘;’, ‘(’, ‘[’, ‘!’, ‘\$’, and ‘\#’. The variables 55-57 contain the average, longest and total run-length of capital
###  letters. Variable 58 indicates the type of the mail and is either "nonspam" or "spam", i.e. unsolicited commercial e-mail.
library(kernlab)
set.seed(12345)
data(spam)
names(spam)
spam[1:10,]




#######   Handwritten digits datasets
library(jpeg)
myurl <- "https://raw.githubusercontent.com/RRighart/Digits/master/HandwrittenDigits.JPG" 
z <- tempfile()
download.file(myurl,z,mode="wb")
img <- readJPEG(z)
file.remove(z)

### show all handwritten digits images together
par(mfrow=c(1,1),
    oma = c(0.5,0.5,0.5,0.5) + 0.1,
    mar = c(0,0,0,0) + 0.1)
image(t(apply(img[c(1:dim(img)[1]), c(1:dim(img)[2]), 1], 2, rev)), col=grey.colors(255), axes=F, asp=1)
mtext("Whole image of handwritten digits", cex=0.6, col="red")





######## Political Blog Datasets
library(igraph)
polg=read.graph("Datasets/polblogs/polblogs.gml",format=c("gml"))
summary(polg)
vcount(polg)
ecount(polg)
E(polg)[sample(1:ecount(polg), 10)]



