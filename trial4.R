#Kaggle competition: Diagnose schizophrenia using multimodal features from MRI scans

# Here I tried using a heuristic C thing to estimate the cost of the logistic regression. Not convinced that it helped (compared to trial3).
# load the training data, split in two parts: testing (25%) and training (75%)
library(caret)
set.seed(3433)
labels_train = read.csv(file='train_labels.csv',head=TRUE,sep=",")
FNC_train = read.csv(file='train_FNC.csv',head=TRUE,sep=",")
SBM_train = read.csv(file='train_SBM.csv',head=TRUE,sep=",")
q<-merge(FNC_train,SBM_train,by="Id")
q<-merge(q,labels_train,by="Id")
inTrain = createDataPartition(q$Class, p = 3/4)[[1]]
training = q[ inTrain,]
testing = q[-inTrain,]

# logistic regression using all features

training$Class<-as.numeric(training$Class)
testing$Class<-as.numeric(testing$Class)
j<-as.matrix(training[1:65,2:411])
w<-as.matrix(testing[,2:411])
library(LiblineaR)
# scale and center the training features
s=scale(j,center=TRUE,scale=TRUE)
# Tune the cost parameter of a logistic regression according to the Joachim's heuristics
co=heuristicC(s)
m=LiblineaR(data=s,labels=training[,412],type=6,cost=co,bias=TRUE,verbose=FALSE)
# scale and center the testing features
s2=scale(w,attr(s,"scaled:center"),attr(s,"scaled:scale"))
# 
pr=FALSE
if(bestType==0 | bestType==7) pr=TRUE
p=predict(m,s2,proba=pr,decisionValues=TRUE)
confusionMatrix(p$predictions,testing$Class)
library(verification)
roc.area(testing$Class,p$predictions)

#load the real testing set, apply the solution and write results to file
FNC_test<-read.csv("test/test_FNC.csv")
SBM_test<-read.csv("test/test_SBM.csv")
Test<-merge(FNC_test,SBM_test,by="Id")
TTest<-as.matrix(Test[,2:411])
s3=scale(TTest,attr(s,"scaled:center"),attr(s,"scaled:scale")) 
predtest<-predict(m,s3,proba=TRUE)
probi<-predtest[[2]][,1]
submission<-read.csv("submission_example.csv")
submission[,2]=probi
write.csv(submission,"trialX.csv",row.names=FALSE)

