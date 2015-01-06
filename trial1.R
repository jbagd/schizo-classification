#Kaggle competition: Diagnose schizophrenia using multimodal features from MRI scans

#very basic magic in R (logistic regression)

#load the training data, split in two parts: testing (25%) and training (75%)
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

#logistic regression using all features
training$Class<-as.numeric(training$Class)
testing$Class<-as.numeric(testing$Class)
j<-as.matrix(training[1:65,2:411])
w<-as.matrix(testing[,2:411])
library(LiblineaR)
libfit<-LiblineaR(j, training[,412], type=6, cost=0.16,bias = TRUE, wi = NULL, verbose = TRUE)
predtest<-predict(libfit,w,proba=TRUE)
confusionMatrix(predtest$predictions, testing$Class)


#load the real testing set, apply the solution and write results to file
FNC_test<-read.csv("test/test_FNC.csv")
SBM_test<-read.csv("test/test_SBM.csv")
Test<-merge(FNC_test,SBM_test,by="Id")
Test<-as.matrix(Test[,2:411])
predtest<-predict(libfit,Test,proba=TRUE)
probi<-predtest[[2]][,1]
submission<-read.csv("submission_example.csv")
submission[,2]=probi
write.csv(submission,"trial1.csv",row.names=FALSE)