#Kaggle competition: Diagnose schizophrenia using multimodal features from MRI scans

# smarter magic that appeared to work better than simple magic (trial1) 
# makes a selection of a best logistic regression model and the optimal cost

# load the training data, split in two parts: testing (25%) and training (75%)
library(caret)
set.seed(3433)
labels_train = read.csv(file='Train/train_labels.csv',head=TRUE,sep=",")
FNC_train = read.csv(file='Train/train_FNC.csv',head=TRUE,sep=",")
SBM_train = read.csv(file='Train/train_SBM.csv',head=TRUE,sep=",")
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

# find best parameters for LiblineaR; try vary cross
t=0
tryTypes=c(6:7)
tryCosts=c(0.7,0.5,0.4,0.3,0.2,0.1,0.01,0.001)
bestCost=NA
bestAcc=0
bestType=NA
for(ty in tryTypes){
for(co in tryCosts){
acc=LiblineaR(data=s,labels=training[,412],type=ty,cost=co,bias=TRUE,cross=10,verbose=FALSE)
cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
if(acc>bestAcc){
bestCost=co
bestAcc=acc
bestType=ty
}
}
}
cat("Best model type is:",bestType,"\n")
cat("Best cost is:",bestCost,"\n")
cat("Best accuracy is:",bestAcc,"\n")

# use best model to refit the data
m=LiblineaR(data=s,labels=training[,412],type=bestType,cost=bestCost,bias=TRUE,verbose=FALSE)

# scale and center the testing features
s2=scale(w,attr(s,"scaled:center"),attr(s,"scaled:scale"))
# predict
p=predict(m,s2,proba=TRUE)  
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
write.csv(submission,"trial3.csv",row.names=FALSE)