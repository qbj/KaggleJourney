
#train_data <-read.csv('C:\\Users\\fu\\Documents\\kaggle\\titanic\\train.csv')
#head(train_data)
#titanic sink

#get train data
train_data<-read.csv('C:\\Users\\fu\\Documents\\kaggle\\titanic\\train.csv')

#pick vip attributes
train_data.pick <- train_data
train_data.pick
train_data.pick$PassengerId = NULL
train_data.pick$Name = NULL
train_data.pick$Embarked = NULL 
train_data.pick$Ticket = NULL 
train_data.pick$Fare = NULL 
names(train_data.pick)

##clean data 
summary(train_data.pick$Survived) #1=yes 0=no
summary(train_data.pick$Pclass)

#female -> 1, male -> 2
summary(train_data.pick$Sex)
v = NULL
for ( i in 1:nrow(train_data.pick)) {
  #print(train_data.pick$Sex[i])
  if(train_data.pick$Sex[i]=='male'){
    v <- c(v,1)
  }
  else{
    v <- c(v,2)
  }
}
train_data.pick['nSex'] = v
train_data.pick$Sex =NULL
names(train_data.pick)

summary(train_data.pick$nSex)
summary(train_data.pick$Age)
#na
t_mean<-mean(na.omit(train_data.pick$Age) )
for (i in 1:nrow(train_data.pick)){
  #print(is.na(train_data.pick$Age[i]))
  #print(i)
  if(is.na(train_data.pick$Age[i])==TRUE){
    train_data.pick$Age[i]<-t_mean
  }
}
summary(train_data.pick$Age)
summary(train_data.pick$SibSp)
summary(train_data.pick$Parch)

#Cabin to number
summary(train_data.pick$Cabin)
nCabin <- NULL
for (i in 1:nrow(train_data.pick)){
  t<-as.character(train_data.pick$Cabin[i])
  contain_alpha <- grepl("[[:alpha:]]",t)
  if ( contain_alpha ){
    alpha_t <- gsub("\\d+|NA", "", t)
    code_t <- utf8ToInt(substr(gsub("\\d+|NA", "", t),1,1))-64
  }else {
    code_t <- 0
  } 
  nCabin <- c(nCabin,code_t)
}
train_data.pick["nCabin"] <-nCabin
train_data.pick$Cabin<-NULL


#SUPPORT VECTOR MACHINE#
library(e1071)#svm


#learning from training
svmmodel<-svm(as.factor(Survived)~., data=train_data.pick, method="C-classification", kernel="radial",cross=5, probability=TRUE)

#get test data
test_data<-read.csv('C:\\NotBackedUp\\fu\\temp\\Titanic Sink\\data\\test.csv')

#pick vip attributes
test_data.pick <- test_data
test_data.pick$PassengerId = NULL
test_data.pick$Name = NULL
test_data.pick$Embarked = NULL 
test_data.pick$Ticket = NULL 
test_data.pick$Fare = NULL 
names(test_data.pick)


##clean data 
summary(test_data.pick$Survived)
summary(test_data.pick$Pclass)

#female -> 1, male -> 2
summary(test_data.pick$Sex)
v = NULL
for ( i in 1:nrow(test_data.pick)) {
  #print(train_data.pick$Sex[i])
  if(test_data.pick$Sex[i]=='male'){
    v <- c(v,1)
  }
  else{
    v <- c(v,2)
  }
}
test_data.pick['nSex'] = v
test_data.pick$Sex =NULL
names(test_data.pick)

summary(test_data.pick$nSex)
summary(test_data.pick$Age)
#na
t_mean<-mean(na.omit(test_data.pick$Age) )
for (i in 1:nrow(test_data.pick)){
  if(is.na(test_data.pick$Age[i])==TRUE){
    test_data.pick$Age[i]<-t_mean
  }
}
summary(test_data.pick$Age)
summary(test_data.pick$SibSp)
summary(test_data.pick$Parch)

#Cabin to number
summary(test_data.pick$Cabin)
nCabin <- NULL
for (i in 1:nrow(test_data.pick)){
  t<-as.character(test_data.pick$Cabin[i])
  contain_alpha <- grepl("[[:alpha:]]",t)
  if ( contain_alpha ){
    alpha_t <- gsub("\\d+|NA", "", t)
    code_t <- utf8ToInt(substr(gsub("\\d+|NA", "", t),1,1))-64
  }else {
    code_t <- 0
  } 
  nCabin <- c(nCabin,code_t)
}
test_data.pick["nCabin"] <-nCabin
test_data.pick$Cabin<-NULL
test_data.pick$nCabin

#predicting the test data
svmmodel.predict<-predict(svmmodel,test_data.pick,decision.values=TRUE)
svmmodel.probs<-attr(svmmodel.predict,"decision.values")
svmmodel.class<-svmmodel.predict[1:nrow(test_data.pick)]



svmmodel.truth
svmmodel.class


#analyzing result
truth<-read.csv('C:\\NotBackedUp\\fu\\temp\\Titanic Sink\\data\\gender_submission.csv')
svmmodel.truth<-truth$Survived

library(SDMTools)#confusion
svmmodel.confusion<-confusion.matrix(svmmodel.truth,svmmodel.class)
svmmodel.accuracy<-prop.correct(svmmodel.confusion) #/t
svmmodel.recall <- #tf/t
  
  
  svmmodel.confusion
#roc analysis for test data
svmmodel.prediction<-prediction(svmmodel.probs,svmmodel.labels)
svmmodel.performance<-performance(svmmodel.prediction,"tpr","fpr")
svmmodel.auc<-performance(svmmodel.prediction,"auc")@y.values[[1]]



final_result<-test_data[1:2]
final_result['Survived']<-svmmodel.class
final_result$Pclass<-NULL



write.csv(final_result,quote = FALSE,
          "C:\\NotBackedUp\\fu\\temp\\Titanic Sink\\data\\my_submission.csv",
          row.names=FALSE
)