rm(list = ls())
data <- read.csv('E-Commerce_shipping_data.csv', header = TRUE) 
#remove the first column 'ID' 
data <- data[,-1]
#transform categorical variables to factor 
data$Warehouse_block<- as.factor(data$Warehouse_block)
data$Mode_of_Shipment <- as.factor(data$Mode_of_Shipment)
data$Gender <- as.factor(data$Gender)
data$Customer_rating <- as.factor(data$Customer_rating)
data$Product_importance <- as.factor(data$Product_importance)
data$Customer_care_calls <- as.factor(data$Customer_care_calls)
data$Reached.on.Time_Y.N <- as.factor(data$Reached.on.Time_Y.N)

dim(data)
summary(data)
#Detect missing values in each column 
colSums(is.na(data))

library(caret)
set.seed(7406)
trainIndex <- createDataPartition(data$Reached.on.Time_Y.N, p=0.8, list = FALSE, times = 1)
df_train <- data[trainIndex,]
df_test <- data[-trainIndex,]

summary(df_train)
str(df_train)

####################EDA###############################3
library(ggplot2)
library(ggcorrplot)
#stacked bar charts - categorical vs categorical 
ggplot(data = df_train, aes(x=Reached.on.Time_Y.N, ..count..))+geom_bar(aes(fill=Reached.on.Time_Y.N),position='dodge')+xlab('Reach on Time (0=Yes, 1=No)')
ggplot(data = df_train, aes(x=Reached.on.Time_Y.N, ..count..))+geom_bar(aes(fill=Product_importance),position='dodge')+xlab('Reach on Time (0=Yes, 1=No)')

#to display which warehouse block has the most volume and shipping methods
ggplot(data = df_train, aes(x="",fill=Mode_of_Shipment))+geom_bar(width = 1)+coord_polar('y')+theme_void()
ggplot(data = df_train, aes(x=Warehouse_block, ..count..))+geom_bar(aes(fill=Warehouse_block),position='dodge')+xlab('Reach on Time (0=Yes, 1=No)')
ggplot(data=df_train,aes(Warehouse_block, fill = Mode_of_Shipment))+geom_bar(position = "stack")

#display customer rating and customer rating vs gender
ggplot(data=df_train,aes(x=Customer_rating, ..count..))+geom_bar(aes(fill=Customer_rating),position='dodge')+xlab('Reach on Time (0=Yes, 1=No)')
ggplot(data=df_train,aes(x=Customer_rating,..count.., fill = Gender))+geom_bar(position = "stack")+xlab('Reach on Time (0=Yes, 1=No)')


#stacked bar chart to see which gender gives higher rating 
ggplot(data=df_train,aes(x=Gender, ..count..))+geom_bar(aes(fill=Gender),position='dodge')+xlab('Reach on Time (0=Yes, 1=No)')
ggplot(data=df_train,aes(x=Reached.on.Time_Y.N, fill = Customer_care_calls))+geom_bar(position = "stack")+xlab('Reach on Time (0=Yes, 1=No)')
ggplot(data=df_train,aes(Customer_care_calls, fill = Gender))+geom_bar(position = "stack")

#boxplots 
ggplot(data = df_train, aes(x=Reached.on.Time_Y.N, y = Cost_of_the_Product, fill=Reached.on.Time_Y.N))+geom_boxplot()+xlab('Reach on Time')+labs(fill='Reach on Time \n(0=Yes, 1=No)')
ggplot(data = df_train, aes(x=Reached.on.Time_Y.N, y = Discount_offered, fill=Reached.on.Time_Y.N))+geom_boxplot()+xlab('Reach on Time')+labs(fill='Reach on Time \n(0=Yes, 1=No)')
ggplot(data = df_train, aes(x=Reached.on.Time_Y.N, y = Weight_in_gms, fill=Reached.on.Time_Y.N))+geom_boxplot()+xlab('Reach on Time')+labs(fill='Reach on Time \n(0=Yes, 1=No)')

#correlation 
corr <- round(cor(df_train[,c(5,9,10)]),2)
ggcorrplot(corr, lab = TRUE)

#######
# Methods
#1. Logistics regression (baseline model: simple and easy to intrepret)
#2. Classification tree 
#3. RF 
#4. Boosting 

TrainErr <- NULL
TestErr <- NULL

####
#1.Logistics 
m1 <- glm(Reached.on.Time_Y.N~., data = df_train, family = binomial(link = "logit"))

#variable selection stepwise BIC (increase prediction accuracy)
m1_reduced<-step(m1, trace = FALSE, k=log(length(df_train)))
summary(m1_reduced)

round(m1_reduced$coefficients,4)

#deviance test 
#result: p value is > 0.05 we fails to reject null hypothesis, so the reduced model is better 
anova(m1_reduced,m1,test = 'Chisq')

pred_tr <- predict(m1_reduced, df_train[,1:10], type = 'response')
predClass <- ifelse(pred_tr>0.5,1,0)
TrainErr1 <- mean(predClass != df_train$Reached.on.Time_Y.N)
TrainErr <- c(TrainErr, TrainErr1)
TrainErr

pred_te <- predict(m1_reduced, df_test[,1:10], type = 'response')
predClass_te <- ifelse(pred_te>0.5,1,0)
TestErr1 <- mean(predClass_te != df_test$Reached.on.Time_Y.N)
TestErr <- c(TestErr, TestErr1)
TestErr

#10 folds cv and resample 10 times 
control <- trainControl(method = 'repeatedcv', number = 10, repeats = 10)
m1_cv <- train(Reached.on.Time_Y.N~Customer_care_calls+Cost_of_the_Product+Prior_purchases+Product_importance+Discount_offered+Weight_in_gms, data = df_train, method = 'glm', family='binomial', trControl = control)
m1_cv

########
#2. Classification Tree
library(rpart)
library(rpart.plot)
set.seed(7406)
m2 <- rpart(Reached.on.Time_Y.N ~ .,data=df_train, method="class", parms=list(split="gini"), xval=10)
printcp(m2)
plotcp(m2)
rpart.plot(m2)

#look for tuning parameter cp 
opt <- which.min(m2$cptable[, "xerror"])
cp1 <- m2$cptable[opt, "CP"]
m2_pruned <- rpart.pruned1 <- prune(m2,cp=cp1)
plot(m2_pruned,compress=TRUE)
text(m2_pruned)
rpart.plot(m2_pruned)

pred2_tr<-  predict(m2_pruned, df_train[,1:10],type="class")
TrainErr2 = mean(pred2_tr != df_train$Reached.on.Time_Y.N)
TrainErr <- c(TrainErr, TrainErr2)
TrainErr

pred2_te<-  predict(m2_pruned, df_test[,1:10],type="class")
TestErr2 = mean(pred2_te != df_test$Reached.on.Time_Y.N)
TestErr <- c(TestErr, TestErr2)
TestErr

#10 folds cv and resample 10 times 
m2_cv <- train(Reached.on.Time_Y.N~., data = df_train, method = 'rpart', trControl = control)
m2_cv 

#######
#3. RF
library(randomForest)
trControl <- trainControl(method = 'cv', number = 10, search = 'random')
m3_fit <- train(Reached.on.Time_Y.N~., data = df_train, method='rf', metric='Accuracy',trControl=trControl)
m3_fit

best_mtry=m3_fit$bestTune$mtry

tuneGrid <- expand.grid(.mtry=best_mtry)
rf_tree <- list()
for (ntree in c(500,1000,1500,2000)) {
  set.seed(7406)
  fit <- train(Reached.on.Time_Y.N~., data = df_train, method='rf', metric='Accuracy',trControl=trControl,tuneGrid=tuneGrid,ntree=ntree)
  key <- toString(ntree)
  rf_tree[[key]] <- fit
}
results_tree <- resamples(rf_tree)
summary(results_tree)

m3 <- randomForest(Reached.on.Time_Y.N~., data=df_train, ntree=2000,mtry=2, importance=TRUE)
importance(m3, type=2)
varImpPlot(m3)

#######
#4. Boosting 
#random search on 5-folds to find hypermeters 
set.seed(7406)
fitControl <- trainControl(method = 'cv', number = 10, search = 'random')
m4_fit <- train(Reached.on.Time_Y.N~., data = df_train, method = 'gbm',trControl = fitControl, verbose = FALSE, tuneLength=5)
m4_fit

library(gbm)
m4 <- gbm(Reached.on.Time_Y.N ~ .,data=df_train,distribution = 'bernoulli', n.trees = 1452, interaction.depth = 4, shrinkage =0.1951, n.minobsinnode = 19)
print(m4)
summary(m4)



