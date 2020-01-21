# bike rental count prediction

#first clean R enviorment
rm(list=ls(all=T))

#set  working directory
setwd("C:/Users/himanshu gupta/Desktop/edwisor/project/2")
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", "sampling", "DataCombine", "inTrees","gridExtra","scales","psych","gplots")

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#lets load the data
bike_rental_data = read.csv("day.csv")

#------------------------------------------------------#
#                 explore the data                     #

dim(bike_rental_data)
names(bike_rental_data)
head(bike_rental_data)
str(bike_rental_data)
summary(bike_rental_data)

#in our dataset some variables have no useful information for our prediction
#so it is better to remove those variables.it helps us to make useful inferences

#lets drop unnecessary variables
bike_rental_data = subset(bike_rental_data,select = -c(instant,dteday,casual,registered))

#------------------------------------------------------#
#                 data-preprocessing                   #

#missing value analysis
sapply(bike_rental_data, function(x) {
  sum(is.na(x))
})
# there are no missing values

#outlier analysis

cnames=c("temp",'atemp','windspeed','hum','cnt')
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bike_rental_data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="green", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="count")+
           ggtitle(paste("Box plot of count for",cnames[i])))
}

#plotting boxplot
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,ncol=1)

#lets remove outliers using boxplot
df = bike_rental_data
for(i in cnames){
  print(i)
  outliers = bike_rental_data[,i][bike_rental_data[,i] %in% boxplot.stats(bike_rental_data[,i])$out]
  print(length(outliers))
  bike_rental_data = bike_rental_data[which(!bike_rental_data[,i] %in% outliers),]
}

#lets plot boxplot after removing outliers
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bike_rental_data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="green", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="cnt")+
           ggtitle(paste("Box plot of cnt for",cnames[i])))
}
#plotting Boxplot after removing outliers
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,ncol=1)

#------------------------------------------------------#
#                 feature selection                    #

#find correlation matrix using corrplot and correlation plot using corrgram library
#FOR NUMERICAL VARIABLES

#save dataset after outlier analysis
df = bike_rental_data
#correlation matrix
cnames=c("temp","atemp","windspeed","hum")
sapply(bike_rental_data,class)
correlation_matrix = cor(bike_rental_data,cnames)
correlation_matrix
#correlation plot
corrgram(bike_rental_data[,cnames],order = F,upper.panel = panel.pie,
         text.panel = panel.txt,main = 'Correlation plot')

#From the correlation plot,we see that temp and atemp variables are correlated to each other
#so we need to remove atemp variable.

#perform annova test for categorical variables
catnames=c('season','yr','mnth','holiday','weekday','workingday','weathersit')
for (i in catnames) {
  print(i)
  anova = summary(aov(formula = cnt~bike_rental_data[,i],bike_rental_data))
  print(anova)
}
#based on the anova result, we can drop three variables named,
# holiday, weekday, workingday
#because these variables having the p-value > 0.05

#Dimension reduction
bike_rental_data = subset(bike_rental_data,select = -c(holiday,weekday,workingday,atemp))

#lets check after dimension reduction
dim(bike_rental_data)
head(bike_rental_data)

#------------------------------------------------------#
#                 feature scaling                      #

#check normality between the varaibles

cnames=c("temp","windspeed","hum","cnt")
for (i in cnames){
  print(i)
  normality = qqnorm(bike_rental_data[,i])
  
}

#already we plotted distrution between these variables,lets recall it
for(i in 1:length(cnames))
{
  assign(paste0("h",i),ggplot(aes_string(x=(cnames[i])),
                              data=subset(bike_rental_data))+
           geom_histogram(fill="blue",colour = "green")+geom_density()+
           scale_y_continuous(breaks =pretty_breaks(n=8))+
           scale_x_continuous(breaks = pretty_breaks(n=8))+
           theme_bw()+xlab(cnames[i])+ylab("Frequency")+
           ggtitle(paste("distribution plot for ",cnames[i])))
}
gridExtra::grid.arrange(h1,h2,h3,h4,ncol = 2)

#summary of the data
for (i in cnames) {
  print(i)
  print(summary(bike_rental_data[,i]))
  
}
#Based on the above inferences and plots,we can see that the variables are normalised.

# bivariate analysis for categorical variables
bivariate_categorical <-
  function(dataset, variable, targetVariable) {
    variable <- enquo(variable)
    targetVariable <- enquo(targetVariable)
    
    ggplot(
      data = dataset,
      mapping = aes_(
        x = rlang::quo_expr(variable),
        y = rlang::quo_expr(targetVariable),
        fill = rlang::quo_expr(variable)
      )
    ) +
      geom_boxplot() +
      theme(legend.position = "bottom") -> p
    plot(p)
    
  }

bivariate_continous <-
  function(dataset, variable, targetVariable) {
    variable <- enquo(variable)
    targetVariable <- enquo(targetVariable)
    ggplot(data = dataset,
           mapping = aes_(
             x = rlang::quo_expr(variable),
             y = rlang::quo_expr(targetVariable)
           )) +
      geom_point() +
      geom_smooth() -> q
    plot(q)
    
  }

bivariate_categorical(bike_rental_data, season, cnt)
bivariate_categorical(bike_rental_data, yr, cnt)
bivariate_categorical(bike_rental_data, mnth, cnt)
bivariate_categorical(bike_rental_data, weathersit, cnt)
bivariate_continous(bike_rental_data, temp, cnt)
bivariate_continous(bike_rental_data, hum, cnt)
bivariate_continous(bike_rental_data, windspeed, cnt)


#------------------------------------------------------#
#                 model devlopment                     #

#we can not pass categorical variables to regression problems
#so convert categorical variables into dummy variables
#saving our preprocessed data
df = bike_rental_data

#create dummies
library(dummies)
catnames = c('season','yr','mnth','weathersit')
bike_rental_data = dummy.data.frame(bike_rental_data,catnames)

#we have created dummies,lets check dimension and top 5 observations
dim(bike_rental_data)
head(bike_rental_data)

#divide the data into train and test
set.seed(1234)
train_index = sample(1:nrow(df), 0.8 * nrow(df))
train_data = bike_rental_data[train_index,]
test_data = bike_rental_data[-train_index,]

#------------------------------------------------------#
#                 (1) linear regression                #

#running regression model
lm_model = lm(cnt~. ,data = bike_rental_data)
#lets check performance of our modedl
summary(lm_model)
#Residual standard error: 787.3 on 696 degrees of freedom
#Multiple R-squared:  0.8388,	Adjusted R-squared:  0.8342 
#F-statistic: 181.1 on 20 and 696 DF,  p-value: < 2.2e-16

# Function for Error metrics to calculate the performance of model
#lets build function for MAPE
#calculate MAPE
MAPE = function(y, y1){
  mean(abs((y - y1)/y))
}

# Function for r2 to calculate the goodness of fit of model
rsquare=function(y,y1){
  cor(y,y1)^2
}

# Function for RMSE value 
RMSE = function(y,y1){
  difference = y - y1
  root_mean_square = sqrt(mean(difference^2))
}

#lets predict for train and test data
Predictions_LR_train = predict(lm_model,train_data[,-25])
Predictions_LR_test = predict(lm_model,test_data[,-25])

#let us check performance of our model

#mape calculation
LR_train_mape = MAPE(Predictions_LR_train,train_data[,25])
LR_test_mape = MAPE(test_data[,25],Predictions_LR_test)

#Rsquare calculation
LR_train_r2 = rsquare(train_data[,25],Predictions_LR_train)
LR_test_r2 = rsquare(test_data[,25],Predictions_LR_test)

#rmse calculation
LR_train_rmse = RMSE(train_data[,25],Predictions_LR_train)
LR_test_rmse = RMSE(test_data[,25],Predictions_LR_test)

print(LR_train_mape) #0.15
print(LR_test_mape) #0.18
print(LR_train_r2) #0.831
print(LR_test_r2) #0.867
print(LR_train_rmse) #789.6
print(LR_test_rmse) #717.2


#------------------------------------------------------#
#                 (2) decision tree                #

library(rpart)
DT_model = rpart(cnt ~ ., data = train_data, method = "anova")
DT_model


#predicting for train and test data
predictions_DT_train= predict(DT_model,train_data[,-25])
predictions_DT_test= predict(DT_model,test_data[,-25])

# MAPE calculation
DT_train_mape = MAPE(train_data[,25],predictions_DT_train)
DT_test_mape = MAPE(test_data[,25],predictions_DT_test)

# Rsquare calculation
DT_train_r2= rsquare(train_data[,25],predictions_DT_train)
DT_test_r2 = rsquare(test_data[,25],predictions_DT_test)

# RMSE calculation
DT_train_rmse = RMSE(train_data[,25],predictions_DT_train)
DT_test_rmse = RMSE(test_data[,25],predictions_DT_test)

print(DT_train_mape) #0.522
print(DT_test_mape) #0.243
print(DT_train_r2) #0.811
print(DT_test_r2) #0.798
print(DT_train_rmse) #833.848
print(DT_test_rmse) #885.59

#------------------------------------------------------#
#                 (3) Random Forest                #

#building random forest model
RF_model = randomForest(cnt~.,data = train_data,n.trees = 600)
print(RF_model)

#lets predict for both train and test data
predictions_RF_train = predict(RF_model,train_data[-25])
predictions_RF_test = predict(RF_model,test_data[-25])

#MAPE calculation
RF_train_mape = MAPE(predictions_RF_train,train_data[,25])
RF_test_mape = MAPE(predictions_RF_test,test_data[,25])

#Rsquare calculation
RF_train_r2 = rsquare(predictions_RF_train,train_data[,25])
RF_test_r2 = rsquare(predictions_RF_test,test_data[,25])

#RMSE calculation
RF_train_rmse = RMSE(train_data[,25],predictions_RF_train)
RF_test_rmse = RMSE(test_data[,25],predictions_RF_test)

print(RF_train_mape) #0.07
print(RF_test_mape) #0.12
print(RF_train_r2) #0.965
print(RF_test_r2) #0.910
print(RF_train_rmse) #371.18
print(RF_test_rmse) #593.84

#------------------------------------------------------#
#                   model selection                    #

Model_name = c('Linear regression',
               'Decision tree',
               'Random forest')

MAPE_train = c(LR_train_mape,DT_train_mape,
               RF_train_mape)

MAPE_test = c(LR_test_mape,DT_test_mape,
              RF_test_mape)

Rsquare_train = c(LR_train_r2,DT_train_r2,
                  RF_train_r2)

Rsquare_test = c(LR_test_r2,DT_test_r2,
                 RF_test_r2)

RMSE_train =  c(LR_train_rmse,DT_train_rmse,
                RF_train_rmse)

RMSE_test = c(LR_test_rmse,DT_test_rmse,
              RF_test_rmse)

FINAL_RESULTS = data.frame(Model_name,MAPE_train,MAPE_test,Rsquare_train,Rsquare_test,
                           RMSE_train,RMSE_test)

print(FINAL_RESULTS)


#Index  Model_name           MAPE_train  MAPE_test    Rsquare_train  Rsquare_test  RMSE_train  RMSE_test
#1      Linear regression    0.15497164  0.1829289    0.8311816      0.8671739     789.6785    717.2833
#2      Decision tree        0.52210598  0.2438791    0.8119266      0.7986807     833.4855    885.5906
#3      Random forest        0.07256787  0.1224177    0.9652738      0.9103488     371.1827    593.8403

# Based on the above inferences,we came to know that Random forest performs very well in our dataset
#so we are finalising that model.

