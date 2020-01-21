
# coding: utf-8

# In[612]:

#import working libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().magic('matplotlib inline')
import statsmodels.api as sm

from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[613]:

#set working directory
os.chdir(r"C:\Users\himanshu gupta\Desktop\edwisor\project\2")


# In[614]:

#load data
bike_rental_data= pd.read_csv("day.csv")


# ### data exploration

# In[615]:

#dimension of data
bike_rental_data.shape


# In[616]:

#checking first 5 rows
bike_rental_data.head(5)


# In[617]:

#checking data type of all variables
bike_rental_data.dtypes


# In[618]:

# checking summary of the dataset
bike_rental_data.describe()


# In[619]:

#converting some useful variables into categorical variabels
categorical_var= ['season','yr','mnth','holiday','weekday','workingday','weathersit']
for a in categorical_var:
    bike_rental_data[a]=bike_rental_data[a].astype("category")


# In[620]:

#checking datatypes again
bike_rental_data.dtypes


# we will not use instant,dateday, casual and registered variable because they are not caryying useful information.

# ### data preprocessing

# ##### target variable distribution

# In[621]:

fig,(ax1,ax2) = plt.subplots(ncols=2)
fig.set_size_inches(10,6)
sn.distplot(bike_rental_data["cnt"],ax=ax1)
stats.probplot(bike_rental_data["cnt"], dist='norm', fit=True, plot=ax2)


# we can cleary see that cnt is very close to normal distribution.

# ##### missing value analysis

# In[622]:

bike_rental_data.isnull().sum()


# there are no missing values.

# ##### outliner analysis

# from above boxplot following things are clear:

# In[623]:

fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(14,14)
sn.boxplot(data=bike_rental_data,y="cnt",orient='v',ax=axes[0][0])
sn.boxplot(data=bike_rental_data,y="cnt",x="season",orient='v',ax=axes[0][1])
sn.boxplot(data=bike_rental_data,y="cnt",x="weekday",orient="v",ax=axes[1][0])
sn.boxplot(data=bike_rental_data,y="cnt",x="workingday",orient="v",ax=axes[1][1])
axes[0][0].set(ylabel='cnt',title = "Boxplot of cnt")
axes[0][1].set(xlabel="season",ylabel="cnt",title="Boxplot for cnt vs season")
axes[1][0].set(xlabel="weekday", ylabel="cnt",title="Boxplot for cnt vs weekday")
axes[1][1].set(xlabel="workingday",ylabel="cnt",title="Boxplot for cnt vs workingday")


# (1)there are no outliers in count.
# (2)demands for bike is very low in spring season.

# from above boxplot following things are clear:
# (1)there are no outliers in count.
# (2)demands for bike is very low in spring season.

# In[624]:

fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(14,14)
sn.boxplot(data=bike_rental_data,y="cnt",x="yr",orient='v',ax=axes[0][0])
sn.boxplot(data=bike_rental_data,y="cnt",x="mnth",orient='v',ax=axes[0][1])
sn.boxplot(data=bike_rental_data,y="cnt",x="holiday",orient='v',ax=axes[1][0])
sn.boxplot(data=bike_rental_data,y="cnt",x="weathersit",orient='v',ax=axes[1][1])
axes[0][0].set(xlabel="yr",ylabel="cnt",title="Boxplot for cnt vs yr")
axes[0][1].set(xlabel="mnth",ylabel="cnt",title="Boxplot for cnt vs mnth")
axes[1][0].set(xlabel="holiday",ylabel="cnt",title="Boxplot for cnt vs holiday")
axes[1][1].set(xlabel="weathersit",ylabel="cnt",title="Boxplot for cnt vs weathersit")


# from above boxplot following things are clear:

# (1) demands for bike is high in year 2011.
# (2) demands for bike is gardually incaresing from january to september and then started to decreasing.
# (3) demands for bike is high when there is clear weather.

# In[625]:

fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(14,14)
sn.boxplot(data=bike_rental_data,y="temp",orient='v',ax=axes[0][0])
sn.boxplot(data=bike_rental_data,y="atemp",orient='v',ax=axes[0][1])
sn.boxplot(data=bike_rental_data,y="hum",orient='v',ax=axes[1][0])
sn.boxplot(data=bike_rental_data,y="windspeed",orient='v',ax=axes[1][1])
axes[0][0].set(ylabel="temp",title="Boxplot for temp")
axes[0][1].set(ylabel="atemp",title="Boxplot for atemp")
axes[1][0].set(ylabel="hum",title="Boxplot for hum")
axes[1][1].set(ylabel="windspeed",title="Boxplot for windspeed")


# From the above boxplot we can cleary see that there are:
# (1) outliers in windspeed.
# (2) inliers in humidity.

# In[626]:

#removal of outliers and inliers
cnames=["hum","windspeed"]
for i in cnames:
    print(i)
    q75, q25 = np.percentile(bike_rental_data.loc[:,i], [75 ,25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)
    bike_rental_data = bike_rental_data.drop(bike_rental_data[bike_rental_data.loc[:,i] < min].index)
    bike_rental_data = bike_rental_data.drop(bike_rental_data[bike_rental_data.loc[:,i] > max].index)
    min = bike_rental_data.loc[bike_rental_data[i] < min,i] 
    max = bike_rental_data.loc[bike_rental_data[i] > max,i]


# subsituted inliers with minimum values and outliers with maximum values.

# In[627]:

#checking humidity and windspeed after removal of inliers and outliers
fig.set_size_inches(14,14)
sn.boxplot(data=bike_rental_data,y="hum").set_title("Boxplot of humidity")


# In[628]:

fig.set_size_inches(14,14)
sn.boxplot(data=bike_rental_data,y="windspeed").set_title("Boxplot of windspeed")


# ##### feature selection

# In[629]:

##Correlation analysis
#numeric variables
cnames=["temp","atemp","hum","windspeed","cnt"]
#Correlation plot
bike_rental_corr = bike_rental_data.loc[:,cnames]


# In[630]:

#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(9, 7))

#Generate correlation matrix
corr = bike_rental_corr.corr()

#Plot using seaborn library
sn.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sn.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# from the above plot,we came to know that both temp and atemp variables are carrying almost same information
# hence there is no need to continue with both variables.so we need to drop any one of the variables
# here we are dropping atemp variable.

# In[631]:

#Anova test for categorical variables(target variable is numeric)
#Save categorical variables
cat_names = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]


# In[632]:

for i in cat_names:
    results = ols('cnt' + '~' + i, data = bike_rental_data).fit()
    aov_table = sm.stats.anova_lm(results, typ = 2)
    print(aov_table)


# based on the anova result, we are going to drop three variables holiday,weekday,workingday
# because these variables have the p-value > 0.05

# In[633]:

# Removing the variables which have p-value > 0.05 and are correlated variable or does not contain useful information and store into a new dataset
df = bike_rental_data.drop(['atemp', 'holiday','weekday','workingday','instant','dteday','casual','registered'], axis=1)
bike_rental_data=df.copy()


# In[634]:

#now check dimension of data
bike_rental_data.shape


# In[635]:

bike_rental_data


# ##### feature scaling

# In[636]:

col=["temp","hum","windspeed","cnt"]


# In[637]:

for i in col:
    print(i)
    sn.distplot(bike_rental_data[i],bins='auto',color='black')
    plt.title("distribution plot for "+i)
    plt.ylabel("density")
    plt.show()


# based on distribution plot we can clearly see that all the numeric variables are normalized.

# ##### bivariate analysis

# In[638]:

# Bivariate analysis of cnt and continous variables

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12,8)

sn.regplot(x="temp",y="cnt",data=bike_rental_data,ax=ax1)
sn.regplot(x="hum",y="cnt",data=bike_rental_data,ax=ax2)
sn.regplot(x="windspeed",y="cnt",data=bike_rental_data,ax=ax3)


# from above boxplot it is clear that bike count has:
# (1) positive linear relationship with temperature.
# (2) slightly negative linear relationship with humidity.
# (3) negative linear reltionship with windspeed.

# # model devlopment

# In[639]:

#In Regression problems, we can't directly pass categorical variables.so we need to convert all categorical variables 
#into dummy variables.
ccol=['season','yr','mnth','weathersit']

#  Converting categorical variables to dummy variables
df = pd.get_dummies(bike_rental_data,columns=ccol)
bike_rental_data=df


# In[640]:

#Divide the data into train and test set 

x= bike_rental_data.drop(['cnt'],axis=1)
y= bike_rental_data['cnt']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.25)


# In[641]:

# Function for Error metrics to calculate the performance of model
def MAPE(y_true,y_prediction):
    mape= np.mean(np.abs(y_true-y_prediction)/y_true)*100
    return mape


# ### linear regression model

# In[642]:

LinearRegression_model= sm.OLS(y_train,x_train).fit()
print(LinearRegression_model.summary())


# In[643]:

# Model prediction on  train data
LinearRegression_train= LinearRegression_model.predict(x_train)

# Model prediction on test data
LinearRegression_test= LinearRegression_model.predict(x_test)

# Model performance on train data
MAPE_train= MAPE(y_train,LinearRegression_train)

# Model performance on test data
MAPE_test= MAPE(y_test,LinearRegression_test)

# r2 value for train data
r2_train= r2_score(y_train,LinearRegression_train)

# r2 value for test data-
r2_test=r2_score(y_test,LinearRegression_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,LinearRegression_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,LinearRegression_test))

print("Mean Absolute % Error for train data="+str(MAPE_train))
print("Mean Absolute % error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[644]:

Error_MetricsLT = {'Model Name': ['Linear Regression'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}

LinearRegression_Results = pd.DataFrame(Error_MetricsLT)


# In[645]:

LinearRegression_Results


# ### random forest

# In[646]:

# Random Forest for regression
RF_model= RandomForestRegressor(n_estimators=80).fit(x_train,y_train)

# Prediction on train data
RF_train= RF_model.predict(x_train)

# Prediction on test data
RF_test= RF_model.predict(x_test)

# MAPE For train data
MAPE_train= MAPE(y_train,RF_train)

# MAPE For test data
MAPE_test= MAPE(y_test,RF_test)

# Rsquare  For train data
r2_train= r2_score(y_train,RF_train)

# Rsquare  For test data
r2_test=r2_score(y_test,RF_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,RF_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RF_test))

print("Mean Absolute % Error for train data="+str(MAPE_train))
print("Mean Absolute % Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[647]:

Error_MetricsRF = {'Model Name': ['Random Forest'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
RandomForest_Results = pd.DataFrame(Error_MetricsRF)
RandomForest_Results


# ### decision tree

# In[648]:

# Decision tree for regression
DecisionTree_model= DecisionTreeRegressor(max_depth=3).fit(x_train,y_train)

# Model prediction on train data
DecisionTree_train= DecisionTree_model.predict(x_train)

# Model prediction on test data
DecisionTree_test= DecisionTree_model.predict(x_test)

# Model performance on train data
MAPE_train= MAPE(y_train,DecisionTree_train)

# Model performance on test data
MAPE_test= MAPE(y_test,DecisionTree_test)

# r2 value for train data
r2_train= r2_score(y_train,DecisionTree_train)

# r2 value for test data
r2_test=r2_score(y_test,DecisionTree_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,DecisionTree_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,DecisionTree_test))

print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str(RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[649]:

Error_MetricsDT = {'Model Name': ['Decision Tree'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
DecisionTree_Results = pd.DataFrame(Error_MetricsDT)
DecisionTree_Results 


# From above results Random Forest & linear regression both model have optimum values and this algorithms are good for our data 

# In[650]:

#saveing the out put of finalized model (random forest)

input = y_test.reset_index()
predicted = pd.DataFrame(RF_test,columns = ['predicted'])
Final_result = predicted.join(input)
Final_result


# In[651]:

Final_result.to_csv("Final_results_python.csv",index=False)


# In[ ]:



