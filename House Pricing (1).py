#!/usr/bin/env python
# coding: utf-8

# # All Lifecycles in a datascience project

# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 5. Model Deployment

# In[256]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

pd.pandas.set_option("display.max_columns",None)      #will show all columns no matter how many


# In[257]:


train = pd.read_csv('train[1].csv')
train.head()


# In[258]:


train.shape


# ## Missing Values

# In[259]:


#check % of nan values present in each feature
#1. list of features having nan values
#2. feature name and % of missing values

features_with_na = [features for features in train.columns if train[features].isnull().sum()>1]

for feature in features_with_na:
    print(feature, np.round(train[feature].isnull().mean(), 4),  "% missing values")


# Since there are many missing values so finding the relationship between missing value features and sales price by plotting some diagram

# In[260]:


for feature in features_with_na:
    data=train.copy()
    
    #creating variable s.t features having nan obsv = 1 else features not having nan values 0
    data[feature] = np.where(data[feature].isnull(),1,0)

    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.title(feature)
    plt.show()
    


# In[ ]:





# ## numerical variables

# In[261]:


numerical_features = [feature for feature in train.columns if train[feature].dtype!= "O"]
print("Number of numerical variables:", len(numerical_features))
train[numerical_features].head()


# ## temporal variable(e.g Datetime variables)

# We have 4 year features in the dataset.

# In[262]:


#features that contain year info

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_feature


# In[263]:


#numerical variables are of 2 types
#discrete and continous

discrete_feature = [feature for feature in numerical_features if len(train[feature].unique())<25 and feature not in year_feature+['Id']]
print('Discrete variable: {}'.format(len(discrete_feature)))
discrete_feature


# In[264]:


#relationship b/w discrete var and output feature

for feature in discrete_feature:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.title(feature)
    plt.ylabel('SalePrice')
    plt.show()


# In[265]:



continous_feature= [feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print('Continous variable: {}'.format(len(continous_feature)))


# In[266]:


for feature in continous_feature:
    data=train.copy()
    data[feature].hist(bins=20)
    plt.xlabel(feature)
    plt.title(feature)
    plt.ylabel('SalePrice')
    plt.show()


# it can be clearly seen that the graphs in continous variables are skewed so we use logarithmic transformation here

# # Exploratory Data Analysis

# In[267]:


#using logarithmic transformation

for feature in continous_feature:
    data=train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# In[268]:


for feature in continous_feature:
    data=train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        


# In[269]:


categorical_features= [feature for feature in train.columns if train[feature].dtypes=='O']
categorical_features


# In[270]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'. format(feature,len(train[feature].unique())))


# In[271]:


for feature in categorical_features:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# # Feature Engineering
Steps in Feature Enginnering:

1) Missing Values
2) Temporal Variables
3) Categorical variables: remove rare labels
4) Standarise the value of variables to rare labels
# ## missing values

# In[272]:


#handling missing values of categorical features

features_nan=[feature for feature in train.columns if train[feature].isnull().sum()>1 and train[feature].dtypes=='O']

for feature in features_nan:
    print("{}, {}% missing values".format(feature,np.round(data[feature].isnull().sum().mean(),4)))


# In[273]:


#replace missing values with a new label

def replace_cat_features(train,features_nan):
    data=train.copy()
    data[features_nan]=data[features_nan].fillna('missing')
    return(data)

train=replace_cat_features(train,features_nan)

train[features_nan].isnull().sum()
    


# In[274]:


train.head()


# In[275]:


#lets check for numerical variables that contain missing values

numerical_with_nan=[feature for feature in train.columns if train[feature].isnull().sum()>1 and train[feature].dtypes!='O']

for feature in numerical_with_nan:
    print("{}: {}% missing values". format(feature,np.round(train[feature].isnull().sum().mean(),4)))


# In[276]:


#replacing numerical missing values using median since there are outliers

for feature in numerical_with_nan:
    median_value=train[feature].median()
    
    #creating new feature for nan values
    train[feature+'nan']=np.where(train[feature].isnull(),1,0)
    train[feature].fillna(median_value,inplace=True)
    
train[numerical_with_nan].isnull().sum()


# In[277]:


train.head(10)


# ## temporal variable (date-time var)

# In[278]:


#handling by changing years into numerics

for feature in(['YearBuilt','YearRemodAdd','GarageYrBlt']):
    train[feature]=train['YrSold']-train[feature]
    
train[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# In[279]:


#handling numerical features using lognormal (features whose graps are skewed)

num_features= ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']

for feature in num_features:
    train[feature]=np.log(train[feature])
    
train.head()


# ## Feature Scaling

# In[280]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in train.columns:
    if train[column_name].dtype == object:
        train[column_name] = le.fit_transform(train[column_name])
    else:
        pass


# In[281]:


feature_scale=[feature for feature in train.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
scalar.fit_transform(train[feature_scale])


# In[282]:


#transform train set and add on Id and SalePrice variables

data=pd.concat([train[['Id','SalePrice']].reset_index(drop=True),
              pd.DataFrame(scalar.fit_transform(train[feature_scale]),columns=feature_scale)],
               axis=1)


# In[283]:


data.head()             #now all features have numerical values


# In[284]:


data.to_csv('x_train.csv',index=False)


# ## Feature Selection

# In[285]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[286]:


y_train=data[['SalePrice']]


# In[287]:


x_train=data.drop(['Id','SalePrice'],axis=1)


# In[288]:


# Applying feature selection using LASSO 
#choosing alpha ,bigger alpha value=less features selected
#same seed should be used for test set

feature_sel_model= SelectFromModel(Lasso(alpha=0.001,random_state=0))
feature_sel_model.fit(x_train,y_train)


# In[289]:


feature_sel_model.get_support()


# In[290]:


#printing no. of total and selcted features

selected_feature=x_train.columns[(feature_sel_model.get_support())]

print('total features:{}'.format((x_train.shape[1])))
print('selected features:{}'.format(len(selected_feature)))

#print('features with coefficients shrank to 0:{}'.format(
    #np.sum(sel_.estimator_.coef_ == 0)))


# In[291]:


selected_feature


# ## Data Modelling

# ### Linear Regression
# 

# In[292]:


x_train=x_train[selected_feature]
x_train.head()


# In[293]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[294]:


x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.30,random_state=12)
linearmodel = LinearRegression()
linearmodel.fit(x_train,y_train)


# In[295]:


predictions = linearmodel.predict(x_test)


# In[296]:


from sklearn.metrics import r2_score
r2_score1 = r2_score(predictions,y_test)
r2_score1


# ### Random Forest

# In[328]:


from sklearn.ensemble import RandomForestRegressor

ranf = RandomForestRegressor()
ranf.fit(x_train,y_train)
predict = ranf.predict(x_test)

ranf.score(x_test,y_test)


# # Analysing test dataset

# In[297]:


test = pd.read_csv('test[1].csv')
test.head()


# In[298]:


test.shape


# In[299]:


features_with_na = [features for features in test.columns if test[features].isnull().sum()>1]

for feature in features_with_na:
    print(feature, np.round(test[feature].isnull().mean(), 4),  "% missing values")


# In[300]:


numerical_features = [feature for feature in test.columns if test[feature].dtype!= "O"]
print("Number of numerical variables:", len(numerical_features))
test[numerical_features].head()


# In[301]:


year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_feature


# In[302]:


discrete_feature = [feature for feature in numerical_features if len(test[feature].unique())<25 and feature not in year_feature+['Id']]
print('Discrete variable: {}'.format(len(discrete_feature)))
discrete_feature


# In[303]:


continous_feature= [feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print('Continous variable: {}'.format(len(continous_feature)))


# In[304]:


categorical_features= [feature for feature in test.columns if test[feature].dtypes=='O']
categorical_features


# In[305]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'. format(feature,len(test[feature].unique())))


# In[306]:


features_nan=[feature for feature in test.columns if test[feature].isnull().sum()>1 and test[feature].dtypes=='O']

for feature in features_nan:
    print("{}, {}% missing values".format(feature,np.round(test[feature].isnull().sum().mean(),4)))


# In[307]:


sns.heatmap(test.isnull(),yticklabels=False)


# In[308]:


def replace_cat_features(test,features_nan):
    data=test.copy()
    data[features_nan]=data[features_nan].fillna('missing')
    return(data)

test=replace_cat_features(test,features_nan)

test[features_nan].isnull().sum()


# In[309]:


numerical_with_nan=[feature for feature in test.columns if test[feature].isnull().sum()>1 and test[feature].dtypes!='O']

for feature in numerical_with_nan:
    print("{}: {}% missing values". format(feature,np.round(test[feature].isnull().sum().mean(),4)))


# In[310]:


for feature in numerical_with_nan:
    median_value=test[feature].median()
    
    #creating new feature for nan values
    test[feature+'nan']=np.where(test[feature].isnull(),1,0)
    test[feature].fillna(median_value,inplace=True)
    
test[numerical_with_nan].isnull().sum()


# In[311]:


sns.heatmap(test.isnull(),yticklabels=False)


# In[312]:


for feature in(['YearBuilt','YearRemodAdd','GarageYrBlt']):
    test[feature]=test['YrSold']-test[feature]
    
test[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# In[313]:


num_features= ['LotFrontage','LotArea','1stFlrSF','GrLivArea']

for feature in num_features:
    test[feature]=np.log(test[feature])
    
test.head()


# In[314]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in test.columns:
    if test[column_name].dtype == object:
        test[column_name] = le.fit_transform(test[column_name])
    else:
        pass


# In[315]:


feature_scale=[feature for feature in test.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
scalar.fit_transform(test[feature_scale])


# In[317]:


data=pd.concat([test[['Id']].reset_index(drop=True),
              pd.DataFrame(scalar.fit_transform(test[feature_scale]),columns=feature_scale)],
               axis=1)


# In[318]:


data.to_csv('X_test.csv',index=False)


# In[321]:


testdata = pd.read_csv('X_test.csv')
testdata.head()


# In[323]:


testdata.shape


# In[325]:


pred_test=linearmodel.predict(testdata)


# In[ ]:




