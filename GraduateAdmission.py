#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Powerful n-dimensional arrays. Numerical computing tools. Interoperable:

import pandas as pd
# python powerfull library used for data manipulating and data analysis:

import matplotlib.pyplot as plt
# mayplotlib is a ploting library we can use this to make awesome graphs:

import seaborn as sns
# seaborn also a ploting library we can use this to make awesome graphs:

get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline sets the backend of matplotlib to the inline:


# In[2]:


# Linear Regression Model 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
from sklearn import metrics


# In[3]:


# Load data:

data = pd.read_csv('Admission_Predict.csv')


# In[4]:


# showing first five rows of data:
data.head()


# In[5]:


# first we check our columns name using df.columns function:
data.columns


# In[6]:


data.drop('Serial No.',axis=1,inplace=True)

# axis = 1 because of its by default axis=0 . axis  = 0 means row wise , axis = 1 means column wise 
# so here we want to remove a column so we uset axis = 1 .

# inplace = True . mean we want this changing in our main dataset -


# In[7]:


data.columns


# In[8]:


data.describe()
# using this function we can simply find [count,mean,std,min,25%,50%,75%,max]


# In[9]:


data.corr()
# dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. 
# Any na values are automatically excluded. For any non-numeric data type columns in the dataframe 
# it is ignored.


# In[10]:


sns.heatmap(data.corr(),annot=True)
# now you can simple check out correlation between x and y variables:


# In[11]:


sns.distplot(data['CGPA'])


# In[12]:


sns.regplot(x='CGPA',y='Chance of Admit ',data=data,ci=None)


# In[13]:


sns.regplot(x='GRE Score',y='Chance of Admit ',data=data,ci=None)


# In[14]:


sns.regplot(x='TOEFL Score',y='Chance of Admit ',data=data,ci=None)


# In[19]:


x = data[['CGPA','GRE Score','TOEFL Score']]
y = data[['Chance of Admit ']]


# In[16]:


# Split data for test and train the model.
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=.20)


# In[17]:


# x shape
x.shape


# In[18]:


# x_train shape 80% of data from x - 20% other data for test
x_train.shape


# In[20]:


# x_train first five rows:
x_train.head()


# In[21]:


#object
linreg = LinearRegression()

# fiting our data for training
linreg.fit(x_train,y_train)


# In[22]:


# our model is ready to predict y.
y_predict = linreg.predict(x_test)

# our model prediction
y_predict[:10]


# In[23]:


# y test
y_test[:10]


# In[24]:


# from sklearn import metrics
# already imported

metrics.mean_absolute_error(y_test,y_predict)


# In[ ]:




