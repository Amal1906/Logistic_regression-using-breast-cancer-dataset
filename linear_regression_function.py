#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


#now we are going to generate random samples using the sklearn library
X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4) #n_samples creates 100 random sample
print(X)                                                                         #n_features means 1 columns of data that is simple linear regression
print(y)                                                                         #noise makes the data less perfect otherwise it will be a linear one 


# In[3]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
print(X_train,X_test,y_train,y_test)


# In[4]:


fig=plt.figure(figsize=(8,6))
plt.scatter(X[:, 0],y,color="b",marker='o',s=30)
plt.show()
"""Okay so far we can understand that the data is graphically representes like this and we need to fit a line 
that fits the data."""


# In[5]:


print(X_train.shape)
print(y_test.shape)


# In[8]:


from linear_regression import Linear_regression

regressor = Linear_regression(lr=0.01)
regressor.fit(X_train,y_train)
predicted= regressor.predict(X_test)

def mse(y_true,y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value=mse(y_test,predicted)
print(mse_value)


# In[11]:


# Predict on the entire dataset for plotting the regression line
y_pred_line = regressor.predict(X)

# Plotting
fig = plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="b", marker='o', s=30, label="Actual data")  # Scatter plot of actual data
plt.plot(X, y_pred_line, color="r", linewidth=2, label="Regression line")  # Regression line
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()


# In[ ]:




