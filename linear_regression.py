#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Linear_regression:
    
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weights=None
        self.bias=None
        
    def fit(self,X,y):
        #consider it as a matrix and we are multipling the inputs with the weights which we initialize with 0
        #the bias we assign as 0
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iter):
            #we know the y_cap=wx+b 
            #using this predicted values we calculate the cost function inorder to update the weights
            y_predicted=np.dot(X,self.weights) + self.bias
            
            #now we need to find the gradient okay!
            #If we find the partial derivative of the cost function which is 1/N(sigma(y-y_predicted))**2 with respect to w
            #also we need to find the patrial derivative of the cost function with respect to bias 'b' to update the weights
            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)
            
            self.weights= self.weights - (self.lr*dw)
            self.bias=self.bias - (self.lr*db)
            
            
    
    def predict(self,X):
        y_predicted  = np.dot(X,self.weights) + self.bias
        return y_predicted
    
    


# In[ ]:




