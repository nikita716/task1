#!/usr/bin/env python
# coding: utf-8

# ## NAME: NIKITA KATHURIA
# ## DATA SCIENCE AND BUSINESS ANALYTICS , GRIP MAY 2021
# ## THE SPARKS FOUNDATION
# 
# 
# 
# ## TASK 1: PREDICTION USING SUPERVISED ML
# 

# ### **Importing Libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Importing Dataset**

# In[3]:


dataframe=pd.read_csv("http://bit.ly/w-data")


# In[5]:


dataframe.head(10)


# In[9]:


plt.scatter(dataframe['Hours'],dataframe['Scores'],color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# In[11]:


dataframe.shape


# There are 25 rows and 2 columns in dataset.

# In[14]:


#checking for missing values(if present)
dataframe.isnull().sum().any()


# In[15]:


x=dataframe.iloc[:,:-1].values
y=dataframe.iloc[:, 1].values


# ### **Splitting the dataset into train and test sets**

# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# ### **Training**

# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[20]:


y_pred=lr.predict(x_test)


# In[22]:


from sklearn.metrics import r2_score


# In[23]:


r2_score(y_test,y_pred)


# ### **Visualisation**

# In[24]:


plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,lr.predict(x_train),color='red')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# ### **Predicting Score**

# In[30]:


prediction=lr.predict([[9.25]])
print('Predicted Score:',prediction)


# The student will score around 91.83% if he/she will study for 9.25 hours per day.

# ### **Evaluating the model**

# In[31]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




