#!/usr/bin/env python
# coding: utf-8

# In[75]:


##installing necessary modules
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import streamlit.components.v1 as components
plt.style.use('dark_background')


# In[2]:


st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title='superb_dashboard', page_icon="random")


# In[94]:


data = pd.read_csv("https://raw.githubusercontent.com/regan-mu/ADS-April-2022/05af51d7b1f9768fbf667e61de568c54579bb3f4/Assignments/Assignment%201/data.csv")


# In[95]:


data.info()


# In[141]:


print(data)


# In[96]:


data.head()


# In[79]:


data["price"]


# In[80]:


data.shape


# In[97]:


data['date']


# In[83]:


data.isnull().sum().sum()


# In[98]:


data.head()


# In[116]:


# custom number of splits
day_date = data['date'].str.split('/', n=1, expand=True)
day_date


# In[32]:


data["company"]


# In[104]:


top_4_price = data.price
top_4_price.head()


# In[23]:


top_4_company = Data.company
top_4_company.head()


# In[105]:


data['price'].value_counts()


# In[108]:


data['company'].value_counts()


# In[109]:


data['date'].value_counts()


# In[110]:


data['ticker'].value_counts()


# In[111]:


data['company'].unique()


# In[112]:


data['price'].unique()


# In[113]:


data['ticker'].unique()


# In[123]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[125]:


data['price'] = le.fit_transform(data['price'])
data['company'] = le.fit_transform(data['company'])
data['ticker'] = le.fit_transform(data['ticker'])
data['date'] = le.fit_transform(data['date'])

data.head()


# In[126]:


data.describe()


# In[135]:


x = data.drop(['price'], axis = 1)
y = data['price']


# In[136]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[156]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)


# In[157]:


diff = y_test - predictions


# In[158]:


print(diff)


# In[159]:


sns.displot(diff)


# In[163]:


import pickle
pickle.dump(lr. open('./model.sav'. 'wb'))


# In[ ]:





# In[ ]:




