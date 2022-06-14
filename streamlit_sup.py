#!/usr/bin/env python
# coding: utf-8

# In[2]:


##installing necessary modules
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import streamlit.components.v1 as components
plt.style.use('dark_background')


# In[4]:


st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title='superb_dashboard', page_icon="random")

st.sidebar.header('superb')

data = st.sidebar.file_uploader("data", type=['csv'])

# In[6]:
data = pd.read_csv("https://raw.githubusercontent.com/regan-mu/ADS-April-2022/05af51d7b1f9768fbf667e61de568c54579bb3f4/Assignments/Assignment%201/data.csv")
# In[8]:


data.info()


# In[141]:


print(data)


# In[9]:


data.head()


# In[10]:


data["price"]


# In[11]:


data.shape


# In[97]:


data['date']


# In[12]:


data.isnull().sum().sum()


# In[98]:


data.head()


# In[13]:


# custom number of splits
day_date = data['date'].str.split('/', n=1, expand=True)
day_date


# In[17]:


data["company"]


# In[18]:


data['ticker']


# In[30]:


len('ticker')


# In[38]:


top_price = data.price
print(top_price)


# In[39]:


top_company = data.company
print(top_company)


# In[40]:


top_ticker = data.ticker
print(top_ticker)


# In[21]:


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


# In[43]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[44]:


data['price'] = le.fit_transform(data['price'])
data['company'] = le.fit_transform(data['company'])
data['ticker'] = le.fit_transform(data['ticker'])
data['date'] = le.fit_transform(data['date'])

data.head()


# In[126]:


data.describe()


# In[45]:


x = data.drop(['price'], axis = 1)
y = data['price']


# In[62]:


from sklearn.model_selection import train_test_split

##x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)

# Fit the model on training set
model = LogisticRegression()
model.fit(x_train, y_train)
# save the model to disk


# In[63]:


from sklearn import model_selection
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)


# In[64]:


diff = y_test - predictions


# In[65]:


print(diff)


# In[66]:


sns.displot(diff)


# In[72]:


filename = data
print(filename)


# In[74]:


pickle.dump(model, open("filename", 'wb'))


# In[75]:


loaded_model = pickle.load(open("filename", 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

