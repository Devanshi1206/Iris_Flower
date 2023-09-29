#!/usr/bin/env python
# coding: utf-8

# # Iris Flower

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ## Importing Dataset

# In[2]:


df=pd.read_csv("Iris Flower Data.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# There are no null values 

# In[6]:


df.describe()


# In[7]:


df['species'].value_counts()


# ## Data Visualisation

# In[8]:


sns.countplot(x='species',data=df)
plt.show


# In[9]:


sns.pairplot(df,hue='species')
plt.show()


# In[10]:


df['species'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}, inplace = True)


# In[11]:


plt.figure(figsize=[6,5])
sns.heatmap(df.corr(),annot = True, cmap="coolwarm")


# ## ML Modeling

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df['species'], test_size = 0.3, random_state = 30)


# In[14]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[15]:


y_pred=model.predict(x_test)
y_pred


# In[16]:


matrix=confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix)

cm_display.plot()
plt.show()


# In[17]:


report=classification_report(y_test,y_pred)
print(report)


# In[18]:


y_pred_train=model.predict(x_train)
acc_score=accuracy_score(y_train,y_pred_train)
print('Accuracy Score is : ',acc_score)


# This model predict the Iris Flower with an accuracy of 99%
