#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('car data.csv')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


print(df['Seller_Type'].unique())


# In[7]:


print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[8]:


df.isnull().sum()


# In[9]:


df.columns


# In[10]:


df.drop('Car_Name',axis =1,inplace = True)


# In[11]:


df


# In[12]:


df['current_year']=2022


# In[13]:


df.head()


# In[14]:


df['no_year'] = df['current_year']-df['Year']


# In[15]:


df.head()


# In[16]:


df.drop(['Year','current_year'],axis=1,inplace=True)


# In[17]:


df


# In[18]:


df.dtypes


# In[19]:


final_dataset= pd.get_dummies(df,drop_first=True)


# In[20]:


final_dataset


# In[21]:


final_dataset.corr()


# In[22]:



import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[23]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[24]:


### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[25]:


print(model.feature_importances_)


# In[31]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[28]:


from sklearn.ensemble import RandomForestRegressor


# In[29]:


regressor=RandomForestRegressor()


# In[32]:


# hyper tuning


# In[33]:


from sklearn.model_selection import RandomizedSearchCV


# In[34]:


import numpy as np


# In[43]:


# randomzide cv lists

# Randamized CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[44]:


## Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[45]:


rf = RandomForestRegressor()


# In[48]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=5,cv=2, verbose=2,random_state=10,n_jobs=-1)


# In[49]:


rf_random.fit(X_train,y_train)


# In[50]:


rf_random.best_params_


# In[51]:


predictions = rf_random.predict(X_test)


# In[52]:


predictions


# In[53]:


from sklearn import metrics


# In[54]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[55]:


# work on rf without hyper ...xgboost , liner,svm --tuniing ---houseprice data


# In[56]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




