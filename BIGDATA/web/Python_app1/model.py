#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 


# In[45]:


df1 = pd.read_csv('Member_data_Allsum.csv')


# In[46]:


df1['고객분류'].value_counts()


# In[47]:


df1 = df1.drop('고객분류', axis = 1)


# In[48]:


df1['Y'] = 0


# In[57]:


df1.loc[df1['총구매횟수'] == 1, 'Y'] = 1


# In[58]:


df1


# In[59]:


Y = df1['Y']
X = df1[['성별', '거주지역', '연령', '결제등록카드', '유입경로', '자녀여부', '구매일로부터기간', '총결제금액']]


# In[16]:


from sklearn.model_selection import train_test_split

from sklearn.compose  import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[60]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=1234)


# In[61]:


numeric_pipe  = make_pipeline(SimpleImputer(strategy='mean'))
category_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())


# In[64]:


preprocessing_pipe = make_column_transformer( (numeric_pipe,  ['연령', '구매일로부터기간', '총결제금액']) ,
                                              (category_pipe, ['성별', '거주지역', '결제등록카드', '유입경로', '자녀여부']))


# In[65]:


model_pipe = make_pipeline(preprocessing_pipe, RandomForestClassifier())


# In[66]:


hyper_list = {'randomforestclassifier__max_depth':range(5,10),
             'randomforestclassifier__min_samples_split':range(10,100, 10),
              'randomforestclassifier__n_estimators' : range(10, 100, 10)
             }

grid_model = GridSearchCV(model_pipe, param_grid=hyper_list , cv=5, n_jobs=-1)
grid_model.fit(X_train, Y_train)


# In[67]:


best_model = grid_model.best_estimator_


# In[68]:


y_pred = best_model.predict(X_test)


# In[69]:


from sklearn.metrics import confusion_matrix, classification_report


# In[70]:


print(classification_report(Y_test,y_pred,digits=3))


# In[71]:


best_model.score(X_train,Y_train)


# In[72]:


best_model.score(X_test,Y_test)


# In[73]:


import pickle


# In[74]:


pickle.dump(best_model, open('model.pkl', 'wb'))


# In[75]:


model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:




