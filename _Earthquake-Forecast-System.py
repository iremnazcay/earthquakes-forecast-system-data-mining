#!/usr/bin/env python
# coding: utf-8

# # Earth Quake Forecast System

# In[70]:


#   Add libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset

# 1911 - 2017 Earthquake Data

# In[71]:


#   Read csv file(data) 
data = pd.read_csv('/home/iremnazcay/data/titanic/randomDataset.csv')


# In[72]:


data.head()


# In[73]:


data['Latitude']=data['Enlem']
data['Longitude']=data['Boylam']
data['Depth']=data['Der']


# In[74]:


data= data.drop('Enlem', axis=1)
data= data.drop('Boylam', axis=1)
data= data.drop('Der', axis=1)


# In[75]:


data['Magnitude']=data['ML']
data['Degree'] = data['Magnitude']


# In[76]:


data= data.drop('ML', axis=1)


# In[77]:


data.head()


# In[78]:


data['Tarih']=pd.to_datetime(data['Tarih'])


# In[79]:


data.head()


# In[80]:


data['Magnitude'].min()


# In[81]:


# Delete rows which 'Magnitude < 0.49'
data.drop( data[ data['Magnitude']  < 0.49 ].index , inplace=True)


# In[82]:


data['Magnitude'].min()


# In[83]:


data['Magnitude'].max()


# In[84]:


# Richter Scale and Class
r = [0.49, 3.5, 4.3, 5.3, 6.5, 7.5]
g = ['Small','Moderate','Medium','Great','Super']

data['Degree'] = pd.cut(data['Degree'], bins=r, labels=g)


# In[85]:


data['Degree'].value_counts()


# In[86]:


data.info()


# In[87]:


data['Magnitude'].groupby(data['Degree']).mean()


# In[88]:


# Reset the index range and add new range
data.reset_index(inplace=True)
data.index = pd.RangeIndex(start=0, stop=11398 , step=1)


# In[89]:


data.info()


# ## Create Training and Test Sets

# In[90]:


# Seperate dataset as training and test %30 - % 70
train= data.iloc[:7979, :]
test= data.iloc[7979:11398 ,:]
test_test=test.copy()


# In[91]:


train.head()


# In[92]:


test.head()


# In[93]:


test['Magnitude'].groupby(test['Degree']).size()


# In[94]:


train['Magnitude'].groupby(train['Degree']).size()


# In[95]:


train.shape, test.shape


# ## Clear unnecessary tags from our dataset

# In[96]:


train.drop('index', axis=1, inplace=True)
test.drop('index', axis=1, inplace=True)
test_test.drop('index', axis=1, inplace=True)

train.drop('No', axis=1, inplace=True)
test.drop('No', axis=1, inplace=True)
test_test.drop('No', axis=1, inplace=True)

train.drop('Tarih', axis=1, inplace=True)
test.drop('Tarih', axis=1, inplace=True)
test_test.drop('Tarih', axis=1, inplace=True)

train.drop('Zaman', axis=1, inplace=True)
test.drop('Zaman', axis=1, inplace=True)
test_test.drop('Zaman', axis=1, inplace=True)

train.drop('Yer', axis=1, inplace=True)
test.drop('Yer', axis=1, inplace=True)
test_test.drop('Yer', axis=1, inplace=True)


# In[97]:


#   Clear Derece column in test set
test.drop('Degree', axis=1, inplace=True)


# In[98]:


train.info()


# In[99]:


test.info()


# In[100]:


test_test.info()


# In[101]:


# Reset the index range and add new range
test.reset_index(inplace=True)
test.index = pd.RangeIndex(start=0, stop=3419  , step=1)


# In[102]:


# Reset the index range and add new range
test_test.reset_index(inplace=True)
test_test.index = pd.RangeIndex(start=0, stop=3419  , step=1)


# In[103]:


test.drop('index', axis=1, inplace=True)
test_test.drop('index', axis=1, inplace=True)


# In[104]:


test.info()


# In[105]:


test_test.info()


# ## VISUALIZATION

# In[106]:


sns.countplot(train['Magnitude'])


# In[107]:


sns.countplot(test['Magnitude'])


# In[108]:


sns.set_style("whitegrid") 
  
sns.boxplot(x = 'Degree', y = 'Magnitude', data = train) 


# In[109]:


sns.set_style("whitegrid") 
  
sns.boxplot(x = 'Degree', y = 'Magnitude', data = test_test)


# In[110]:


train['Magnitude'].groupby(train['Degree']).size()


# In[111]:


test_test['Magnitude'].groupby(test_test['Degree']).size()


# In[112]:


def bar_chart(feature):
    Small = train[train['Degree'] == 'Small'][feature].value_counts()
    Moderate = train[train['Degree'] == 'Moderate'][feature].value_counts()
    Medium = train[train['Degree'] == 'Medium'][feature].value_counts()
    Great = train[train['Degree'] == 'Great'][feature].value_counts()
    Super = train[train['Degree'] == 'Super'][feature].value_counts()
    
    df = pd.DataFrame([Small,Moderate,Medium, Great, Super])
    df.index = ['Small','Moderate','Medium', 'Great', 'Super']
    df.plot(kind='bar', stacked=True, figsize=(10,10))


# In[113]:


bar_chart('Degree')


# In[114]:


def bar_chart(feature):
    Small = test_test[test_test['Degree'] == 'Small'][feature].value_counts()
    Moderate = test_test[test_test['Degree'] == 'Moderate'][feature].value_counts()
    Medium = test_test[test_test['Degree'] == 'Medium'][feature].value_counts()
    Great = test_test[test_test['Degree'] == 'Great'][feature].value_counts()
    Super = test_test[test_test['Degree'] == 'Super'][feature].value_counts()
    
    df = pd.DataFrame([Small,Moderate,Medium, Great, Super])
    df.index = ['Small','Moderate','Medium', 'Great', 'Super']
    df.plot(kind='bar', stacked=True, figsize=(10,10))


# In[115]:


bar_chart('Degree')


# In[116]:


facet = sns.FacetGrid(train,hue="Degree",aspect=3)
facet.map(sns.kdeplot,'Magnitude',shade=True)
facet.add_legend()

plt.xlim(0 ,8)


# In[117]:


facet = sns.FacetGrid(test_test,hue="Degree",aspect=3)
facet.map(sns.kdeplot,'Magnitude',shade=True)
facet.add_legend()

plt.xlim(0 ,8)


# ## MODELLING

# In[118]:


# Creating feature matrix and target vector
train_data = train.drop('Degree', axis=1)
target = train['Degree'] #target has training data's degrees
train_data.shape, target.shape


# In[119]:


train_data.head()


# In[120]:


target.head()


# In[121]:


train.drop('Magnitude', axis=1, inplace=True) # latitude, longitude, depth, degree
test.drop('Magnitude', axis=1, inplace=True) # has only latitude, longitude and depth
train_data.drop('Magnitude', axis=1, inplace=True) # has only latitude, longitude and depth

#test_test.drop('Magnitude', axis=1, inplace=True) #daha sonra yazdırıcağım için kalıcak


# In[122]:


train.head()


# In[123]:


test.head()


# In[124]:


train_data.head()


# In[125]:


test_test.head()


# ## RANDOM FOREST

# In[134]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=360,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train_data, target)
print("Score: %.4f" % rf.oob_score_)


# In[139]:


# Prediction
predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Predicted Degree'])
# Derece - Degree
predictions = pd.concat((test_test.iloc[:, 4], predictions), axis = 1)
predictions = pd.concat((test_test.iloc[:, 3], predictions), axis = 1)

predictions.to_csv('result.csv', sep="-", index = False)


# In[140]:


predictionResult = pd.read_csv('/home/iremnazcay/fastai/courses/ml1/result.csv')


# In[141]:


predictionResult.head()


# ## Performance Metrics for Classification

# ##### Confusion Matrix

# In[142]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_test.iloc[:, 4], rf.predict(test))


#  Derece
# Small       1118
# Moderate    1370
# Medium       839
# Great         87
# Super          5

# Great, Medium, Modarate, Small, Super


# In[143]:


y_actu = pd.Series(test_test.iloc[:, 4], name='Actual')
y_pred = pd.Series( rf.predict(test), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)


# In[144]:


df_confusion = pd.crosstab(y_actu,y_pred ,rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion


# In[145]:


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',  cmap=None, normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.show()


# In[147]:


plot_confusion_matrix(cm           =  np.array([[   665,    378,   73,  1,    1],
                                                 [   232,   942,  192,  4,    0],
                                                 [   112,   239, 478, 9,    1],
                                                 [   14,    20,  48,  4,    1],
                                                [   0,    1,    4,   0,    0]]),
                      
                      normalize    = False,
                      target_names = ['Small', 'Moderate','Medium','Great','Super'],
                      title        = "Confusion Matrix")

# Great, Medium, Modarate, Small, Super


# In[148]:


plot_confusion_matrix(cm            =  np.array([[   665,    378,   73,  1,    1],
                                                 [   232,   942,  192,  4,    0],
                                                 [   112,   239, 478, 9,    1],
                                                 [   14,    20,  48,  4,    1],
                                                [   0,    1,    4,   0,    0]]),
                      normalize    = True,
                      target_names = ['Small', 'Moderate','Medium','Great','Super'],
                      title        = "Confusion Matrix, Normalized")


# ##### Accuracy

# In[150]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_test.iloc[:, 4], rf.predict(test))
print('Accuracy: %f' % accuracy)


# ##### Recall

# In[151]:


from sklearn.metrics import recall_score

# r: tp / (tp + fn)
recall=recall_score(test_test.iloc[:, 4], rf.predict(test), pos_label='positive', average='micro')
print('Recall: %f' % recall)


# ##### Precision

# In[152]:


from sklearn.metrics import precision_score

# p: tp / (tp + fp)
precision=precision_score(test_test.iloc[:, 4], rf.predict(test),pos_label='positive', average='micro')
print('Precision: %f' % precision)


# ##### F1 score

# In[153]:


from sklearn.metrics import f1_score

# f1: 2 tp / (2 tp + fp + fn)
# f1: 2*(p*r) / p+r
f1 = f1_score(test_test.iloc[:, 4], rf.predict(test),pos_label='positive', average='micro')
print('F1 score: %f' % f1)


# In[ ]:




