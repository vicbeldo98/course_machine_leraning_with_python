import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

# ### About dataset

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/cbb.csv')
df['windex'] = np.where(df.WAB > 7, 'True', 'False')
df1 = df[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
Feature = df1[['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D']]
Feature = pd.concat([Feature,pd.get_dummies(df1['POSTSEASON'])], axis=1)
Feature.drop(['S16'], axis = 1,inplace=True)
X = Feature
y = df1['POSTSEASON'].values
X= preprocessing.StandardScaler().fit(X).transform(X)

# ## Training and Validation 
# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the validation set  to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 

# # K Nearest Neighbor(KNN)
# 
# <b>Question  1 </b> Build a KNN model using a value of k equals three, find the accuracy on the validation data (X_val and y_val)

# You can use <code> accuracy_score</cdoe>

# In[17]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
k = 3
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_val)
from sklearn import metrics
print("Validation Data Accuracy: ", metrics.accuracy_score(y_val, yhat))


# <b>Question  2</b> Determine the accuracy for the first 15 values of k the on the validation data:

# In[19]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 15
for k in range(1,Ks+1):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat = neigh.predict(X_val)
    print( "With k = ",k, ",Validation Data Accuracy is ", metrics.accuracy_score(y_val, yhat)) 


# # Decision Tree

# The following lines of code fit a <code>DecisionTreeClassifier</code>:

# In[42]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# <b>Question  3</b> Determine the minumum   value for the parameter <code>max_depth</code> that improves results 

# In[44]:


max_depth = 18
for depth in range(1,max_depth+1):
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = depth)
    drugTree.fit(X_train,y_train)
    predTree = drugTree.predict(X_val)
    print( "With k = ",depth, ",Validation Data Accuracy is ", metrics.accuracy_score(y_val, predTree)) 
    


# # Support Vector Machine

# <b>Question  4</b>Train the following linear  support  vector machine model and determine the accuracy on the validation data 

# In[20]:


from sklearn import svm


# In[21]:


clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_val)
from sklearn import metrics
print("Validation Data Accuracy: ", metrics.accuracy_score(y_val, yhat))


# # Logistic Regression

# <b>Question 5</b> Train a logistic regression model and determine the accuracy of the validation data (set C=0.01)

# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_val)
from sklearn import metrics
print("Validation Data Accuracy: ", metrics.accuracy_score(y_val, yhat))


# # Model Evaluation using Test set

# In[24]:


from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[25]:


def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1


# <b>Question  5</b> Calculate the  F1 score and Jaccard Similarity score for each model from above. Use the Hyperparameter that performed best on the validation data.

# ### Load Test set for evaluation 

# In[26]:


test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv',error_bad_lines=False)
test_df.head()


# In[27]:


test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_df1. head()
test_df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)
test_Feature = test_df1[['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df1['POSTSEASON'])], axis=1)
test_Feature.drop(['S16'], axis = 1,inplace=True)
test_Feature.head()
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]


# In[28]:


test_y = test_df1['POSTSEASON'].values
test_y[0:5]


# KNN

# In[50]:


from sklearn.neighbors import KNeighborsClassifier
k =6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(test_X)
print("Jaccard Coefficient: ", jaccard_index(test_y,yhat))
print("F1-Score Coefficient:" ,f1_score(test_y, yhat, average='weighted'))


# Decision Tree

# In[45]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(test_X)
print("Jaccard Coefficient: ", jaccard_index(test_y, predTree))
print("F1-Score Coefficient:" ,f1_score(test_y, predTree, average='weighted'))


# SVM

# In[34]:


clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
yhat = clf.predict(test_X)
print("Jaccard Coefficient: ", jaccard_index(yhat,test_y))
print("F1-Score Coefficient:" ,f1_score(test_y, yhat, average='weighted'))


# Logistic Regression

# In[35]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(test_X)
yhat_prob = LR.predict_proba(test_X)
print("Jaccard Coefficient: ", jaccard_index(yhat,test_y))
print("F1-Score Coefficient: " ,f1_score(test_y, yhat, average='weighted'))
from sklearn.metrics import log_loss
print("Log-loss Coefficient: ",log_loss(test_y, yhat_prob) )


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                |.93      |.93       | NA      |
# | Decision Tree      | 1       | 1        | NA      |
# | SVM                | 1       | 1        | NA      |
# | LogisticRegression | 1       | 1        | .97     |