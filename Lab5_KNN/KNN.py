import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

#Load and show dataset
df = pd.read_csv('teleCust1000t.csv')
print(df.head())

#Counts number of clases
n_clases = df['custcat'].value_counts()
print('Number of clases: ')
print(n_clases)

#Visualize data
#df.hist(column='income', bins=50)
#plt.show()

#Look the features
#print(df.columns)

#Convert into a numpy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
#print(X[0:5])

#Take labels of clasification
y = df['custcat'].values

#Standarize data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
#print(X[0:5])

#Split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
#print ('Train set:', X_train.shape,  y_train.shape)
#print ('Test set:', X_test.shape,  y_test.shape)

#Import k-nearest-neighbour vote
from sklearn.neighbors import KNeighborsClassifier

#Train Model
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#Predict with the model
yhat = neigh.predict(X_test)

#Computing accuracy
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#Try with different K and store accuracy to see which is the most accurate
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1,Ks):
    
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
