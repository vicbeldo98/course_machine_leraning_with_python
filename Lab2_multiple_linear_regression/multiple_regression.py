import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Reading the dataset
df = pd.read_csv("FuelConsumption.csv", encoding='utf8')

#Showing the dataset firsts rows
df.head()

#Shows statistics on the dataset columns
df.describe()

#Selects some features to explore more
cdf = df[['MODELYEAR','MAKE','MODEL','CLASS','ENGINE_SIZE','CYLINDERS','TRANSMISSION','FUEL_CONSUMPTION_CITY','FUEL_CONSUMPTION_HWY','FUEL_CONSUMPTION_COMB','CO2_EMISSIONS']]
cdf.head(1000)

#Plot these features
viz = cdf[['MAKE','MODEL','CLASS','ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION_CITY','FUEL_CONSUMPTION_HWY','FUEL_CONSUMPTION_COMB','CO2_EMISSIONS']]
#viz.hist()
#plt.show()

#Plot linearly
'''plt.scatter(cdf.FUEL_CONSUMPTION_COMB, cdf.CO2_EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()'''

plt.scatter(cdf.ENGINE_SIZE, cdf.CO2_EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

'''plt.scatter(cdf.CYLINDERS, cdf.CO2_EMISSIONS, color='yellow')
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()'''

#Create a partition of the dataset and plot an estimatation
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#Plot train dataset to assure linear relationship
plt.scatter(train.ENGINE_SIZE, train.CO2_EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Creating a model
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2_EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

#Prediction
y_hat= regr.predict(test[['ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2_EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))
