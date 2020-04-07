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

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINE_SIZE']])
train_y = np.asanyarray(train[['CO2_EMISSIONS']])

test_x = np.asanyarray(test[['ENGINE_SIZE']])
test_y = np.asanyarray(test[['CO2_EMISSIONS']])
poly = PolynomialFeatures(degree=2)
#Now train_x_poly has three columns, first with 1, second with x and third with x ^ 2
train_x_poly = poly.fit_transform(train_x)
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

plt.scatter(train.ENGINE_SIZE, train.CO2_EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )