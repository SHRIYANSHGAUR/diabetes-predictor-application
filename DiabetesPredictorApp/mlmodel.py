
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import pickle

df = pd.read_csv("data.csv")


cdf = df[['Age','Glucose','BloodPressure','BMI', 'Outcome']]


x = cdf.iloc[:, :4]
y = cdf.iloc[:, -1]


xtrain , xtest, ytrain, ytest = train_test_split(x, y, test_size=0.02)

#POLYNOMIAL  linear REGRESSION MODEL
poly_reg=PolynomialFeatures(degree=2)

x_poly= poly_reg.fit_transform(xtrain)

regressor=LinearRegression()
regressor.fit(x_poly,ytrain)



#LINEAR MODEL
#regressor = LinearRegression()
#regressor.fit(x, y)
#regressor.fit(xtrain, ytrain)

pickle.dump(regressor, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))



ypred=regressor.predict(poly_reg.fit_transform(xtest))


#ypred=np.array(ypred)
#ytest=np.array(ytest)
#np.set_printoptions(precision=2)

#print((ytest-ypred)+1)

#print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))

'''
#CHECK
ypred= regressor.predict(xtest)
ypred=np.array(ypred)
ytest=np.array(ytest)
np.set_printoptions(precision=2)

print(((ypred-ytest)/ypred) * 100)

print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))
'''