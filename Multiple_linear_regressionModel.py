import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"C:\Users\suraj\Downloads\Investment.csv")
 
x= dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


y_pred=regressor.predict(x_test)

regressor.coef_

print(regressor.intercept_)

x=np.append(arr=np.full((50,1),42467),values=x,axis=1)

# till now we build ml linear regression model
# what is next? I hava



import statsmodels.api as sm
x_otp=x[:,[0,1,2,3,4,5]]

regressor_OLS=sm.OLS(endog=y,exog=x_otp).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_otp=x[:,[0,1,2,3,5]]

regressor_OLS=sm.OLS(endog=y,exog=x_otp).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_otp=x[:,[0,1,2,3]]

regressor_OLS=sm.OLS(endog=y,exog=x_otp).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_otp=x[:,[0,1,2]]

regressor_OLS=sm.OLS(endog=y,exog=x_otp).fit()
regressor_OLS.summary()



