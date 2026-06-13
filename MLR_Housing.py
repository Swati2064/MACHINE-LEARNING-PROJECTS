import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# importing dataset using pandas
dataset=pd.read_csv(r"C:\Users\suraj\Downloads\House_data.csv")
dataset.head()

# checking if any value is missing
print(dataset.isnull().any())

# checking for categorical data
print(dataset.dtypes)

# deopping the id and data column
dataset = dataset.drop(['id','date'], axis=1, errors='ignore')

#understanding the distribution with seaborn
with sns.plotting_context("notebook",font_scale=2.5):
    g=sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
                   hue='bedrooms',palette='tab20',size=6)
g.set(xticklabels=[]);

# separating independent and dependent variable
x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values
#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# predicting the Test set results
y_pred=regressor.predict(x_test)


# please use the method step for backward elimination and do not use the fun. part for backward elimition.

import statsmodels.api as sm
x_otp=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
# ordinary Least squares
regressor_OLS=sm.OLS(endog=y,exog=x_otp).fit()
regressor_OLS.summary()

# Backward Elimination (do not do the function part for backward elimination)
import statsmodels.api as sm
def backwardElimination(x,SL):
    numVars=len(x[0])
    temp=np.zeros((21613,19)).astype(int)
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(y,x).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
x_Modeled = backwardElimination(x_opt, SL)