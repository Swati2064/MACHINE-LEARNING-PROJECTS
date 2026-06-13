import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\suraj\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Linear Regression --- linear algorithm(degree-1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[5]])
print(lin_model_pred)

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

# Create polynomial model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Plot
plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Prediction using polynomial regression
poly_model_pred = lin_reg_2.predict( poly_reg.fit_transform([[5]]))
print(poly_model_pred)