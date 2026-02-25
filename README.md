# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets.
6. Create an SGD Regressor model.
7. Fit the model on the training data.
8. Evaluate the model performance.
9. Make predictions and visualize the results.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: K RAGAPRIYAN
RegisterNumber: 212225040323

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data= pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

data = data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first = True)

x=data.drop('price',axis=1)
y=data['price']

scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

sgd_model = SGDRegressor(max_iter=1000 , tol=1e-3)

sgd_model.fit(x_train,y_train)

y_pred = sgd_model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print('Name : K RAGAPRIYAN')
print('Reg No : 212225040323')
print(f"MSE : {mse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R_2 : {r2:.4f}")

print("\nModel Coefficients :")
print("Coefficients : ",sgd_model.coef_)
print("Intercept : ",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Prdicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
*/
```

## Output:

![alt text](<Screenshot 2026-02-25 085855.png>)
![alt text](<Screenshot 2026-02-25 085811.png>)
![alt text](<Screenshot 2026-02-25 085621.png>)
![alt text](<Screenshot 2026-02-25 085639.png>)
![alt text](<Screenshot 2026-02-25 085652.png>)

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
