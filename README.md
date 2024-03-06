# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
### Step-1:
To train, initialize theta and iteratively update it using gradient descent.
### Step-2:
For preprocessing, read and scale data.
### Step-3:
for modeling, train the linear regression model.
### Step-4:
To predict data values, scale a new data and predict.
### Step-5:
Print the prediction.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SHYAM S
RegisterNumber: 212223240156
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
  X=np.c_[np.ones(len(X1)), X1]
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions= (X).dot(theta).reshape(-1,1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta
data=pd.read_csv('50_Startups.csv',header=None)
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")

```

## Output:
![alt text](image.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
