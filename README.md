# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values and import linear regression from sklearn.
3.Assign the points for representing in the graph.
4.Predict the regression for marks by using the representation of the graph and compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by:SATHYANARAYANAN M

RegisterNumber:  212224040300

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
# display the content in file
print(df.head())
print(df.tail())
# segeragating the values in data
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
# splitting train and test data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
# graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# Mean Absolute Error
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:





## print(df.head())


<img width="177" height="143" alt="image" src="https://github.com/user-attachments/assets/62a34262-7719-47ed-957f-a7a83005f35a" />

## print(df.tail())


<img width="196" height="138" alt="image" src="https://github.com/user-attachments/assets/8df43b7f-4afa-4217-80b5-54298d5c8564" />

## segeragating the values in data

<img width="790" height="617" alt="image" src="https://github.com/user-attachments/assets/6bcba71b-e05b-4af3-b4a6-9e7010585c91" />

## splitting train and test data
<img width="743" height="101" alt="image" src="https://github.com/user-attachments/assets/87b7dc4f-3162-4f83-a3cd-8fb5ab0a0503" />

## graph plot for training data
<img width="789" height="610" alt="image" src="https://github.com/user-attachments/assets/c9ac07e6-9c3f-4715-af12-3aff6ea66beb" />

## Graph plot for test data
<img width="884" height="607" alt="image" src="https://github.com/user-attachments/assets/726cf2e6-0d34-428e-b9d6-d7dccfa0564c" />

## Mean Absolute Error

<img width="292" height="83" alt="image" src="https://github.com/user-attachments/assets/a4c5bb19-00d4-467b-a95f-afb1fce0ecd4" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
