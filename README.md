# KAATRU-Data-Scientist-Assesment
Predicted the cont of the shared bikes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dat=pd.read_csv("D:/Datasets/day.csv")
x=dat.iloc[:,[7,11,14]].values
y=dat.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.predict(x_test)

y_pred=regressor.predict(x_test)
regressor.score(x_train,y_train)
dt=[[1,90.754,654]]
d=pd.DataFrame(dt)
regressor.predict(d)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


