import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("advertising.csv")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x=dataset[['TV']]
y=dataset['Sales']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
slr=LinearRegression()
slr.fit(x_train,y_train)
print('Intercept: ',slr.intercept_)
print('Coefficient: ',slr.coef_)
plt.scatter(x_train,y_train)
plt.plot(x_train,6.948+0.054*x_train,'r')
plt.show()
y_pred_slr=slr.predict(x_test)
x_pred_slr=slr.predict(x_train)
print("Prediction for test set: {}".format(y_pred_slr))



slr_diff= pd.DataFrame({'Actual value': y_test,'Predicted value':y_pred_slr})
slr_diff



from sklearn.metrics import accuracy_score
print('R squared value of the model: {:.2f}'.format(slr.score(x,y)*100))