#support vector regression on boston housing prices

import numpy as np
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

boston=datasets.load_boston()
x=boston.data
y=boston.target

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, random_state=42)

def evaluate_data(model=None):
    pred = model.predict(x_test)
    sum_of_squared_error=0
    for i in range(len(y_test)):
        err=(pred[i]-y_test[i])**2
        sum_of_squared_error+=err
    mean_squared_error=sum_of_squared_error/len(y_test)
    RMSE=np.sqrt(mean_squared_error)

    return RMSE

kernels=('linear','rbf')
RMSE_vec=[]

for index, kernel in enumerate(kernels):
    model=svm.SVR(kernel=kernel)
    model.fit(x_train,y_train)
    RMSE=evaluate_data(model)
    RMSE_vec.append(RMSE)
    print("RMSE={} obtained with kernel={}".format(RMSE,kernel))

